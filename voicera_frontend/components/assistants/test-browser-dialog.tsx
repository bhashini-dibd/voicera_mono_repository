"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Orb, type AgentState as OrbAgentState } from "@/components/ui/orb"
import type { Agent } from "@/lib/api"
import { Loader2, Mic, MicOff, PhoneOff, Radio } from "lucide-react"

interface TestBrowserDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  agent: Agent | null
  getAgentDisplayName: (agent: Agent) => string
}

const MIC_SAMPLE_RATE = 16000
const BHASHINI_TTS_SAMPLE_RATE = 44100

function rms(samples: Float32Array): number {
  if (!samples.length) return 0
  let sum = 0
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i]
    sum += s * s
  }
  return Math.sqrt(sum / samples.length)
}

function normalizeVolume(v: number): number {
  return Math.max(0, Math.min(1, v * 5))
}

function floatToInt16Base64(float32: Float32Array): string {
  const int16 = new Int16Array(float32.length)
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]))
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
  }
  const bytes = new Uint8Array(int16.buffer)
  let binary = ""
  const chunk = 0x8000
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk))
  }
  return btoa(binary)
}

function base64ToInt16(base64: string): Int16Array {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return new Int16Array(bytes.buffer)
}

function base64ToUint8(base64: string): Uint8Array {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return bytes
}

function int16ToFloat32(input: Int16Array): Float32Array {
  const out = new Float32Array(input.length)
  for (let i = 0; i < input.length; i++) {
    out[i] = input[i] / 0x8000
  }
  return out
}

function muLawByteToLinearSample(muLaw: number): number {
  const sample = (~muLaw) & 0xff
  const sign = sample & 0x80
  const exponent = (sample >> 4) & 0x07
  const mantissa = sample & 0x0f
  let pcm = ((mantissa << 3) + 0x84) << exponent
  pcm -= 0x84
  return sign ? -pcm : pcm
}

function muLawBase64ToFloat32(base64: string): Float32Array {
  const muLawBytes = base64ToUint8(base64)
  const out = new Float32Array(muLawBytes.length)
  for (let i = 0; i < muLawBytes.length; i++) {
    out[i] = muLawByteToLinearSample(muLawBytes[i]) / 0x8000
  }
  return out
}

function downsampleTo16k(input: Float32Array, inputRate: number): Float32Array {
  if (inputRate === MIC_SAMPLE_RATE) return input
  if (inputRate < MIC_SAMPLE_RATE) return input

  const ratio = inputRate / MIC_SAMPLE_RATE
  const newLength = Math.round(input.length / ratio)
  const result = new Float32Array(newLength)
  let offsetResult = 0
  let offsetBuffer = 0

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio)
    let accum = 0
    let count = 0
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < input.length; i++) {
      accum += input[i]
      count++
    }
    result[offsetResult] = count > 0 ? accum / count : 0
    offsetResult++
    offsetBuffer = nextOffsetBuffer
  }

  return result
}

function parseSampleRate(contentType?: string | null): number | null {
  if (!contentType) return null
  const match = contentType.match(/rate\s*=\s*(\d+)/i)
  if (!match) return null
  const rate = Number(match[1])
  return Number.isFinite(rate) && rate > 0 ? rate : null
}

function decodeIncomingAudio(payloadB64: string, contentType?: string | null): Float32Array {
  const normalizedType = (contentType || "").toLowerCase()
  if (normalizedType.includes("mulaw") || normalizedType.includes("mu-law")) {
    return muLawBase64ToFloat32(payloadB64)
  }
  return int16ToFloat32(base64ToInt16(payloadB64))
}

function isBhashiniAgent(agent: Agent | null): boolean {
  const tts = agent?.agent_config?.tts_model
  const haystack = `${tts?.name || ""} ${tts?.model || ""} ${tts?.speaker || ""}`.toLowerCase()
  return haystack.includes("bhashini")
}

async function requestMicrophoneStream(): Promise<MediaStream> {
  if (typeof window === "undefined") {
    throw new Error("Microphone access is only available in the browser")
  }

  const nav = window.navigator as Navigator & {
    mediaDevices?: {
      getUserMedia: (constraints: MediaStreamConstraints) => Promise<MediaStream>
    }
    webkitGetUserMedia?: (
      constraints: MediaStreamConstraints,
      success: (stream: MediaStream) => void,
      failure: (error: unknown) => void,
    ) => void
    mozGetUserMedia?: (
      constraints: MediaStreamConstraints,
      success: (stream: MediaStream) => void,
      failure: (error: unknown) => void,
    ) => void
    msGetUserMedia?: (
      constraints: MediaStreamConstraints,
      success: (stream: MediaStream) => void,
      failure: (error: unknown) => void,
    ) => void
    getUserMedia?: (
      constraints: MediaStreamConstraints,
      success: (stream: MediaStream) => void,
      failure: (error: unknown) => void,
    ) => void
  }

  if (nav.mediaDevices?.getUserMedia) {
    return nav.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    })
  }

  const legacyGetUserMedia = nav.getUserMedia || nav.webkitGetUserMedia || nav.mozGetUserMedia || nav.msGetUserMedia
  if (!legacyGetUserMedia) {
    throw new Error(
      "This browser does not expose microphone capture APIs. Try opening the test in a secure browser tab with microphone permission enabled.",
    )
  }

  return new Promise<MediaStream>((resolve, reject) => {
    legacyGetUserMedia.call(
      nav,
      {
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      },
      resolve,
      reject,
    )
  })
}

function getBrowserWsUrl(agentId: string): string {
  const explicitWsBase = process.env.NEXT_PUBLIC_JOHNAIC_WEBSOCKET_URL
  const serverBase = process.env.NEXT_PUBLIC_JOHNAIC_SERVER_URL

  let wsBase = explicitWsBase || ""
  if (!wsBase && serverBase) {
    wsBase = serverBase.replace(/^http:\/\//i, "ws://").replace(/^https:\/\//i, "wss://")
  }
  if (!wsBase) {
    wsBase = "ws://localhost:7860"
  }

  return `${wsBase.replace(/\/$/, "")}/browser/agent/${encodeURIComponent(agentId)}`
}

export function TestBrowserDialog({
  open,
  onOpenChange,
  agent,
  getAgentDisplayName,
}: TestBrowserDialogProps) {
  const [isConnecting, setIsConnecting] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [isMuted, setIsMuted] = useState(true)
  const [error, setError] = useState("")
  const [orbState, setOrbState] = useState<OrbAgentState>(null)
  const [transcripts, setTranscripts] = useState<Array<{ id: string; role: "user" | "assistant"; content: string }>>([])
  const [latencyEvents, setLatencyEvents] = useState<Array<{
    id: string
    service: string
    metric: string
    value_ms: number
    stage?: string | null
    details?: Record<string, any>
  }>>([])
  const [latencySummary, setLatencySummary] = useState<Record<string, any> | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const isMutedRef = useRef(false)
  const audioContextRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null)
  const playbackTimeRef = useRef(0)
  const sessionIdRef = useRef("")
  const inputVolumeRef = useRef(0)
  const outputVolumeRef = useRef(0)
  const lastOutputAtRef = useRef(0)
  const transcriptViewportRef = useRef<HTMLDivElement | null>(null)

  const sessionLabel = useMemo(() => {
    if (isConnected) return "Live"
    if (isConnecting) return "Connecting"
    return "Idle"
  }, [isConnected, isConnecting])

  const microphoneSupported = useMemo(() => {
    if (typeof window === "undefined") return false
    const nav = window.navigator as Navigator & {
      mediaDevices?: { getUserMedia?: unknown }
      webkitGetUserMedia?: unknown
      mozGetUserMedia?: unknown
      msGetUserMedia?: unknown
      getUserMedia?: unknown
    }
    return Boolean(
      nav.mediaDevices?.getUserMedia ||
      nav.getUserMedia ||
      nav.webkitGetUserMedia ||
      nav.mozGetUserMedia ||
      nav.msGetUserMedia,
    )
  }, [])

  const teardown = useCallback(async () => {
    const ws = wsRef.current
    wsRef.current = null
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close(1000, "client-end")
    } else if (ws) {
      ws.close()
    }

    if (processorNodeRef.current) {
      processorNodeRef.current.disconnect()
      processorNodeRef.current.onaudioprocess = null
      processorNodeRef.current = null
    }

    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect()
      sourceNodeRef.current = null
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop())
      mediaStreamRef.current = null
    }

    const ctx = audioContextRef.current
    audioContextRef.current = null
    if (ctx && ctx.state !== "closed") {
      await ctx.close()
    }

    playbackTimeRef.current = 0
    sessionIdRef.current = ""
    isMutedRef.current = false
    inputVolumeRef.current = 0
    outputVolumeRef.current = 0
    lastOutputAtRef.current = 0
    setIsConnected(false)
    setIsConnecting(false)
    setOrbState(null)
    setTranscripts([])
  }, [])

  const stopSession = useCallback(async () => {
    await teardown()
  }, [teardown])

  const handleIncomingAudio = (
    payloadB64: string,
    sampleRate: number | null = BHASHINI_TTS_SAMPLE_RATE,
    contentType?: string | null,
  ) => {
    const ctx = audioContextRef.current
    if (!ctx) return

    const normalizedContentType = (contentType || "").toLowerCase()
    const float32 = decodeIncomingAudio(payloadB64, contentType)
    if (!float32.length) return
    outputVolumeRef.current = normalizeVolume(rms(float32))
    lastOutputAtRef.current = performance.now()
    const bufferSampleRate =
      sampleRate ||
      (normalizedContentType.includes("mulaw") || normalizedContentType.includes("mu-law") ? 8000 : null) ||
      (isBhashiniAgent(agent) ? BHASHINI_TTS_SAMPLE_RATE : MIC_SAMPLE_RATE)
    const buffer = ctx.createBuffer(1, float32.length, bufferSampleRate)
    buffer.copyToChannel(float32, 0)

    const source = ctx.createBufferSource()
    source.buffer = buffer
    source.connect(ctx.destination)

    const now = ctx.currentTime + 0.02
    const startAt = Math.max(now, playbackTimeRef.current)
    source.start(startAt)
    playbackTimeRef.current = startAt + buffer.duration
  }

  const startSession = async () => {
    if (!agent?.agent_id || isConnecting || isConnected) return

    setError("")
    setIsConnecting(true)
    sessionIdRef.current = `browser-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    setTranscripts([])
    setLatencyEvents([])
    setLatencySummary(null)

    try {
      const stream = await requestMicrophoneStream()
      mediaStreamRef.current = stream

      const ctx = new AudioContext()
      audioContextRef.current = ctx
      await ctx.resume()

      const ws = new WebSocket(getBrowserWsUrl(agent.agent_id))
      wsRef.current = ws

      ws.onopen = () => {
        // Start muted by default; user must click Unmute to send mic audio.
        isMutedRef.current = true
        setIsMuted(true)

        ws.send(
          JSON.stringify({
            event: "start",
            start: {
              callSid: sessionIdRef.current,
              streamSid: sessionIdRef.current,
            },
          }),
        )

        const src = ctx.createMediaStreamSource(stream)
        const processor = ctx.createScriptProcessor(1024, 1, 1)
        sourceNodeRef.current = src
        processorNodeRef.current = processor

        processor.onaudioprocess = (ev) => {
          const socket = wsRef.current
          if (!socket || socket.readyState !== WebSocket.OPEN || isMutedRef.current) return

          const channelData = ev.inputBuffer.getChannelData(0)
          const downsampled = downsampleTo16k(channelData, ctx.sampleRate)
          if (!downsampled.length) return
          inputVolumeRef.current = normalizeVolume(rms(downsampled))

          const payload = floatToInt16Base64(downsampled)
          socket.send(
            JSON.stringify({
              event: "media",
              media: {
                contentType: "audio/x-l16",
                sampleRate: MIC_SAMPLE_RATE,
                payload,
              },
            }),
          )
        }

        src.connect(processor)
        processor.connect(ctx.destination)
        setIsConnecting(false)
        setIsConnected(true)
        setOrbState("listening")
      }

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data)
          if (msg?.event === "playAudio" && msg?.media?.payload) {
            const mediaSampleRate = Number(msg?.media?.sampleRate)
            handleIncomingAudio(
              msg.media.payload,
              Number.isFinite(mediaSampleRate) && mediaSampleRate > 0
                ? mediaSampleRate
                : parseSampleRate(msg?.media?.contentType),
              msg?.media?.contentType,
            )
          } else if (msg?.event === "transcript" && msg?.content) {
            const role = msg.role === "assistant" ? "assistant" : "user"
            const content = String(msg.content).trim()
            if (!content) return
            setTranscripts((prev) => {
              const last = prev[prev.length - 1]
              if (last && last.role === role && last.content === content) return prev
              const next = {
                id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
                role,
                content,
              }
              const merged = [...prev, next]
              return merged.length > 120 ? merged.slice(merged.length - 120) : merged
            })
          } else if (msg?.event === "latency" && msg?.metric) {
            const metricValue = Number(msg?.value_ms)
            const entry = {
              id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
              service: String(msg.service || "unknown"),
              metric: String(msg.metric || "unknown"),
              value_ms: Number.isFinite(metricValue) ? metricValue : 0,
              stage: msg.stage ? String(msg.stage) : null,
              details: (msg.details && typeof msg.details === "object") ? msg.details : undefined,
            }
            setLatencyEvents((prev) => {
              const merged = [...prev, entry]
              return merged.length > 80 ? merged.slice(merged.length - 80) : merged
            })
          } else if (msg?.event === "latency_summary" && msg?.summary) {
            setLatencySummary(msg.summary)
          }
        } catch {
          // Ignore malformed frames.
        }
      }

      ws.onerror = () => {
        setError("WebSocket connection failed")
      }

      ws.onclose = () => {
        void teardown()
      }
    } catch (err) {
      await teardown()
      setError(err instanceof Error ? err.message : "Failed to start browser test")
    }
  }

  useEffect(() => {
    const id = window.setInterval(() => {
      if (!isConnected) {
        setOrbState(null)
        return
      }

      if (performance.now() - lastOutputAtRef.current > 220) {
        outputVolumeRef.current *= 0.68
      }
      inputVolumeRef.current *= 0.9

      if (outputVolumeRef.current > 0.08) {
        setOrbState("talking")
      } else {
        setOrbState("listening")
      }
    }, 60)

    return () => window.clearInterval(id)
  }, [isConnected])

  useEffect(() => {
    return () => {
      void stopSession()
    }
  }, [stopSession])

  useEffect(() => {
    const viewport = transcriptViewportRef.current
    if (!viewport) return
    viewport.scrollTop = viewport.scrollHeight
  }, [transcripts])

  const handleDialogOpenChange = (nextOpen: boolean) => {
    if (!nextOpen) {
      void stopSession()
      setError("")
      setIsMuted(true)
      isMutedRef.current = true
    }
    onOpenChange(nextOpen)
  }

  return (
    <Dialog open={open} onOpenChange={handleDialogOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{agent ? getAgentDisplayName(agent) : "Test Agent on Browser"}</DialogTitle>
          <DialogDescription>
            Talk directly with your agent in real time from this browser.
          </DialogDescription>
        </DialogHeader>

        <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
          <div className="mb-3 flex justify-center">
            <div className="h-36 w-36 rounded-full bg-slate-100 p-1 shadow-[inset_0_2px_8px_rgba(0,0,0,0.08)]">
              <div className="h-full w-full overflow-hidden rounded-full bg-white shadow-[inset_0_0_12px_rgba(0,0,0,0.05)]">
                <Orb
                  agentState={orbState}
                  volumeMode="manual"
                  inputVolumeRef={inputVolumeRef}
                  outputVolumeRef={outputVolumeRef}
                  colors={["#CADCFC", "#A0B9D1"]}
                  className="h-full w-full"
                />
              </div>
            </div>
          </div>

          <div
            ref={transcriptViewportRef}
            className="mb-3 max-h-52 overflow-y-auto rounded-lg border border-slate-200 bg-white p-3"
          >
            {transcripts.length > 0 && (
              <div className="space-y-2">
                {transcripts.map((t) => (
                  <div key={t.id} className={`flex ${t.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`max-w-[90%] rounded-2xl px-3 py-2 text-sm ${t.role === "user"
                        ? "bg-slate-900 text-white"
                        : "bg-slate-100 text-slate-900"
                        }`}
                    >
                      <p className="mb-1 text-[10px] font-medium uppercase tracking-wide opacity-70">
                        {t.role === "user" ? "You" : "Agent"}
                      </p>
                      <p className="whitespace-pre-wrap">{t.content}</p>
                    </div>
                  </div>
                ))}
                {transcripts.length === 0 && (
                  <div className="flex justify-center items-center h-full">
                    <p className="text-sm text-slate-500">No transcripts yet</p>
                  </div>
                )}

              </div>
            )}
          </div>

          <div className="flex items-center">
            <div className="flex items-center gap-2 text-sm text-slate-700">
              <Radio className={`h-4 w-4 ${isConnected ? "text-green-600" : "text-slate-400"}`} />
              <span>Status: {sessionLabel}</span>
            </div>
          </div>
          {!microphoneSupported && (
            <p className="mt-2 text-xs text-amber-700">
              This browser environment does not expose microphone capture APIs, so the browser test cannot start here.
            </p>
          )}
          {error && <p className="mt-3 text-xs text-red-600">{error}</p>}
        </div>

        {(latencyEvents.length > 0 || latencySummary) && (
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-900">Latency Feed</h3>
              <span className="text-[11px] uppercase tracking-wide text-slate-500">
                {latencyEvents.length} events
              </span>
            </div>
            {latencySummary && (
              <div className="mb-4 grid grid-cols-1 gap-2 rounded-lg bg-slate-50 p-3 text-xs text-slate-700 md:grid-cols-2">
                <div>Call total: {latencySummary.call?.run_bot_total_ms ?? "N/A"} ms</div>
                <div>Service init: {latencySummary.call?.service_initialization_ms ?? "N/A"} ms</div>
                <div>STT TTFT: {latencySummary.stt?.first_transcript_ms ?? "N/A"} ms</div>
                <div>LLM TTFT: {latencySummary.llm?.ttft_ms ?? "N/A"} ms</div>
                <div>TTS TTFT: {latencySummary.tts?.ttft_ms ?? "N/A"} ms</div>
                <div>Orchestrator gap: {latencySummary.orchestrator?.user_transcript_to_first_tts_audio_ms ?? "N/A"} ms</div>
              </div>
            )}
            <div className="max-h-56 overflow-y-auto rounded-lg border border-slate-200">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-50 text-slate-500">
                  <tr>
                    <th className="px-3 py-2">Service</th>
                    <th className="px-3 py-2">Metric</th>
                    <th className="px-3 py-2">Value</th>
                    <th className="px-3 py-2">Stage</th>
                  </tr>
                </thead>
                <tbody>
                  {latencyEvents.slice().reverse().map((event) => (
                    <tr key={event.id} className="border-t border-slate-100">
                      <td className="px-3 py-2 font-medium text-slate-800">{event.service}</td>
                      <td className="px-3 py-2 text-slate-700">{event.metric}</td>
                      <td className="px-3 py-2 text-slate-700">{event.value_ms.toFixed(1)} ms</td>
                      <td className="px-3 py-2 text-slate-500">{event.stage || "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <DialogFooter className="gap-2 justify-center sm:justify-center">
          {!isConnected ? (
            <Button type="button" onClick={startSession} disabled={isConnecting || !agent?.agent_id || !microphoneSupported}>
              {isConnecting ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Mic className="h-4 w-4 mr-2" />}
              {isConnecting ? "Connecting..." : "Start Browser Test"}
            </Button>
          ) : (
            <Button type="button" variant="destructive" onClick={stopSession}>
              <PhoneOff className="h-4 w-4 mr-2" />
              End Session
            </Button>
          )}
          <Button
            type="button"
            variant="outline"
            disabled={!isConnected}
            onClick={() =>
              setIsMuted((v) => {
                const next = !v
                isMutedRef.current = next
                return next
              })
            }
            className={
              isMuted
                ? "border-red-300 bg-red-50 text-red-700 hover:bg-red-100 hover:text-red-800"
                : "border-slate-200 bg-white text-slate-900 hover:bg-slate-50"
            }
          >
            {isMuted ? <MicOff className="h-4 w-4 mr-1" /> : <Mic className="h-4 w-4 mr-1" />}
            {isMuted ? "Unmute" : "Mute"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
