"use client"

import type { ComponentType } from "react"
import { useEffect, useMemo, useState } from "react"
import {
  Activity,
  AlertTriangle,
  Bot,
  Clock3,
  Loader2,
  Mic,
  RefreshCw,
  Sparkles,
  TimerReset,
  Volume2,
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { getMeetings, type Meeting } from "@/lib/api"

type NumericMetricMap = Record<string, number | null>

const POLL_INTERVAL_MS = 15000
const RECENT_CALL_LIMIT = 8

function normalizeKey(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "")
}

function collectNumericMetrics(value: unknown, prefix = ""): Array<{ key: string; value: number }> {
  if (typeof value === "number" && Number.isFinite(value)) {
    return [{ key: prefix || "value", value }]
  }

  if (!value || typeof value !== "object") return []

  if (Array.isArray(value)) {
    return value.flatMap((entry, index) => collectNumericMetrics(entry, `${prefix}[${index}]`))
  }

  return Object.entries(value as Record<string, unknown>).flatMap(([key, nested]) => {
    const nextPrefix = prefix ? `${prefix}.${key}` : key
    return collectNumericMetrics(nested, nextPrefix)
  })
}

function findEventMetric(
  summary: Record<string, any> | undefined,
  service: string,
  metric: string,
): number | null {
  const events = summary?.events
  if (!Array.isArray(events)) return null

  const normalizedService = normalizeKey(service)
  const normalizedMetric = normalizeKey(metric)

  for (let i = events.length - 1; i >= 0; i -= 1) {
    const entry = events[i] as Record<string, any>
    if (!entry) continue
    if (normalizeKey(String(entry.service || "")) !== normalizedService) continue
    if (normalizeKey(String(entry.metric || "")) !== normalizedMetric) continue
    if (typeof entry.value_ms === "number" && Number.isFinite(entry.value_ms)) {
      return entry.value_ms
    }
  }

  return null
}

function getMetric(
  summary: Record<string, any> | undefined,
  candidates: string[],
  service?: string,
): number | null {
  if (!summary) return null
  const flattened = collectNumericMetrics(summary)

  for (const candidate of candidates) {
    const normalizedCandidate = normalizeKey(candidate)
    const match = flattened.find(({ key }) => {
      const normalizedKey = normalizeKey(key)
      return (
        normalizedKey === normalizedCandidate ||
        normalizedKey.endsWith(normalizedCandidate) ||
        normalizedKey.includes(normalizedCandidate)
      )
    })
    if (match) return match.value
  }

  if (service) {
    for (const candidate of candidates) {
      const eventMetric = findEventMetric(summary, service, candidate)
      if (eventMetric !== null) return eventMetric
    }
  }

  return null
}

function formatMs(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A"
  return `${Math.round(value)} ms`
}

function formatRelativeTime(value?: string) {
  if (!value) return "N/A"
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return "N/A"
  return parsed.toLocaleTimeString()
}

function getMeetingLabel(meeting: Meeting) {
  return meeting.agent_type || meeting.agent_category || meeting.meeting_id
}

function getMeetingDate(meeting: Meeting) {
  const anyMeeting = meeting as Record<string, any>
  const raw =
    anyMeeting.end_time_utc ||
    anyMeeting.start_time_utc ||
    anyMeeting.created_at ||
    anyMeeting.updated_at ||
    anyMeeting.timestamp_utc ||
    anyMeeting.timestamp ||
    ""

  if (!raw) return null
  const parsed = new Date(raw)
  if (Number.isNaN(parsed.getTime())) return null
  return parsed
}

function getMeetingUpdatedAt(meeting: Meeting) {
  const anyMeeting = meeting as Record<string, any>
  return (
    anyMeeting.updated_at ||
    anyMeeting.updatedAt ||
    anyMeeting.created_at ||
    anyMeeting.createdAt ||
    anyMeeting.timestamp ||
    anyMeeting.timestamp_utc ||
    ""
  )
}

function buildLatencyMap(meeting: Meeting): NumericMetricMap {
  const summary = meeting.latency_summary || {}
  return {
    pipeline_build_ms: getMetric(summary, ["pipeline_build_ms"]),
    client_connected_after_run_bot_ms: getMetric(summary, ["client_connected_after_run_bot_ms"]),
    run_bot_total_ms: getMetric(summary, ["run_bot_total_ms"]),
    stt_ws_open_ms: getMetric(summary, ["stt_ws_open_ms", "ws_open_ms"], "stt"),
    stt_first_transcript_ms: getMetric(
      summary,
      [
        "stt_first_transcript_ms",
        "first_transcript_ms",
        "speech_to_first_transcript_ms",
        "voice_to_first_transcript_ms",
      ],
      "stt",
    ),
    llm_ttft_ms: getMetric(summary, ["llm_ttft_ms", "llm_first_token_ms", "first_chunk_ms"], "llm"),
    llm_stream_complete_ms: getMetric(summary, ["llm_stream_complete_ms"], "llm"),
    tts_ttft_ms: getMetric(summary, ["tts_ttft_ms", "tts_started_to_first_audio_ms"], "tts"),
    stt_segment_duration_ms: getMetric(summary, ["segment_duration_ms"], "stt"),
  }
}

function mean(values: Array<number | null | undefined>) {
  const numeric = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v))
  if (numeric.length === 0) return null
  return numeric.reduce((sum, value) => sum + value, 0) / numeric.length
}

function sum(values: Array<number | null | undefined>) {
  const numeric = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v))
  if (numeric.length === 0) return null
  return numeric.reduce((total, value) => total + value, 0)
}

function formatPercent(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A"
  return `${Math.round(value)}%`
}

function getServiceBreakdown(entry: {
  metrics: NumericMetricMap
}) {
  const setup = sum([
    entry.metrics.pipeline_build_ms,
    entry.metrics.client_connected_after_run_bot_ms,
  ])
  const stt = entry.metrics.stt_first_transcript_ms ?? null
  const llm = entry.metrics.llm_ttft_ms ?? null
  const tts = entry.metrics.tts_ttft_ms ?? null
  const endToEnd = entry.metrics.run_bot_total_ms ?? null
  const measured = sum([setup, stt, llm, tts])
  const gap =
    typeof endToEnd === "number" && typeof measured === "number"
      ? Math.max(endToEnd - measured, 0)
      : null

  const services = [
    { key: "setup", label: "Orchestrator setup", value: setup },
    { key: "stt", label: "STT", value: stt },
    { key: "llm", label: "LLM", value: llm },
    { key: "tts", label: "TTS", value: tts },
  ].filter((item) => typeof item.value === "number" && Number.isFinite(item.value)) as Array<{
    key: string
    label: string
    value: number
  }>

  const bottleneck = services.reduce<{
    key: string
    label: string
    value: number
  } | null>((current, candidate) => {
    if (!current || candidate.value > current.value) return candidate
    return current
  }, null)

  const shareBase = typeof measured === "number" && measured > 0 ? measured : null
  const shares = services.map((item) => ({
    ...item,
    share: shareBase ? (item.value / shareBase) * 100 : null,
  }))

  return {
    setup,
    stt,
    llm,
    tts,
    endToEnd,
    measured,
    gap,
    shares,
    bottleneck,
    gapShare:
      typeof gap === "number" && typeof endToEnd === "number" && endToEnd > 0
        ? (gap / endToEnd) * 100
        : null,
  }
}

function getDiagnosis(breakdown: ReturnType<typeof getServiceBreakdown>) {
  const labels = breakdown.shares.map((item) => `${item.label}: ${formatMs(item.value)}`).join(", ")
  if (breakdown.llm === null && breakdown.stt !== null) {
    return {
      title: "LLM telemetry still missing",
      text:
        "The call shows a strong STT signal, but LLM TTFT is not yet captured. Fix the LLM telemetry path first so we can compare the model stream against STT and TTS.",
    }
  }

  if (typeof breakdown.gapShare === "number" && breakdown.gapShare > 25) {
    return {
      title: "Large untracked gap",
      text:
        "A sizable part of end-to-end time is not explained by the measured service timings. Focus on queueing, network latency, or any hop that is not being instrumented yet.",
    }
  }

  switch (breakdown.bottleneck?.key) {
    case "stt":
      return {
        title: "STT is the bottleneck",
        text:
          "ASR is taking the most time. Focus on websocket connect time, VAD segmenting, pre-roll buffering, and the model endpoint itself.",
      }
    case "llm":
      return {
        title: "LLM is the bottleneck",
        text:
          "The first token is the slowest part. Reduce prompt size, check streaming start latency, and confirm the provider is returning tokens immediately.",
      }
    case "tts":
      return {
        title: "TTS is the bottleneck",
        text:
          "Text-to-speech is dominating the turn. Focus on model startup, chunk generation, and the first-audio path.",
      }
    case "setup":
      return {
        title: "Orchestrator setup is the bottleneck",
        text:
          "Most of the delay happens before the user turn is even processed. Focus on pipeline build, service initialization, and connection handoff.",
      }
    default:
      return {
        title: "Balanced or incomplete data",
        text: labels || "No service timings were captured for this call.",
      }
  }
}

function MetricCard({
  title,
  icon: Icon,
  value,
  description,
}: {
  title: string
  icon: ComponentType<{ className?: string }>
  value: string
  description: string
}) {
  return (
    <Card className="border-slate-200 shadow-sm">
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-1">
            <p className="text-sm font-medium text-slate-600">{title}</p>
            <p className="text-2xl font-semibold tracking-tight text-slate-900">{value}</p>
            <p className="text-sm text-slate-500">{description}</p>
          </div>
          <div className="rounded-xl border bg-slate-50 p-2 text-slate-700">
            <Icon className="h-5 w-5" />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default function TelemetryPage() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [meetings, setMeetings] = useState<Meeting[]>([])
  const [lastUpdatedAt, setLastUpdatedAt] = useState<string | null>(null)
  const [dateFrom, setDateFrom] = useState("")
  const [dateTo, setDateTo] = useState("")
  const [fromNumber, setFromNumber] = useState("")
  const [sessionId, setSessionId] = useState("")

  useEffect(() => {
    let mounted = true
    let timer: ReturnType<typeof setInterval> | null = null

    const fetchTelemetry = async () => {
      try {
        const data = await getMeetings()
        if (!mounted) return
        setMeetings(data)
        setError(null)
        setLastUpdatedAt(new Date().toISOString())
      } catch (err) {
        if (!mounted) return
        setError(err instanceof Error ? err.message : "Failed to load telemetry")
      } finally {
        if (mounted) setLoading(false)
      }
    }

    fetchTelemetry()
    timer = setInterval(fetchTelemetry, POLL_INTERVAL_MS)

    return () => {
      mounted = false
      if (timer) clearInterval(timer)
    }
  }, [])

  const clearFilters = () => {
    setDateFrom("")
    setDateTo("")
    setFromNumber("")
    setSessionId("")
  }

  const filteredMeetings = useMemo(() => {
    const fromQuery = fromNumber.trim().toLowerCase()
    const sessionQuery = sessionId.trim().toLowerCase()
    const start = dateFrom ? new Date(`${dateFrom}T00:00:00`) : null
    const end = dateTo ? new Date(`${dateTo}T23:59:59.999`) : null

    return meetings.filter((meeting) => {
      if (fromQuery) {
        const candidate = String(meeting.from_number || "").toLowerCase()
        if (!candidate.includes(fromQuery)) return false
      }

      if (sessionQuery) {
        const meetingId = String(meeting.meeting_id || "").toLowerCase()
        const callSid = String((meeting as Record<string, any>).call_sid || "").toLowerCase()
        if (!meetingId.includes(sessionQuery) && !callSid.includes(sessionQuery)) return false
      }

      const meetingDate = getMeetingDate(meeting)
      if (start && meetingDate && meetingDate < start) return false
      if (end && meetingDate && meetingDate > end) return false

      return true
    })
  }, [meetings, dateFrom, dateTo, fromNumber, sessionId])

  const latencyMeetings = useMemo(() => {
    return filteredMeetings
      .map((meeting) => ({
        meeting,
        metrics: buildLatencyMap(meeting),
        updatedAt: getMeetingUpdatedAt(meeting),
      }))
      .filter((entry) => Object.values(entry.metrics).some((value) => typeof value === "number"))
      .slice(0, RECENT_CALL_LIMIT)
  }, [filteredMeetings])

  const averages = useMemo(() => {
    return {
      stt_first_transcript_ms: mean(latencyMeetings.map((entry) => entry.metrics.stt_first_transcript_ms)),
      llm_ttft_ms: mean(latencyMeetings.map((entry) => entry.metrics.llm_ttft_ms)),
      tts_ttft_ms: mean(latencyMeetings.map((entry) => entry.metrics.tts_ttft_ms)),
      orchestrator_ms: mean(
        latencyMeetings.map(
          (entry) =>
            (entry.metrics.pipeline_build_ms ?? 0) +
            (entry.metrics.client_connected_after_run_bot_ms ?? 0),
        ),
      ),
      end_to_end_ms: mean(latencyMeetings.map((entry) => entry.metrics.run_bot_total_ms)),
    }
  }, [latencyMeetings])

  const latestMeeting = latencyMeetings[0]
  const latestBreakdown = useMemo(() => {
    if (!latestMeeting) return null
    return getServiceBreakdown(latestMeeting)
  }, [latestMeeting])

  const coverage = useMemo(() => {
    const total = latencyMeetings.length || 1
    const count = (selector: (entry: (typeof latencyMeetings)[number]) => boolean) =>
      latencyMeetings.filter(selector).length

    return {
      stt: count((entry) => typeof entry.metrics.stt_first_transcript_ms === "number"),
      llm: count((entry) => typeof entry.metrics.llm_ttft_ms === "number"),
      tts: count((entry) => typeof entry.metrics.tts_ttft_ms === "number"),
      setup: count(
        (entry) =>
          typeof entry.metrics.pipeline_build_ms === "number" ||
          typeof entry.metrics.client_connected_after_run_bot_ms === "number",
      ),
      total,
    }
  }, [latencyMeetings])

  const latestDiagnosis = useMemo(() => {
    if (!latestBreakdown) return null
    return getDiagnosis(latestBreakdown)
  }, [latestBreakdown])

  const lastUpdatedLabel = lastUpdatedAt ? formatRelativeTime(lastUpdatedAt) : "N/A"
  const hasActiveFilters = Boolean(dateFrom || dateTo || fromNumber || sessionId)

  return (
    <div className="flex h-screen flex-col bg-gradient-to-br from-slate-50 via-white to-slate-100">
      <header className="border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="flex h-16 items-center justify-between gap-4 px-6">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-slate-700" />
            <h1 className="text-lg font-semibold text-slate-900">Telemetry</h1>
          </div>
          <Badge variant="outline" className="gap-2 rounded-full px-3 py-1 text-xs">
            <RefreshCw className="h-3.5 w-3.5" />
            Last update: {lastUpdatedLabel}
          </Badge>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto p-4 md:p-6">
        <div className="mx-auto max-w-7xl space-y-5">
          <Card className="border-slate-200 bg-white/80 shadow-sm">
            <CardHeader className="space-y-2">
              <CardTitle className="text-xl">Call Latency Telemetry</CardTitle>
              <CardDescription className="max-w-3xl">
                This dashboard focuses on the latency we control: STT first transcript time, LLM TTFT,
                TTS TTFT, and orchestrator overhead. End-to-end means the full bot runtime from run start
                to run finish. New calls now measure STT from voice start to first transcript; older calls may
                still reflect the previous call-start proxy.
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="border-slate-200 bg-white/80 shadow-sm">
            <CardContent className="grid grid-cols-1 gap-3 p-4 md:grid-cols-2 xl:grid-cols-5">
              <div className="space-y-1">
                <label className="text-xs font-medium uppercase tracking-wide text-slate-500">Date from</label>
                <Input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium uppercase tracking-wide text-slate-500">Date to</label>
                <Input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium uppercase tracking-wide text-slate-500">From number</label>
                <Input
                  placeholder="+91..."
                  value={fromNumber}
                  onChange={(e) => setFromNumber(e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium uppercase tracking-wide text-slate-500">Session ID</label>
                <Input
                  placeholder="meeting_id / call_sid"
                  value={sessionId}
                  onChange={(e) => setSessionId(e.target.value)}
                />
              </div>
              <div className="flex items-end">
                <Button variant="outline" className="w-full" onClick={clearFilters} disabled={!hasActiveFilters}>
                  Clear filters
                </Button>
              </div>
            </CardContent>
          </Card>

          {loading && (
            <Card className="border-slate-200">
              <CardContent className="flex items-center justify-center gap-2 py-10 text-slate-600">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading telemetry...
              </CardContent>
            </Card>
          )}

          {!loading && error && (
            <Card className="border-red-200 bg-red-50/60">
              <CardContent className="flex items-start gap-3 py-6 text-sm text-red-700">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                <div>
                  <p className="font-medium">Telemetry feed unavailable</p>
                  <p>{error}</p>
                </div>
              </CardContent>
            </Card>
          )}

          {!loading && !error && filteredMeetings.length === 0 && (
            <Card className="border-slate-200">
              <CardContent className="py-10 text-center">
                <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full border bg-slate-50 text-slate-600">
                  <Sparkles className="h-5 w-5" />
                </div>
                <p className="text-base font-medium text-slate-900">
                  {hasActiveFilters ? "No calls match these filters" : "No latency summaries yet"}
                </p>
                <p className="mt-2 text-sm text-slate-500">
                  {hasActiveFilters
                    ? "Try widening the date range or clearing the session/from-number filters."
                    : "Start a call and we&apos;ll populate this page with STT, LLM, TTS, and orchestrator timings."}
                </p>
              </CardContent>
            </Card>
          )}

          {!loading && !error && latencyMeetings.length > 0 && (
            <>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5">
                <MetricCard
                  title="STT first transcript"
                  icon={Mic}
                  value={formatMs(averages.stt_first_transcript_ms)}
                  description="Voice start to first transcript"
                />
                <MetricCard
                  title="LLM TTFT"
                  icon={Bot}
                  value={formatMs(averages.llm_ttft_ms)}
                  description="Average time to first assistant token"
                />
                <MetricCard
                  title="TTS TTFT"
                  icon={Volume2}
                  value={formatMs(averages.tts_ttft_ms)}
                  description="Average time to first synthesized audio"
                />
                <MetricCard
                  title="Orchestrator setup"
                  icon={Clock3}
                  value={formatMs(averages.orchestrator_ms)}
                  description="Pipeline build + client connect"
                />
                <MetricCard
                  title="End-to-end"
                  icon={TimerReset}
                  value={formatMs(averages.end_to_end_ms)}
                  description="Total bot runtime from run start to run finish"
                />
              </div>

              <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
                <Card className="border-slate-200 bg-white/80 shadow-sm xl:col-span-2">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base">Latest call diagnosis</CardTitle>
                    <CardDescription>
                      This summarizes where time is going in the most recent call and what to fix first.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {!latestMeeting || !latestBreakdown || !latestDiagnosis ? (
                      <p className="text-sm text-slate-500">No call selected.</p>
                    ) : (
                      <>
                        <div className="rounded-2xl border bg-slate-50 p-4">
                          <div className="flex flex-wrap items-start justify-between gap-3">
                            <div>
                              <div className="text-sm font-semibold text-slate-900">
                                {getMeetingLabel(latestMeeting.meeting)}
                              </div>
                              <div className="mt-1 text-xs text-slate-500">{latestMeeting.meeting.meeting_id}</div>
                            </div>
                            <Badge variant="outline" className="rounded-full">
                              {latestBreakdown.bottleneck
                                ? `${latestBreakdown.bottleneck.label} dominates`
                                : "Incomplete data"}
                            </Badge>
                          </div>

                          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                            {latestBreakdown.shares.map((item) => (
                              <div key={item.key} className="rounded-xl border bg-white p-3">
                                <div className="flex items-center justify-between gap-2">
                                  <p className="text-sm font-medium text-slate-900">{item.label}</p>
                                  <Badge variant="outline" className="rounded-full">
                                    {formatMs(item.value)}
                                  </Badge>
                                </div>
                                <div className="mt-2 h-2 rounded-full bg-slate-100">
                                  <div
                                    className="h-2 rounded-full bg-slate-900"
                                    style={{ width: `${Math.min(item.share ?? 0, 100)}%` }}
                                  />
                                </div>
                                <p className="mt-2 text-xs text-slate-500">
                                  {formatPercent(item.share)} of measured latency
                                </p>
                              </div>
                            ))}
                          </div>

                          <div className="mt-4 grid gap-3 md:grid-cols-2">
                            <div className="rounded-xl border bg-white p-3">
                              <p className="text-xs font-medium uppercase tracking-wide text-slate-500">What this means</p>
                              <p className="mt-2 text-sm font-medium text-slate-900">{latestDiagnosis.title}</p>
                              <p className="mt-1 text-sm text-slate-600">{latestDiagnosis.text}</p>
                            </div>
                            <div className="rounded-xl border bg-white p-3">
                              <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Measured vs end-to-end</p>
                              <div className="mt-2 space-y-2 text-sm text-slate-700">
                                <div className="flex items-center justify-between">
                                  <span>Measured service time</span>
                                  <span className="font-medium">{formatMs(latestBreakdown.measured)}</span>
                                </div>
                                <div className="flex items-center justify-between">
                                  <span>Unexplained gap</span>
                                  <span className="font-medium">{formatMs(latestBreakdown.gap)}</span>
                                </div>
                                <div className="flex items-center justify-between">
                                  <span>Gap share</span>
                                  <span className="font-medium">{formatPercent(latestBreakdown.gapShare)}</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </>
                    )}
                  </CardContent>
                </Card>

                <Card className="border-slate-200 bg-white/80 shadow-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base">Signal quality</CardTitle>
                    <CardDescription>How much of the recent data is actually instrumented.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {[
                      { label: "STT", value: coverage.stt },
                      { label: "LLM", value: coverage.llm },
                      { label: "TTS", value: coverage.tts },
                      { label: "Setup", value: coverage.setup },
                    ].map((item) => (
                      <div key={item.label} className="rounded-xl border p-3">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-slate-900">{item.label}</span>
                          <Badge variant="outline" className="rounded-full">
                            {item.value}/{coverage.total}
                          </Badge>
                        </div>
                        <div className="mt-2 h-2 rounded-full bg-slate-100">
                          <div
                            className="h-2 rounded-full bg-slate-700"
                            style={{ width: `${Math.round((item.value / coverage.total) * 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}

                    <div className="rounded-xl border bg-slate-50 p-3 text-sm text-slate-600">
                      LLM TTFT missing here usually means the callback path is not wired for that provider yet.
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Card className="border-slate-200 bg-white/80 shadow-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Recent calls</CardTitle>
                  <CardDescription>
                    The latest calls with the latency summary captured by the orchestrator.
                  </CardDescription>
                </CardHeader>
                <CardContent className="overflow-hidden">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="border-b bg-slate-50 text-slate-600">
                        <tr>
                          <th className="px-3 py-2 text-left font-medium">Call</th>
                          <th className="px-3 py-2 text-left font-medium">From</th>
                          <th className="px-3 py-2 text-left font-medium">Session</th>
                          <th className="px-3 py-2 text-left font-medium">Bottleneck</th>
                          <th className="px-3 py-2 text-left font-medium">STT</th>
                          <th className="px-3 py-2 text-left font-medium">LLM</th>
                          <th className="px-3 py-2 text-left font-medium">TTS</th>
                          <th className="px-3 py-2 text-left font-medium">End-to-end</th>
                          <th className="px-3 py-2 text-left font-medium">Updated</th>
                        </tr>
                      </thead>
                      <tbody>
                        {latencyMeetings.map((entry) => {
                          const breakdown = getServiceBreakdown(entry)
                          return (
                            <tr key={entry.meeting.meeting_id} className="border-b last:border-b-0">
                              <td className="px-3 py-3 align-top">
                                <div className="font-medium text-slate-900">{getMeetingLabel(entry.meeting)}</div>
                                <div className="text-xs text-slate-500">{entry.meeting.meeting_id}</div>
                              </td>
                              <td className="px-3 py-3 align-top text-slate-700">
                                {entry.meeting.from_number || "N/A"}
                              </td>
                              <td className="px-3 py-3 align-top text-slate-700">
                                {(entry.meeting as Record<string, any>).call_sid || entry.meeting.meeting_id}
                              </td>
                              <td className="px-3 py-3 align-top">
                                <Badge variant="outline" className="rounded-full">
                                  {breakdown.bottleneck ? breakdown.bottleneck.label : "N/A"}
                                </Badge>
                              </td>
                              <td className="px-3 py-3 align-top">{formatMs(entry.metrics.stt_first_transcript_ms)}</td>
                              <td className="px-3 py-3 align-top">{formatMs(entry.metrics.llm_ttft_ms)}</td>
                              <td className="px-3 py-3 align-top">{formatMs(entry.metrics.tts_ttft_ms)}</td>
                              <td className="px-3 py-3 align-top">{formatMs(entry.metrics.run_bot_total_ms)}</td>
                              <td className="px-3 py-3 align-top text-slate-600">{formatRelativeTime(entry.updatedAt)}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </main>
    </div>
  )
}
