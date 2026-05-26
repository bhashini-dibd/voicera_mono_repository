"use client"

import { useState, useEffect } from "react"
import { useRouter, useParams } from "next/navigation"
import { Separator } from "@/components/ui/separator"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { getMeeting, type Meeting } from "@/lib/api"
import {
  ChevronLeft,
  Phone,
  PhoneIncoming,
  PhoneOutgoing,
  Calendar,
  Clock,
  User,
  FileText,
  Volume2,
  Loader2,
  Download,
} from "lucide-react"

export default function MeetingDetailPage() {
  const router = useRouter()
  const params = useParams()
  const meetingId = params.meeting_id as string
  const [meeting, setMeeting] = useState<Meeting | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [activeTab, setActiveTab] = useState("info")

  useEffect(() => {
    async function fetchMeeting() {
      if (!meetingId || meetingId === 'undefined') {
        console.error("Meeting ID is missing:", meetingId)
        setIsLoading(false)
        router.push("/assistants")
        return
      }

      try {
        const meetingData = await getMeeting(meetingId)
        setMeeting(meetingData)
      } catch (error) {
        console.error("Failed to fetch meeting:", error)
        router.push("/assistants")
      } finally {
        setIsLoading(false)
      }
    }

    fetchMeeting()
  }, [meetingId, router])

  const formatDate = (dateString?: string) => {
    if (!dateString) return "N/A"
    try {
      const date = new Date(dateString)
      return date.toLocaleString()
    } catch {
      return dateString
    }
  }

  const formatDuration = (seconds?: number) => {
    if (!seconds) return "N/A"
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}m ${secs}s`
  }

  const calculateDuration = () => {
    if (!meeting?.start_time_utc || !meeting?.end_time_utc) return "N/A"
    try {
      const start = new Date(meeting.start_time_utc)
      const end = new Date(meeting.end_time_utc)
      const diff = (end.getTime() - start.getTime()) / 1000
      return formatDuration(diff)
    } catch {
      return "N/A"
    }
  }

  const latencySummary = meeting?.latency_summary as Record<string, any> | undefined

  const formatLatencyValue = (value: any) => {
    if (typeof value === "number") return `${value.toFixed(1)} ms`
    if (value === null || value === undefined || value === "") return "N/A"
    return String(value)
  }

  return (
    <div className="flex flex-col h-screen bg-slate-50/50">
      {/* Header */}
      <header className="flex h-14 items-center gap-4 border-b border-slate-200 bg-white px-5 lg:px-8 sticky top-0 z-10">
        <nav className="flex items-center gap-1.5 text-sm">
          <button
            onClick={() => router.back()}
            className="text-slate-500 hover:text-slate-900 transition-colors"
          >
            Back
          </button>
          <ChevronLeft className="h-4 w-4 text-slate-400" />
          <span className="text-slate-900 font-medium">Meeting Details</span>
        </nav>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-auto p-6 lg:p-8">
        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-6 w-6 animate-spin text-slate-400" />
            <span className="ml-2 text-slate-500">Loading meeting details...</span>
          </div>
        )}

        {/* Meeting Details */}
        {!isLoading && meeting && (
          <div className="max-w-6xl mx-auto">
            {/* Header Card */}
            <div className="bg-white rounded-xl border border-slate-200 p-6 mb-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  {meeting.inbound ? (
                    <PhoneIncoming className="h-6 w-6 text-blue-500" />
                  ) : (
                    <PhoneOutgoing className="h-6 w-6 text-green-500" />
                  )}
                  <div>
                    <h1 className="text-2xl font-semibold text-slate-900">
                      {meeting.inbound ? "Inbound" : "Outbound"} Call
                    </h1>
                    <p className="text-sm text-slate-500 mt-1">
                      {meeting.agent_type}
                    </p>
                  </div>
                </div>
                <Button
                  variant="outline"
                  onClick={() => router.back()}
                  className="gap-2"
                >
                  <ChevronLeft className="h-4 w-4" />
                  Back
                </Button>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-slate-500 mb-1">Start Time</p>
                  <p className="font-medium text-slate-900">
                    {formatDate(meeting.start_time_utc || meeting.created_at)}
                  </p>
                </div>
                {meeting.end_time_utc && (
                  <div>
                    <p className="text-slate-500 mb-1">End Time</p>
                    <p className="font-medium text-slate-900">
                      {formatDate(meeting.end_time_utc)}
                    </p>
                  </div>
                )}
                <div>
                  <p className="text-slate-500 mb-1">Duration</p>
                  <p className="font-medium text-slate-900">
                    {meeting.duration ? formatDuration(meeting.duration) : calculateDuration()}
                  </p>
                </div>
                {meeting.from_number && (
                  <div>
                    <p className="text-slate-500 mb-1">
                      {meeting.inbound ? "From" : "To"}
                    </p>
                    <p className="font-medium text-slate-900">{meeting.from_number}</p>
                  </div>
                )}
              </div>
            </div>

            {/* Tabs */}
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-3 mb-6">
                <TabsTrigger value="info" className="gap-2">
                  <FileText className="h-4 w-4" />
                  Call Info
                </TabsTrigger>
                <TabsTrigger value="audio" className="gap-2">
                  <Volume2 className="h-4 w-4" />
                  Audio
                </TabsTrigger>
                <TabsTrigger value="transcript" className="gap-2">
                  <FileText className="h-4 w-4" />
                  Transcript
                </TabsTrigger>
              </TabsList>

              {/* Call Info Tab */}
              <TabsContent value="info" className="space-y-4">
                <div className="bg-white rounded-xl border border-slate-200 p-6">
                  <h2 className="text-lg font-semibold text-slate-900 mb-4">Call Information</h2>
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-slate-500 mb-1">Meeting ID</p>
                        <p className="font-mono text-sm text-slate-900 break-all">
                          {meeting.meeting_id}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-slate-500 mb-1">Agent Type</p>
                        <p className="text-slate-900">{meeting.agent_type}</p>
                      </div>
                      {meeting.agent_category && (
                        <div>
                          <p className="text-sm text-slate-500 mb-1">Agent Category</p>
                          <p className="text-slate-900">{meeting.agent_category}</p>
                        </div>
                      )}
                      {meeting.org_id && (
                        <div>
                          <p className="text-sm text-slate-500 mb-1">Organization ID</p>
                          <p className="text-slate-900">{meeting.org_id}</p>
                        </div>
                      )}
                      {meeting.from_number && (
                        <div>
                          <p className="text-sm text-slate-500 mb-1">From Number</p>
                          <p className="text-slate-900">{meeting.from_number}</p>
                        </div>
                      )}
                      {meeting.to_number && (
                        <div>
                          <p className="text-sm text-slate-500 mb-1">To Number</p>
                          <p className="text-slate-900">{meeting.to_number}</p>
                        </div>
                      )}
                      <div>
                        <p className="text-sm text-slate-500 mb-1">Call Direction</p>
                        <p className="text-slate-900">
                          {meeting.inbound ? "Inbound" : "Outbound"}
                        </p>
                      </div>
                      {meeting.created_at && (
                        <div>
                          <p className="text-sm text-slate-500 mb-1">Created At</p>
                          <p className="text-slate-900">{formatDate(meeting.created_at)}</p>
                        </div>
                      )}
                      {meeting.start_time_utc && (
                        <div>
                          <p className="text-sm text-slate-500 mb-1">Start Time (UTC)</p>
                          <p className="text-slate-900">{formatDate(meeting.start_time_utc)}</p>
                        </div>
                      )}
                      {meeting.end_time_utc && (
                        <div>
                          <p className="text-sm text-slate-500 mb-1">End Time (UTC)</p>
                          <p className="text-slate-900">{formatDate(meeting.end_time_utc)}</p>
                        </div>
                      )}
                      {meeting.duration && (
                        <div>
                          <p className="text-sm text-slate-500 mb-1">Duration (seconds)</p>
                          <p className="text-slate-900">{meeting.duration}s</p>
                        </div>
                      )}
                    </div>
                    {latencySummary && (
                      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                        <h3 className="mb-3 text-sm font-semibold text-slate-900">Latency Summary</h3>
                        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                          <div>
                            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Call</p>
                            <div className="space-y-1 text-sm text-slate-700">
                              <div>Run total: {formatLatencyValue(latencySummary.call?.run_bot_total_ms)}</div>
                              <div>Service init: {formatLatencyValue(latencySummary.call?.service_initialization_ms)}</div>
                              <div>Pipeline build: {formatLatencyValue(latencySummary.call?.pipeline_build_ms)}</div>
                              <div>Client connected: {formatLatencyValue(latencySummary.call?.client_connected_after_run_bot_ms)}</div>
                            </div>
                          </div>
                          <div>
                            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Service TTFT</p>
                            <div className="space-y-1 text-sm text-slate-700">
                              <div>STT first transcript: {formatLatencyValue(latencySummary.stt?.first_transcript_ms)}</div>
                              <div>LLM TTFT: {formatLatencyValue(latencySummary.llm?.ttft_ms)}</div>
                              <div>TTS TTFT: {formatLatencyValue(latencySummary.tts?.ttft_ms)}</div>
                              <div>Orchestrator gap: {formatLatencyValue(latencySummary.orchestrator?.user_transcript_to_first_tts_audio_ms)}</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </TabsContent>

              {/* Audio Tab */}
              <TabsContent value="audio" className="space-y-4">
                <div className="bg-white rounded-xl border border-slate-200 p-6">
                  <h2 className="text-lg font-semibold text-slate-900 mb-4">Audio Recording</h2>
                  {meeting.recording_url ? (
                    <div className="space-y-4">
                      <div>
                        <p className="text-sm text-slate-500 mb-2">Recording URL</p>
                        <a
                          href={meeting.recording_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:text-blue-700 break-all text-sm"
                        >
                          {meeting.recording_url}
                        </a>
                      </div>
                      <div>
                        <audio controls className="w-full">
                          <source src={meeting.recording_url} type="audio/wav" />
                          <source src={meeting.recording_url} type="audio/mpeg" />
                          Your browser does not support the audio element.
                        </audio>
                      </div>
                      <Button
                        variant="outline"
                        onClick={() => window.open(meeting.recording_url, "_blank")}
                        className="gap-2"
                      >
                        <Download className="h-4 w-4" />
                        Download Recording
                      </Button>
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <Volume2 className="h-12 w-12 text-slate-300 mx-auto mb-4" />
                      <p className="text-slate-500">No audio recording available</p>
                    </div>
                  )}
                </div>
              </TabsContent>

              {/* Transcript Tab */}
              <TabsContent value="transcript" className="space-y-4">
                <div className="bg-white rounded-xl border border-slate-200 p-6">
                  <h2 className="text-lg font-semibold text-slate-900 mb-4">Transcript</h2>
                  {meeting.transcript_content ? (
                    <div className="space-y-4">
                      <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                        <pre className="whitespace-pre-wrap text-sm text-slate-900 font-mono">
                          {meeting.transcript_content}
                        </pre>
                      </div>
                      {meeting.transcript_url && (
                        <div>
                          <p className="text-sm text-slate-500 mb-2">Transcript URL</p>
                          <a
                            href={meeting.transcript_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-600 hover:text-blue-700 break-all text-sm"
                          >
                            {meeting.transcript_url}
                          </a>
                        </div>
                      )}
                      <Button
                        variant="outline"
                        onClick={() => {
                          const blob = new Blob([meeting.transcript_content || ""], {
                            type: "text/plain",
                          })
                          const url = URL.createObjectURL(blob)
                          const a = document.createElement("a")
                          a.href = url
                          a.download = `${meeting.meeting_id}_transcript.txt`
                          a.click()
                          URL.revokeObjectURL(url)
                        }}
                        className="gap-2"
                      >
                        <Download className="h-4 w-4" />
                        Download Transcript
                      </Button>
                    </div>
                  ) : meeting.transcript_url ? (
                    <div className="space-y-4">
                      <div className="text-center py-8">
                        <FileText className="h-12 w-12 text-slate-300 mx-auto mb-4" />
                        <p className="text-slate-500 mb-4">
                          Transcript content not available, but URL is provided
                        </p>
                        <a
                          href={meeting.transcript_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:text-blue-700"
                        >
                          View Transcript
                        </a>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <FileText className="h-12 w-12 text-slate-300 mx-auto mb-4" />
                      <p className="text-slate-500">No transcript available</p>
                    </div>
                  )}
                </div>
              </TabsContent>
            </Tabs>
          </div>
        )}

        {/* Error State */}
        {!isLoading && !meeting && (
          <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
            <Phone className="h-12 w-12 text-slate-300 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-slate-900 mb-2">
              Meeting not found
            </h3>
            <p className="text-slate-500 mb-4">
              The meeting you&apos;re looking for doesn&apos;t exist or you don&apos;t have access to it.
            </p>
            <Button onClick={() => router.back()} variant="outline">
              Go Back
            </Button>
          </div>
        )}
      </main>
    </div>
  )
}
