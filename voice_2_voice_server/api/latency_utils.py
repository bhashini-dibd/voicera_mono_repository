"""Shared latency helpers for voice call telemetry."""

from __future__ import annotations

import time
from typing import Awaitable, Callable, Optional

from loguru import logger


LatencyCallback = Optional[Callable[[dict], Awaitable[None]]]


def record_latency_metric(
    timing_state: Optional[dict],
    *,
    service: str,
    metric: str,
    value_ms: float,
    stage: Optional[str] = None,
    details: Optional[dict] = None,
) -> dict:
    """Store a structured latency event in the shared timing state."""
    entry = {
        "service": service,
        "metric": metric,
        "value_ms": round(float(value_ms), 1),
        "stage": stage,
        "details": details or {},
        "timestamp_monotonic": time.monotonic(),
    }

    if timing_state is not None:
        metrics = timing_state.setdefault("latency_metrics", [])
        metrics.append(entry)
        latest = timing_state.setdefault("latency_latest", {})
        latest[f"{service}.{metric}"] = entry

    logger.info(
        "Latency | service={} metric={}{} value_ms={:.1f}{}",
        service,
        metric,
        f" stage={stage}" if stage else "",
        value_ms,
        f" details={details}" if details else "",
    )
    return entry


def build_latency_summary(timing_state: Optional[dict]) -> dict:
    """Return a call-level latency summary suitable for persistence or UI display."""
    timing_state = timing_state or {}
    metrics = timing_state.get("latency_metrics", []) or []

    def latest_value(service: str, metric: str):
        entry = (timing_state.get("latency_latest", {}) or {}).get(f"{service}.{metric}")
        if not entry:
            return None
        return entry.get("value_ms")

    return {
        "call": {
            "run_bot_total_ms": _elapsed_ms(timing_state, "run_bot_started_at", "run_bot_finished_at"),
            "service_initialization_ms": _elapsed_ms(timing_state, "run_bot_started_at", "services_ready_at"),
            "pipeline_build_ms": _elapsed_ms(timing_state, "services_ready_at", "pipeline_built_at"),
            "client_connected_after_run_bot_ms": _elapsed_ms(timing_state, "run_bot_started_at", "client_connected_at"),
        },
        "orchestrator": {
            "user_transcript_to_llm_start_ms": latest_value("orchestrator", "user_transcript_to_llm_start_ms"),
            "user_transcript_to_tts_start_ms": latest_value("orchestrator", "user_transcript_to_tts_start_ms"),
            "user_transcript_to_first_tts_audio_ms": latest_value("orchestrator", "user_transcript_to_first_tts_audio_ms"),
            "greeting_tts_ttft_ms": latest_value("orchestrator", "greeting_tts_ttft_ms"),
        },
        "stt": {
            "ws_open_ms": latest_value("stt", "ws_open_ms"),
            "first_transcript_ms": latest_value("stt", "first_transcript_ms"),
            "final_transcript_ms": latest_value("stt", "final_transcript_ms"),
            "segment_duration_ms": latest_value("stt", "segment_duration_ms"),
        },
        "llm": {
            "ttft_ms": latest_value("llm", "ttft_ms"),
            "stream_complete_ms": latest_value("llm", "stream_complete_ms"),
            "chunk_count": latest_value("llm", "chunk_count"),
        },
        "tts": {
            "ttft_ms": latest_value("tts", "ttft_ms"),
            "stream_complete_ms": latest_value("tts", "stream_complete_ms"),
            "chunk_count": latest_value("tts", "chunk_count"),
        },
        "events": metrics,
    }


def _elapsed_ms(timing_state: dict, start_key: str, end_key: str) -> Optional[float]:
    start = timing_state.get(start_key)
    end = timing_state.get(end_key)
    if start is None or end is None:
        return None
    return round((end - start) * 1000.0, 1)
