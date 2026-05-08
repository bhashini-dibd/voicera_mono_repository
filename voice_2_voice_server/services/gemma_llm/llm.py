"""NVIDIA NVCF-hosted Google Gemma LLM service for Pipecat voice agents.

This service is intentionally voice-first:
- credentials and endpoint come from environment variables
- ``stream`` defaults to ``true`` so the model can stream SSE chunks when the
  upstream endpoint supports it
- streamed text is chunked into speakable pieces so the caller hears the answer
  sooner, instead of waiting for one large completion
"""

from __future__ import annotations

import codecs
import json
import os
import re
import time
from typing import Any, Optional

import aiohttp
from loguru import logger
from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_ENDPOINT = (
    "https://95dc20d6-e270-459f-a421-74d5ca30576b"
    ".invocation.api.nvcf.nvidia.com/v1/chat/completions"
)
_DEFAULT_MODEL = "google/gemma-4-26B-A4B-it"
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TOP_P = 0.9
_DEFAULT_TIMEOUT_SECONDS = 60
_DEFAULT_STREAM = True
_STREAM_FLUSH_CHARS = 60
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class NvidiaGemmaLLM(OpenAILLMService):
    """OpenAI-compatible Gemma service backed by an NVIDIA NVCF endpoint."""

    def __init__(self, model: Optional[str] = None, **kwargs: Any) -> None:
        gemma_kwargs = dict(kwargs)
        api_key = str(gemma_kwargs.pop("api_key", "") or os.getenv("NVIDIA_GEMMA_LLM_API_KEY", "")).strip()
        endpoint = str(
            gemma_kwargs.pop("endpoint", "")
            or os.getenv("NVIDIA_GEMMA_LLM_ENDPOINT", _DEFAULT_ENDPOINT)
        ).strip()
        model_name = str(
            model
            or gemma_kwargs.pop("model", "")
            or os.getenv("NVIDIA_GEMMA_LLM_MODEL", _DEFAULT_MODEL)
        ).strip()
        temperature = float(
            gemma_kwargs.pop("temperature", os.getenv("NVIDIA_GEMMA_LLM_TEMPERATURE", str(_DEFAULT_TEMPERATURE)))
        )
        max_tokens = int(
            gemma_kwargs.pop("max_tokens", os.getenv("NVIDIA_GEMMA_LLM_MAX_TOKENS", str(_DEFAULT_MAX_TOKENS)))
        )
        top_p = float(
            gemma_kwargs.pop("top_p", os.getenv("NVIDIA_GEMMA_LLM_TOP_P", str(_DEFAULT_TOP_P)))
        )
        stream_enabled = str(
            gemma_kwargs.pop("stream", os.getenv("NVIDIA_GEMMA_LLM_STREAM", str(_DEFAULT_STREAM)))
        ).strip().lower() not in {"0", "false", "no", "off"}
        timeout_seconds = int(
            gemma_kwargs.pop("timeout_seconds", os.getenv("NVIDIA_GEMMA_LLM_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT_SECONDS)))
        )

        super().__init__(**gemma_kwargs)

        self._api_key = api_key
        self._endpoint = endpoint
        self._model = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._stream_enabled = stream_enabled
        self._timeout_seconds = timeout_seconds

        self._session: Optional[aiohttp.ClientSession] = None

        if not self._api_key:
            logger.warning(
                "NvidiaGemmaLLM | NVIDIA_GEMMA_LLM_API_KEY is not set; requests will fail"
            )

        logger.info(
            "NvidiaGemmaLLM initialized | endpoint={} | model={} | stream={} | timeout={}s",
            self._endpoint,
            self._model,
            self._stream_enabled,
            self._timeout_seconds,
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout_seconds)
            )
        return self._session

    @staticmethod
    def _extract_text(response_json: dict[str, Any]) -> str:
        choices = response_json.get("choices") or []
        if not choices:
            logger.warning("NvidiaGemmaLLM | response has no choices")
            return ""
        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
            return "".join(parts)
        delta = choice.get("delta") or {}
        delta_content = delta.get("content")
        if isinstance(delta_content, str):
            return delta_content
        return ""

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        if not text:
            return ""
        cleaned = text.replace(_THINK_OPEN, " ").replace(_THINK_CLOSE, " ")
        cleaned = re.sub(r"<\/?think>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _finalize_text(text: str) -> str:
        if not text:
            return ""
        candidate = text.strip()
        if _THINK_CLOSE in candidate:
            candidate = candidate.split(_THINK_CLOSE)[-1].strip()
        elif _THINK_OPEN in candidate:
            quoted = re.findall(r'"([^"\n]{2,220})"', candidate)
            for item in reversed(quoted):
                cleaned = item.strip()
                if cleaned and "<" not in cleaned and ">" not in cleaned:
                    return cleaned
            candidate = candidate.split(_THINK_OPEN)[-1]

        candidate = re.sub(r"</?think>", " ", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"[ \t]+", " ", candidate)
        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        if not lines:
            return ""
        return lines[-1]

    @staticmethod
    def _yield_voice_chunks(text: str, max_chars: int = _STREAM_FLUSH_CHARS):
        buffer = ""
        for token in re.split(r"(\s+)", text):
            if not token:
                continue
            buffer += token
            if (
                len(buffer) >= max_chars
                or token.endswith((".", "!", "?", ":", ";", "\n"))
            ):
                chunk = buffer.strip()
                if chunk:
                    yield chunk
                buffer = ""
        chunk = buffer.strip()
        if chunk:
            yield chunk

    async def _emit_text(self, text: str) -> None:
        clean = self._strip_think_blocks(text)
        for chunk in self._yield_voice_chunks(clean):
            await self.push_frame(LLMTextFrame(text=chunk))

    async def _stream_response(
        self,
        session: aiohttp.ClientSession,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> bool:
        """Return True when a streaming response was consumed successfully."""
        t0 = time.perf_counter()
        async with session.post(self._endpoint, json=payload, headers=headers) as response:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            content_type = (response.headers.get("Content-Type") or "").lower()

            if response.status != 200:
                error_body = await response.text()
                logger.error(
                    "NvidiaGemmaLLM | HTTP {} in {:.0f}ms | error: {}",
                    response.status,
                    elapsed_ms,
                    error_body[:500],
                )
                raise RuntimeError(
                    f"NvidiaGemmaLLM API error {response.status}: {error_body[:200]}"
                )

            if "text/event-stream" not in content_type:
                logger.debug(
                    "NvidiaGemmaLLM | non-SSE response content-type={}, falling back to JSON handling",
                    content_type or "unknown",
                )
                return False

            logger.info(
                "NvidiaGemmaLLM | streaming SSE response from {}",
                self._endpoint,
            )

            raw_buffer = ""
            in_think = False
            chunk_count = 0
            first_chunk_at = None
            decoder = codecs.getincrementaldecoder("utf-8")("replace")

            async for raw_bytes in response.content.iter_any():
                raw_buffer += decoder.decode(raw_bytes, final=False)
                while "\n" in raw_buffer:
                    line, raw_buffer = raw_buffer.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        raw_buffer = ""
                        break

                    try:
                        payload_chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    chunk_text = self._extract_text(payload_chunk)
                    if not chunk_text:
                        continue

                    raw_text = chunk_text
                    visible = ""

                    while raw_text:
                        if in_think:
                            close_idx = raw_text.find(_THINK_CLOSE)
                            if close_idx == -1:
                                break
                            raw_text = raw_text[close_idx + len(_THINK_CLOSE) :]
                            in_think = False
                            continue

                        open_idx = raw_text.find(_THINK_OPEN)
                        if open_idx == -1:
                            visible += raw_text
                            raw_text = ""
                            break

                        visible += raw_text[:open_idx]
                        raw_text = raw_text[open_idx + len(_THINK_OPEN) :]
                        in_think = True

                    if visible:
                        if first_chunk_at is None:
                            first_chunk_at = time.perf_counter()
                            logger.info(
                                "NvidiaGemmaLLM | first streamed text at {:.0f}ms",
                                (first_chunk_at - t0) * 1000,
                            )
                        chunk_count += 1
                        await self._emit_text(visible)

            if raw_buffer.strip() and not in_think:
                await self._emit_text(raw_buffer)

            logger.info(
                "NvidiaGemmaLLM | stream complete | chunks={} | elapsed={:.0f}ms",
                chunk_count,
                (time.perf_counter() - t0) * 1000,
            )
            return True

    async def _process_context(
        self, context: OpenAILLMContext | LLMContext
    ) -> None:
        messages = context.get_messages()
        if not messages:
            logger.warning("NvidiaGemmaLLM | _process_context called with empty messages")
            return

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": self._top_p,
            "stream": self._stream_enabled,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        masked_key = (
            ("*" * (len(self._api_key) - 4) + self._api_key[-4:])
            if len(self._api_key) > 4
            else "****"
        )
        logger.info(
            "NvidiaGemmaLLM | → POST {} | model={} | messages={} | stream={} | key=...{}",
            self._endpoint,
            self._model,
            len(messages),
            self._stream_enabled,
            masked_key,
        )

        session = await self._get_session()

        try:
            if self._stream_enabled:
                try:
                    streamed = await self._stream_response(session, payload, headers)
                    if streamed:
                        return
                except Exception as stream_error:
                    logger.warning(
                        "NvidiaGemmaLLM | streaming path failed, retrying without SSE: {}",
                        stream_error,
                    )

            t0 = time.perf_counter()
            async with session.post(self._endpoint, json=payload, headers=headers) as response:
                elapsed_ms = (time.perf_counter() - t0) * 1000

                if response.status != 200:
                    error_body = await response.text()
                    logger.error(
                        "NvidiaGemmaLLM | ← HTTP {} in {:.0f}ms | error body: {}",
                        response.status,
                        elapsed_ms,
                        error_body[:500],
                    )
                    raise RuntimeError(
                        f"NvidiaGemmaLLM NVCF API error {response.status}: {error_body[:200]}"
                    )

                response_json: dict[str, Any] = await response.json(content_type=None)
                raw_text = self._extract_text(response_json)
                clean_text = self._finalize_text(raw_text)

                usage = response_json.get("usage", {})
                logger.info(
                    "NvidiaGemmaLLM | ← HTTP 200 in {:.0f}ms | prompt_tokens={} completion_tokens={} | raw_chars={} clean_chars={}",
                    elapsed_ms,
                    usage.get("prompt_tokens", "?"),
                    usage.get("completion_tokens", "?"),
                    len(raw_text),
                    len(clean_text),
                )

                if not clean_text:
                    logger.warning(
                        "NvidiaGemmaLLM | post-processed text is empty (raw={!r})",
                        raw_text[:120],
                    )
                    return

                await self.push_frame(LLMTextFrame(text=clean_text))

        except aiohttp.ClientError as exc:
            logger.error("NvidiaGemmaLLM | network error: {}", exc)
            raise

    async def cleanup(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("NvidiaGemmaLLM | aiohttp session closed")


def create_gemma_llm(model: Optional[str] = None, **kwargs: Any) -> NvidiaGemmaLLM:
    """Factory for the Gemma voice LLM."""

    return NvidiaGemmaLLM(model=model, **kwargs)


__all__ = [
    "NvidiaGemmaLLM",
    "create_gemma_llm",
    "_DEFAULT_ENDPOINT",
    "_DEFAULT_MODEL",
]
