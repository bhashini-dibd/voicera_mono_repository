"""NVIDIA NVCF-hosted Google Gemma LLM service for Pipecat voice agents.

Endpoint: NVIDIA Cloud Functions (NVCF) — OpenAI-compatible /v1/chat/completions
Model:    google/gemma-4-26B-A4B-it

Environment variables (set in .env):
    NVIDIA_GEMMA_LLM_API_KEY   — Bearer token for NVCF authorization (required)
    NVIDIA_GEMMA_LLM_ENDPOINT  — Full NVCF invocation URL (optional, defaults below)
    NVIDIA_GEMMA_LLM_MODEL     — Model name override (optional)
    NVIDIA_GEMMA_LLM_TEMPERATURE        — Sampling temperature (default: 0.7)
    NVIDIA_GEMMA_LLM_MAX_TOKENS         — Max output tokens (default: 1024)
    NVIDIA_GEMMA_LLM_TIMEOUT_SECONDS    — HTTP timeout in seconds (default: 60)
"""

from __future__ import annotations

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
_DEFAULT_TIMEOUT_SECONDS = 60


class NvidiaGemmaLLM(OpenAILLMService):
    """Pipecat LLM service that calls the NVIDIA NVCF Gemma endpoint.

    The NVCF API is OpenAI-compatible (/v1/chat/completions), so we override
    ``_process_context`` to call it directly with ``aiohttp`` — the same
    approach used by SarvamLLM — instead of relying on the openai SDK client,
    which would require matching base-URL gymnastics.

    All requests use ``stream: false`` because the NVCF endpoint for this
    model does not currently support SSE streaming.  The full response arrives
    as a single JSON payload which is then forwarded downstream as an
    ``LLMTextFrame`` so Pipecat's TTS stage picks it up immediately.
    """

    def __init__(self, model: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._api_key: str = os.getenv("NVIDIA_GEMMA_LLM_API_KEY", "").strip()
        self._endpoint: str = os.getenv(
            "NVIDIA_GEMMA_LLM_ENDPOINT", _DEFAULT_ENDPOINT
        ).strip()
        self._model: str = model or os.getenv(
            "NVIDIA_GEMMA_LLM_MODEL", _DEFAULT_MODEL
        )
        self._temperature: float = float(
            os.getenv("NVIDIA_GEMMA_LLM_TEMPERATURE", str(_DEFAULT_TEMPERATURE))
        )
        self._max_tokens: int = int(
            os.getenv("NVIDIA_GEMMA_LLM_MAX_TOKENS", str(_DEFAULT_MAX_TOKENS))
        )
        self._timeout_seconds: int = int(
            os.getenv("NVIDIA_GEMMA_LLM_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT_SECONDS))
        )

        self._session: Optional[aiohttp.ClientSession] = None

        if not self._api_key:
            logger.warning(
                "NvidiaGemmaLLM | NVIDIA_GEMMA_LLM_API_KEY is not set — "
                "all requests will fail with 401"
            )

        logger.info(
            "NvidiaGemmaLLM initialized | endpoint={} | model={} | "
            "temperature={} | max_tokens={} | timeout={}s",
            self._endpoint,
            self._model,
            self._temperature,
            self._max_tokens,
            self._timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return (or lazily create) a shared aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout_seconds)
            )
        return self._session

    # ------------------------------------------------------------------
    # Response parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response_json: dict[str, Any]) -> str:
        """Pull assistant content out of an OpenAI-compatible response JSON."""
        choices = response_json.get("choices") or []
        if not choices:
            logger.warning("NvidiaGemmaLLM | response has no choices field")
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Multi-part content blocks (rare but handle gracefully)
            parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("text")
            ]
            return "".join(parts)
        logger.warning(
            "NvidiaGemmaLLM | unexpected content type: {}", type(content)
        )
        return ""

    @staticmethod
    def _post_process_text(raw: str) -> str:
        """Strip reasoning/think blocks and return clean user-facing text.

        Gemma 4 may include internal reasoning wrapped in <think>…</think>
        tags.  We strip those and return only the visible answer.
        """
        if not raw:
            return ""
        text = raw.strip()

        # Prefer content after a completed </think> block
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        elif "<think>" in text:
            # Truncated reasoning — attempt to extract quoted final answer
            quoted = re.findall(r'"([^"\n]{2,220})"', text)
            for candidate in reversed(quoted):
                c = candidate.strip()
                if c and "<" not in c and ">" not in c:
                    return c
            text = text.split("<think>")[-1]

        # Remove any residual think tags and normalise whitespace
        text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[ \t]+", " ", text)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""
        return lines[-1] if len(lines) > 0 else text.strip()

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    async def _process_context(
        self, context: OpenAILLMContext | LLMContext
    ) -> None:
        """Fetch a completion from NVCF and push the result as an LLMTextFrame."""
        messages = context.get_messages()
        if not messages:
            logger.warning("NvidiaGemmaLLM | _process_context called with empty messages")
            return

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        # Log the outgoing request (mask all but last 4 chars of key)
        masked_key = (
            ("*" * (len(self._api_key) - 4) + self._api_key[-4:])
            if len(self._api_key) > 4
            else "****"
        )
        logger.info(
            "NvidiaGemmaLLM | → POST {} | model={} | messages={} | key=...{}",
            self._endpoint,
            self._model,
            len(messages),
            masked_key,
        )

        t0 = time.perf_counter()
        session = await self._get_session()

        try:
            async with session.post(
                self._endpoint, json=payload, headers=headers
            ) as response:
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
                clean_text = self._post_process_text(raw_text)

                # Log usage metadata if present
                usage = response_json.get("usage", {})
                logger.info(
                    "NvidiaGemmaLLM | ← HTTP 200 in {:.0f}ms | "
                    "prompt_tokens={} completion_tokens={} | "
                    "raw_chars={} clean_chars={}",
                    elapsed_ms,
                    usage.get("prompt_tokens", "?"),
                    usage.get("completion_tokens", "?"),
                    len(raw_text),
                    len(clean_text),
                )

                if not clean_text:
                    logger.warning(
                        "NvidiaGemmaLLM | post-processed text is empty "
                        "(raw was {!r})",
                        raw_text[:120],
                    )
                    return

                await self.push_frame(LLMTextFrame(text=clean_text))

        except aiohttp.ClientError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "NvidiaGemmaLLM | network error after {:.0f}ms: {}", elapsed_ms, exc
            )
            raise

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self) -> None:
        """Close the shared aiohttp session on shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("NvidiaGemmaLLM | aiohttp session closed")


def create_gemma_llm(
    model: Optional[str] = None,
    **kwargs: Any,
) -> NvidiaGemmaLLM:
    """Convenience factory — builds an NvidiaGemmaLLM with environment defaults.

    Pass any extra kwargs through to ``OpenAILLMService.__init__`` if needed
    (e.g. ``retry_timeout_secs``).
    """
    return NvidiaGemmaLLM(model=model, **kwargs)


__all__ = [
    "NvidiaGemmaLLM",
    "create_gemma_llm",
    "_DEFAULT_ENDPOINT",
    "_DEFAULT_MODEL",
]
