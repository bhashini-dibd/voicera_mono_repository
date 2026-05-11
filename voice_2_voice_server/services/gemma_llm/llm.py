"""Direct Gemma LLM service using NVIDIA's OpenAI-compatible chat completions API."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Optional

import aiohttp
from loguru import logger
from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService


DEFAULT_GEMMA_MODEL = "google/gemma-4-26B-A4B-it"
DEFAULT_GEMMA_ENDPOINT = (
    "https://95dc20d6-e270-459f-a421-74d5ca30576b.invocation.api.nvcf.nvidia.com/v1/chat/completions"
)


class GemmaLLMService(OpenAILLMService):
    """Direct REST Gemma service for voice calls.

    This bypasses the OpenAI client wrapper and calls the NVIDIA endpoint
    directly so the request shape stays identical to the working curl example.
    """

    supports_developer_role = True

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_GEMMA_MODEL,
        temperature: float = 0.15,
        top_p: float = 0.8,
        max_tokens: int = 192,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        timeout_seconds: float | None = 60.0,
        stream: bool | None = None,
        **kwargs: Any,
    ):
        # Keep the parent service initialized for compatibility, but we do not
        # rely on its OpenAI client path for Gemma requests.
        super().__init__(**kwargs)

        self._api_key = (api_key or os.getenv("NVIDIA_GEMMA_LLM_API_KEY", "")).strip()
        self._endpoint = self._normalize_endpoint(
            endpoint or base_url or os.getenv("NVIDIA_GEMMA_LLM_ENDPOINT", DEFAULT_GEMMA_ENDPOINT)
        )
        self._model = model or os.getenv("NVIDIA_GEMMA_LLM_MODEL", DEFAULT_GEMMA_MODEL)
        self._temperature = float(os.getenv("NVIDIA_GEMMA_LLM_TEMPERATURE", str(temperature)))
        self._top_p = float(os.getenv("NVIDIA_GEMMA_LLM_TOP_P", str(top_p)))
        self._max_tokens = int(os.getenv("NVIDIA_GEMMA_LLM_MAX_TOKENS", str(max_tokens)))
        self._frequency_penalty = float(
            os.getenv("NVIDIA_GEMMA_LLM_FREQUENCY_PENALTY", str(frequency_penalty))
        )
        self._presence_penalty = float(
            os.getenv("NVIDIA_GEMMA_LLM_PRESENCE_PENALTY", str(presence_penalty))
        )
        self._timeout_seconds = int(
            os.getenv("NVIDIA_GEMMA_LLM_TIMEOUT_SECONDS", str(timeout_seconds or 60))
        )
        self._requested_stream = stream
        self._session: Optional[aiohttp.ClientSession] = None

        if not self._api_key:
            logger.warning("NVIDIA_GEMMA_LLM_API_KEY is not set; Gemma requests will fail")

        logger.info(
            "GemmaLLM initialized | endpoint={} | model={} | max_tokens={} | temp={} | top_p={}",
            self._endpoint,
            self._model,
            self._max_tokens,
            self._temperature,
            self._top_p,
        )

    @staticmethod
    def _normalize_endpoint(url: str) -> str:
        cleaned = (url or "").strip().rstrip("/")
        if cleaned.endswith("/chat/completions"):
            return cleaned
        if cleaned.endswith("/v1"):
            return f"{cleaned}/chat/completions"
        return f"{cleaned}/v1/chat/completions"

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
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(text)
            return "".join(parts)
        return ""

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").replace("\u200b", " ").replace("\ufeff", " ")
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([।!?.,:;])", r"\1", text)
        text = re.sub(r"([।!?.,:;])(?!\s|$)", r"\1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    async def _post_messages(self, messages: list[dict[str, Any]]) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "stream": False,
        }

        if self._frequency_penalty:
            payload["frequency_penalty"] = self._frequency_penalty
        if self._presence_penalty:
            payload["presence_penalty"] = self._presence_penalty

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        session = await self._get_session()
        async with session.post(self._endpoint, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(
                    "Gemma LLM API error {}: {}",
                    response.status,
                    error_text[:1000],
                )
                raise Exception(f"Gemma LLM API error {response.status}")

            response_json = await response.json(content_type=None)
            text = self._extract_text(response_json)
            return self._normalize_text(text)

    async def _process_context(self, context: OpenAILLMContext | LLMContext):
        messages = context.get_messages()
        if not messages:
            logger.warning("No messages in context for GemmaLLM")
            return

        text = await self._post_messages(messages)
        if not text:
            logger.warning("GemmaLLM returned empty response text")
            return

        await self.push_frame(LLMTextFrame(text=text))

    async def cleanup(self):
        if self._session and not self._session.closed:
            await self._session.close()


def create_gemma_llm(
    *,
    model: str = DEFAULT_GEMMA_MODEL,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.15,
    top_p: float = 0.8,
    max_tokens: int = 192,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    timeout_seconds: float | None = 60.0,
    stream: bool | None = None,
    **kwargs: Any,
) -> GemmaLLMService:
    """Backward-compatible factory for Gemma."""
    return GemmaLLMService(
        api_key=api_key,
        endpoint=endpoint,
        base_url=base_url,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        timeout_seconds=timeout_seconds,
        stream=stream,
        **kwargs,
    )


# Backward-compatible aliases for older imports.
GemmaLLM = GemmaLLMService
GemmaLLMModel = GemmaLLMService
NvidiaGemmaLLM = GemmaLLMService


__all__ = [
    "DEFAULT_GEMMA_ENDPOINT",
    "DEFAULT_GEMMA_MODEL",
    "GemmaLLM",
    "GemmaLLMModel",
    "GemmaLLMService",
    "NvidiaGemmaLLM",
    "create_gemma_llm",
]
