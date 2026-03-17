import os
import re
from typing import Any, Optional

import aiohttp
from loguru import logger
from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService


class SarvamLLM(OpenAILLMService):
    """OpenAI-compatible Sarvam LLM over NVIDIA NVCF endpoint."""

    def __init__(self, model: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._api_key = os.getenv("SARVAM_LLM_API_KEY", "").strip()
        self._endpoint = os.getenv(
            "SARVAM_LLM_ENDPOINT",
            "https://e75f29cc-857c-4a05-ac2c-70f4291e0bef.invocation.api.nvcf.nvidia.com/v1/chat/completions",
        ).strip()
        self._model = model or os.getenv("SARVAM_LLM_MODEL", "sarvamai/sarvam-30b")
        self._temperature = float(os.getenv("SARVAM_LLM_TEMPERATURE", "0.8"))
        self._max_tokens = int(os.getenv("SARVAM_LLM_MAX_TOKENS", "2048"))
        self._repetition_penalty = float(os.getenv("SARVAM_LLM_REPETITION_PENALTY", "1.0"))
        self._timeout_seconds = int(os.getenv("SARVAM_LLM_TIMEOUT_SECONDS", "60"))

        self._session: Optional[aiohttp.ClientSession] = None

        if not self._api_key:
            logger.warning("SARVAM_LLM_API_KEY is not set; Sarvam LLM requests will fail")

        logger.info(f"SarvamLLM initialized | endpoint={self._endpoint} | model={self._model}")

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
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(text)
            return "".join(parts)
        return ""

    @staticmethod
    def _post_process_text(raw_text: str) -> str:
        """
        Return only user-facing final answer text.
        Removes reasoning blocks and picks the best final candidate when model truncates.
        """
        if not raw_text:
            return ""

        text = raw_text.strip()

        # Prefer text after a completed </think> block if present.
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        else:
            # If we only have <think> content (possibly truncated), try quoted final answers.
            quoted = re.findall(r'"([^"\n]{2,220})"', text)
            if quoted:
                # Pick the last non-empty, sentence-like candidate.
                for candidate in reversed(quoted):
                    c = candidate.strip()
                    if c and "<" not in c and ">" not in c:
                        return c

            # Fallback: keep only content after "<think>" marker if present.
            if "<think>" in text:
                text = text.split("<think>")[-1]

        # Remove any remaining think tags and normalize whitespace.
        text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[ \t]+", " ", text)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""

        # Prefer the last concise sentence-like line.
        for line in reversed(lines):
            cleaned = re.sub(r"^[\-\*\d\.\)\s:]+", "", line).strip()
            if cleaned and len(cleaned) <= 240:
                return cleaned

        return lines[-1]

    async def _process_context(self, context: OpenAILLMContext | LLMContext):
        messages = context.get_messages()
        if not messages:
            logger.warning("No messages in context for SarvamLLM")
            return

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "repetition_penalty": self._repetition_penalty,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        session = await self._get_session()
        async with session.post(self._endpoint, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Sarvam LLM API error {response.status}: {error_text}")
                raise Exception(f"Sarvam LLM API error {response.status}")

            response_json = await response.json(content_type=None)
            text = self._post_process_text(self._extract_text(response_json))

            if not text:
                logger.warning("SarvamLLM returned empty response text")
                return

            await self.push_frame(LLMTextFrame(text=text))

    async def cleanup(self):
        if self._session and not self._session.closed:
            await self._session.close()
