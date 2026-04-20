"""Bhashini TTS service backed by NVIDIA Triton Inference Server over gRPC streaming.

Required environment variables (set in .env):
  BHASHINI_TRITON_URL          gRPC endpoint of the Triton server
                                e.g. grpc.nvcf.nvidia.com:443
  BHASHINI_TRITON_API_KEY      Bearer token / NVCF API key for authentication
  BHASHINI_TRITON_FUNCTION_ID  NVCF function-id header value

Optional environment variables:
  BHASHINI_TRITON_FUNCTION_VERSION_ID  NVCF function-version-id (if required)
  BHASHINI_TRITON_MODEL_NAME           Triton model name (default: indicparler_tts)
  BHASHINI_TRITON_USE_PLAINTEXT        Set to "true" to disable TLS (default: TLS on)
  BHASHINI_TRITON_TIMEOUT_S            Per-request timeout in seconds (default: 120)
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncGenerator

import numpy as np
import tritonclient.grpc as grpcclient
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    """Return env var value or raise a clear error at service startup."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(
            f"BhashiniTTSService: required environment variable '{name}' is not set. "
            "Add it to your .env file."
        )
    return value


def _optional_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _make_triton_inputs(prompt: str, description: str) -> list:
    """Build the two Triton InferInput tensors expected by indicparler_tts."""
    prompt_input = grpcclient.InferInput("PROMPT", [1], "BYTES")
    desc_input = grpcclient.InferInput("DESCRIPTION", [1], "BYTES")
    prompt_input.set_data_from_numpy(np.array([prompt], dtype=object))
    desc_input.set_data_from_numpy(np.array([description], dtype=object))
    return [prompt_input, desc_input]


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class BhashiniTTSService(TTSService):
    """Text-to-speech via NVIDIA Triton gRPC streaming (indicparler_tts model).

    All connection parameters are read from environment variables so that no
    endpoint or credential is hard-coded in source.  See module docstring for
    the full list of variables.
    """

    def __init__(
        self,
        *,
        speaker: str = "Divya",
        description: str = "A clear, natural voice with good audio quality.",
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        # --- Connection config from .env ---
        self._triton_url = _require_env("BHASHINI_TRITON_URL")
        self._api_key = _require_env("BHASHINI_TRITON_API_KEY")
        self._function_id = _require_env("BHASHINI_TRITON_FUNCTION_ID")
        self._function_version_id = _optional_env("BHASHINI_TRITON_FUNCTION_VERSION_ID")
        self._model_name = _optional_env("BHASHINI_TRITON_MODEL_NAME", "indicparler_tts")
        self._use_tls = _optional_env("BHASHINI_TRITON_USE_PLAINTEXT", "false").lower() != "true"
        self._timeout_s = float(_optional_env("BHASHINI_TRITON_TIMEOUT_S", "120"))

        # --- Voice config ---
        self._speaker = speaker
        self._description = description

        logger.info(
            "BhashiniTTSService initialised | url={} model={} tls={} timeout={}s",
            self._triton_url,
            self._model_name,
            self._use_tls,
            self._timeout_s,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict:
        headers = {
            "authorization": f"Bearer {self._api_key}",
            "function-id": self._function_id,
        }
        if self._function_version_id:
            headers["function-version-id"] = self._function_version_id
        return headers

    def _full_description(self) -> str:
        """Build the description string passed to the model.

        indicparler_tts conditions voice style on the description field.
        Prepend speaker name when configured, matching the load-test convention.
        """
        if self._speaker:
            return f"{self._speaker}. {self._description}"
        return self._description

    # ------------------------------------------------------------------
    # TTSService implementation
    # ------------------------------------------------------------------

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        if not text.strip():
            return

        loop = asyncio.get_event_loop()
        result_queue: asyncio.Queue = asyncio.Queue()

        # Triton gRPC callbacks run on a background thread; push results onto
        # an asyncio queue so the async generator can yield them safely.
        def _on_result(result, error):
            loop.call_soon_threadsafe(result_queue.put_nowait, (result, error))

        client: grpcclient.InferenceServerClient | None = None
        try:
            client = grpcclient.InferenceServerClient(
                url=self._triton_url,
                ssl=self._use_tls,
                verbose=False,
            )
            client.start_stream(callback=_on_result, headers=self._build_headers())

            inputs = _make_triton_inputs(text, self._full_description())
            request_id = f"bhashini-tts-{id(text)}"
            client.async_stream_infer(
                model_name=self._model_name,
                inputs=inputs,
                request_id=request_id,
            )

            yield TTSStartedFrame()

            chunk_count = 0
            deadline = loop.time() + self._timeout_s

            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    yield ErrorFrame("Bhashini TTS request timed out")
                    return

                try:
                    result, error = await asyncio.wait_for(
                        result_queue.get(), timeout=min(remaining, 10.0)
                    )
                except asyncio.TimeoutError:
                    yield ErrorFrame("Bhashini TTS request timed out")
                    return

                if error is not None:
                    logger.error("Bhashini Triton gRPC error: {}", error)
                    yield ErrorFrame(f"Triton gRPC error: {error}")
                    return

                status = _decode(result.as_numpy("STATUS")[0])
                is_final = bool(result.as_numpy("IS_FINAL")[0])
                audio_chunk: np.ndarray = result.as_numpy("AUDIO_CHUNK")

                if status == "audio" and audio_chunk is not None and audio_chunk.size > 0:
                    # AUDIO_CHUNK arrives as int16 PCM from indicparler_tts
                    pcm_bytes = audio_chunk.astype(np.int16).tobytes()
                    chunk_count += 1
                    logger.debug(
                        "Bhashini TTS chunk {} | {} bytes", chunk_count, len(pcm_bytes)
                    )
                    yield TTSAudioRawFrame(
                        audio=pcm_bytes,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )

                if is_final:
                    logger.info(
                        "Bhashini TTS complete | chunks={} text_len={}", chunk_count, len(text)
                    )
                    break

            yield TTSStoppedFrame()

        except Exception as e:
            logger.error("Bhashini TTS unexpected error: {}", e)
            yield ErrorFrame(f"Bhashini TTS error: {e}")
        finally:
            if client is not None:
                try:
                    client.stop_stream()
                except Exception:
                    pass
