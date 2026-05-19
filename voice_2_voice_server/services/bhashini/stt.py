"""Bhashini HTTP REST STT Service for Pipecat"""

import base64
import io
import time
import wave
from typing import AsyncGenerator, Optional
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    ErrorFrame,
    StartFrame,
    EndFrame,
    CancelFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.utils.time import time_now_iso8601

try:
    import aiohttp
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("Install with: pip install aiohttp")
    raise Exception(f"Missing module: {e}")


class BhashiniSTTService(STTService):
    """Bhashini STT using the /services/inference/pipeline REST API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://dhruva-api.bhashini.gov.in",
        service_id: str = "bhashini/ai4bharat/conformer-multilingual-asr",
        language: str = "hi",
        sample_rate: int = 16000,
        audio_channels: int = 1,
        audio_format: str = "wav",
        vad_analyzer: Optional[VADAnalyzer] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._endpoint = f"{base_url.rstrip('/')}/services/inference/pipeline"
        self._service_id = service_id
        self._language = language
        self._sample_rate = sample_rate
        self._audio_channels = audio_channels
        self._audio_format = audio_format

        self._session: Optional[aiohttp.ClientSession] = None

        # Buffer raw PCM bytes while the user is speaking; flush on stop.
        self._audio_buffer: bytes = b""
        self._is_speaking = False

        # Interim / VAD-aware state (ported from IndicConformerRESTSTTService)
        self._vad_analyzer: Optional[VADAnalyzer] = vad_analyzer
        self._text_chunks: list[str] = []
        self._stopping_start_time: Optional[float] = None
        self._stopping_triggered = False
        self._STOPPING_DURATION_MS = 10

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": "oMowmlxsAdR3TqmLaY7mG_SBYTjGmg_p124n8G6FKA67dkmoDLWBtNElIpQQqxHk",
                "Content-Type": "application/json",
                "Accept": "*/*",
            }
        )
        self._is_speaking = False
        self._audio_buffer = b""
        self._text_chunks = []
        self._stopping_start_time = None
        self._stopping_triggered = False
        logger.info("Bhashini STT service started")

    async def stop(self, frame: EndFrame):
        await self._close_session()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._audio_buffer = b""
        self._text_chunks = []
        await self._close_session()
        await super().cancel(frame)

    async def _close_session(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug("Bhashini HTTP session closed")

    # ------------------------------------------------------------------
    # Audio handling
    # ------------------------------------------------------------------

    def _pcm_to_wav_b64(self, pcm_data: bytes) -> str:
        """Wrap raw PCM bytes into a WAV file and return base64 string."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self._audio_channels)
            wf.setsampwidth(2)          # 16-bit PCM
            wf.setframerate(self._sample_rate)
            wf.writeframes(pcm_data)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _build_payload(self, audio_b64: str) -> dict:
        return {
            "pipelineTasks": [
                {
                    "taskType": "asr",
                    "config": {
                        "language": {"sourceLanguage": self._language},
                        "serviceId": self._service_id,
                    },
                }
            ],
            "inputData": {
                "audio": [{"audioContent": audio_b64}]
            },
        }

    async def _transcribe_buffer(self) -> str:
        if not self._audio_buffer or len(self._audio_buffer) < 3200:
            return ""
        try:
            audio_b64 = self._pcm_to_wav_b64(self._audio_buffer)
            return await self._transcribe(audio_b64) or ""
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    async def _transcribe(self, audio_b64: str) -> Optional[str]:
        """POST to Bhashini pipeline and return the transcript string."""
        if not self._session:
            logger.error("No active HTTP session")
            return None

        payload = self._build_payload(audio_b64)
        try:
            async with self._session.post(self._endpoint, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Bhashini API error {resp.status}: {text}")
                    return None

                data = await resp.json()

            pipeline_response = data.get("pipelineResponse", [])
            if not pipeline_response:
                return None

            outputs = pipeline_response[0].get("output", [])
            if not outputs:
                return None

            # Join all output chunks into a single transcript
            transcript = ". ".join(
                chunk.get("source", "").strip()
                for chunk in outputs
                if chunk.get("source", "").strip()
            )
            return transcript or None

        except aiohttp.ClientError as e:
            logger.error(f"Bhashini HTTP request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Bhashini transcription error: {e}")
            return None

    # ------------------------------------------------------------------
    # VAD stopping-state detection (ported from IndicConformerRESTSTTService)
    # ------------------------------------------------------------------

    def _check_stopping_state(self) -> bool:
        if self._vad_analyzer is None:
            return False

        try:
            vad_state = self._vad_analyzer._vad_state

            if vad_state == VADState.STOPPING:
                current_time = time.time() * 1000

                if self._stopping_start_time is None:
                    self._stopping_start_time = current_time
                    return False

                duration_ms = current_time - self._stopping_start_time

                if duration_ms >= self._STOPPING_DURATION_MS and not self._stopping_triggered:
                    self._stopping_triggered = True
                    return True

                return False
            else:
                self._stopping_start_time = None
                self._stopping_triggered = False
                return False

        except AttributeError:
            return False

    # ------------------------------------------------------------------
    # STTService interface
    # ------------------------------------------------------------------

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not audio:
            return

        try:
            self._audio_buffer += audio

            if self._check_stopping_state():
                logger.info("STOPPING state triggered, transcribing buffer")
                text = await self._transcribe_buffer()
                if text:
                    self._text_chunks.append(text)
                    accumulated = " ".join(self._text_chunks)
                    logger.info(f"Interim: {accumulated}")
                    yield InterimTranscriptionFrame(
                        text=accumulated,
                        user_id=getattr(self, "_user_id", ""),
                        timestamp=str(int(time.time() * 1000)),
                    )
                self._audio_buffer = b""

        except Exception as e:
            logger.error(f"STT processing error: {e}")
            yield ErrorFrame(f"STT processing failed: {str(e)}")

    # ------------------------------------------------------------------
    # Speaking detection + frame ordering
    # (UserStoppedSpeakingFrame handled BEFORE super() so TranscriptionFrame
    #  reaches the aggregator before the stop frame — ported from IndicConformer)
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("User stopped speaking — flushing buffer to Bhashini")
            self._is_speaking = False
            self._stopping_start_time = None
            self._stopping_triggered = False

            # Transcribe any remaining buffered audio
            if self._audio_buffer:
                transcript = await self._transcribe_buffer()
                if transcript:
                    self._text_chunks.append(transcript)
            self._audio_buffer = b""

            # Push final accumulated transcript BEFORE UserStoppedSpeakingFrame
            if self._text_chunks:
                accumulated = " ".join(self._text_chunks)
                logger.info(f"Final: {accumulated}")
                await self.push_frame(TranscriptionFrame(
                    text=accumulated,
                    user_id=getattr(self, "_user_id", ""),
                    timestamp=time_now_iso8601(),
                ))
                self._text_chunks = []

        # Now let super() push UserStoppedSpeakingFrame downstream
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("User started speaking — buffering audio")
            self._is_speaking = True
            self._stopping_start_time = None
            self._stopping_triggered = False
            self._audio_buffer = b""
            self._text_chunks = []

    # ------------------------------------------------------------------
    # Runtime config changes
    # ------------------------------------------------------------------

    async def set_language(self, language: str):
        logger.info(f"Switching language to: {language}")
        self._language = language

    async def set_model(self, service_id: str):
        logger.info(f"Switching service to: {service_id}")
        self._service_id = service_id

    def can_generate_metrics(self) -> bool:
        return True