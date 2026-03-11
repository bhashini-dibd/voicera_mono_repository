import asyncio
import base64
import os
from typing import AsyncGenerator, Optional

from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

try:
    import socketio
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("Install with: pip install python-socketio[asyncio_client] aiohttp")
    raise Exception(f"Missing module: {e}")


class BhashiniTTSService(TTSService):
    """Bhashini real-time TTS over Socket.IO websocket."""

    def __init__(
        self,
        *,
        speaker: str = "Divya",
        description: str = "A clear, natural voice with good audio quality.",
        sample_rate: int = 24000,
        api_key: Optional[str] = None,
        socket_url: Optional[str] = None,
        socket_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = (api_key or os.getenv("BHASHINI_API_KEY", "")).strip()
        if not self._api_key:
            raise ValueError("BHASHINI_API_KEY environment variable not set")

        self._socket_url = (
            socket_url
            or os.getenv("BHASHINI_TTS_SERVER_URL")
            or os.getenv("BHASHINI_SOCKET_URL")
            or "wss://dhruva-api.bhashini.gov.in"
        ).strip()
        self._socket_path = (socket_path or os.getenv("BHASHINI_TTS_SOCKET_PATH", "/socket_tts.io")).strip()

        self._speaker = speaker
        self._description = description
        self._request_timeout = int(os.getenv("BHASHINI_TTS_TIMEOUT_SECONDS", "90"))

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        if not text.strip():
            return

        queue: asyncio.Queue = asyncio.Queue()
        stream_finished = asyncio.Event()
        stream_sample_rate = self.sample_rate
        sio = socketio.AsyncClient(reconnection=False)

        @sio.event
        async def connect():
            logger.debug(f"Bhashini TTS connected: {self._socket_url}")
            await sio.emit("start")

        @sio.event
        async def connect_error(data):
            await queue.put(("error", f"Connection error: {data}"))
            stream_finished.set()

        @sio.on("ready")
        async def on_ready():
            payload = {
                "text": text,
                "description": self._description,
                "speaker": self._speaker,
            }
            await sio.emit("data", payload)

        @sio.on("response")
        async def on_response(data):
            nonlocal stream_sample_rate
            stream_sample_rate = int(data.get("sampleRate", stream_sample_rate))
            audio_b64 = data.get("audioContent")
            if audio_b64:
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    await queue.put(("audio", audio_bytes))
                except Exception as e:
                    await queue.put(("error", f"Audio decode error: {e}"))
                    stream_finished.set()
                    return

            if data.get("isFinal", False):
                stream_finished.set()

        @sio.on("abort")
        async def on_abort(data):
            await queue.put(("error", f"Server aborted stream: {data}"))
            stream_finished.set()

        @sio.on("terminate")
        async def on_terminate():
            stream_finished.set()

        @sio.event
        async def disconnect():
            logger.debug("Bhashini TTS disconnected")
            stream_finished.set()

        try:
            yield TTSStartedFrame()

            await sio.connect(
                f"{self._socket_url}?output_sample_rate={self.sample_rate}",
                transports=["websocket"],
                socketio_path=self._socket_path,
                headers={"Authorization": self._api_key},
            )

            while True:
                if stream_finished.is_set() and queue.empty():
                    break
                try:
                    kind, payload = await asyncio.wait_for(queue.get(), timeout=0.5)
                    if kind == "audio":
                        yield TTSAudioRawFrame(
                            audio=payload,
                            sample_rate=stream_sample_rate,
                            num_channels=1,
                        )
                    elif kind == "error":
                        yield ErrorFrame(str(payload))
                        break
                except asyncio.TimeoutError:
                    continue

            yield TTSStoppedFrame()

        except asyncio.TimeoutError:
            yield ErrorFrame("Bhashini TTS request timeout")
        except Exception as e:
            yield ErrorFrame(f"Bhashini TTS error: {e}")
        finally:
            try:
                if sio.connected:
                    await sio.disconnect()
            except Exception as e:
                logger.warning(f"Bhashini TTS disconnect error: {e}")
