import asyncio
from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class MarathiIdlePromptFilter(FrameProcessor):
    """For Marathi calls, plays an idle prompt if user stays silent after bot speech."""

    IDLE_PROMPT_TEXT = "हॅलो, तुम्ही अजून कॉलवर आहात का"

    def __init__(self, timeout_secs: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self._timeout_secs = timeout_secs
        self._idle_task: Optional[asyncio.Task] = None
        self._ignore_next_bot_stop = False

    def _cancel_idle_timer(self) -> None:
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = None

    def _schedule_idle_timer(self) -> None:
        self._cancel_idle_timer()

        async def _timer() -> None:
            try:
                await asyncio.sleep(self._timeout_secs)
                logger.info(
                    f"User silence detected for {self._timeout_secs}s, sending Marathi idle prompt"
                )
                self._ignore_next_bot_stop = True
                # This processor is placed after TTS in the pipeline, so send
                # the speak frame upstream to ensure it flows through TTS.
                await self.push_frame(
                    TTSSpeakFrame(self.IDLE_PROMPT_TEXT),
                    FrameDirection.UPSTREAM,
                )
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error(f"Failed to send Marathi idle prompt: {exc}")

        self._idle_task = asyncio.create_task(_timer())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._cancel_idle_timer()

        elif isinstance(frame, BotStartedSpeakingFrame):
            self._cancel_idle_timer()

        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._ignore_next_bot_stop:
                self._ignore_next_bot_stop = False
            else:
                self._schedule_idle_timer()

        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._cancel_idle_timer()

        await self.push_frame(frame, direction)