"""Utilities for submitting call recording data to the backend API."""

import os
import re
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from storage.minio_client import MinIOStorage


def _extract_llm_responses(transcript_content: Optional[str], agent_config: dict) -> List[Dict[str, Any]]:
    """Extract assistant turns from the saved transcript content."""
    if not transcript_content:
        return []

    llm_model = agent_config.get("llm_model", {}) or {}
    provider_name = llm_model.get("name")
    model_name = llm_model.get("model")

    responses: List[Dict[str, Any]] = []
    for raw_line in transcript_content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = re.match(r"^\[([^\]]+)\]\s*(user|assistant|agent|human|bot):\s*(.+)$", line, re.IGNORECASE)
        if match:
            timestamp, role, message = match.groups()
            if role.lower() in ("assistant", "agent", "bot"):
                responses.append(
                    {
                        "role": "assistant",
                        "content": message.strip(),
                        "timestamp": timestamp.strip(),
                        "provider": provider_name,
                        "model": model_name,
                    }
                )
            continue

        if line.lower().startswith(("assistant:", "agent:", "bot:")):
            content = re.sub(r"^(agent|assistant|bot):\s*", "", line, flags=re.IGNORECASE).strip()
            if content:
                responses.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "provider": provider_name,
                        "model": model_name,
                    }
                )

    return responses


async def submit_call_recording(
    call_sid: str,
    agent_type: str,
    agent_config: dict,
    storage: MinIOStorage,
    call_start_time: float,
) -> None:
    """
    Submit call recording data to the backend API after a call ends.

    This function reads the transcript from MinIO, builds the recording URLs,
    and sends all call metadata to the backend API endpoint.
    """
    try:
        logger.info(f"Submitting call recording data to backend after call ends: {call_sid}")
        call_end_time = time.monotonic()
        call_duration = call_end_time - call_start_time
        end_time_utc = datetime.utcnow().isoformat()

        recording_url = f"minio://recordings/{call_sid}.wav"
        transcript_url = f"minio://transcripts/{call_sid}.txt"

        transcript_content = None
        try:
            response = await storage.get_object("transcripts", f"{call_sid}.txt")
            transcript_content = response.read().decode("utf-8")
            response.close()
            response.release_conn()
        except Exception as e:
            logger.warning(f"Could not read transcript: {e}")

        llm_responses = _extract_llm_responses(transcript_content, agent_config)

        backend_url = os.getenv("VOICERA_BACKEND_URL", "http://localhost:8000")
        api_endpoint = f"{backend_url}/api/v1/call-recordings"

        payload = {
            "call_sid": call_sid,
            "recording_url": recording_url,
            "transcript_url": transcript_url,
            "transcript_content": transcript_content,
            "llm_responses": llm_responses,
            "agent_type": agent_type,
            "call_duration": call_duration,
            "end_time_utc": end_time_utc,
        }

        if "org_id" in agent_config:
            payload["org_id"] = agent_config["org_id"]

        logger.info(f"Sending call recording data to backend: {call_sid}")
        response = requests.post(api_endpoint, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"Call recording data saved successfully: {call_sid}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send call recording data: {e}")
    except Exception as e:
        logger.error(f"Error processing call recording data: {e}")
        logger.debug(traceback.format_exc())
