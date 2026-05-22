import asyncio
import base64
import binascii
import logging
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn

import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
import torchaudio
from dotenv import load_dotenv

load_dotenv()

# =========================
# Logging setup
# =========================

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai4bharat_stt")

# =========================
# FastAPI setup
# =========================

app = FastAPI()

# =========================
# Request/Response Models
# =========================

class TranscribeRequest(BaseModel):
    audio_b64: str
    language_id: str = "hi"


class TranscribeResponse(BaseModel):
    text: str

# =========================
# Constants
# =========================

TARGET_SAMPLE_RATE      = 16000
MIN_SAMPLES             = 1600      # 100 ms at 16 kHz — minimum for useful ASR
MIN_STFT_SAFE_SAMPLES   = 513       # n_fft=512 → 256 reflect-pad; input must exceed this
QUEUE_MAXSIZE           = 256
MAX_BATCH_SIZE          = 16
BATCH_TIMEOUT           = 0.100     # seconds — batch collection window
RESPONSE_TIMEOUT        = 30.0      # seconds — max wait for worker response

BHILI_ENABLE = "no"

device     = "cuda:0" if torch.cuda.is_available() else "cpu"
main_model = None
bhili_model = None


# =========================
# Model loading
# =========================

def _required_model_path(env_var_name: str) -> Path:
    env_value = (os.environ.get(env_var_name) or "").strip()
    if not env_value:
        raise RuntimeError(
            f"Missing required environment variable: {env_var_name}. "
            f"Please set it in ai4bharat_stt_server/.env"
        )
    path = Path(env_value).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise RuntimeError(
            f"Invalid {env_var_name}: file not found at {path}. "
            "Please update ai4bharat_stt_server/.env"
        )
    return path


def load_main_model():
    model_path = _required_model_path("INDIC_NEMO_PATH")
    logger.info("Loading main ASR model from: %s", model_path)
    t0 = time.perf_counter()
    model = nemo_asr.models.ASRModel.restore_from(restore_path=str(model_path))
    model = model.to(device)
    model.freeze()
    model.cur_decoder = "rnnt"
    elapsed = time.perf_counter() - t0
    logger.info(
        "Main ASR model loaded | device=%s | decoder=rnnt | load_time=%.2fs",
        device, elapsed,
    )
    return model


def load_bhili_model():
    model_path = _required_model_path("BHILI_NEMO_PATH")
    logger.info("Loading Bhili model from: %s", model_path)
    t0 = time.perf_counter()
    model = EncDecHybridRNNTCTCBPEModel.restore_from(str(model_path)).to(device).eval()
    elapsed = time.perf_counter() - t0
    logger.info(
        "Bhili model loaded | device=%s | load_time=%.2fs", device, elapsed
    )
    return model


# =========================
# Audio helpers
# =========================

def _decode_audio_b64(audio_b64: str) -> np.ndarray:
    """
    Decode base64-encoded raw int16 PCM → float32 normalised to [-1, 1].
    Raises HTTP 400 on malformed base64 or odd-length payload.
    """
    b64_len = len(audio_b64)
    logger.debug("Decoding audio | base64_chars=%d (~%.1f KB)", b64_len, b64_len / 1365)

    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except binascii.Error as exc:
        logger.warning("Base64 decode failed: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid base64 audio") from exc

    byte_count = len(audio_bytes)
    if byte_count % np.dtype(np.int16).itemsize != 0:
        logger.warning(
            "Odd byte length rejected | bytes=%d", byte_count
        )
        raise HTTPException(
            status_code=400,
            detail="PCM16 audio payload has an odd byte length",
        )

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    samples   = len(audio_np)
    duration_ms = samples / TARGET_SAMPLE_RATE * 1000

    logger.info(
        "Audio decoded | bytes=%d | samples=%d | duration=%.1fms (%.3fs) | rate=%dHz",
        byte_count, samples, duration_ms, duration_ms / 1000, TARGET_SAMPLE_RATE,
    )

    return audio_np


def _sanitize_audio(audio_np: np.ndarray) -> np.ndarray:
    """
    Ensure float32, 1-D, finite values, C-contiguous.
    Logs any corrections applied.
    """
    original_shape = audio_np.shape
    audio_np = np.asarray(audio_np, dtype=np.float32).squeeze()

    if audio_np.ndim != 1:
        logger.warning(
            "Audio sanitize: non-1D after squeeze | original_shape=%s → returning empty",
            original_shape,
        )
        return np.asarray([], dtype=np.float32)

    if original_shape != audio_np.shape:
        logger.debug(
            "Audio sanitize: squeezed shape %s → %s", original_shape, audio_np.shape
        )

    nan_inf_count = int(np.sum(~np.isfinite(audio_np)))
    if nan_inf_count > 0:
        logger.warning(
            "Audio sanitize: replaced %d NaN/Inf values with zeros", nan_inf_count
        )
        audio_np = np.nan_to_num(audio_np)

    return np.ascontiguousarray(audio_np, dtype=np.float32)


def _enqueue_request(
    request_queue: queue.Queue, audio_np: np.ndarray, language_id: str
) -> queue.Queue:
    response_queue = queue.Queue(maxsize=1)
    request_item   = {
        "audio_np":       audio_np,
        "language_id":    language_id,
        "response_queue": response_queue,
        "enqueue_time":   time.perf_counter(),  # for end-to-end latency tracking
    }

    queue_size_before = request_queue.qsize()
    try:
        request_queue.put(request_item, timeout=1.0)
    except queue.Full:
        logger.error(
            "Request queue full | language_id=%s | queue_size=%d",
            language_id, queue_size_before,
        )
        raise HTTPException(status_code=503, detail="STT queue is full")

    logger.debug(
        "Enqueued request | language_id=%s | samples=%d | duration=%.1fms | queue_depth=%d",
        language_id,
        len(audio_np),
        len(audio_np) / TARGET_SAMPLE_RATE * 1000,
        request_queue.qsize(),
    )
    return response_queue


# =========================
# Queues
# =========================

main_request_queue  = queue.Queue(maxsize=QUEUE_MAXSIZE)
bhili_request_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)


# =========================
# Inference functions
# =========================

def _bhili_language_id(language_id: str) -> str:
    if (language_id or "").strip().lower() == "bhb":
        return "mr"
    return language_id or "mr"


def main_infer(audio_arrays, language_ids):
    results         = [""] * len(audio_arrays)
    valid_audio     = []
    valid_positions = []

    logger.debug("main_infer: received batch of %d item(s)", len(audio_arrays))

    for idx, audio_np in enumerate(audio_arrays):
        audio_np    = _sanitize_audio(audio_np)
        samples     = len(audio_np)
        duration_ms = samples / TARGET_SAMPLE_RATE * 1000
        lang        = language_ids[idx]

        if samples < MIN_SAMPLES:
            logger.info(
                "main_infer: SKIP too-short clip | idx=%d | samples=%d | "
                "duration=%.1fms | language_id=%s | min_required=%d samples (%.0fms)",
                idx, samples, duration_ms, lang,
                MIN_SAMPLES, MIN_SAMPLES / TARGET_SAMPLE_RATE * 1000,
            )
            continue

        if samples < MIN_STFT_SAFE_SAMPLES:
            logger.debug(
                "main_infer: padding clip below STFT safe floor | idx=%d | "
                "samples=%d → %d", idx, samples, MIN_STFT_SAFE_SAMPLES,
            )
            audio_np = np.pad(audio_np, (0, MIN_STFT_SAFE_SAMPLES - samples))

        logger.debug(
            "main_infer: queuing clip for inference | idx=%d | samples=%d | "
            "duration=%.1fms (%.3fs) | language_id=%s",
            idx, samples, duration_ms, duration_ms / 1000, lang,
        )
        valid_audio.append(audio_np)
        valid_positions.append(idx)

    if not valid_audio:
        logger.info(
            "main_infer: all %d clip(s) filtered — returning empty results",
            len(audio_arrays),
        )
        return results

    logger.info(
        "main_infer: running inference | valid_clips=%d/%d | language_id=%s",
        len(valid_audio), len(audio_arrays), language_ids[valid_positions[0]],
    )

    infer_t0 = time.perf_counter()
    with torch.no_grad():
        transcriptions = main_model.transcribe(
            audio=valid_audio,
            batch_size=len(valid_audio),
            language_id=language_ids[valid_positions[0]],
        )[0]
    infer_elapsed_ms = (time.perf_counter() - infer_t0) * 1000

    total_audio_ms = sum(
        len(valid_audio[i]) / TARGET_SAMPLE_RATE * 1000
        for i in range(len(valid_audio))
    )
    logger.info(
        "main_infer: inference complete | clips=%d | inference_time=%.1fms | "
        "total_audio=%.1fms | RTF=%.3f",
        len(valid_audio),
        infer_elapsed_ms,
        total_audio_ms,
        infer_elapsed_ms / total_audio_ms if total_audio_ms > 0 else 0,
    )

    for pos, (idx, text) in enumerate(zip(valid_positions, transcriptions)):
        text       = str(text).strip() if text is not None else ""
        samples    = len(valid_audio[pos])
        dur_ms     = samples / TARGET_SAMPLE_RATE * 1000
        results[idx] = text
        logger.info(
            "main_infer: transcript | idx=%d | samples=%d | duration=%.1fms | "
            "language_id=%s | transcript=%r",
            idx, samples, dur_ms, language_ids[idx], text,
        )

    return results


def bhili_infer(audio_arrays, language_ids):
    results = []
    logger.debug("bhili_infer: received batch of %d item(s)", len(audio_arrays))

    for idx, (audio_np, language_id) in enumerate(zip(audio_arrays, language_ids)):
        audio_np    = _sanitize_audio(audio_np)
        samples     = len(audio_np)
        duration_ms = samples / TARGET_SAMPLE_RATE * 1000

        if samples < MIN_SAMPLES:
            logger.info(
                "bhili_infer: SKIP too-short clip | idx=%d | samples=%d | "
                "duration=%.1fms | language_id=%s",
                idx, samples, duration_ms, language_id,
            )
            results.append("")
            continue

        logger.debug(
            "bhili_infer: processing clip | idx=%d | samples=%d | "
            "duration=%.1fms | language_id=%s",
            idx, samples, duration_ms, language_id,
        )

        waveform = torch.from_numpy(audio_np).float().unsqueeze(0)
        tmp_path = f"/tmp/bhili_{threading.get_ident()}_{time.time_ns()}.wav"
        torchaudio.save(tmp_path, waveform, TARGET_SAMPLE_RATE)

        try:
            nemo_lid = _bhili_language_id(language_id)
            logger.debug(
                "bhili_infer: running inference | idx=%d | nemo_lid=%s | tmp=%s",
                idx, nemo_lid, tmp_path,
            )
            infer_t0 = time.perf_counter()
            with torch.no_grad():
                out = bhili_model.transcribe([tmp_path], language_id=nemo_lid)
            infer_ms = (time.perf_counter() - infer_t0) * 1000

            text = out[0][0] if (out and out[0]) else out
            text = str(text).strip() if text is not None else ""
            results.append(text)

            logger.info(
                "bhili_infer: transcript | idx=%d | samples=%d | duration=%.1fms | "
                "inference_time=%.1fms | language_id=%s | transcript=%r",
                idx, samples, duration_ms, infer_ms, language_id, text,
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return results


# =========================
# Batch worker thread
# =========================

def batch_worker(request_queue, infer_fn):
    """
    Collects requests into batches, groups by language, runs inference,
    returns results or exceptions to each caller's response_queue.
    """
    worker_name = threading.current_thread().name
    logger.info("Batch worker started | thread=%s | fn=%s", worker_name, infer_fn.__name__)

    while True:
        batch = []
        start = time.time()

        while len(batch) < MAX_BATCH_SIZE:
            remaining = BATCH_TIMEOUT - (time.time() - start)
            if remaining <= 0:
                break
            try:
                item = request_queue.get(timeout=remaining)
                batch.append(item)
            except queue.Empty:
                break

        if not batch:
            continue

        logger.debug(
            "batch_worker [%s]: collected batch | size=%d | languages=%s",
            worker_name,
            len(batch),
            list({(item["language_id"] or "").strip().lower() for item in batch}),
        )

        # Group by language — NeMo accepts one language_id per batch call
        grouped: dict[str, list] = {}
        for item in batch:
            lang = (item["language_id"] or "").strip().lower()
            grouped.setdefault(lang, []).append(item)

        for lang, grouped_batch in grouped.items():
            audio_arrays = [item["audio_np"]    for item in grouped_batch]
            language_ids = [item["language_id"] for item in grouped_batch]
            total_ms     = sum(
                len(a) / TARGET_SAMPLE_RATE * 1000 for a in audio_arrays
            )

            logger.info(
                "batch_worker [%s]: inferring group | language=%s | clips=%d | "
                "total_audio=%.1fms",
                worker_name, lang, len(grouped_batch), total_ms,
            )

            group_t0 = time.perf_counter()
            try:
                transcriptions = infer_fn(audio_arrays, language_ids)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - group_t0) * 1000
                logger.exception(
                    "batch_worker [%s]: inference FAILED | language=%s | clips=%d | "
                    "elapsed=%.1fms | error=%s",
                    worker_name, lang, len(grouped_batch), elapsed_ms, exc,
                )
                for item in grouped_batch:
                    item["response_queue"].put(exc)
                continue

            elapsed_ms = (time.perf_counter() - group_t0) * 1000
            logger.info(
                "batch_worker [%s]: group done | language=%s | clips=%d | "
                "wall_time=%.1fms",
                worker_name, lang, len(grouped_batch), elapsed_ms,
            )

            for item, text in zip(grouped_batch, transcriptions):
                # Log end-to-end latency from enqueue to result ready
                e2e_ms = (time.perf_counter() - item["enqueue_time"]) * 1000
                logger.debug(
                    "batch_worker [%s]: result ready | language=%s | e2e_latency=%.1fms | "
                    "transcript=%r",
                    worker_name, lang, e2e_ms, text,
                )
                item["response_queue"].put(text)


async def _wait_for_response(response_queue: queue.Queue) -> str:
    """
    Await worker result with timeout.
    Raises HTTP 504 on timeout, HTTP 500 on worker exception.
    """
    wait_t0 = time.perf_counter()
    try:
        result = await asyncio.to_thread(
            response_queue.get, True, RESPONSE_TIMEOUT
        )
    except queue.Empty as exc:
        wait_ms = (time.perf_counter() - wait_t0) * 1000
        logger.error(
            "_wait_for_response: TIMEOUT after %.1fms (limit=%.0fs)",
            wait_ms, RESPONSE_TIMEOUT,
        )
        raise HTTPException(status_code=504, detail="STT inference timed out") from exc

    wait_ms = (time.perf_counter() - wait_t0) * 1000
    if isinstance(result, Exception):
        logger.error(
            "_wait_for_response: worker returned exception after %.1fms | error=%s",
            wait_ms, result,
        )
        raise HTTPException(status_code=500, detail="STT inference failed") from result

    logger.debug("_wait_for_response: got result after %.1fms", wait_ms)
    return result


def _start_workers():
    threading.Thread(
        target=batch_worker,
        args=(main_request_queue, main_infer),
        name="worker-main",
        daemon=True,
    ).start()
    if BHILI_ENABLE == "yes":
        threading.Thread(
            target=batch_worker,
            args=(bhili_request_queue, bhili_infer),
            name="worker-bhili",
            daemon=True,
        ).start()


# =========================
# Startup
# =========================

@app.on_event("startup")
async def startup_event():
    global main_model, bhili_model
    logger.info("Server starting up | device=%s", device)

    main_model = load_main_model()
    if BHILI_ENABLE == "yes":
        bhili_model = load_bhili_model()
    else:
        bhili_model = None
        logger.info("Bhili model disabled (BHILI_ENABLE=no)")

    _start_workers()
    logger.info("All workers started — server ready")


# =========================
# Routes
# =========================

@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    req_t0 = time.perf_counter()
    logger.info(
        "POST /transcribe | language_id=%s | b64_len=%d",
        request.language_id, len(request.audio_b64),
    )

    audio_np = _decode_audio_b64(request.audio_b64)
    samples  = len(audio_np)
    dur_ms   = samples / TARGET_SAMPLE_RATE * 1000

    logger.info(
        "POST /transcribe | audio_ready | samples=%d | duration=%.1fms (%.3fs) | "
        "language_id=%s",
        samples, dur_ms, dur_ms / 1000, request.language_id,
    )

    response_queue = _enqueue_request(main_request_queue, audio_np, request.language_id)
    result         = await _wait_for_response(response_queue)

    total_ms = (time.perf_counter() - req_t0) * 1000
    logger.info(
        "POST /transcribe | DONE | samples=%d | audio_duration=%.1fms | "
        "total_latency=%.1fms | language_id=%s | transcript=%r",
        samples, dur_ms, total_ms, request.language_id, result,
    )

    return TranscribeResponse(text=result)


@app.post("/transcribe/bhili", response_model=TranscribeResponse)
async def transcribe_bhili(request: TranscribeRequest):
    req_t0 = time.perf_counter()
    logger.info(
        "POST /transcribe/bhili | language_id=%s | b64_len=%d",
        request.language_id, len(request.audio_b64),
    )

    if BHILI_ENABLE != "yes":
        raise HTTPException(status_code=503, detail="Bhili model is disabled")
    if bhili_model is None:
        raise HTTPException(status_code=503, detail="Bhili model not loaded")

    audio_np = _decode_audio_b64(request.audio_b64)
    samples  = len(audio_np)
    dur_ms   = samples / TARGET_SAMPLE_RATE * 1000

    logger.info(
        "POST /transcribe/bhili | audio_ready | samples=%d | duration=%.1fms | "
        "language_id=%s",
        samples, dur_ms, request.language_id,
    )

    response_queue = _enqueue_request(bhili_request_queue, audio_np, request.language_id)
    result         = await _wait_for_response(response_queue)

    total_ms = (time.perf_counter() - req_t0) * 1000
    logger.info(
        "POST /transcribe/bhili | DONE | samples=%d | audio_duration=%.1fms | "
        "total_latency=%.1fms | language_id=%s | transcript=%r",
        samples, dur_ms, total_ms, request.language_id, result,
    )

    return TranscribeResponse(text=result)


@app.get("/health")
def health():
    logger.debug("GET /health")
    return {
        "status":           "healthy",
        "device":           device,
        "bhili_enabled":    BHILI_ENABLE,
        "main_loaded":      main_model is not None,
        "bhili_loaded":     bhili_model is not None,
        "main_queue_size":  main_request_queue.qsize(),
        "bhili_queue_size": bhili_request_queue.qsize(),
        "max_batch_size":   MAX_BATCH_SIZE,
        "batch_timeout_ms": int(BATCH_TIMEOUT * 1000),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
