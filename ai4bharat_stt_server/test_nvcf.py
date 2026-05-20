"""
test_nvcf.py — AI4Bharat STT Server · NVCF Test Suite
======================================================
Usage examples
--------------
# Test deployed NVCF endpoint with a real audio file:
    python test_nvcf.py --url https://<nvcf-endpoint> --audio sample.wav --api-key <NGC_KEY>

# Test locally running container:
    python test_nvcf.py --audio sample.wav

# Change language:
    python test_nvcf.py --audio sample.wav --language ta

# Run only specific tests:
    python test_nvcf.py --audio sample.wav --tests health transcribe

# Benchmark with more requests:
    python test_nvcf.py --audio sample.wav --tests benchmark --bench-n 50

# Concurrent load test:
    python test_nvcf.py --audio sample.wav --tests concurrent --bench-n 40 --bench-workers 8

Arguments
---------
  --url           NVCF or local server base URL  (default: http://localhost:8000)
  --audio         Path to audio file (.wav / .flac / .mp3 etc.)
  --api-key       NGC / NVCF Bearer token (optional for local, required for NVCF)
  --language      Language code sent to the model (default: hi)
  --tests         Which tests to run (default: all)
  --bench-n       Number of requests for benchmark / load tests (default: 20)
  --bench-workers Parallel threads for concurrent test (default: 8)
"""

import argparse
import base64
import io
import os
import sys
import time
import threading
import statistics
import wave
from typing import Optional

import numpy as np
import requests

# ── Parse args early so helpers can reference them ───────────
parser = argparse.ArgumentParser(
    description="AI4Bharat STT NVCF test suite",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument("--url",           default=os.environ.get("NVCF_URL", "http://localhost:8000"),
                    help="Server base URL (default: http://localhost:8000)")
parser.add_argument("--audio",         default=os.environ.get("TEST_AUDIO_PATH", ""),
                    help="Path to audio file to transcribe")
parser.add_argument("--api-key",       default=os.environ.get("NVCF_API_KEY", ""),
                    help="NGC / NVCF API key (Bearer token)")
parser.add_argument("--language",      default="hi",
                    help="Language code: hi, ta, te, bn, mr, gu, kn, ml, pa, ... (default: hi)")
parser.add_argument("--tests",         nargs="*",
                    default=["health", "transcribe", "bhili", "benchmark", "concurrent"],
                    choices=["health", "transcribe", "bhili", "benchmark", "concurrent"],
                    help="Tests to run (default: all)")
parser.add_argument("--bench-n",       type=int, default=20,
                    help="Requests for benchmark / concurrent test (default: 20)")
parser.add_argument("--bench-workers", type=int, default=8,
                    help="Thread workers for concurrent test (default: 8)")
args = parser.parse_args()

BASE_URL = args.url.rstrip("/")
HEADERS  = {"Content-Type": "application/json"}
if args.api_key:
    HEADERS["Authorization"] = f"Bearer {args.api_key}"

TIMEOUT = 120   # seconds — model warm-up on NVCF can be slow


# ─────────────────────────────────────────────────────────────
# Terminal helpers
# ─────────────────────────────────────────────────────────────

def _c(code, text): return f"\033[{code}m{text}\033[0m"
def ok(m):      print(_c(32, f"  ✓  {m}"))
def err(m):     print(_c(31, f"  ✗  {m}"))
def info(m):    print(_c(36, f"  ·  {m}"))
def warn(m):    print(_c(33, f"  !  {m}"))
def section(t): print(f"\n{'─'*58}\n  {_c(1,t)}\n{'─'*58}")


# ─────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────

def _load_audio_b64(path: str) -> tuple[str, str]:
    """
    Load an audio file, convert to mono 16 kHz int16 PCM,
    return (base64_string, source_description).
    Falls back to a synthetic sine wave if no path given.
    """
    if path and os.path.isfile(path):
        try:
            import soundfile as sf
        except ImportError:
            err("soundfile not installed — run: pip install soundfile")
            sys.exit(1)

        data, sr = sf.read(path, dtype="int16", always_2d=False)

        # Stereo → mono
        if data.ndim > 1:
            data = data[:, 0]

        # Resample to 16 kHz if needed
        if sr != 16000:
            try:
                import librosa
                data = (
                    librosa.resample(
                        data.astype(np.float32) / 32768.0,
                        orig_sr=sr,
                        target_sr=16000,
                    ) * 32767
                ).astype(np.int16)
                info(f"Resampled {sr} Hz → 16000 Hz")
            except ImportError:
                warn("librosa not installed — audio sent at original sample rate "
                     "(may affect accuracy). Install: pip install librosa")

        duration_s = len(data) / 16000
        b64 = base64.b64encode(data.tobytes()).decode()
        desc = f"{os.path.basename(path)}  ({duration_s:.2f}s, {len(b64)/1024:.1f} KB base64)"
        return b64, desc

    else:
        if path:
            warn(f"Audio file not found: {path!r}  — falling back to synthetic audio")
        n   = 16000  # 1 second
        t   = np.linspace(0, 1.0, n, endpoint=False)
        pcm = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype(np.int16)
        b64 = base64.b64encode(pcm.tobytes()).decode()
        return b64, "synthetic 1-second 440 Hz sine (no --audio file provided)"


def _post(endpoint: str, payload: dict, label: str = "") -> Optional[requests.Response]:
    url = f"{BASE_URL}{endpoint}"
    try:
        return requests.post(url, json=payload, headers=HEADERS, timeout=TIMEOUT)
    except requests.exceptions.ConnectionError:
        err(f"{label or endpoint}: connection refused — is the server reachable at {BASE_URL}?")
    except requests.exceptions.Timeout:
        err(f"{label or endpoint}: timed out after {TIMEOUT}s")
    return None


# ─────────────────────────────────────────────────────────────
# Test 1 — Health check
# ─────────────────────────────────────────────────────────────

def test_health() -> bool:
    section("1 · Health Check")
    try:
        t0 = time.perf_counter()
        r  = requests.get(f"{BASE_URL}/health", headers=HEADERS, timeout=TIMEOUT)
        ms = (time.perf_counter() - t0) * 1000
    except requests.exceptions.ConnectionError:
        err(f"Cannot reach {BASE_URL} — server not running?")
        return False
    except requests.exceptions.Timeout:
        err(f"Timed out after {TIMEOUT}s")
        return False

    if r.status_code != 200:
        err(f"HTTP {r.status_code}  {r.text[:200]}")
        return False

    d = r.json()
    ok(f"HTTP 200  ({ms:.0f} ms)")

    status      = d.get("status", "unknown")
    device      = d.get("device", "unknown")
    main_loaded = d.get("main_loaded")

    (ok if status == "healthy" else warn)(f"status        = {status}")
    info(f"device        = {device}")
    (ok if main_loaded else warn)(f"main_loaded   = {main_loaded}")
    info(f"bhili_loaded  = {d.get('bhili_loaded')}")
    info(f"main_queue_sz = {d.get('main_queue_size')}")
    info(f"max_batch     = {d.get('max_batch_size')}")
    info(f"batch_timeout = {d.get('batch_timeout_ms')} ms")

    if not main_loaded:
        warn("Main model not loaded — check INDIC_NEMO_PATH inside the container")
    if "cuda" not in str(device).lower():
        warn("Running on CPU — CUDA/GPU not detected")

    return status == "healthy"


# ─────────────────────────────────────────────────────────────
# Test 2 — Single transcription
# ─────────────────────────────────────────────────────────────

def test_transcribe() -> bool:
    section(f"2 · Transcription  (language={args.language})")
    audio_b64, desc = _load_audio_b64(args.audio)
    info(f"audio  : {desc}")

    t0 = time.perf_counter()
    r  = _post("/transcribe",
               {"audio_b64": audio_b64, "language_id": args.language},
               label="transcribe")
    ms = (time.perf_counter() - t0) * 1000

    if r is None:
        return False
    if r.status_code != 200:
        err(f"HTTP {r.status_code}: {r.text[:300]}")
        return False

    text = r.json().get("text", "")
    ok(f"HTTP 200  ({ms:.0f} ms)")
    print(f"\n  {'─'*50}")
    print(f"  Transcript : {text}")
    print(f"  {'─'*50}\n")
    return True


# ─────────────────────────────────────────────────────────────
# Test 3 — Bhili endpoint
# ─────────────────────────────────────────────────────────────

def test_transcribe_bhili() -> bool:
    section("3 · Bhili Transcription  (/transcribe/bhili)")
    try:
        bhili_enabled = requests.get(
            f"{BASE_URL}/health", headers=HEADERS, timeout=10
        ).json().get("bhili_enabled")
        if bhili_enabled != "yes":
            warn("Bhili disabled on this deployment — skipping (BHILI_ENABLE=yes to enable)")
            return True
    except Exception:
        pass

    audio_b64, desc = _load_audio_b64(args.audio)
    info(f"audio  : {desc}")

    t0 = time.perf_counter()
    r  = _post("/transcribe/bhili",
               {"audio_b64": audio_b64, "language_id": "bhb"},
               label="bhili")
    ms = (time.perf_counter() - t0) * 1000

    if r is None:
        return False
    if r.status_code == 503:
        warn(f"503: {r.json().get('detail')} — Bhili not loaded")
        return True
    if r.status_code != 200:
        err(f"HTTP {r.status_code}: {r.text[:300]}")
        return False

    ok(f"HTTP 200  ({ms:.0f} ms)")
    print(f"\n  Transcript : {r.json().get('text', '')}\n")
    return True


# ─────────────────────────────────────────────────────────────
# Test 4 — Sequential latency benchmark
# ─────────────────────────────────────────────────────────────

def test_benchmark() -> bool:
    n = args.bench_n
    section(f"4 · Latency Benchmark  ({n} sequential requests)")
    audio_b64, desc = _load_audio_b64(args.audio)
    info(f"audio  : {desc}")
    payload    = {"audio_b64": audio_b64, "language_id": args.language}
    latencies  = []

    for i in range(n):
        t0 = time.perf_counter()
        r  = _post("/transcribe", payload, label=f"bench[{i}]")
        ms = (time.perf_counter() - t0) * 1000
        if r is None or r.status_code != 200:
            err(f"Request {i} failed — aborting")
            return False
        latencies.append(ms)
        print(f"    [{i+1:3d}/{n}]  {ms:7.1f} ms", end="\r", flush=True)

    print()
    p = sorted(latencies)
    ok(f"All {n} requests succeeded")
    info(f"mean   : {statistics.mean(latencies):.1f} ms")
    info(f"median : {statistics.median(latencies):.1f} ms")
    info(f"p95    : {p[int(0.95*len(p))]:.1f} ms")
    info(f"p99    : {p[min(int(0.99*len(p)), len(p)-1)]:.1f} ms")
    info(f"min    : {min(latencies):.1f} ms")
    info(f"max    : {max(latencies):.1f} ms")
    return True


# ─────────────────────────────────────────────────────────────
# Test 5 — Concurrent load test
# ─────────────────────────────────────────────────────────────

def test_concurrent() -> bool:
    n, w = args.bench_n, args.bench_workers
    section(f"5 · Concurrent Load Test  ({n} requests · {w} threads)")
    audio_b64, desc = _load_audio_b64(args.audio)
    info(f"audio  : {desc}")
    payload  = {"audio_b64": audio_b64, "language_id": args.language}
    results  = []
    errors   = []
    lock     = threading.Lock()

    def _worker(idx):
        t0 = time.perf_counter()
        try:
            r  = requests.post(f"{BASE_URL}/transcribe",
                               json=payload, headers=HEADERS, timeout=TIMEOUT)
            ms = (time.perf_counter() - t0) * 1000
            with lock:
                (results if r.status_code == 200 else errors).append(
                    ms if r.status_code == 200 else f"[{idx}] HTTP {r.status_code}"
                )
        except Exception as exc:
            with lock:
                errors.append(f"[{idx}] {exc}")

    sem     = threading.Semaphore(w)
    threads = []
    wall_t0 = time.perf_counter()

    for i in range(n):
        sem.acquire()
        def _run(i=i):
            try: _worker(i)
            finally: sem.release()
        t = threading.Thread(target=_run)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    wall = time.perf_counter() - wall_t0
    ok(f"{len(results)}/{n} succeeded  ({len(errors)} failed)")
    for e in errors[:5]:
        err(str(e))
    if results:
        p = sorted(results)
        info(f"mean      : {statistics.mean(results):.1f} ms")
        info(f"p95       : {p[int(0.95*len(p))]:.1f} ms")
        info(f"wall time : {wall:.2f}s  →  {len(results)/wall:.1f} req/s")
    return len(errors) == 0


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

TEST_FNS = {
    "health":     test_health,
    "transcribe": test_transcribe,
    "bhili":      test_transcribe_bhili,
    "benchmark":  test_benchmark,
    "concurrent": test_concurrent,
}

print(f"\n{'═'*58}")
print(f"  AI4Bharat STT · NVCF Test Suite")
print(f"  Target   : {BASE_URL}")
print(f"  Audio    : {args.audio or '(synthetic — no --audio given)'}")
print(f"  Language : {args.language}")
print(f"  Auth     : {'Bearer token set' if args.api_key else 'none (local mode)'}")
print(f"{'═'*58}")

passed = failed = 0
for name in args.tests:
    if TEST_FNS[name]():
        passed += 1
    else:
        failed += 1

section("Summary")
(ok if failed == 0 else err)(f"{passed} passed  /  {failed} failed")
print()
sys.exit(0 if failed == 0 else 1)
