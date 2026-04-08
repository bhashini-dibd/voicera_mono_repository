"""
Parallel WebSocket TTS clients with a fixed gap between each client's start.

Each client sends one (prompt, description) pair chosen at random from
EXAMPLE_UTTERANCES (prompts in Hindi, Telugu, or Kannada; descriptions in English).

Run server first, e.g. `python server.py`, then:
  python tests/multi_ws_smoke.py
  python tests/multi_ws_smoke.py -n 10 --gap-ms 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import websockets
from scipy.io import wavfile

OUT_DIR = Path(__file__).resolve().parent / "files"

# Ten fixed (prompt, description) pairs — regional prompts, English descriptions. Random per request.
EXAMPLE_UTTERANCES: list[tuple[str, str]] = [
    # Hindi
    (
        "नमस्ते, आप कैसे हैं? आज दिन कैसा रहा?",
        "A calm, clear female voice speaking at a normal pace.",
    ),
    (
        "आज मौसम बहुत सुहावना है, बाहर घूमने का मन कर रहा है।",
        "A warm, friendly male voice.",
    ),
    (
        "कृपया धीरे और साफ़ बोलें, मैं सुन रहा हूँ।",
        "A projected, articulate voice with crisp pronunciation.",
    ),
    (
        "यह एक छोटा परीक्षण वाक्य है, सब ठीक से सुनाई दे रहा है क्या?",
        "A soft, gentle female voice speaking slowly.",
    ),
    # Telugu
    (
        "నమస్కారం, మీరు ఎలా ఉన్నారు? ఈ రోజు ఎలా గడిచింది?",
        "A peaceful female voice with clear articulation.",
    ),
    (
        "ఈ రోజు వాతావరణం చాలా బాగుంది, బయటకు వెళ్ళడానికి మంచి రోజు.",
        "A strong, friendly male voice.",
    ),
    (
        "దయచేసి నెమ్మదిగా మాట్లాడండి, నేను వింటున్నాను.",
        "A delicate, relaxed speaking tone.",
    ),
    # Kannada
    (
        "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ? ಇಂದು ದಿನ ಹೇಗೆ ಕಳೆಯಿತು?",
        "A steady female voice with precise, clear speech.",
    ),
    (
        "ಇಂದು ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ, ಹೊರಗೆ ಹೋಗುವುದಕ್ಕೆ ಒಳ್ಳೆಯ ದಿನ.",
        "A warm male voice with an upbeat, positive tone.",
    ),
    (
        "ದಯವಿಟ್ಟು ನಿಧಾನವಾಗಿ ಹೇಳಿ, ನಾನು ಕೇಳುತ್ತಿದ್ದೇನೆ.",
        "A soft, clear voice at a slow, easy pace.",
    ),
]


def safe_filename_from_prompt(prompt: str, max_len: int = 120) -> str:
    s = prompt.strip()
    s = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", s)
    s = re.sub(r"\s+", "_", s)
    s = s.strip("._") or "output"
    return s[:max_len]


async def run_one_request(
    index: int,
    gap_s: float,
    uri: str,
    prompt: str,
    description: str,
    out_dir: Path,
    strict: bool,
) -> tuple[int, float | None, float | None, Path | None, str]:
    """Sleep ``index * gap_s``, then one utterance.

    Returns (index, ttft_ms or None, mean_chunk_gap_ms or None, wav_path or None, prompt).
    """
    await asyncio.sleep(index * gap_s)

    chunks: list[np.ndarray] = []
    ttft_ms: float | None = None
    inter_chunk_ms: list[float] = []
    last_recv_mono: float | None = None

    async with websockets.connect(uri) as ws:
        t_before_send = time.monotonic()
        await ws.send(json.dumps({"prompt": prompt, "description": description}))

        meta = json.loads(await ws.recv())
        if meta["type"] != "meta":
            raise RuntimeError(f"expected meta, got {meta}")
        sample_rate = int(meta.get("sample_rate", 24000))
        pid = str(meta.get("pid", f"req{index}"))

        while True:
            msg = await ws.recv()
            now = time.monotonic()
            if isinstance(msg, str):
                body = json.loads(msg)
                if body["type"] == "error":
                    raise RuntimeError(f"request {index}: server error {body!r}")
                if body["type"] != "done":
                    raise RuntimeError(f"expected done, got {body}")
                break
            if ttft_ms is None:
                ttft_ms = (now - t_before_send) * 1000.0
            elif last_recv_mono is not None:
                inter_chunk_ms.append((now - last_recv_mono) * 1000.0)
            last_recv_mono = now
            chunks.append(np.frombuffer(msg, dtype=np.float32))

    pcm = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

    if pcm.size == 0:
        msg = (
            f"request {index}: no PCM (meta then done; use server --decode-every 1 "
            f"or a longer prompt — short runs with --decode-every 60 often skip DAC)"
        )
        if strict:
            raise RuntimeError(msg)
        print(f"WARN {msg}")
        return index, None, None, None, prompt

    base = safe_filename_from_prompt(prompt)
    out_path = out_dir / f"{base}_{index:02d}_{pid}.wav"
    wavfile.write(out_path, sample_rate, pcm)

    mean_chunk_ms: float | None
    if inter_chunk_ms:
        mean_chunk_ms = float(np.mean(inter_chunk_ms))
    else:
        mean_chunk_ms = None  # only one audio chunk

    return index, ttft_ms, mean_chunk_ms, out_path, prompt


async def async_main(
    n_requests: int,
    gap_ms: float,
    uri: str,
    strict: bool,
    rng: random.Random,
) -> None:
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    gap_s = gap_ms / 1000.0

    pairs = [rng.choice(EXAMPLE_UTTERANCES) for _ in range(n_requests)]

    tasks = [
        asyncio.create_task(
            run_one_request(
                i, gap_s, uri, pairs[i][0], pairs[i][1], out_dir, strict,
            ),
        )
        for i in range(n_requests)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    failures = [r for r in results if isinstance(r, BaseException)]
    if failures:
        for r in failures:
            print(f"ERROR {r}")
        if strict:
            raise failures[0]

    oks = [r for r in results if not isinstance(r, BaseException)]
    oks.sort(key=lambda r: r[0])
    print(f"uri={uri} n={n_requests} gap_ms={gap_ms}\n")
    for idx, ttft_ms, mean_chunk_ms, path, prompt in oks:
        snippet = prompt if len(prompt) <= 50 else prompt[:47] + "..."
        if path is None:
            print(
                f"[{idx:02d}] ttft_ms=n/a  mean_inter_chunk_ms=n/a  -> (no wav)  | {snippet}",
            )
            continue
        chunk_str = f"{mean_chunk_ms:.2f}" if mean_chunk_ms is not None else "n/a (single chunk)"
        ttft_str = f"{ttft_ms:.2f}" if ttft_ms is not None else "n/a"
        print(
            f"[{idx:02d}] ttft_ms={ttft_str}  mean_inter_chunk_ms={chunk_str}  "
            f"-> {path.name}  | {snippet}",
        )
    if failures and not strict:
        print(f"finished with {len(failures)} error(s); use --strict to fail fast")
    elif not failures:
        print("ok")


def main() -> None:
    p = argparse.ArgumentParser(description="Many staggered WS TTS clients + latency stats")
    p.add_argument("-n", "--requests", type=int, default=10, help="Number of parallel clients (default 10)")
    p.add_argument(
        "--gap-ms",
        type=float,
        default=20.0,
        help="Delay between starting each client: client i sleeps i * gap (default 20)",
    )
    p.add_argument("--uri", default="ws://127.0.0.1:8002")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed so random prompt/description picks are reproducible",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Raise on server errors or if an utterance returns no PCM",
    )
    args = p.parse_args()
    if args.requests < 1:
        p.error("--requests must be >= 1")
    if args.gap_ms < 0:
        p.error("--gap-ms must be >= 0")

    rng = random.Random(args.seed)

    asyncio.run(
        async_main(
            n_requests=args.requests,
            gap_ms=args.gap_ms,
            uri=args.uri,
            strict=args.strict,
            rng=rng,
        ),
    )


if __name__ == "__main__":
    main()
