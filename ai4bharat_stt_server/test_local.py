import base64
import sys
import numpy as np
import soundfile as sf
import requests

# ── Config ───────────────────────────────────────
AUDIO_PATH  = "/raid_new/bhashiniuser/triton_testing/asr/22_language_audios/data_sub/Hindi/rv3/valid/5348024557564075.wav"   # change this
LANGUAGE_ID = "hi"               # change this (hi, ta, te, bn, mr, ...)
SERVER_URL  = "http://localhost:8032/transcribe"
# ─────────────────────────────────────────────────

# Read audio and convert to mono 16 kHz int16
data, sr = sf.read(AUDIO_PATH, dtype="int16", always_2d=False)
if data.ndim > 1:
    data = data[:, 0]                          # stereo → mono

if sr != 16000:
    # resample if needed
    import librosa
    data = (librosa.resample(
                data.astype(np.float32) / 32768.0,
                orig_sr=sr, target_sr=16000
            ) * 32767).astype(np.int16)

audio_b64 = base64.b64encode(data.tobytes()).decode()

resp = requests.post(SERVER_URL,
                     json={"audio_b64": audio_b64, "language_id": LANGUAGE_ID})
resp.raise_for_status()
print("Transcript:", resp.json()["text"])