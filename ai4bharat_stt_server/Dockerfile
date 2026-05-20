# ============================================================
# AI4Bharat STT Server — NVIDIA H100 / NVCF Deployment
# Base  : nvcr.io/nvidia/pytorch:25.12-py3
#         (ships with PyTorch, CUDA, cuDNN, Python 3 — no manual
#          torch install needed)
# NeMo  : AI4Bharat fork (nemo-v2 branch)
# Port  : 8000  (NVCF default)
# ============================================================

FROM nvcr.io/nvidia/pytorch:25.12-py3

# ── Runtime env ──────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    PYTHONPATH=/opt/NeMo:/app \
    # Model checkpoint — baked into the image
    INDIC_NEMO_PATH=/app/models/indic_conformer.nemo

# ── System dependencies ──────────────────────────────────────
# Python, pip, torch, CUDA, cuDNN are all already in the base image.
# We only add audio processing tools and build utilities that NeMo needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    sox \
    libsox-fmt-all \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip setuptools wheel

# ── PyTorch stack (pinned versions) ──────────────────────────
# Force exact versions from the cu129 wheel index, overriding
# whatever torch the base image shipped with.
RUN pip install --force-reinstall \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129

# ── Python dependencies ──────────────────────────────────────
COPY requirements_nvcf.txt /tmp/requirements_nvcf.txt
RUN pip install -r /tmp/requirements_nvcf.txt

# ── AI4Bharat NeMo (from source, nemo-v2 branch) ─────────────
# TORCH_INSTALL_TYPE=skip tells reinstall.sh not to touch the
# torch version already present in the base image.
WORKDIR /opt
RUN git clone --depth 1 --branch nemo-v2 \
    https://github.com/AI4Bharat/NeMo.git \
    && cd NeMo \
    && TORCH_INSTALL_TYPE="skip" bash reinstall.sh

# ── Application code ─────────────────────────────────────────
WORKDIR /app
COPY server.py        ./server.py
COPY model.py         ./model.py
COPY .env.example     ./.env.example

# ── Model checkpoint (baked into the image) ───────────────────
# Place indic_conformer.nemo next to the Dockerfile before building.
# It is copied to /app/models/ and INDIC_NEMO_PATH points there.
RUN mkdir -p /app/models
COPY indic_conformer.nemo /app/models/indic_conformer.nemo

# ── Expose & healthcheck ──────────────────────────────────────
EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=15s \
    --start-period=180s \
    --retries=5 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────
# NVCF listens on $PORT (default 8000).
# We pass --port explicitly so server.py's hardcoded 8001 is ignored.
CMD ["sh", "-c", \
    "uvicorn server:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --log-level info"]
