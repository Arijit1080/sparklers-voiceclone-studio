# syntax=docker/dockerfile:1.7
#
# Sparklers VoiceClone Studio — Dockerfile (F5-TTS edition)
# ---------------------------------------------------------
# Targets NVIDIA Jetson Orin on JetPack 6.x (L4T R36.x), CUDA 12.6.
#
# Two big design choices:
#   1. PyTorch is fetched directly from jetson-ai-lab.io by exact wheel
#      URL — pip's resolver otherwise prefers the CPU-only wheel from
#      PyPI when the version strings collide.
#   2. F5-TTS's load_checkpoint() is patched at build time so big
#      safetensors don't trip Tegra's NVML caching-allocator assert
#      (CUDACachingAllocator.cpp:1131). The patches are applied with
#      tools/patch_f5_load.py and tools/patch_f5_load_v4.py.
#
# Layers are ordered slowest-to-change → fastest-to-change so editing
# Python code rebuilds in seconds, not minutes.

ARG L4T_TAG=r36.4.0
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# ----- 1. System packages -----
# ffmpeg is needed by Whisper's pipeline for arbitrary-format audio
# decoding even though we mostly feed it 16 kHz mono WAV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-setuptools python3-wheel \
        libopenblas0-pthread libsndfile1 ffmpeg \
        alsa-utils libasound2-plugins libportaudio2 \
        unzip wget git \
        tzdata ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN command -v tegrastats >/dev/null 2>&1 \
    || echo "WARN: tegrastats not in image — sys monitor will be empty"

# ----- 2. Python: numpy first so downstream wheels link cleanly -----
RUN pip3 install --upgrade pip wheel setuptools && \
    pip3 install 'numpy<2'

# ----- 3. PyTorch + torchaudio for Jetson JP6/CU126 (CUDA-enabled) -----
# Direct URL — pip would otherwise prefer the PyPI CPU build.
RUN pip3 install \
        "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl" \
        "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/81a/775c8af36ac85/torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl"

# ----- 4. The rest of the Python deps -----
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# ----- 5. Vendor F5-TTS + apply Tegra allocator patches -----
WORKDIR /opt
RUN git clone --depth=1 https://github.com/SWivid/F5-TTS F5-TTS

COPY tools/patch_f5_load.py     /tmp/patch_f5_load.py
COPY tools/patch_f5_load_v4.py  /tmp/patch_f5_load_v4.py
RUN python3 /tmp/patch_f5_load.py    /opt/F5-TTS/src/f5_tts/infer/utils_infer.py && \
    python3 /tmp/patch_f5_load_v4.py /opt/F5-TTS/src/f5_tts/infer/utils_infer.py && \
    pip3 install --no-build-isolation --no-deps -e /opt/F5-TTS

# ----- 6. Pre-download model checkpoints — bakes ~1.7 GB into image -----
#  - F5-TTS v1 base safetensors (~1.35 GB) + Vocos vocoder (~50 MB)
#  - Whisper tiny.en (~150 MB) for reference transcription
RUN python3 - <<'PY'
import time, torch
print("downloading F5-TTS v1 + Vocos…", flush=True)
from huggingface_hub import hf_hub_download
hf_hub_download("SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors")
hf_hub_download("charactr/vocos-mel-24khz", "config.yaml")
hf_hub_download("charactr/vocos-mel-24khz", "pytorch_model.bin")
print("downloading openai/whisper-tiny.en…", flush=True)
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
AutoProcessor.from_pretrained("openai/whisper-tiny.en")
AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny.en")
print("done", flush=True)
PY

# ----- 7. App code (LAST so edits don't bust the heavy layers above) -----
WORKDIR /app
COPY audio/ /app/audio/
COPY voice/ /app/voice/
COPY tools/ /app/tools/
COPY web/   /app/web/
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

VOLUME ["/app/data", "/app/models"]

ENV PYTHONPATH=/app \
    SPARKLERS_DATA_DIR=/app/data \
    SPARKLERS_MODELS_DIR=/app/models \
    NLTK_DATA=/usr/local/share/nltk_data

EXPOSE 8083

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8083"]
