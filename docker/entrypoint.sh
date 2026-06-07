#!/usr/bin/env bash
# Sparklers VoiceClone Studio — container entrypoint.
#
# On every start:
#   1. Symlink the baked-in OpenVoice v2 checkpoints into the runtime
#      models dir so the user's volume mount doesn't shadow them.
#   2. Make sure the writable dirs exist on the mounted volumes.
#   3. Surface that tegrastats is available (it ships in the L4T BSP
#      and is bind-mounted via docker-compose).
#   4. exec the CMD (uvicorn by default).

set -euo pipefail

log() { echo "[entrypoint] $*"; }

DATA_DIR=${SPARKLERS_DATA_DIR:-/app/data}
MODELS_DIR=${SPARKLERS_MODELS_DIR:-/app/models}

mkdir -p \
  "${DATA_DIR}/enrollments" "${DATA_DIR}/out" \
  "${MODELS_DIR}/voices"

# F5-TTS + Vocos + Whisper checkpoints are baked into the image under
# /root/.cache/huggingface — they live with the container, not the
# volumes, so no symlinking is needed.

# tegrastats comes from the L4T BSP on the host (bind-mounted).
if command -v tegrastats >/dev/null 2>&1; then
    log "tegrastats: available"
else
    log "tegrastats: NOT available — dashboard will show empty sys panel"
fi

log "data:    ${DATA_DIR}"
log "models:  ${MODELS_DIR}"
log "running: $*"
exec "$@"
