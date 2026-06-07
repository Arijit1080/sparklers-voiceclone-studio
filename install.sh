#!/usr/bin/env bash
# Sparklers VoiceClone Studio — one-shot installer for Jetson Orin.
#
# Pulls the prebuilt container from GHCR and starts it via docker
# compose.  After install, the UI is at http://<jetson-ip>:8083
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Arijit1080/sparklers-voiceclone-studio/main/install.sh | bash

set -e

REPO_URL="https://raw.githubusercontent.com/Arijit1080/sparklers-voiceclone-studio/main"
INSTALL_DIR="${INSTALL_DIR:-$HOME/sparklers-voiceclone-studio}"

echo "==> Sparklers VoiceClone Studio installer"
echo "    target: $INSTALL_DIR"

# 1) sanity: docker + compose v2
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found — install docker first (apt install docker.io)" >&2
    exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: docker compose v2 plugin not found — install docker-compose-plugin" >&2
    exit 1
fi

# 2) folder layout
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# 3) compose file
if [ ! -f docker-compose.yml ]; then
    echo "==> fetching docker-compose.yml"
    curl -fsSL "$REPO_URL/docker-compose.yml" -o docker-compose.yml
fi

# 4) pull + up
echo "==> pulling image…"
docker compose pull
echo "==> starting container…"
docker compose up -d

# 5) tiny readiness check
echo "==> waiting for the UI to respond on :8083 (up to 60 s)…"
for i in $(seq 1 60); do
    if curl -sf -o /dev/null http://127.0.0.1:8083/healthz; then
        echo
        echo "✨ done!"
        ip=$(hostname -I 2>/dev/null | awk '{print $1}')
        echo "    UI:  http://${ip:-<jetson-ip>}:8083"
        echo "    logs:  docker compose logs -f"
        exit 0
    fi
    sleep 1
    printf "."
done
echo
echo "WARN: didn't get a healthy /healthz in 60 s — check 'docker compose logs'."
exit 1
