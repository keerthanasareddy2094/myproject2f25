#!/usr/bin/env bash
set -euo pipefail

IMAGE="onboarding-phase1"
NAME="onboarding-phase1-container"
PORT="8501"
OLLAMA_PORT="11434"
DETACH="${DETACH:-false}"

# Convert Windows line endings if needed
if command -v sed >/dev/null 2>&1; then
  sed -i 's/\r$//' startup.sh cleanup.sh entrypoint.sh 2>/dev/null || true
fi

chmod +x entrypoint.sh cleanup.sh || true

echo "🔧 Building image: ${IMAGE}"
docker build -t "${IMAGE}" .

RUN_ARGS=(-p "${PORT}:${PORT}"
          -e "STREAMLIT_SERVER_PORT=${PORT}"
          -e "MODEL_NAME=qwen2:0.5b"
          -e "OLLAMA_HOST=http://host.docker.internal:${OLLAMA_PORT}")

if [[ "${DETACH}" == "true" ]]; then
  echo "🚀 Running container detached..."
  docker run -d --name "${NAME}" "${RUN_ARGS[@]}" "${IMAGE}"
  echo "✅ Visit http://localhost:${PORT}"
else
  echo "🚀 Running container in foreground..."
  docker run --rm "${RUN_ARGS[@]}" "${IMAGE}"
fi
