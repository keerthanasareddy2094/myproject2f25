#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-onboarding-phase1-container}"
IMAGE="${IMAGE:-onboarding-phase1}"
REMOVE_IMAGE="${REMOVE_IMAGE:-false}"
PORT="8501"

echo "🧹 Cleaning up container(s)..."
docker rm -f "${NAME}" >/dev/null 2>&1 || true

# Stop anything else on the same port
CID=$(docker ps --filter "publish=${PORT}" --format "{{.ID}}" || true)
if [[ -n "${CID}" ]]; then
  echo "Stopping container publishing port ${PORT} (${CID})"
  docker stop "${CID}" >/dev/null 2>&1 || true
fi

if [[ "${REMOVE_IMAGE}" == "true" ]]; then
  echo "Removing image ${IMAGE}"
  docker rmi "${IMAGE}" >/dev/null 2>&1 || true
fi

echo "✅ Cleanup complete."
