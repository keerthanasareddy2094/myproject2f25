#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-onboarding-phase1}"

echo "🔧 Building image: ${IMAGE}"
docker compose build

echo "🚀 Starting stack"
docker compose up
