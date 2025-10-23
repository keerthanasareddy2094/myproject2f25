#!/usr/bin/env bash
set -euo pipefail

: "${STREAMLIT_SERVER_PORT:=8501}"
: "${MODEL_NAME:=qwen2:0.5b}"
: "${OLLAMA_HOST:=http://127.0.0.1:11434}"

echo "🌐 OLLAMA_HOST = ${OLLAMA_HOST}"
echo "🤖 MODEL_NAME  = ${MODEL_NAME}"

# Convert line endings (safety)
if command -v sed >/dev/null 2>&1; then
  sed -i 's/\r$//' /app/entrypoint.sh || true
fi

# Optional: pull model if Ollama accessible
if command -v curl >/dev/null 2>&1 && curl -s "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
  echo "Checking for model ${MODEL_NAME}..."
  if ! curl -s "${OLLAMA_HOST}/api/tags" | grep -q "\"name\":\"${MODEL_NAME}\""; then
    echo "Pulling ${MODEL_NAME}..."
    curl -s -X POST -H "Content-Type: application/json" \
         -d "{\"name\":\"${MODEL_NAME}\"}" "${OLLAMA_HOST}/api/pull" || true
  fi
else
  echo "⚠️  Ollama service not reachable at ${OLLAMA_HOST} (continuing anyway)"
fi

# Start Streamlit
exec streamlit run app.py \
  --server.port "${STREAMLIT_SERVER_PORT}" \
  --server.address "0.0.0.0" \
  --browser.gatherUsageStats false
