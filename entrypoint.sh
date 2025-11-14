#!/usr/bin/env bash
set -euo pipefail

# Set defaults for all environment variables
export STREAMLIT_SERVER_PORT="${STREAMLIT_SERVER_PORT:-5002}"
export STREAMLIT_SERVER_BASE_URL_PATH="${STREAMLIT_SERVER_BASE_URL_PATH:-team2f25}"
export MODEL_NAME="${MODEL_NAME:-qwen2.5:0.5b}"
export USE_OLLAMA="${USE_OLLAMA:-1}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export BACKEND_PORT="${BACKEND_PORT:-8000}"

if command -v sed >/dev/null 2>&1; then
  sed -i 's/\r$//' entrypoint.sh || true
fi

echo "=== CSUSB Internship Finder - Startup ==="
echo "OLLAMA_HOST: $OLLAMA_HOST"
echo "MODEL_NAME: $MODEL_NAME"
echo "BACKEND_PORT: $BACKEND_PORT"
echo "STREAMLIT_PORT: $STREAMLIT_SERVER_PORT"
echo ""

# ============================================================================
# 1. START OLLAMA
# ============================================================================
echo "[1/4] Checking Ollama..."
if command -v ollama >/dev/null 2>&1; then
  echo "Ollama found. Starting service..."
  (ollama serve >/tmp/ollama.log 2>&1 &) || true
  sleep 3
  
  echo "Waiting for Ollama to be ready..."
  MAX_RETRIES=30
  RETRY=0
  while [ $RETRY -lt $MAX_RETRIES ]; do
    if curl -s "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      echo "Ollama is ready!"
      break
    fi
    echo "  Retry $((RETRY+1))/$MAX_RETRIES..."
    sleep 1
    RETRY=$((RETRY+1))
  done
  
  if [ $RETRY -ge $MAX_RETRIES ]; then
    echo "ERROR: Ollama failed to start"
    cat /tmp/ollama.log
    exit 1
  fi
  
  # Pull model if needed
  if ! curl -s "${OLLAMA_HOST}/api/tags" | grep -q "\"name\":\"${MODEL_NAME}\""; then
    echo "Pulling model: $MODEL_NAME"
    ollama pull "${MODEL_NAME}" || {
      echo "ERROR: Failed to pull model"
      exit 1
    }
  else
    echo "Model $MODEL_NAME already available"
  fi
else
  echo "WARNING: Ollama not found. Backend will fail to connect."
fi

# ============================================================================
# 2. START BACKEND
# ============================================================================
echo ""
echo "[2/4] Starting backend navigator..."
python -m uvicorn backend_navigator:app \
  --host 0.0.0.0 \
  --port "$BACKEND_PORT" \
  --log-level info \
  >/tmp/backend.log 2>&1 &

BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Wait for backend to be ready
echo "Waiting for backend to be ready..."
MAX_RETRIES=30
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
  if curl -s "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1; then
    echo "Backend is ready!"
    break
  fi
  echo "  Retry $((RETRY+1))/$MAX_RETRIES..."
  sleep 1
  RETRY=$((RETRY+1))
done

if [ $RETRY -ge $MAX_RETRIES ]; then
  echo "ERROR: Backend failed to start"
  cat /tmp/backend.log
  kill $BACKEND_PID 2>/dev/null || true
  exit 1
fi

# ============================================================================
# 3. TEST BACKEND CONNECTIVITY
# ============================================================================
echo ""
echo "[3/4] Testing backend connectivity..."
RESPONSE=$(curl -s -X POST "http://localhost:${BACKEND_PORT}/fetch" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.google.com"}' 2>&1 || echo "FAILED")

if echo "$RESPONSE" | grep -q "FAILED\|error\|refused"; then
  echo "ERROR: Backend is not responding correctly"
  echo "Response: $RESPONSE"
  kill $BACKEND_PID 2>/dev/null || true
  exit 1
else
  echo "Backend connectivity OK"
fi

# ============================================================================
# 4. START STREAMLIT
# ============================================================================
echo ""
echo "[4/4] Starting Streamlit..."
echo "Access the app at: http://localhost:${STREAMLIT_SERVER_PORT}/${STREAMLIT_SERVER_BASE_URL_PATH}"
echo ""

exec streamlit run app.py \
  --server.port "$STREAMLIT_SERVER_PORT" \
  --server.baseUrlPath "$STREAMLIT_SERVER_BASE_URL_PATH" \
  --browser.gatherUsageStats false
