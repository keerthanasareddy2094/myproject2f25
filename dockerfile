# Use slim Python base
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (curl for health checks, locales optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Requirements first (for better Docker layer caching)
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install Playwright browsers + OS deps for Chromium inside the image
# (playwright has built-in helper for apt dependencies)
RUN python -m playwright install --with-deps chromium

# Copy app code
COPY app.py auto_apply.py /app/

# Port for Streamlit
EXPOSE 8501

# Default env (can be overridden)
ENV OLLAMA_HOST=http://ollama:11434 \
    MODEL_NAME=qwen2:0.5b \
    NUM_CTX=2048 \
    MAX_TOKENS=256 \
    SYSTEM_PROMPT="Answer concisely (2–5 sentences)."

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
