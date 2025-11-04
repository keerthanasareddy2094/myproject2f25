# Requirements first
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Playwright browsers + deps
RUN python -m playwright install --with-deps chromium

# Copy ALL code now
COPY . /app/

EXPOSE 8501

ENV OLLAMA_HOST=http://ollama:11434 \
    MODEL_NAME=qwen2:0.5b \
    NUM_CTX=2048 \
    MAX_TOKENS=256 \
    SYSTEM_PROMPT="Answer concisely (2–5 sentences)."

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
