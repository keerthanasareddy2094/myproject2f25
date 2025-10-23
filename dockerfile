FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# OPTIONAL: enable Playwright for JS-heavy career sites
# Uncomment next 3 lines if you want Playwright inside the container.
# RUN pip install --no-cache-dir playwright==1.47.0
# RUN playwright install chromium
# RUN playwright install-deps chromium

COPY app.py scraper.py search.py styles.css ./

ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    MODEL_NAME=qwen2:0.5b

EXPOSE 8501
CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]
