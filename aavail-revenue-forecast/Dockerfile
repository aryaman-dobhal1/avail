FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AAVAIL_DATA_DIR=/app/data/cs-train \
    AAVAIL_MODEL_DIR=/app/models \
    AAVAIL_LOG_PATH=/app/logs/runtime.jsonl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home appuser \
    && mkdir -p /app/data /app/models /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8080/logfile', timeout=3)"

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", \
     "--timeout", "900", "app:create_app()"]
