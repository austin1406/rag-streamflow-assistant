FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/austin1406/rag-streamflow-assistant"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY rag ./rag
COPY eval ./eval
COPY templates ./templates
COPY static ./static
COPY data/raw ./data/raw

# Bake the vector index, the evaluation database, and the embedding model into the image
# so the container starts with no downloads and no extra setup step.
RUN python -m rag.ingest && python -m rag.eval_db

RUN useradd --create-home appuser && chown -R appuser /app /opt/hf-cache
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "2", "--timeout", "180", "app:app"]
