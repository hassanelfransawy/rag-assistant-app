# Container image for the FastAPI backend.
#
# Build from the backend/ directory:
#     docker build -t rag-backend .
#
# Run it (Ollama runs on the HOST, not in this container, so we point the
# container at the host's Ollama):
#     docker run -p 8000:8000 \
#       -e OLLAMA_HOST=http://host.docker.internal:11434 \
#       -v "$(pwd)/data/vector_store:/app/data/vector_store:ro" \
#       rag-backend

FROM python:3.11-slim

# Keeps Python from writing .pyc files and makes logs appear immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install dependencies first, in their own layer. Docker can then reuse this
# layer whenever only application code changes, which makes rebuilds fast.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# The vector store is mounted at runtime (see the -v flag above) rather than
# baked into the image, so the image stays small and the index can be updated
# without a rebuild.
RUN mkdir -p /app/data/vector_store /app/.cache/huggingface

# Do not run as root.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
