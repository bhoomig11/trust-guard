FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    TRUSTGUARD_CHROMA_DIR=/app/chroma_data \
    TRUSTGUARD_LOG_PATH=/app/logs/governance.jsonl \
    TRUSTGUARD_DATA_DIR=/app/data/policies

WORKDIR /app

# CPU-only PyTorch keeps the image far smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install . && python -m spacy download en_core_web_md

# Bake the embedding model into the image so startup needs no network access.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY data ./data
COPY scripts ./scripts
# Public FinCEN guidance is optional; the build continues if the download fails.
RUN python scripts/download_public_docs.py || echo "Public documents not downloaded"

RUN useradd --create-home --uid 1000 trustguard \
    && mkdir -p /app/chroma_data /app/logs \
    && chown -R trustguard:trustguard /app
USER trustguard

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "trustguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
