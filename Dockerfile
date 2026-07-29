# =============================================================================
# OpenCodeReview — Dockerfile
# =============================================================================
# Multi-stage build:
#   1. builder — installs all Python dependencies
#   2. runtime — minimal image with just the app + deps
#
# Usage
# -----
#   docker build -t opencodereview .
#   docker run --rm opencodereview review --smoke
#   docker run --rm -e GROQ_API_KEY=gsk_... -e GITHUB_TOKEN=ghp_... \
#       opencodereview review --repo org/repo --pr 42
#
# To include sentence-transformers (needed for vector-store retrieval):
#   docker build --target runtime-full -t opencodereview:full .
#
# Environment variables
# ---------------------
#   GROQ_API_KEY      Required for AI reviewers (correctness, security, etc.)
#   GITHUB_TOKEN      Required for fetching PRs and posting comments
# =============================================================================

# -- Stage 1: Builder (core + optional deps, except heavy ML) -----------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Install system build deps needed by chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Core dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional dependencies (chromadb for vector store, langchain-groq for LLM)
# ragas and litellm are installed here (not just via requirements.txt)
# so pip resolves all transitive deps (numpy, etc.) together with
# chromadb, avoiding version conflicts at runtime.
RUN pip install --no-cache-dir \
    chromadb \
    langchain-groq \
    langfuse \
    langchain-google-genai \
    "ragas>=0.3,<0.4" \
    litellm

# -- Stage 2: Runtime (default) -----------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# chromadb/onnx needs libgomp at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Fallback: multi-stage COPY may not transfer packages reliably on HF Spaces build infra.
# Install ragas+litellm normally (with deps) so transitive dependencies like
# numpy, datasets, etc. are available at import time.
# Pin numpy<2 to stay compatible with chromadb (uses numpy.float_ removed in 2.x)
RUN pip install --no-cache-dir langfuse "ragas>=0.3,<0.4" "numpy<2" litellm

# Verify ragas + chromadb are both importable (numpy version compatibility check)
RUN python -c "import ragas, chromadb; print(f'ragas {ragas.__version__} + chromadb OK')"

# Copy application code
COPY . .
COPY docker-entrypoint.sh /docker-entrypoint.sh

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app \
    && chmod +x /docker-entrypoint.sh \
    && mkdir -p /tmp/opencodereview_vectors \
    && chown -R appuser:appuser /tmp/opencodereview_vectors

# The entrypoint runs as root to fix volume permissions, then drops to appuser.
# Default: run the Gradio web UI on port 7860 (HF Spaces expects this).
# Override CMD for CLI usage:
#   docker run --rm opencodereview python -m main review --smoke
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "app.py"]

# -- Stage 3: Full runtime (includes sentence-transformers for embeddings) -----
# Build with: docker build --target runtime-full -t opencodereview:full .
FROM runtime AS runtime-full

USER root
RUN pip install --no-cache-dir sentence-transformers
# Note: No USER appuser here — ENTRYPOINT (inherited from runtime) handles privilege dropping

# Note: sentence-transformers pulls in PyTorch (~800 MB), making the image
# significantly larger. Only use this target when you need the retrieval node
# to build vector indexes from scratch.
