###############################################################################
# Dockerfile
# Multi-stage build for the Energy Trading MLOps pipeline image.
# Stage 1: builder (installs deps)
# Stage 2: runtime (lean production image)
###############################################################################

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for Python packages with C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="mlops-team@energytrading.io"
LABEL description="Energy Trading DataOps/MLOps Pipeline"
LABEL version="1.0.0"

# Non-root user for security
RUN groupadd --gid 1001 mlops \
 && useradd --uid 1001 --gid mlops --shell /bin/bash --create-home mlops

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY --chown=mlops:mlops src/ ./src/
COPY --chown=mlops:mlops config/ ./config/
COPY --chown=mlops:mlops tests/ ./tests/

# Create directories for MLflow artifacts and logs
RUN mkdir -p /app/mlruns /app/logs \
 && chown -R mlops:mlops /app

USER mlops

ENV PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

# Health check - verifies the pipeline config loads
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "from src.utils.helpers import load_pipeline_config; load_pipeline_config()" \
    || exit 1

# Default command: run the ingestion module
CMD ["python", "-m", "src.ingestion.ingest"]
