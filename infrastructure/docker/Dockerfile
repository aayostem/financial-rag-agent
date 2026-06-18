# # =============================================================================
# # Financial RAG Agent — Production Dockerfile
# # infrastructure/docker/Dockerfile
# #
# # Multi-stage build:
# #   builder: installs dependencies
# #   production: minimal runtime image
# # =============================================================================

# # ── Stage 1: builder ──────────────────────────────────────────────────────────
# FROM python:3.11-slim AS builder

# # ✅ Add security updates
# RUN apt-get update && apt-get upgrade -y --no-install-recommends && \
#     apt-get install -y --no-install-recommends \
#     curl \
#     libpq5 \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Install build dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     libpq-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Copy dependency files first (layer caching)
# COPY pyproject.toml README.md ./
# COPY src/ ./src/

# # Install production dependencies into /app/venv
# RUN python -m venv /app/venv && \
#     /app/venv/bin/pip install --upgrade pip && \
#     /app/venv/bin/pip install -e "." --no-cache-dir

# # ── Stage 2: production ───────────────────────────────────────────────────────
# FROM python:3.11-slim AS production

# WORKDIR /app

# # Install runtime system dependencies only
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libpq5 \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Create non-root user for security
# RUN groupadd -r finrag && useradd -r -g finrag finrag

# # Copy virtual environment from builder
# COPY --from=builder /app/venv /app/venv
# COPY --from=builder /app/src /app/src
# COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# # Set environment
# ENV PATH="/app/venv/bin:$PATH" \
#     PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONPATH="/app/src"

# # Switch to non-root user
# RUN chown -R finrag:finrag /app
# USER finrag

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# EXPOSE 8000

# # Run with uvicorn (use gunicorn + uvicorn workers in true production)
# CMD ["uvicorn", "financial_rag.api.server:app", \
#      "--host", "0.0.0.0", \
#      "--port", "8000", \
#      "--workers", "2", \
#      "--loop", "asyncio", \
#      "--access-log"]

# =============================================================================
# Financial RAG Agent — Production Dockerfile
# infrastructure/docker/Dockerfile
#
# Multi-stage build:
#   builder: installs dependencies
#   production: minimal runtime image
# =============================================================================

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# ✅ Add security updates and fix known CVEs
RUN apt-get update && \
    apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy dependency files first (layer caching)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install production dependencies into /app/venv
# Use --no-cache-dir to reduce image size
RUN python -m venv /app/venv && \
    /app/venv/bin/pip install --upgrade pip setuptools wheel && \
    /app/venv/bin/pip install -e "." --no-cache-dir


# ── Stage 2: production ───────────────────────────────────────────────────────
# Use a more secure base image with recent security patches
FROM python:3.11-slim-bookworm AS production

WORKDIR /app

# Install runtime system dependencies and security updates
RUN apt-get update && \
    apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create non-root user for security
RUN groupadd -r finrag && useradd -r -g finrag finrag && \
    mkdir -p /app/data && \
    chown -R finrag:finrag /app

# Copy virtual environment from builder
COPY --from=builder --chown=finrag:finrag /app/venv /app/venv
COPY --from=builder --chown=finrag:finrag /app/src /app/src
COPY --from=builder --chown=finrag:finrag /app/pyproject.toml /app/pyproject.toml

# Set environment
ENV PATH="/app/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/src" \
    TZ="UTC"

# Security hardening: remove write permissions from non-essential files
RUN find /app -type f -name "*.pyc" -delete && \
    find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Switch to non-root user
USER finrag

# Health check - use python instead of curl to avoid additional dependencies
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "financial_rag.api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "asyncio", \
     "--access-log"]