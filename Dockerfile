# Supreme System V5 - Multi-stage Production Dockerfile
# Optimized for production deployment with minimal image size

# Stage 1: Base Python environment
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 supreme && \
    useradd --uid 1000 --gid supreme --shell /bin/bash --create-home supreme

# Stage 2: Dependencies installation
FROM base as dependencies

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Application build
FROM dependencies as builder

# Copy source code
COPY . .

# Set ownership
RUN chown -R supreme:supreme /app

# Stage 4: Production runtime
FROM python:3.12-slim as production

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    PROMETHEUS_PORT=9090 \
    LOG_LEVEL=info

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 supreme && \
    useradd --uid 1000 --gid supreme --shell /bin/bash --create-home supreme

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder --chown=supreme:supreme /app .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/config && \
    chown -R supreme:supreme /app

# Switch to non-root user
USER supreme

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/api/v1/health || exit 1

# Expose ports
EXPOSE ${API_PORT} ${PROMETHEUS_PORT}

# Default command
CMD ["python", "-m", "src.api.server"]

# Metadata
LABEL maintainer="Supreme System V5 Team" \
      version="5.0.0" \
      description="Revolutionary AI-Powered Trading System" \
      org.opencontainers.image.title="Supreme System V5" \
      org.opencontainers.image.description="World's First Neuromorphic Trading System" \
      org.opencontainers.image.version="5.0.0" \
      org.opencontainers.image.vendor="Supreme System Development Team" \
      org.opencontainers.image.licenses="MIT"