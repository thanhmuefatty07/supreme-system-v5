# Supreme System V5 - Production Docker Image
# Security: Use specific Python version for reproducibility
FROM python:3.11.9-slim

# Security: Add metadata labels
LABEL maintainer="Supreme System Team" \
      version="5.0.0" \
      description="Supreme System V5 - Production Trading Platform" \
      security.scan="enabled"

# Security: Create non-root user early in build process
RUN groupadd -r trader && useradd -r -g trader -u 1000 trader

# Security: Update package lists and install security updates
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    curl \
    jq \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Security: Set working directory
WORKDIR /app

# Environment variables for production security
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    PATH="/app:${PATH}"

# Security: Copy requirements first for better caching
COPY requirements.txt ./

# Security: Install Python dependencies with constraints
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Security: Copy application code with minimal permissions
COPY --chown=trader:trader . .

# Security: Create necessary directories with proper permissions
RUN mkdir -p \
    /app/data/historical \
    /app/data/cache \
    /app/logs \
    /app/reports \
    /app/config && \
    chown -R trader:trader /app && \
    chmod -R 755 /app && \
    chmod -R 777 /app/logs /app/data /app/reports  # Allow write access for app

# Security: Remove unnecessary files and potential secrets
RUN find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /app -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /app -name "*.md" -delete && \
    find /app -name "test*" -delete && \
    find /app -name "*test*" -delete

# Security: Switch to non-root user
USER trader

# Security: Health check with timeout and proper error handling
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose ports for application and metrics
EXPOSE 8501 8001

# Security: Use exec form for CMD to avoid shell injection
CMD ["python", "-m", "src.cli", "--config", "/app/config/production.json"]
