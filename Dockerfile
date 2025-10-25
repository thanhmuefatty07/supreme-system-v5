# Use lockfile if present for deterministic builds
# Fallback to requirements.txt

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.lock requirements.txt ./

RUN pip install --upgrade pip && \
    if [ -f requirements.lock ] && [ -s requirements.lock ]; then \
        pip install -r requirements.lock; \
    else \
        pip install -r requirements.txt; \
    fi

COPY . .

# ========================= RUNTIME =========================
FROM python:3.11-slim AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UVICORN_WORKERS=2 \
    PORT=8000

# Create non-root user
RUN addgroup --system app && adduser --system --ingroup app app
WORKDIR /app

# Copy installed packages from builder (optional: use venv for slimmer images)
COPY --from=builder /usr/local /usr/local
COPY . .

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python - <<'PY' || exit 1
import json,urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3) as r:
        print('ok', r.status)
except Exception as e:
    print('fail', e)
    raise
PY

CMD ["python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
