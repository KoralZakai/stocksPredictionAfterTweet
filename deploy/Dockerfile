# CPU-only image (classical-ML track). Same image runs every Nebius Job and the
# /predict endpoint, so batch and serve share one code path (§3.2).
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .
ENV PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

# DEPRECATED shared image (System-A scaffold). The CANONICAL build is now two
# purpose-built images: jobs/backtest/Dockerfile (the Job) and serving/Dockerfile
# (the Endpoint). This file is kept for the older System-A jobs only.
# Default command points at the SHIPPED System-B endpoint for consistency:
#   docker run IMG uvicorn serving.app:app --host 0.0.0.0 --port 8080
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
