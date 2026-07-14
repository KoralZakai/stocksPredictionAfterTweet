# Nebius Serverless AI Endpoint: /predict — the SHIPPED serving image. CPU-only.
# Serves the validated raw 70B call; verifies the manifest hash at boot.
# Build from repo root:  docker build -f serving/Dockerfile -t predict .
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .
ENV PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    MANIFEST_PATH=/data/reports/validation_manifest.json

EXPOSE 8080

# Import-time boot verification (serving/app.py) refuses to start on a missing
# manifest, a corpus-hash mismatch, or a prompt-hash drift. Runs with NO market
# keys (market provider degrades to null). Nebius secrets supply NEBIUS_API_KEY.
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
