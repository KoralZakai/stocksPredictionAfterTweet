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

# Endpoint by default; Jobs override the command, e.g.:
#   docker run IMG python jobs/data_ingestion.py --tweets ... --bars ... --out runs/mvp
#   docker run IMG python jobs/dataset_build.py  --in runs/mvp --out runs/mvp/dataset.json
#   docker run IMG python jobs/evaluation.py     --dataset runs/mvp/dataset.json --out runs/mvp
CMD ["python", "serving/endpoint.py"]
