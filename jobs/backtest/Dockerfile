# Nebius Serverless AI Job: backtest-and-validate. CPU-only, run-to-completion.
# Build from the REPO ROOT:  docker build -f jobs/backtest/Dockerfile -t backtest .
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

# Manifest traceability: no git in the image, so the sha is baked in at build time.
ARG CODE_REV=unknown
ENV CODE_REV=${CODE_REV}
ENV PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

# Idempotent, resumable batch job. Writes macro_dataset.csv + validation_manifest.json
# to the mounted artifact store. Runs with NO Alpaca keys (intraday horizons skipped).
#   $0 smoke:  docker run backtest python jobs/backtest/entrypoint.py --limit 10
#   from cache: docker run backtest python jobs/backtest/entrypoint.py --from-results
ENTRYPOINT ["python", "jobs/backtest/entrypoint.py"]
CMD ["--from-results"]
