# Convenience targets for the Nebius Serverless submission.
# Windows note: needs GNU make + git-bash. Every target also has a raw one-liner
# in the README, so make is optional.

PY := ./.venv/Scripts/python.exe
export PYTHONPATH := .

.PHONY: smoke smoke-live manifest test lint serve help

help:
	@echo "smoke       - regenerate dataset + manifest from cached teacher outputs (offline, \$$0)"
	@echo "smoke-live  - classify 10 tweets live via Nebius, then validate (~\$$0, needs NEBIUS_API_KEY)"
	@echo "manifest    - full re-score + manifest from cached results (offline)"
	@echo "test        - run the full pytest suite"
	@echo "lint        - ruff check ."
	@echo "serve       - boot the /predict endpoint on :8080 (verifies the manifest hash)"

# $0, no keys, fully offline: proves the pipeline + manifest end-to-end.
smoke:
	$(PY) jobs/backtest/entrypoint.py --from-results

# ~$0: 10 live Nebius classifications (needs NEBIUS_API_KEY in .env).
smoke-live:
	$(PY) jobs/backtest/entrypoint.py --live --limit 10

manifest:
	$(PY) jobs/backtest/entrypoint.py --from-results

test:
	$(PY) -m pytest -q

lint:
	./.venv/Scripts/ruff.exe check .

serve:
	./.venv/Scripts/uvicorn.exe serving.app:app --host 0.0.0.0 --port 8080
