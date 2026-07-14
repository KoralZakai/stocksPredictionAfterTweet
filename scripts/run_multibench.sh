#!/usr/bin/env bash
# Run the v-multibench pipeline end to end, locally: label -> train/test -> report.
# Same entrypoints the Nebius jobs call, so a reviewer reproduces it with no cloud.
#
#   ./scripts/run_multibench.sh            # horizon 3d
#   ./scripts/run_multibench.sh 1w         # any label horizon (eod/2d/3d/1w/2w/3w/1mo)
#
# Prereqs (one-time): daily bars incl. QQQ+DIA in data/real/bars.csv
#   PYTHONPATH=. python scripts/fetch_extra_bars.py QQQ DIA

set -euo pipefail
HORIZON="${1:-3d}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY=(python); command -v uv >/dev/null 2>&1 && PY=(uv run python)

echo "==> 1 label  (tweet x stock x horizon, multi-benchmark)"
"${PY[@]}" scripts/build_multibench.py

echo "==> 2 train/test  @ $HORIZON"
"${PY[@]}" jobs/train_multibench.py --horizon "$HORIZON" --out runs/real/multibench_model

echo "==> 3 per-tweet dashboard"
"${PY[@]}" scripts/build_multibench_report.py --top 36

echo "==> done. serve it:"
echo "   MB_MODEL_DIR=runs/real/multibench_model BARS_CSV=data/real/bars.csv python serving/endpoint.py"
echo "   curl -X POST localhost:8080/predict_stock -d '{\"tweet_text\":\"...\",\"timestamp\":\"2025-08-11T14:00:00Z\"}'"
