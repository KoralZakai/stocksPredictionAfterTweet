#!/usr/bin/env bash
# Run the exact Job DAG that deploy/nebius/deploy.sh submits, but locally.
#
# Same entrypoints, same argument shapes, same order — so a reviewer can
# reproduce the whole pipeline with no Nebius account, and so the serverless
# manifests are exercised (in shape) before they ever hit a tenant.
#
#   ./scripts/run_dag.sh fixture   # 10-tweet synthetic fixture (seconds)
#   ./scripts/run_dag.sh real      # real posts + real bars (minutes)
#
# Times each stage and prints a summary — that table is what the README's
# runtime/cost section is derived from.

set -euo pipefail

MODE="${1:-fixture}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY=(python)
command -v uv >/dev/null 2>&1 && PY=(uv run python)

case "$MODE" in
  fixture)
    TWEETS=data/fixtures/tweets.csv; BARS=data/fixtures/bars.csv
    EVENTS=data/real/stock_event_dataset.csv
    RUNS=runs/local-fixture ;;
  real)
    TWEETS=data/real/tweets.csv; BARS=data/real/bars.csv
    EVENTS=data/real/stock_event_dataset.csv
    RUNS=runs/local-real ;;
  *) echo "usage: $0 {fixture|real}" >&2; exit 2 ;;
esac

mkdir -p "$RUNS"
declare -a NAMES=() SECS=()

stage() {
  local name="$1"; shift
  printf '\033[36m==> %s\033[0m\n' "$name"
  local t0 t1
  t0=$(date +%s)
  "$@"
  t1=$(date +%s)
  NAMES+=("$name"); SECS+=($((t1 - t0)))
}

[ -f "$TWEETS" ] || stage "make-fixture" "${PY[@]}" data/fixtures/make_fixture.py

stage "1 data_ingestion" "${PY[@]}" jobs/data_ingestion.py --tweets "$TWEETS" --bars "$BARS" --out "$RUNS"
stage "2 llm_features"   "${PY[@]}" jobs/llm_features.py   --in "$RUNS" --out "$RUNS/llm_signals.json"
stage "3 dataset_build"  "${PY[@]}" jobs/dataset_build.py  --in "$RUNS" --out "$RUNS/dataset.json"
stage "4 training"       "${PY[@]}" jobs/training.py       --dataset "$RUNS/dataset.json" --out "$RUNS/model" --horizon 3
stage "5 evaluation"     "${PY[@]}" jobs/evaluation.py     --dataset "$RUNS/dataset.json" --out "$RUNS"
stage "6 reporting"      "${PY[@]}" jobs/reporting.py      --events "$EVENTS" --signals "$RUNS/llm_signals.json" --out "$RUNS/eden_dashboard.html"

echo
printf '\033[36m==> DAG complete (%s)\033[0m\n' "$MODE"
total=0
for i in "${!NAMES[@]}"; do
  printf '  %-20s %4ss\n' "${NAMES[$i]}" "${SECS[$i]}"
  total=$((total + SECS[i]))
done
printf '  %-20s %4ss\n' "TOTAL" "$total"
echo "  artifacts -> $RUNS"
