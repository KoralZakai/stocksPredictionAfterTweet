#!/usr/bin/env bash
# Deploy the SHIPPED System-B pipeline to Nebius Serverless AI:
#   1 batch JOB  (jobs/backtest -> writes validation_manifest.json to the bucket)
#   1 ENDPOINT   (serving/app.py -> loads + hash-verifies that manifest, serves /predict)
# Two purpose-built CPU images, one repo, so batch and serve cannot drift (3.2).
#
# Supersedes the deleted System-A DAG script.
#
# Usage (run from git-bash; authenticate the Nebius CLI first):
#   nebius init                                            # login + pick tenant/project
#   cp deploy/nebius/env.example deploy/nebius/.env && $EDITOR deploy/nebius/.env
#   DRY_RUN=1 ./deploy/nebius/deploy.sh all              # print every command, touch nothing
#   ./deploy/nebius/deploy.sh image                      # build + push BOTH images
#   ./deploy/nebius/deploy.sh job                        # run backtest-and-validate ($0)
#   ./deploy/nebius/deploy.sh endpoint                   # stand up /predict
#   ./deploy/nebius/deploy.sh verify                     # curl /health
#
# Docs: https://docs.nebius.com/serverless/jobs/manage
#       https://docs.nebius.com/serverless/endpoints/manage

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# Strip CR: the .env is edited on Windows, but this also runs under WSL, where a
# trailing \r rides into every value ("1h\r" is not a duration).
[ -f "$HERE/.env" ] && set -a && . <(tr -d '\r' < "$HERE/.env") && set +a

DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
# Bucket mount point. /app/runs/real is where the code already caches the LLM
# predictions (scripts/nebius_macro_backtest.py: CACHE = runs/real/nebius_predictions.json,
# relative to WORKDIR=/app), so mounting the bucket THERE makes a live run's
# classifications survive the container — a rerun re-reads them for $0.
DATA=/app/runs/real
MANIFEST_IN_BUCKET="$DATA/validation_manifest.json"

# CPU-only. Overridden from .env. These are the values the shipped deployment
# actually ran on (aijob-e00j2dskdm71qp3b3k / aiendpoint-e00fs92qgq1h4wzb2s).
NB_PLATFORM="${NB_PLATFORM:-cpu-d3}"
NB_PRESET="${NB_PRESET:-4vcpu-16gb}"
NB_TIMEOUT="${NB_TIMEOUT:-1h}"

# Image tags: one registry path, two tags (job vs endpoint).
IMG_JOB="${NB_IMAGE_JOB:-${NB_IMAGE:-}-backtest}"
IMG_API="${NB_IMAGE_API:-${NB_IMAGE:-}-predict}"

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m==>\033[0m %s\n' "$*"; }

# Print the command always; execute it only when DRY_RUN=0. %q keeps it copy-pasteable.
# Secrets are redacted on the way out — this output lands in terminals and CI logs.
run() {
  printf '\033[2m$'
  printf ' %q' "$@" | sed -E 's/(NEBIUS_API_KEY=|ANTHROPIC_API_KEY=|NB_ENDPOINT_TOKEN=)[^ ]*/\1<redacted>/g'
  printf '\033[0m\n'
  [ "$DRY_RUN" = "1" ] || "$@"
}
require() {
  for v in "$@"; do
    [ -n "${!v:-}" ] || die "$v is unset. Copy deploy/nebius/env.example to deploy/nebius/.env and fill it in."
  done
}

# --- images -------------------------------------------------------------------
cmd_image() {
  require NB_IMAGE
  info "build + push $IMG_JOB and $IMG_API (linux/amd64 — Nebius runs x86)"
  run docker build --platform linux/amd64 -f "$ROOT/jobs/backtest/Dockerfile" -t "$IMG_JOB" "$ROOT"
  run docker push "$IMG_JOB"
  run docker build --platform linux/amd64 -f "$ROOT/serving/Dockerfile" -t "$IMG_API" "$ROOT"
  run docker push "$IMG_API"
}

# --- job: backtest-and-validate ----------------------------------------------
# Thin CLI over alpha/ (the serverless layer holds zero science).
#
# JOB_ARGS picks the mode:
#   --live --limit N   classify N tweets on Nebius, then score  (needs NEBIUS_API_KEY)
#   --from-results     score the cached predictions, no LLM, $0
# Mount by bucket RESOURCE ID (storagebucket-...), NOT s3://<name>: the s3:// form
# needs AWS-profile/MysteryBox credentials, the ID form is authorised by the job's
# own service account. `nebius storage bucket get --name koral-bucket` prints the id.
JOB_ARGS="${JOB_ARGS:---live --limit ${LIMIT:-1000}}"

cmd_job() {
  require NB_PROJECT_ID NB_SUBNET_ID NB_BUCKET_ID
  [ -n "${IMG_JOB// }" ] || die "NB_IMAGE (or NB_IMAGE_JOB) is unset."
  local key="${NEBIUS_API_KEY:-${EXPO_PUBLIC_NEBIUS_API_KEY:-}}"
  case "$JOB_ARGS" in
    *--live*) [ -n "$key" ] || die "JOB_ARGS has --live but NEBIUS_API_KEY is unset — the run cannot classify." ;;
  esac
  info "run backtest-and-validate (run id $RUN_ID, args: $JOB_ARGS) -> $MANIFEST_IN_BUCKET"
  run nebius ai job create \
    --parent-id "$NB_PROJECT_ID" \
    --name "backtest-and-validate-${RUN_ID}" \
    --image "$IMG_JOB" \
    --container-command python \
    --args "jobs/backtest/entrypoint.py ${JOB_ARGS} --manifest ${MANIFEST_IN_BUCKET} --dataset-out ${DATA}/macro_dataset.csv" \
    --env "PYTHONPATH=/app" \
    --env "PYTHONUNBUFFERED=1" \
    --env "RUN_ID=$RUN_ID" \
    --env "NEBIUS_API_KEY=${key}" \
    --volume "${NB_BUCKET_ID}:${DATA}:rw" \
    --platform "$NB_PLATFORM" \
    --preset "$NB_PRESET" \
    --timeout "$NB_TIMEOUT" \
    --subnet-id "$NB_SUBNET_ID"
  info "job submitted. Wait for it to finish (writes the manifest) BEFORE the endpoint."
}

# --- endpoint: /predict -------------------------------------------------------
# Hash-verifies the manifest at boot; refuses to start on a corpus/prompt-hash
# mismatch. NEBIUS_API_KEY (live /predict calls the 70B) comes from .env, never
# committed.
#
# ponytail: no --volume. The manifest is baked into the image (serving/Dockerfile
# defaults MANIFEST_PATH=/app/reports/...), so the Endpoint does not depend on the
# Job having run first, and cannot fail on an unmounted /data. Mount the bucket and
# override MANIFEST_PATH only if you want the Job's freshly-written manifest.
#
# The image ENTRYPOINT/CMD already runs uvicorn on 8080 — no --container-command
# override needed. Nebius wants --container-port <port>/<protocol>, not --port.
cmd_endpoint() {
  require NB_PROJECT_ID NB_SUBNET_ID
  [ -n "${IMG_API// }" ] || die "NB_IMAGE (or NB_IMAGE_API) is unset."
  # serving/app.py reads NEBIUS_API_KEY or EXPO_PUBLIC_NEBIUS_API_KEY; /health is
  # fine without one but /predict 500s, so fail loudly here instead of at runtime.
  local key="${NEBIUS_API_KEY:-${EXPO_PUBLIC_NEBIUS_API_KEY:-}}"
  [ -n "$key" ] || die "NEBIUS_API_KEY (or EXPO_PUBLIC_NEBIUS_API_KEY) unset — /predict cannot call the model."
  info "deploy /predict endpoint"
  run nebius ai endpoint create \
    --parent-id "$NB_PROJECT_ID" \
    --name "predict-${RUN_ID}" \
    --image "$IMG_API" \
    --container-port 8080/http \
    --public \
    --auth none \
    --env "PYTHONPATH=/app" \
    --env "MANIFEST_PATH=/app/reports/validation_manifest.json" \
    --env "NEBIUS_API_KEY=${key}" \
    --platform "$NB_PLATFORM" \
    --preset "$NB_PRESET" \
    --subnet-id "$NB_SUBNET_ID"
}

# --- verify -------------------------------------------------------------------
cmd_verify() {
  [ -n "${ENDPOINT_URL:-}" ] || die "set ENDPOINT_URL=https://... (from: nebius ai endpoint get ...)"
  info "GET ${ENDPOINT_URL}/health"
  run curl -fsS "${ENDPOINT_URL}/health"
  echo
  info "POST ${ENDPOINT_URL}/predict (sample tweet)"
  run curl -fsS -X POST "${ENDPOINT_URL}/predict" -H 'content-type: application/json' \
    -d '{"tweet_text":"We will impose massive tariffs on all Chinese semiconductors.","t0_utc":"2025-03-03T14:00:00+00:00","author":"realDonaldTrump"}'
  echo
}

case "${1:-}" in
  image)    cmd_image ;;
  job)      cmd_job ;;
  endpoint) cmd_endpoint ;;
  verify)   cmd_verify ;;
  all)      cmd_image; cmd_job; cmd_endpoint ;;
  *) die "usage: $0 {image|job|endpoint|verify|all}   (DRY_RUN=1 to preview)" ;;
esac
