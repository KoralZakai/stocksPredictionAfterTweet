#!/usr/bin/env bash
# Deploy the SHIPPED System-B pipeline to Nebius Serverless AI:
#   1 batch JOB  (jobs/backtest -> writes validation_manifest.json to the bucket)
#   1 ENDPOINT   (serving/app.py -> loads + hash-verifies that manifest, serves /predict)
# Two purpose-built CPU images, one repo, so batch and serve cannot drift (3.2).
#
# This supersedes deploy.sh (which drives the deprecated System-A DAG).
#
# Usage (run from git-bash; authenticate the Nebius CLI first):
#   nebius init                                            # login + pick tenant/project
#   cp deploy/nebius/env.example deploy/nebius/.env && $EDITOR deploy/nebius/.env
#   DRY_RUN=1 ./deploy/nebius/deploy_b.sh all              # print every command, touch nothing
#   ./deploy/nebius/deploy_b.sh image                      # build + push BOTH images
#   ./deploy/nebius/deploy_b.sh job                        # run backtest-and-validate ($0)
#   ./deploy/nebius/deploy_b.sh endpoint                   # stand up /predict
#   ./deploy/nebius/deploy_b.sh verify                     # curl /health
#
# Docs: https://docs.nebius.com/serverless/jobs/manage
#       https://docs.nebius.com/serverless/endpoints/manage

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
[ -f "$HERE/.env" ] && set -a && . "$HERE/.env" && set +a

DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATA=/data                                   # shared bucket mount, identical in job + endpoint
MANIFEST_IN_BUCKET="$DATA/reports/validation_manifest.json"

# CPU-only. Overridden from .env.
NB_PLATFORM="${NB_PLATFORM:-cpu-e2}"
NB_PRESET="${NB_PRESET:-2vcpu-8gb}"
NB_TIMEOUT="${NB_TIMEOUT:-1h}"

# Image tags: one registry path, two tags (job vs endpoint).
IMG_JOB="${NB_IMAGE_JOB:-${NB_IMAGE:-}-backtest}"
IMG_API="${NB_IMAGE_API:-${NB_IMAGE:-}-predict}"

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m==>\033[0m %s\n' "$*"; }

# Print the command always; execute it only when DRY_RUN=0. %q keeps it copy-pasteable.
run() {
  printf '\033[2m$'; printf ' %q' "$@"; printf '\033[0m\n'
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
# Thin CLI over alpha/ (the serverless layer holds zero science). --from-results
# reproduces the committed manifest for $0; drop it (and add the NEBIUS_API_KEY
# secret) for a full ~443-tweet live classification (a few cents, ~15-25 min).
cmd_job() {
  require NB_PROJECT_ID NB_SUBNET_ID NB_BUCKET
  [ -n "${IMG_JOB// }" ] || die "NB_IMAGE (or NB_IMAGE_JOB) is unset."
  info "run backtest-and-validate (run id $RUN_ID) -> $MANIFEST_IN_BUCKET"
  run nebius ai job create \
    --parent-id "$NB_PROJECT_ID" \
    --name "backtest-and-validate-${RUN_ID}" \
    --image "$IMG_JOB" \
    --container-command python \
    --args "jobs/backtest/entrypoint.py --from-results --manifest ${MANIFEST_IN_BUCKET} --dataset-out ${DATA}/reports/macro_dataset.csv" \
    --env "PYTHONPATH=/app" \
    --env "RUN_ID=$RUN_ID" \
    --volume "s3://${NB_BUCKET}:${DATA}" \
    --platform "$NB_PLATFORM" \
    --preset "$NB_PRESET" \
    --timeout "$NB_TIMEOUT" \
    --subnet-id "$NB_SUBNET_ID"
  info "job submitted. Wait for it to finish (writes the manifest) BEFORE the endpoint."
}

# --- endpoint: /predict -------------------------------------------------------
# Loads + hash-verifies the manifest from the bucket at boot; refuses to start on
# a corpus/prompt-hash mismatch. NEBIUS_API_KEY (live /predict calls the 70B) and
# NB_ENDPOINT_TOKEN come from Nebius secrets, never committed.
cmd_endpoint() {
  require NB_PROJECT_ID NB_SUBNET_ID NB_BUCKET NB_ENDPOINT_TOKEN
  [ -n "${IMG_API// }" ] || die "NB_IMAGE (or NB_IMAGE_API) is unset."
  info "deploy /predict endpoint"
  run nebius ai endpoint create \
    --parent-id "$NB_PROJECT_ID" \
    --name "predict-${RUN_ID}" \
    --image "$IMG_API" \
    --container-command uvicorn \
    --args "serving.app:app --host 0.0.0.0 --port 8080" \
    --port 8080 \
    --env "PYTHONPATH=/app" \
    --env "MANIFEST_PATH=${MANIFEST_IN_BUCKET}" \
    --env "NEBIUS_API_KEY=${NEBIUS_API_KEY:-}" \
    --env "ENDPOINT_TOKEN=${NB_ENDPOINT_TOKEN}" \
    --volume "s3://${NB_BUCKET}:${DATA}" \
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
