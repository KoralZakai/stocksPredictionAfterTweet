#!/usr/bin/env bash
# Deploy the pipeline to Nebius Serverless AI: 6 batch Jobs (the DAG) + 1
# Endpoint (/predict). One CPU image serves all of them, so batch and serve
# cannot drift (§3.2, §12).
#
# NOT YET RUN AGAINST A TENANT. Every command below is written from the public
# Nebius CLI docs, but nothing here has been executed on a real project. Run it
# in DRY_RUN first, read the commands, then drop DRY_RUN.
#
#   cp deploy/nebius/env.example deploy/nebius/.env && $EDITOR deploy/nebius/.env
#   DRY_RUN=1 ./deploy/nebius/deploy.sh all     # print every command, touch nothing
#   ./deploy/nebius/deploy.sh image             # build + push the image
#   ./deploy/nebius/deploy.sh jobs              # submit the DAG, in order
#   ./deploy/nebius/deploy.sh endpoint          # stand up /predict
#   ./deploy/nebius/deploy.sh urls              # print endpoint URL for the README
#
# Docs: https://docs.nebius.com/serverless/jobs/manage
#       https://docs.nebius.com/serverless/endpoints/manage

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
[ -f "$HERE/.env" ] && set -a && . "$HERE/.env" && set +a

DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATA=/data                    # shared bucket mount, identical in every job
RUNS="$DATA/runs/$RUN_ID"

# CPU-only. Overridden from .env.
NB_PLATFORM="${NB_PLATFORM:-cpu-e2}"
NB_PRESET="${NB_PRESET:-2vcpu-8gb}"
NB_TIMEOUT="${NB_TIMEOUT:-1h}"

die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m==>\033[0m %s\n' "$*"; }

# Print the command always; execute it only when DRY_RUN=0.
# %q-quote each word so the printed line is copy-pasteable and multi-word
# arguments (notably --args "jobs/x.py --flag v") are visibly single tokens.
run() {
  printf '\033[2m$'
  printf ' %q' "$@"
  printf '\033[0m\n'
  [ "$DRY_RUN" = "1" ] || "$@"
}

require() {
  for v in "$@"; do
    [ -n "${!v:-}" ] || die "$v is unset. Copy deploy/nebius/env.example to deploy/nebius/.env and fill it in."
  done
}

# --- image --------------------------------------------------------------------
cmd_image() {
  require NB_IMAGE
  info "build + push $NB_IMAGE (linux/amd64 — Nebius runs x86, your Mac may not)"
  run docker build --platform linux/amd64 -f "$ROOT/deploy/Dockerfile" -t "$NB_IMAGE" "$ROOT"
  run docker push "$NB_IMAGE"
}

# --- jobs ---------------------------------------------------------------------
# Each job is a thin CLI over the pure modules (§13: the serverless layer holds
# zero science). They chain by artifact I/O under $RUNS on the shared bucket.
job() {
  local name="$1"; shift
  require NB_PROJECT_ID NB_SUBNET_ID NB_IMAGE NB_BUCKET
  run nebius ai job create \
    --parent-id "$NB_PROJECT_ID" \
    --name "${name}-${RUN_ID}" \
    --image "$NB_IMAGE" \
    --container-command python \
    --args "$*" \
    --env "PYTHONPATH=/app" \
    --env "RUN_ID=$RUN_ID" \
    --volume "s3://${NB_BUCKET}:${DATA}" \
    --platform "$NB_PLATFORM" \
    --preset "$NB_PRESET" \
    --timeout "$NB_TIMEOUT" \
    --subnet-id "$NB_SUBNET_ID"
}

cmd_jobs() {
  info "submitting the DAG (run id $RUN_ID) — each job consumes the previous one's artifacts"

  # 1. ingest: validate raw (reject tz-naive / duplicate keys) -> snapshot
  job data-ingestion jobs/data_ingestion.py \
    --tweets "$DATA/real/tweets.csv" --bars "$DATA/real/bars.csv" --out "$RUNS"

  # 2. llm_features: OFFLINE signal extraction, once per post, content-addressed
  #    cache. Without ANTHROPIC_API_KEY this deterministically falls back to the
  #    keyword extractor rather than failing.
  job llm-features jobs/llm_features.py \
    --in "$RUNS" --out "$RUNS/llm_signals.json"

  # 3. labels + point-in-time features via the single decide() path
  job dataset-build jobs/dataset_build.py \
    --in "$RUNS" --out "$RUNS/dataset.json"

  # 4. fit GBT + calibrate the conformal abstainer on a time-ordered tail
  job training jobs/training.py \
    --dataset "$RUNS/dataset.json" --out "$RUNS/model" --horizon 3

  # 5. purged + embargoed walk-forward CV, baselines, permutation null, BH, power gate
  job evaluation jobs/evaluation.py \
    --dataset "$RUNS/dataset.json" --out "$RUNS"

  # 6. THE deliverable: the signal-or-null dashboard
  job reporting jobs/reporting.py \
    --events "$DATA/real/stock_event_dataset.csv" \
    --signals "$RUNS/llm_signals.json" \
    --out "$RUNS/eden_dashboard.html"

  info "follow a job:  nebius ai job logs <job_id> --follow"
  info "list them:     nebius ai job list --parent-id \$NB_PROJECT_ID"
}

# --- endpoint -----------------------------------------------------------------
cmd_endpoint() {
  require NB_PROJECT_ID NB_SUBNET_ID NB_IMAGE NB_BUCKET NB_ENDPOINT_TOKEN
  info "creating /predict endpoint (token auth, public)"

  # Only pass registry credentials when they are actually set — an empty
  # --registry-username is a malformed flag, not "no credentials".
  local -a creds=()
  if [ -n "${NB_REGISTRY_USERNAME:-}" ]; then
    creds+=(--registry-username "$NB_REGISTRY_USERNAME"
            --registry-password "${NB_REGISTRY_PASSWORD:-}")
  fi

  # Serves the SAME decide() the batch jobs use. Abstains when no model is
  # mounted, which is the honest cold-start output rather than a guess.
  run nebius ai endpoint create \
    --parent-id "$NB_PROJECT_ID" \
    --name "tweet-signal-predict" \
    --image "$NB_IMAGE" \
    "${creds[@]+"${creds[@]}"}" \
    --container-command python \
    --args "serving/endpoint.py" \
    --container-port 8080 \
    --env "PYTHONPATH=/app" \
    --env "BARS_CSV=${DATA}/real/bars.csv" \
    --env "MODEL_DIR=${RUNS}/model" \
    --volume "s3://${NB_BUCKET}:${DATA}" \
    --auth token \
    --token "$NB_ENDPOINT_TOKEN" \
    --platform "$NB_PLATFORM" \
    --preset "$NB_PRESET" \
    --subnet-id "$NB_SUBNET_ID" \
    --public
}

cmd_urls() {
  require NB_PROJECT_ID
  info "endpoint URL (paste into the README's proof-of-execution section)"
  run nebius ai endpoint list --parent-id "$NB_PROJECT_ID" --format json
  cat <<'EOF'

Then smoke-test it:

  curl -s -X POST "$ENDPOINT_URL/predict" \
    -H "Authorization: Bearer $NB_ENDPOINT_TOKEN" \
    -H 'Content-Type: application/json' \
    -d '{"tweet_text":"drill baby drill energy dominance","timestamp":"2025-02-20T22:00:00Z"}'

Expected shape (ABSTAIN is correct when no model is mounted):
  {"ticker":"XLE","direction":"ABSTAIN","confidence":0.0,"abstain":true,"map_confidence":1.0}
EOF
}

case "${1:-}" in
  image)    cmd_image ;;
  jobs)     cmd_jobs ;;
  endpoint) cmd_endpoint ;;
  urls)     cmd_urls ;;
  all)      cmd_image; cmd_jobs; cmd_endpoint; cmd_urls ;;
  *) die "usage: $0 {image|jobs|endpoint|urls|all}   (prefix with DRY_RUN=1 to print only)" ;;
esac
