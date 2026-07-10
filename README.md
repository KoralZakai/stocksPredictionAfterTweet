# Political Tweets → Sector-ETF Signal: A Serverless Null-Result Pipeline

A reproducible **statistical ML research pipeline** for the
[Nebius Serverless AI Builders Challenge](serverlessChallange.txt). It asks one
question, rigorously:

> Do political tweets contain measurable, statistically significant short-term
> directional information about sector-ETF returns, under strict causal
> constraints?

**This is not a trading system.** Predictive accuracy is *not* a success
metric. A rigorous **null result — "no signal" — is a full success.** The whole
system is built to detect a weak-or-absent signal *truthfully* rather than to
manufacture a positive one. See [`CLAUDE.md`](CLAUDE.md) for the full
scientific specification and hard correctness invariants.

## What it measures

With **daily OHLCV only**, we measure the 1–3 trading-day *drift-association*
between a tweet and its rule-mapped sector ETF. We do **not** claim to measure
the causal reaction (priced in seconds, invisible to daily bars). Tickers:
`XLK XLE XLF XLI XLV XLP XLY XLB SMH ITA`, benchmark `SPY`.

## Why it can't fool itself

- **Point-in-time.** Every feature for a tweet at `t0` uses only data whose
  session *closed strictly before* `t0`. A future-injection canary test proves
  features are blind to any bar dated ≥ `t0`.
- **No train/serve skew.** Exactly one pure `decide(tweet, market_state)` feeds
  the batch jobs *and* the `/predict` endpoint — a test asserts identical
  features on both paths.
- **Leak-free labels.** Entry = `open(s0)` where `s0` is the first session open
  strictly after `t0`; returns run forward from there; label bands use
  backward-only volatility.
- **Honest evaluation.** Purged + embargoed walk-forward CV (no random KFold),
  majority / market-follow / **permutation-null** baselines, permutation
  p-values, **Benjamini–Hochberg** correction over the full test registry, and
  a **power/MDE gate** that declares the study underpowered when it is.

## Architecture (Nebius Serverless)

The serverless layer is orchestration only — it wraps pure modules and adds no
second feature path.

```
Jobs (batch):  data_ingestion → dataset_build → evaluation/reporting
Endpoint:      POST /predict  ── same decide() ──┘
```

| Component | Path | Role |
|---|---|---|
| `data_ingestion` job | `jobs/data_ingestion.py` | Validate raw (reject tz-naive / dup keys) → snapshot |
| `dataset_build` job | `jobs/dataset_build.py` | Labels + point-in-time features → `dataset.json` |
| `evaluation` job | `jobs/evaluation.py` | Signal-or-null report → `report.txt/json` |
| `/predict` endpoint | `serving/endpoint.py` | Same `decide()`; abstains below calibrated confidence |
| Core | `core/` | Calendar, `market_state_as_of`, features, `decide()` |
| Labeling | `labeling/` | Entry anchor, `ret_1d/2d/3d`, vol-scaled bands |
| Eval | `eval/` | Splits, baselines, significance, power, report |

**Why pandas (not polars):** the working set is tiny (N ≈ hundreds of rows);
pandas' `merge_asof` and ubiquitous ecosystem win over polars' large-data
throughput here. DuckDB backs the storage/validation layer.

## Setup

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                       # install (add --no-dev for runtime only)
uv run python data/fixtures/make_fixture.py   # generate the synthetic MVP-10 fixture
```

Quality gate (mandatory correctness-invariant tests run here):

```bash
uv run ruff check . && uv run mypy . && uv run pytest -q
```

### Run the pipeline (locally or as Nebius Jobs)

```bash
uv run python jobs/data_ingestion.py --tweets data/fixtures/tweets.csv \
    --bars data/fixtures/bars.csv --out runs/mvp
uv run python jobs/dataset_build.py --in runs/mvp --out runs/mvp/dataset.json
uv run python jobs/evaluation.py   --dataset runs/mvp/dataset.json --out runs/mvp
```

### Serve `/predict`

```bash
BARS_CSV=data/fixtures/bars.csv uv run python serving/endpoint.py   # :8080
curl -s -X POST localhost:8080/predict \
  -d '{"tweet_text":"drill baby drill energy dominance","timestamp":"2024-02-20T22:00:00Z"}'
# {"ticker":"XLE","direction":"ABSTAIN","confidence":0.0,"abstain":true,"map_confidence":1.0}
```

### Docker (one CPU image for jobs and endpoint)

```bash
docker build -f deploy/Dockerfile -t tweet-signal .
docker run -p 8080:8080 tweet-signal                                   # endpoint
docker run tweet-signal python jobs/evaluation.py --dataset ... --out ...   # a job
```

## Expected output

The current dataset is a **synthetic, seeded fixture** (10 tweets covering
before-open / intraday / after-close / weekend / holiday cases). Phase 0 is
plumbing-correctness only — outcomes carry **zero** evidential weight. The
report prints a signal-or-null table; at N = 10 the honest headline is:

```
HEADLINE: 0 of 3 cells survive BH correction (alpha=0.05).
          A null result here is a valid, expected outcome.
```

Real Trump tweets + real ETF OHLCV drop into the same pipeline unchanged.

## Reproduce the whole DAG (no Nebius account needed)

`scripts/run_dag.sh` runs the exact six Jobs that `deploy/nebius/deploy.sh`
submits — same entrypoints, same argument shapes, same order — so a reviewer can
reproduce the pipeline locally and the serverless manifests are exercised in
shape before they touch a tenant.

```bash
./scripts/run_dag.sh fixture   # synthetic 10-tweet fixture, seconds
./scripts/run_dag.sh real      # real posts + real bars
```

## Deploy to Nebius Serverless

Six batch **Jobs** (the DAG) + one **Endpoint** (`/predict`), all from the one
CPU image, so batch and serve cannot drift (§3.2).

```bash
cp deploy/nebius/env.example deploy/nebius/.env   # fill in project/subnet/image/bucket
DRY_RUN=1 ./deploy/nebius/deploy.sh all           # print every command, change nothing
./deploy/nebius/deploy.sh image                   # build + push (linux/amd64)
./deploy/nebius/deploy.sh jobs                    # submit the DAG in order
./deploy/nebius/deploy.sh endpoint                # stand up /predict
./deploy/nebius/deploy.sh urls                    # endpoint URL + smoke-test curl
```

`.env` is gitignored; no secrets are committed. Follow a job with
`nebius ai job logs <job_id> --follow`.

> **Not yet executed against a tenant.** The commands are written from the public
> [Jobs](https://docs.nebius.com/serverless/jobs/manage) and
> [Endpoints](https://docs.nebius.com/serverless/endpoints/manage) docs, and the
> full `DRY_RUN` output has been reviewed, but no job has run on real Nebius
> infrastructure yet. Proof-of-execution (job logs, endpoint URL) goes below once
> it has.

### Proof of execution

<!-- Paste job IDs / log screenshots / the endpoint URL here after the first real run. -->
_Pending first tenant run._

## Hardware, runtime, cost

- **CPU-only** (classical-ML track). No GPU. Platform `cpu-e2`, preset
  `2vcpu-8gb` — the smallest Nebius offers, and more than this workload needs.
- Measured on the fixture DAG (`./scripts/run_dag.sh fixture`, Apple M-series):

  | Job | Runtime |
  |---|---|
  | `data_ingestion` | 2 s |
  | `llm_features` | <1 s (keyword fallback; ~seconds/1k posts with Claude) |
  | `dataset_build` | 1 s |
  | `training` | <1 s |
  | `evaluation` | 8 s |
  | `reporting` (23,236 real rows) | 2 s |
  | **total** | **13 s** |

- Test suite: ~7 s (84 tests).
- **Cost:** the DAG is minutes of `2vcpu-8gb` CPU time per run — negligible, and
  within the free trial credits. Rates: [Nebius compute pricing](https://docs.nebius.com/compute/resources/pricing).
  The one variable cost is `llm_features` if you supply `ANTHROPIC_API_KEY`; it
  is a one-time, cached, offline pass over the corpus (unset the key and it uses
  the deterministic keyword fallback for free).

## Status & roadmap

Built and green (84 tests): config, storage, calendar, point-in-time market
state, labeling, sector mapping, dataset build, full eval harness (purged CV,
baselines, permutation null, BH, power gate, **text-ablation**), GBT + conformal
abstention, serverless jobs + endpoint, offline LLM signal layer (`llm/`), and
the `reporting` job that regenerates the dashboard from data.

**Not yet built:** the transformer overfit canary; intraday (30m/1h) horizons;
the prediction log + nightly outcome-backfill loop. See `CLAUDE.md` §10–11.

## License

[MIT](LICENSE).
