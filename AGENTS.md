# AGENTS.md — codebase map for AI agents

Orientation for an agent editing this repo. **`CLAUDE.md` is the scientific spec
and the source of truth for *why*; this file is the map of *what is implemented
and where*.** Section refs like `§3.2` point into `CLAUDE.md`.

## What this is (one paragraph)

An offline, point-in-time ML research pipeline that tests whether political
tweets carry short-term directional information about sector-ETF returns. It is
built to report a **null result honestly**, not to predict prices. It runs as
Nebius Serverless AI **Jobs** (batch) plus one **Endpoint** (`/predict`). All
data is local CSV/parquet — nothing polls a live API.

## The one rule that governs everything

> There is exactly **one feature path**: `core.decide.decide(tweet, market_state)`.
> The batch jobs, the `/predict` endpoint, and `dataset.build` all call it. Never
> add a second way to compute features — the no-skew test (`tests/test_decide.py`)
> exists to fail if you do (§3.2). `jobs/` and `serving/` are thin I/O wrappers
> with **no** science in them.

## Data flow

```
tweet (text, t0)
   │  sector_mapping.rules.map_tweet         -> ticker or NONE
   ▼
core.calendar.TradingCalendar.resolve_s0     -> s0 = first session open strictly after t0   (§3.3)
   │
   ├─ FEATURE side (data with close < t0):
   │     core.market_state.market_state_as_of -> MarketState (prior bars only, §3.1)
   │     core.features.build_features          -> structured features
   │     core.decide.decide                    -> Decision (Phase 0: abstain)
   │
   └─ LABEL side (data with open/close > t0):
         labeling.windows.compute_outcome      -> entry=open(s0), ret_1d/2d/3d (§3.4)
         labeling.thresholds.backward_vol+label-> UP/DOWN/NEUTRAL band ±k·σ (§3.5)
   ▼
dataset.build.build_dataset  -> list[Row]   (features ⋈ labels; the training set)
   ▼
eval.report.run_report       -> per-horizon macro-F1 vs baselines + permutation p
                                 + Benjamini-Hochberg + power gate  -> signal-or-null table (§4,§7)
```

The FEATURE and LABEL sets never overlap: features use bars closed **before**
`t0`; labels use the open/closes **after** `t0`. That gap is the leakage guard.

## Module reference

| Path | Role | Key symbols | Invariant it enforces |
|---|---|---|---|
| `config/settings.py` | Pre-registered config, frozen | `Settings`, `SETTINGS` | one place for k/α/horizons/embargo (§5) |
| `data/sources/interfaces.py` | Source contracts + row types | `Tweet`, `DailyBar`, `TweetSource`, `PriceSource` | read-only, offline (§2) |
| `data/sources/local.py` | CSV/parquet adapters | `LocalTweetSource`, `LocalPriceSource` | normalize to tz-aware UTC (§12) |
| `data/storage/store.py` | DuckDB store | `Store` (`ingest_tweets/bars`, `load_*`) | atomic; reject tz-naive + dup keys |
| `core/calendar.py` | Trading sessions | `TradingCalendar` (`resolve_s0`, `open_utc`, `close_utc`, `session_at_offset`) | s0 = first open strictly after t0 (§3.3) |
| `core/market_state.py` | Point-in-time view | `MarketState`, `market_state_as_of` | only bars closed `< t0` (§3.1) |
| `core/features.py` | Structured features | `build_features` | no future data; structured-only (§8) |
| `core/decide.py` | **The single decision fn** | `Decision`, `decide`, `decide_batch` | no train/serve skew (§3.2) |
| `labeling/windows.py` | Entry + forward returns | `Outcome`, `compute_outcome` | every close ts `> t0` (§3.4) |
| `labeling/thresholds.py` | Vol-scaled labels | `backward_vol`, `label`, `class_balance` | σ backward-only (§3.5) |
| `sector_mapping/rules.py` | Tweet -> ETF | `SECTOR_KEYWORDS`, `Mapping`, `map_tweet` | rule-based only, no ML mapper (§6) |
| `dataset/build.py` | Assemble training rows | `Row`, `build_dataset`, `rows_to_json`, `rows_from_json` | features via decide() (§3.2) |
| `eval/metrics.py` | Metrics | `macro_f1`, `accuracy`, `CLASSES` | macro-averaged (imbalance-proof) |
| `eval/baselines.py` | Baselines + null | `majority_class`, `market_follow`, `permutation_null` | first-class baselines (§4) |
| `eval/splits.py` | CV splitter | `purged_walk_forward` | purged + embargo ≥3, no random KFold (§3.6) |
| `eval/significance.py` | Inference | `permutation_pvalue`, `benjamini_hochberg` | BH over full registry (§4) |
| `eval/registry.py` | Test registry | `Registry`, `Entry` | BH denominator = registry size (§4) |
| `eval/power.py` | Power/MDE gate | `mde_gate`, `PowerResult` | declare underpowered honestly (§4) |
| `eval/report.py` | Signal-or-null table | `run_report`, `format_report`, `ReportRow` | the deliverable (§7) |
| `jobs/data_ingestion.py` | Nebius Job | `run`, `main` | validate + snapshot (thin) |
| `jobs/dataset_build.py` | Nebius Job | `run`, `main` | -> `dataset.json` (thin) |
| `jobs/evaluation.py` | Nebius Job | `run`, `main` | -> `report.txt/json` (thin) |
| `serving/endpoint.py` | `/predict` endpoint | `predict`, `main` | same decide(); abstain w/o history (§2,§13) |
| `scripts/phase0_mvp.py` | MVP-10 harness | `main` | Phase-0 correctness demo (§10) |
| `scripts/phase1_report.py` | Report runner | `main` | prints signal-or-null table |

### Real-data pipeline (DATASEARCH.md, runs in this order)

| Script | Input -> Output | Notes |
|---|---|---|
| `scripts/normalize_trump_tweets.py` | 2 raw archives -> `data/real/tweets.csv` | 2009-2021; snowflake UTC; pre-snowflake rows use empirically-tz'd Time col (`ts_confidence=min`) |
| `scripts/build_corpus.py` | tweets + HF Truth parquet -> `corpus.csv` | unified schema, dedup earliest-wins |
| `scripts/build_corpus_v3.py` | + CNN Truth parquet -> `corpus_v3.csv` | dedup prefers ms-precision; 2009-2026 |
| `scripts/fetch_real_bars.py` | yfinance -> `data/real/bars.csv` | 47 tickers (10 ETFs+SPY+35 stocks+DJT), 2008-2026 |
| `scripts/build_stock_dataset_v3.py` | corpus_v3 + bars -> `market_events.csv` + `stock_event_dataset.csv` | relevance filter + (post x asset) events; SPY always included; sector-adjusted `sec_h`; provenance + `explanation` per row |
| `scripts/final_results.py` | events -> `reports/final_results.txt` | summary + global insights + 20 examples + signal-or-noise verdict |
| `scripts/explore_v3.py` | legacy `stock_events_v3.csv` | older exploratory view (superseded by final_results) |

## Entry points

```bash
# gate (run after any change) — see "Gotchas" for why the paths are like this
.venv/Scripts/ruff.exe check . && .venv/Scripts/mypy.exe . && .venv/Scripts/pytest.exe -q

# regenerate the synthetic fixture (10 tweets + ETF bars + SPY)
PYTHONPATH=. .venv/Scripts/python.exe data/fixtures/make_fixture.py

# job DAG
PYTHONPATH=. python jobs/data_ingestion.py --tweets data/fixtures/tweets.csv --bars data/fixtures/bars.csv --out runs/mvp
PYTHONPATH=. python jobs/dataset_build.py  --in runs/mvp --out runs/mvp/dataset.json
PYTHONPATH=. python jobs/evaluation.py     --dataset runs/mvp/dataset.json --out runs/mvp

# endpoint
BARS_CSV=data/fixtures/bars.csv PYTHONPATH=. python serving/endpoint.py   # POST /predict on :8080
```

## Dependencies (and why each exists)

Runtime deps are minimal on purpose; most logic uses the standard library.

**Runtime** (`[project].dependencies`):
| Package | Used by | Why / rationale |
|---|---|---|
| `duckdb` | `data/storage/store.py` | schema-versioned store; atomic ingest validation; the ASOF-join engine for bulk ops (§12). |
| `pandas` | `data/sources/local.py`, `jobs/*` | CSV/parquet IO and column filtering only. Chosen over polars because N is tiny and `merge_asof`/ecosystem win here (README). **pandas 3.x pulls pyarrow.** |
| `pydantic` | `config/settings.py` | frozen, validated pre-registered config; field validators enforce embargo≥3, k>0. |

**Standard library doing real work** (no dep needed): `zoneinfo` (calendar
DST/sessions), `statistics` (vol + metrics), `random` (permutation null + power
sim, always seeded), `http.server` (the `/predict` endpoint — no web framework),
`json`/`csv`/`dataclasses`.

**Dev** (`[dependency-groups].dev`): `mypy` (strict), `pytest`, `ruff`.

**Not present yet (intentional):**
- No `xgboost`/`lightgbm` — the model (build step 8) isn't wired; `/predict`
  abstains until it is. Adding it requires `uv` (see Gotchas) to keep `uv.lock`
  synced for the Docker `--frozen` build.
- No `fastapi`/`uvicorn` — the endpoint is stdlib `http.server`.
- No `yfinance`/X-API client — real data collection isn't built; the fixture is
  synthetic. Real data would be a one-off prep script, not live polling (§2).

Before adding **any** dependency, confirm with the user (CLAUDE.md build order,
last line) and run `uv add <pkg>` so `uv.lock` stays in sync.

## Conventions an agent must keep

- **Timezone:** every `datetime` is tz-aware UTC in Python. `calendar.py` owns
  market-local (ET) session logic. Storage keeps naive-UTC `TIMESTAMP` and
  reattaches UTC on load (duckdb `TIMESTAMPTZ` would need `pytz`).
- **No full-series statistics.** Volatility and any label threshold use
  backward-only windows. A stat that peeks at the future is a leak bug.
- **Config is frozen.** Change a scientific choice in `config/settings.py`
  (a pre-registration change in git), never inline.
- **Determinism.** Seed everything from `SETTINGS.seed`; permutation/power sims
  take an explicit seed.
- **Every non-trivial module has a test.** The correctness-invariant tests
  (point-in-time, no-skew, close>t0) are mandatory — do not weaken them to make
  a change pass.

## Gotchas (local dev)

- `uv` is **not on PATH** and the venv has no `pip`. Run tools directly:
  `.venv/Scripts/{ruff,mypy,pytest,python}.exe`.
- Scripts need `PYTHONPATH=.` (pytest gets it from `[tool.pytest.ini_options]`).
- mypy runs under `python_version=3.12` so it can parse numpy's bundled stubs;
  code stays 3.11-safe via ruff + `requires-python`.
- Write CSVs with `encoding="utf-8"` (Windows default cp1252 breaks em-dashes).
- `runs/` and `.env` are gitignored; never commit secrets or run artifacts.

## Current status

Build steps 1–7 + serverless packaging (step 10) are done and green (42 tests).
Not built: the XGBoost model + conformal abstention (step 8), transformer canary,
and text-ablation. See `CLAUDE.md` §10–11 for the full build order.
