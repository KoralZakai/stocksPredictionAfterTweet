# Architecture v2 — Multi-Sector Abnormal-Return Response to Political Tweets

Refines the v1 spec in `CLAUDE.md`. Where the two disagree, the **Hard Correctness
Invariants and Non-Goals in `CLAUDE.md` still win** — the decisions below were taken
*with* those guardrails, not against them.

## Reframed goal

Not "predict if a stock goes up/down after a tweet."
Instead: **estimate the multi-sector _abnormal-return_ response conditioned on the
pre-event market regime + tweet semantics.**

A tweet is a market **event**. For each event we ask, per candidate sector ETF and
per horizon: did this sector move *beyond what the whole market (SPY) did*, given how
that sector was *already behaving before the tweet*?

The honest prior stands: signal is weak-to-absent. A rigorous **null result is a full
success**. Predictive accuracy is not a success metric.

## Locked decisions (this session)

1. **Sector candidates: rule + keyword (+ optional entity) rules only.** Deterministic,
   reproducible, human-verifiable. **No embedding similarity / learned retrieval in the
   causal chain** (§6). Rationale: keeps a null result interpretable — low performance =
   model/data, not hidden ML routing ambiguity.
2. **Text features: structured-only in the main pipeline** (topic flags, sentiment
   lexicon, structural counts). Embeddings/transformer remain the **deliberate overfit
   canary** (§8), off by default, explored later only as a *separate ablation*.
3. **PnL-style metrics are diagnostics only.** Success stays BH-corrected significance +
   calibrated conformal abstention (§1, §4). No trading simulation (§2).
4. **Evaluation split: purged + embargoed walk-forward CV** (§3.6) — stronger than a
   single train/val/test split and kept as the mandatory protocol.

## Core change vs v1: single-sector → multi-sector

- v1: one tweet → one ETF (argmax), one direction.
- v2: one tweet → **zero, one, or many** candidate ETFs, each scored independently.
- Dataset becomes a **cross-product**: one row per `(tweet × candidate_ETF × horizon)`.
- **Purge by `tweet_id`** so all rows spawned by one tweet stay in the same CV fold
  (no leakage across the cross-product). This is already how purging works — preserve it.

## Labeling — abnormal return (the false-attribution fix)

For each `(tweet, ETF, horizon h)` with entry anchored at `s0` (first session open
strictly after `t0`, unchanged from §3.3):

```
raw_ret_h      = ETF_close(s0 + h-1) / ETF_open(s0) - 1
spy_ret_h      = SPY_close(s0 + h-1) / SPY_open(s0) - 1
abnormal_ret_h = raw_ret_h - spy_ret_h          # PRIMARY target
```

Label from **abnormal** return, threshold **configurable per horizon**:

```
UP       if abnormal_ret_h >  +thr[h]
DOWN     if abnormal_ret_h <  -thr[h]
NEUTRAL  otherwise
```

- `thr[h]` tighter for 1d, looser for 5d. Still expressible as `k · σ_backward` where
  σ is the sector's backward-only vol — keep it vol-scaled and backward-only (§3.5).
- Report class balance per (sector, horizon) — degenerate split = labeling failure.
- **Horizons: 1d / 3d / 5d** (was 1/2/3). Optional 1-week = 5d.

## Features

All market features are **point-in-time, strictly `< t0`** (§3.1). The scanner test
must cover every new feature.

### A. Text / event semantics (structured, from tweet text)
- **Topic multi-label flags**: tariffs/trade-war, China, oil/energy, OPEC/drilling,
  Fed/rates, inflation, banking/finance-reg, defense/military, healthcare/drug-pricing,
  semiconductors/chips, big-tech/AI, manufacturing/reshoring, autos/consumer.
  Binary or soft score. These double as the **sector candidate generator**.
- **Sentiment / tone**: lexicon-based polarity + urgency markers ("immediately",
  "massive", "huge", "crisis"). Stdlib word lists — no model.
- **Structural**: length, count of numbers/%, exclamation/emphasis count,
  named-entity flags (countries, companies).

### B. Pre-event market regime (per candidate ETF, strictly pre-`t0`)
- **Trend**: returns over 1/3/5/10/20 trading days before.
- **Volatility**: rolling vol 5/10/20d + low/med/high regime bucket.
- **Relative to SPY**: ETF−SPY return over the same windows; rolling beta.
- **Momentum**: short-vs-long MA slope; 5d-vs-20d spread.
- **Stress**: current drawdown from recent peak; rebound-vs-decline state.
- **Volume**: volume z-score vs recent average; unusual-activity flag.

### C. Event-time context
- Market-hours / after-hours / weekend bucket; proximity to open/close;
  day-of-week. (`core/calendar.py` already resolves the session placement.)

## Model

- **Primary: GBT (XGBoost) multiclass** over the structured feature vector, predicting
  `{UP, NEUTRAL, DOWN}` per `(ETF, horizon)`. ETF identity + its pre-event features are
  part of the input, so one model serves all sectors (no per-sector model explosion).
- **Outputs per (tweet, ETF, horizon)**: class probabilities, predicted label,
  optional regression head for expected abnormal return.
- **Abstention**: class-conditional (Mondrian) conformal, pre-registered coverage (§8).
  Not softmax-thresholded.
- Transformer = overfit canary only (§8), unchanged.

## Endpoint output (multi-sector, ranked)

`POST /predict  { tweet_text, timestamp }` → ranked list; each item:

```json
{
  "etf": "XLI", "sector": "Industrials", "relevance_score": 0.87,
  "predictions": {
    "1d": { "label": "UP",
            "probabilities": {"UP": 0.64, "NEUTRAL": 0.20, "DOWN": 0.16},
            "abnormal_return_estimate": 0.012 } },
  "explanation": {
    "matched_topics": ["tariffs", "China", "manufacturing"],
    "market_context": {"pre_5d_momentum": 0.014, "pre_20d_volatility": 0.19,
                       "spy_regime": "neutral"} }
}
```

- `relevance_score` = deterministic keyword/entity match strength (not learned).
- Return top-K (≈3–8) candidates, ranked by relevance then confidence.
- Still calls the single `decide()` (§3.2). No live polling; abstain if no bars `< t0`.

## Evaluation

- Purged + embargoed walk-forward CV; embargo ≥ 3 sessions; purge by tweet id.
- Baselines: majority, **market-follow**, **permutation null**.
- **Text-ablation is primary**: full vs market-only. If market-only matches full,
  the tweet adds nothing → claim dead (§7).
- Significance: effect size + permutation p, **BH-corrected over the full registry**
  (sectors × horizons × models — now larger, so a higher bar; that is correct).
- **Power / MDE gate first** (§4) — likely under-powered at N≈150–300 × sectors.
- **Diagnostics (not success bar)**: macro-F1, balanced accuracy, calibration; and the
  financial-flavored ones (avg abnormal return of UP calls, hit rate, extreme-move
  precision, top-k / NDCG sector ranking).

## Serverless (unchanged shape, §13)

- **Jobs** (offline): data_ingestion → labeling → feature_engineering → dataset_build →
  training → evaluation → reporting. Thin CLIs over the pure modules; zero science.
- **Endpoint** (online): thin handler → `market_state_as_of(t0)` → `decide()` → ranked
  multi-sector response. Same Docker image as jobs.

## Delta map — what changes in code

| Module | Change |
|---|---|
| `config/settings.py` | horizons → (1,3,5); per-horizon thresholds; benchmark-adjust flag |
| `labeling/windows.py` | add SPY-adjusted `abnormal_ret_h` |
| `labeling/thresholds.py` | per-horizon `thr[h]`, label off abnormal return |
| `core/features.py` | pre-event regime block (trend/vol/rel-SPY/momentum/drawdown/volume) + structured text (topics/sentiment/structural) + event-time |
| `sector_mapping/rules.py` | emit **multiple** ranked candidates + relevance score |
| `dataset/build.py` | cross-product tweet × candidate ETF × horizon; purge key = tweet id |
| `models/gbt.py` | multiclass probs + optional abnormal-return regression head |
| `serving/endpoint.py` | ranked multi-sector response + explanation |
| `eval/*` | registry over (sector×horizon×model); ranking + financial diagnostics |
| `data/` | real yfinance bars (done: `data/real/bars.csv`); real Trump tweets (todo) |

## Status

- ✅ Real daily bars pulled: `data/real/bars.csv` (11 tickers incl SPY, ~3y, 8283 rows).
- ⬜ Real Trump tweets (free archive) — still synthetic fixture.
- ⬜ Everything in the delta map above.
