# Dual-Track Predictor — Macro vs Micro (design blueprint)

Every incoming tweet is **routed to one of two tracks** before any prediction. The two tracks
share the same brain (Nebius LLM classification, intraday+daily market fetch, relative-alpha
scoring, tweet-level collapse, 3-way split, reasoning layer, cards) and differ only in **what
benchmark the basket is measured against**.

```
tweet ──► ROUTER (macro vs micro) ──► MACRO track  ─┐
                                    └► MICRO track ─┴► shared: LLM reasoning + fetch +
                                                        relative-alpha hit + split + cards
```

## 0. Router — `classify_track(tweet)`

Decides the track from **tweet-time-only** signals (no price data):
- **Micro** when the tweet targets a *single company*: `sector_mapping.entities.entity_matches(text)`
  returns a direct company ticker **and** the Nebius signal has `names_company = true`. Target = that ticker.
- **Macro** otherwise (wars, peace, rates, universal tariffs, "the economy"): no single-company
  subject. Target = the LLM's scenario basket.
- Tie-break: a tweet that names a company *inside* a macro policy ("tariffs will crush Apple")
  routes **Micro** on the named stock but keeps the macro scenario as a feature.

The LLM already returns `scenario`, `names_company`, `intensity`, `summary`, `macro_link` — the
router reads those + the deterministic entity gazetteer. No new model call.

## 1. Macro Track (broad tweets) — *already built*

| | |
|---|---|
| **When** | wars, global peace, interest rates, universal tariffs, "the economy" |
| **Benchmark** | **SPY only** (broad-market beta) |
| **Predictions** | market sentiment (SPY/QQQ/DIA), volatility (VIXY), commodities (USO/WEAT/CORN) |
| **Label** | Option B relative alpha: instrument beats SPY in the predicted direction (`abn = ret − SPY_ret`) |
| **Code** | `scripts/nebius_macro_validate.py` (`validate`, `relative_hit`) + `scripts/nebius_macro_backtest.py` |

This is the pipeline the 150-tweet run exercises now. No changes.

## 2. Micro Track (single-stock tweets) — *dynamic benchmark, mostly built*

| | |
|---|---|
| **When** | a tweet targets one company (Intel, Tesla, Boeing) |
| **Benchmark** | **auto-resolved per stock** — mean(its indices) + its sector ETF(s) + its peer group |
| **Predictions** | relative alpha: does the stock beat index AND sector AND peers, at 30m / 1h / 1mo |
| **Label** | stock beats **all three** benchmark groups beyond a vol-band, in the LLM-predicted direction |

**The dynamic benchmark already exists** and is reused verbatim:
- `config/membership.py::benchmarks_for(ticker) -> BenchmarkSet(indices, sectors, peers)`
  - `INTC` → indices `[SPY, QQQ]`, sector `[SMH]`, peers `[NVDA, AMD, TSM, AVGO]`
  - `TSLA` → indices `[SPY, QQQ]`, sector `[XLY]`, peers `[AMZN, HD, MCD, NKE]`
  - `BA` → indices `[SPY, DIA]`, sectors `[XLI, ITA]`, peers `[CAT, GE, RTX, LMT, NOC, GD, UNP]`
- `labeling/benchmarks.py::compute_bench_outcome` already computes per horizon:
  `abn_index = ret − mean(indices)`, `abn_sector = ret − mean(sector ETFs)`,
  `abn_peer = ret − median(peers)`, and folds them into a label requiring **beat-all-three**.

## 3. Shared components (both tracks call these — no train/serve skew)

- **LLM classification + reasoning**: `scripts/nebius_macro_validate.py::classify_tweet` (Nebius,
  OpenAI-compatible). Returns scenario, intensity, summary, macro_link, hypothesis_short/long,
  instruments — plus (micro) the named ticker.
- **Market fetch**: `daily_returns` (yfinance) + `intraday_returns` (Alpaca 1-min, next-open
  anchored → captures the Monday-open shock). Memoised per (ticker, date).
- **Relative-alpha hit**: `relative_hit(pred, ret, bench_ret, band)` — macro passes `bench = SPY`;
  micro passes `bench = mean(index/sector/peer set)`. Same function, different benchmark.
- **Honest stats**: tweet-level majority collapse (`_tweet_hit`), binomial vs 50%.
- **Split**: chronological 60/20/20 (`_assign_splits`), sacred test.
- **Reasoning cards**: `reportgen/macro_card.py` (`render_narrative`, `render_gallery`).

## 4. Code plan (what to write when we build it)

1. `router.py` — `classify_track(text, signal) -> ("macro", None) | ("micro", ticker)`. Reuses
   `entity_matches` + the LLM `names_company`.
2. `scripts/nebius_micro_validate.py` — the micro analogue of `validate()`: for the named stock,
   resolve `benchmarks_for(ticker)`, compute `abn_index/abn_sector/abn_peer` per horizon, hit =
   beat-all-three in the predicted direction. **Reuse** `compute_bench_outcome` for daily; wire the
   intraday layer for 30m/1h.
3. `scripts/nebius_dual_backtest.py` — one loop that routes each tweet, dispatches to the macro or
   micro validator, and writes **two labelled datasets** (`macro_dataset.csv`, `micro_dataset.csv`)
   each with its own 60/20/20 split. Tag every row with `track`.
4. Cards/gallery unchanged — micro cards show the index/sector/peer lines that
   `reportgen/tweet_chart.py` already renders.

## 5. The one real data gap

Daily (EOD→1mo) micro benchmarks are fully ready. **Intraday (30m/1h) peer/index** benchmarks
need Alpaca minute bars for the *benchmark & peer* tickers too (QQQ, SMH, AMD, NVDA…), not just the
stock + sector. That's a bounded bulk fetch (extend `scripts/fetch_alpaca_30m.py` to the peer/index
symbols) — scope it before running the micro 30m/1h horizons; until then micro intraday is
sector-relative only (honest, flagged), micro daily is full three-benchmark.

## 6. Verification (when built)
- Router unit check: a peace-deal tweet → macro; "Intel is doing great" → micro/INTC.
- Micro `validate` on the two Intel examples (2025-08-11 positive, 2025-08-07 negative) — abn vs
  mean(SPY,QQQ)+SMH+peers should match the hand-computed numbers.
- Dual backtest emits two datasets with disjoint tracks and clean ~50% relative baselines.
