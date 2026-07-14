# Data Provenance

Challenge rule: **public datasets only, fully reproducible without private
keys.** This file records, for every file in `data/real/`, its origin, whether
it is publicly redistributable, and whether a stranger with no private keys can
regenerate it.

**Bottom line:** the shipped pipeline runs on the **daily-bar path**, which is
100% reproducible from public sources (yfinance + public tweet archives). The
Alpaca intraday files are an **optional committed cache** used only for the
higher-resolution (30m/1h) *diagnostic* horizons; the batch job and the endpoint
both run correctly with **no Alpaca keys**, in which case intraday horizons are
skipped and the manifest says so.

## Summary table

| File(s) | Origin | Tool / source | Public? | Stranger-reproducible (no private keys)? |
|---|---|---|---|---|
| `bars.csv` | Daily OHLCV, ~53 tickers, 2008-11 → 2026-07 | yfinance (`scripts/fetch_real_bars.py`, `fetch_extra_bars.py`) | ✅ Yahoo public data | ✅ **Yes** — no key |
| `corpus_v3.csv`, `corpus.csv`, `tweets.csv` | Trump posts, Twitter + Truth Social | Twitter archive + HF `stiles/trump-truth-social-archive` + CNN public parquet mirror (`scripts/build_corpus_v3.py`) | ✅ public archives (see below) | ✅ **Yes** — public files, no key |
| `trump_in_office_raw.csv`, `trump_bf_office_raw.csv`, `truth_social_raw.parquet`, `truth_cnn_raw.parquet` | Raw per-source dumps feeding the corpus | same public archives as above | ✅ | ✅ Yes |
| `bars_1h.csv`, `bars_1m_events.csv`, `bars_1m_gas_tweet.csv`, `intraday_reactions.csv` | Intraday bars + derived 30m/1h reactions | **Alpaca IEX free feed** (`scripts/fetch_alpaca_30m.py`) | ❌ requires an Alpaca account/key | ❌ **No** — needs your own (free) Alpaca key; committed here as a convenience **cache** |
| `labeled*.csv`, `stock_event*.csv`, `market_events.csv`, `entity_results.csv` | Derived modeling/EDA frames | built by `scripts/build_*.py` from the above | inherits | inherits from inputs |

## Public tweet-archive sources (redistributable)

The corpus is assembled from **public posts of a public figure**, drawn from
public archives — not from a private scrape:

- **Twitter archive** (`source_dataset=twitter_archive`) — the public Trump
  Twitter Archive of his @realDonaldTrump posts (2009 → 2021).
- **Truth Social** (`source_dataset=truth_social_dataset` / `cnn_truth_archive`)
  — the Hugging Face dataset `stiles/trump-truth-social-archive` and its
  CNN-hosted mirror `https://ix.cnn.io/data/truth-social/truth_archive.parquet`
  (public file, no key, millisecond timestamps).

These are public redistributions of public statements. We redistribute only the
derived research columns; no private API, login, or paid scrape is involved.
`ts_confidence` records timestamp precision (`ms`/`s`/`min`) per row so the
point-in-time entry anchor (§3.1) is auditable.

## Alpaca intraday — explicitly optional

`intraday_reactions.csv` and the `bars_1m*`/`bars_1h.csv` caches come from the
**Alpaca IEX feed**, which needs an account key (free tier is sufficient). A
stranger **cannot regenerate them without their own key**, so they are **not**
on the reproducible critical path:

- The batch job (`jobs/backtest`) runs with **no `ALPACA_*` keys** and simply
  skips the 30m/1h horizons, recording that in `validation_manifest.json`.
- The **shipped** decision horizon (see the manifest's `shipped_horizons`) comes
  from the **daily** path (EOD), which is fully public/reproducible.
- To regenerate the intraday cache yourself: set `ALPACA_API_KEY` /
  `ALPACA_API_SECRET` (or `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`) and run
  `scripts/fetch_alpaca_30m.py`.

## Reproduce the public path

```bash
# daily bars (public, no key):
PYTHONPATH=. .venv/Scripts/python.exe scripts/fetch_real_bars.py
# tweet corpus (public archives, no key):
PYTHONPATH=. .venv/Scripts/python.exe scripts/build_corpus_v3.py
```
