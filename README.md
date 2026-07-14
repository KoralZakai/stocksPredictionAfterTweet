# Political Tweets → Sector-ETF Alpha: an honest, reproducible serverless pipeline

A **statistical ML research pipeline** for the **Nebius Serverless AI Builders
Challenge**. It asks one question rigorously: *does a raw large-language-model
read of a political tweet carry measurable, statistically significant short-term
directional information about US sector ETFs — beyond market beta?*

**The honest result.** Yes, at short horizons, and we ship exactly what survives
scrutiny. A raw zero-shot **Llama-3.3-70B** classification, scored as
**relative alpha ("beat SPY")**, beats the market on a held-out, chronological
test set: **EOD 61.8%** (n=89, Wilson 95% CI [0.514, 0.712], BH-corrected
p=0.033). The intraday edge is even stronger (**1h 76%, p≈0**) but requires a
private data feed, so it is reported as a diagnostic, not shipped. Everything
that could inflate the result — market beta, per-tweet confidence scoring,
overlapping windows — was tested and **rejected**: a second-stage meta-model that
tried to predict *which* calls land failed the sacred test three times (text Val
AUC 0.593 → **Test 0.431**). So we ship the **raw call**, never a per-tweet
probability. A rigorous, well-bounded result is the deliverable.

---

## Architecture — a Job produces the numbers, an Endpoint cites them

```
                                 ┌──────────────────────────────────────┐
  data/real/corpus_v3.csv  ──►   │  Nebius Serverless AI JOB (CPU)       │
  data/real/bars.csv  ──────►    │  jobs/backtest/entrypoint.py          │
                                 │   classify (Nebius 70B, cached) →     │
                                 │   beat-SPY relative-hit (signed) →    │
                                 │   chronological 60/20/20 →            │
                                 │   Benjamini-Hochberg + Wilson CI      │
                                 └───────────────┬──────────────────────┘
                                                 │ writes artifacts
                                                 ▼
                                 ┌──────────────────────────────────────┐
                                 │  shared bucket (/data)                │
                                 │   reports/macro_dataset.csv           │
                                 │   reports/validation_manifest.json ◄──┼── the CONTRACT
                                 └───────────────┬──────────────────────┘
                                                 │ loaded + hash-verified at boot
                                                 ▼
   POST /predict  {tweet_text,   ┌──────────────────────────────────────┐
   t0_utc, author}          ──►  │  Nebius Serverless AI ENDPOINT (CPU)  │
                                 │  serving/app.py  (FastAPI, two-plane) │
                                 │   DECISION plane: text → 70B → route  │
                                 │   MARKET plane:  post-hoc enrichment  │
                                 └──────────────────────────────────────┘
```

The serverless layer contains **zero science**. Both the Job and the Endpoint
call the same pure engine in [`alpha/`](alpha/); `jobs/` and `serving/` only
marshal I/O. This is what makes the result trustworthy rather than plumbing.

### Two integrity guarantees you can check

1. **Validated model = served model.** The Job records
   `prompt_template_hash` (sha256 of the exact classification prompt) and the
   corpus `sha256` in the manifest. The Endpoint **refuses to boot** if its live
   prompt or corpus doesn't match. No hit-rate is hardcoded in the serving code —
   the Job produces the numbers, the Endpoint only cites them.
2. **The leakage firewall.** `/predict` has two planes with one-way flow. The
   **decision plane** sees *tweet text only* (the 76%/62% edge holds precisely
   because the LLM never saw market data). The **market plane** runs *after* the
   decision to enrich the response and can never feed back into it. A test
   (`test_decision_invariant_to_market_data`) monkeypatches the market provider to
   raise, time out, and return absurd values, and asserts the decision is
   byte-identical across all of them.

---

## Reproduce it

### Prerequisites
- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv).
- A Nebius AI Studio key for live classification (`NEBIUS_API_KEY`). **Not needed**
  for the offline `$0` path below.
- `cp .env.example .env` and fill in `NEBIUS_API_KEY` (Alpaca keys optional).

### Run it for **$0** (offline, no keys)
The Job re-scores from cached teacher outputs and regenerates the manifest —
proving the full statistical path without spending a cent:
```bash
make smoke
# = PYTHONPATH=. .venv/Scripts/python.exe jobs/backtest/entrypoint.py --from-results
```
Or classify **10 tweets live for ~$0**:
```bash
make smoke-live      # jobs/backtest/entrypoint.py --live --limit 10   (needs NEBIUS_API_KEY)
```

### Build the two images (CPU-only, no GPU)
```bash
docker build -f jobs/backtest/Dockerfile -t backtest .     # the batch Job
docker build -f serving/Dockerfile        -t predict   .   # the Endpoint
```

### Launch on Nebius (illustrative — match your CLI/tenant)
```bash
# 1. Batch Job: writes macro_dataset.csv + validation_manifest.json to the bucket.
nebius ai job create --name backtest-and-validate \
    --image $NB_REGISTRY/backtest:latest \
    --preset cpu-... --mount $NB_BUCKET:/data \
    --command "python jobs/backtest/entrypoint.py --from-results"

# 2. Endpoint: loads + hash-verifies the manifest at boot, then serves.
nebius ai endpoint create --name predict \
    --image $NB_REGISTRY/predict:latest \
    --preset cpu-... --mount $NB_BUCKET:/data --port 8080
```

### Call `/predict`
```bash
curl -s -X POST http://<endpoint>/predict -H 'content-type: application/json' -d '{
  "tweet_text": "We will impose massive tariffs on all Chinese semiconductors.",
  "t0_utc": "2025-03-03T14:00:00+00:00", "author": "realDonaldTrump"
}'
```
```jsonc
{
  "decision": "SHORT",
  "instruments": [{"ticker": "SMH", "direction": "down", "benchmark": "SPY"},
                  {"ticker": "XLK", "direction": "down", "benchmark": "SPY"}],
  "scenario": "Trade War", "reasoning": "tariffs raise input costs for chipmakers",
  "horizon": "EOD",
  "cohort_base_rate": {
    "value": 0.618, "ci95": [0.514, 0.712], "n": 89, "horizon": "EOD",
    "note": "Historical hit-rate of ALL calls of this type on a held-out chronological
             test set. This is NOT a probability for THIS tweet. We tested per-tweet
             confidence; it did not generalize."
  },
  "market_context": {"provider": "yfinance", "session_phase": "premarket",
                     "entry_anchor_utc": "2025-03-03T13:30:00+00:00", "quotes": [],
                     "realized_alpha_since_t0": []},   // null on any market-plane failure
  "manifest_version": "7852708", "disclaimer": "Research output. Not investment advice."
}
```
`GET /health` returns the manifest version, corpus + prompt hashes, and shipped
horizons. `GET /market/{ticker}` returns a quote + benchmark (market plane).

**Abstention is a first-class success.** `/predict` returns `ABSTAIN` when the
tweet isn't market-relevant, no whitelisted instrument resolves, or `t0` is
unanchorable.

---

## Hardware, outputs, runtime, cost

| | |
|---|---|
| **Hardware** | CPU-only (classical-ML track). No GPU. Both images build from `python:3.14-slim`. A small CPU preset (≈2 vCPU / 4 GiB) is sufficient. |
| **Expected outputs** | `reports/macro_dataset.csv` (443 scored tweets) and `reports/validation_manifest.json` (`shipped_horizons: ["EOD"]`, per-horizon n/hit-rate/CI/p_bh, prompt+corpus hashes). |
| **Approx runtime** | Offline `--from-results`: **~10 s**. Live `--limit 10`: **~1 min** (10 Nebius calls + daily-bar fetch). Full 443-tweet live classification: **~15–25 min** (cached + resumable). Endpoint cold boot: **a few seconds** (manifest verify). |
| **Approx cost** | Offline path: **$0**. `--limit 10`: **≈$0** (a few 70B calls). Full corpus classification: a **few cents** of Nebius inference (cached, so paid once). Endpoint: pay-per-request, scale-to-zero. |

---

## Experimental extension (NOT shipped): GPU knowledge distillation

[`experiments/distill/`](experiments/distill/) contains an optional downstream
optimization: distilling the 70B teacher's *classifications* (not the noisy market
outcome) into a small 7–8B student via QLoRA, to cut inference cost. It is
**clearly quarantined**: separate deps, trains on the **train split only**, and its
serving code labels the cohort rates as the *teacher's* with the student marked
**unvalidated** until re-tested on the sacred split. It does **not** touch the
shipped path or the validation manifest. See its
[README](experiments/distill/README.md).

---

## Repository layout

| Path | Role |
|---|---|
| [`alpha/`](alpha/) | The one shared engine: `classify`, `benchmark`, `route`, `schema`, `stats`. |
| [`jobs/backtest/`](jobs/backtest/) | The Nebius **Job**: entrypoint + Dockerfile. |
| [`serving/app.py`](serving/app.py) | The Nebius **Endpoint** (shipped, two-plane). |
| [`market/`](market/) | Market-plane price providers (alpaca → yfinance → null). |
| [`experiments/distill/`](experiments/distill/) | Experimental GPU distillation (not shipped). |
| [`archive/`](archive/) | The rejected meta-model + its Val→Test AUC record. |
| `core/`, `eval/`, `labeling/`, `serving/endpoint.py` | **Deprecated System-A** scaffold (rule-based `decide()`); superseded by the System-B pipeline above, kept for history. |

License: [MIT](LICENSE). Data provenance: [`data/PROVENANCE.md`](data/PROVENANCE.md).

## Limitations & threats to validity

- **Association, not causation.** With daily bars we measure 1-day drift, not the
  seconds-scale causal reaction. Macro confounding is documented, not deconfounded
  (N≈443 observational rows cannot identify causality).
- **Reproducibility boundary.** The shipped EOD horizon is fully public
  (yfinance + public tweet archives). The stronger 1h/30m edge needs a private
  Alpaca feed, so it is **excluded from `shipped_horizons`**.
- **Small n.** The chronological test split is 89 tweets; we use Wilson intervals
  and exact binomial p-values (not the normal approximation) and BH-correct across
  the whole horizon registry.
- **Selection & survivorship.** Deleted tweets are gone and may be systematically
  the moved-then-retracted ones; the corpus is public archives of a public figure.
- **No per-tweet confidence.** By design. The meta-model that would produce one
  failed the sacred test; the response schema has no probability field.
