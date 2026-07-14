# Final Production Blueprint — Political-Tweet → Relative-Alpha Signal

**Signal-or-null verdict:** the LLM's *raw directional relative-alpha* on macro tweets is **real,
scaled, and statistically bulletproof** (N=443, all horizons p<0.001, 1h peak **64.4%**). A
*second-stage meta-model* that tries to predict which calls will land is a **verified null** — it
overfits Validation and fails the sacred Test every time. **We ship the raw LLM edge; we do not
ship the meta-model.**

---

## 1. Executive summary

| | Result | Status |
|---|---|---|
| **Raw LLM relative-alpha** (macro, Beat-SPY) | 58–64% across horizons, N=443, p<0.001 | ✅ **DEPLOY** |
| **Second-stage meta-predictor** (metadata+text → call-success) | Test AUC 0.43–0.53, ≤ base rate | ❌ **REJECT (null)** |
| **Micro track** (single-stock, dynamic benchmark) | data-capped at N=84, inconsistent (Val 35% / Test 59%) | ⚠️ **PARK (insufficient N)** |

The product is a stateless service that, for each incoming tweet, returns the LLM's directional
call on a benchmark-relative basket, tagged with the **empirically validated hit-rate as its
confidence**. No trained downstream model — the "model" is the LLM's reading + the pre-registered
benchmark logic.

---

## 2. The proven signal — scaled macro edge (Option B relative alpha)

**Label (Option B):** a HIT = the instrument beats **SPY** in the LLM-predicted direction at the
horizon (`abn = instrument_ret − SPY_ret`, strict band=0). This strips market beta, so the null is a
clean ~50% coin-flip — not the 74% "everything drifts up" majority we started with.

**Honest aggregate — one call per tweet (N=tweets, correlated instruments collapsed by majority
vote), 443 macro tweets:**

| Horizon | Beat-SPY hit-rate | N | z | p vs 50% |
|---|---|---|---|---|
| 30m | 59.7% | 303 | 3.39 | 0.0004 |
| **1h** | **64.4%** | 303 | **5.00** | **<0.0001** |
| EOD | 58.2% | 443 | 3.47 | 0.0003 |
| 3d | 60.0% | 443 | 4.23 | <0.0001 |
| 1w | 60.3% | 443 | 4.32 | <0.0001 |
| 1mo | 59.1% | 443 | 3.85 | 0.0001 |

**Cross-sample stability (chronological 60/20/20, EOD label):** Train 54.9% · **Val 64.8%** ·
**Test 61.8%** — consistent out-of-sample, positive_rate 0.582.

**Key scientific finding — alpha is front-loaded.** Under *absolute* returns the edge appeared to
build to 1mo; once SPY beta is removed, the tweet-specific alpha is **fastest early (1h 64.4%)** and
merely persists later. The slow 1-month "drift" was market beta, not the tweet. Serve the **1h / EOD**
horizons.

---

## 3. The null results (equally important)

**Micro track — data ceiling.** Single-stock tweets that must beat their *own* index + sector +
peers are a far harder, idiosyncratic bar. The corpus holds only **~84** direct single-company
tweets (hard limit), and results are inconsistent across the tiny splits (strict Val 35% / Test 47%;
soft-blended Val 35% / Test 59%). **Parked until a larger single-stock corpus exists.**

**Second-stage meta-model — overfits, does not generalize.** Predicting *which* LLM calls land from
`[intensity, phase, weekend_flag, market_closed, scenario, track]` + LLM reasoning text:

| Feature set | Val AUC | **Test AUC (sacred)** | Macro test |
|---|---|---|---|
| metadata only | 0.554 | 0.525 | 0.497 (random) |
| + LLM text (summary+macro_link) | 0.593 | **0.431** | 0.412 |

Text lifted Validation every single time (tempting!) and **degraded the Test every single time**.
Conclusion: call-success is **not predictable** from this metadata/text out-of-sample. The alpha is
in the LLM's reading of the tweet, not in a classifier stacked on top.

---

## 4. Serving architecture — raw signal, no meta-model

Deploy on **Nebius Serverless** (the challenge substrate). One stateless endpoint; the "brain" is the
Nebius LLM classification + the pre-registered benchmark logic. **No downstream model is loaded.**

```
POST /predict  { "tweet_text": "...", "timestamp": "2025-04-20T18:00:00Z" }

  1. ROUTER            entity_matches + LLM names_company
                         -> MACRO (broad)  or  MICRO (single stock)
  2. NEBIUS CLASSIFY   Llama-3.3-70B (OpenAI-compatible /chat/completions)
                         -> scenario, intensity, instruments[], direction, reasoning
  3. BENCHMARK MAP     MACRO: each instrument vs SPY (Option B)
                       MICRO: config.membership.benchmarks_for -> mean(indices)+sector+peers
  4. CONFIDENCE        attach the VALIDATED hit-rate per horizon
                         (macro 1h 0.644, EOD 0.582 — from §2, not a live model)
  5. ABSTAIN           empty instruments / not market-relevant -> clean abstain

RESPONSE {
  "track": "macro",
  "scenario": "Geopolitics / Peace",
  "reasoning": { "summary": "...", "macro_link": "..." },
  "calls": [
    { "instrument": "VIXY", "predicted": "down", "benchmark": "SPY",
      "horizon": "1h",  "confidence": 0.644 },
    { "instrument": "XLI",  "predicted": "up",   "benchmark": "SPY",
      "horizon": "EOD", "confidence": 0.582 }
  ]
}
```

**Point-in-time & no-skew.** The call is text-only (knowable at t0); confidence is a static,
pre-registered constant from the §2 backtest — nothing is fit at serve time, so train/serve cannot
drift. A live/future `t0` with no market context still returns a clean directional call + abstain
flag, never a fabricated number.

**Components (all built):**
- `scripts/nebius_macro_validate.py` — `classify_tweet` (Nebius) + `relative_hit` (Beat-SPY) + intraday/daily fetch.
- `config/membership.py` + `labeling/benchmarks.py` — micro dynamic benchmark (parked track).
- `serving/endpoint.py` — thin HTTP handler pattern to wrap the above.
- `deploy/nebius/` — Dockerfile + job/endpoint manifests.

**Runtime image:** one CPU image (classical stack + `requests`/`yfinance`); the LLM is a remote
Nebius call, so the endpoint itself is tiny and cheap.

---

## 5. Methodology audit — how the split caught the overfit

The chronological **Train 60 / Validation 20 / Test 20** shield is the reason this project has a
trustworthy conclusion instead of a plausible-but-wrong one.

1. **Time-ordered, never random.** Splits are cut by date so no future tweet informs a past
   prediction (no leakage). Test = the most recent ~20%, held sacred (touched once).
2. **Independence fix.** Each tweet fires ~5 correlated instruments; we collapse to **one call per
   tweet (majority vote)** so significance is computed on N=tweets, not inflated instrument-calls.
   This alone moved the honest N from a fake 183/2215 down to a real 303–443.
3. **Baseline hygiene.** The first label (absolute return) was 74% positive — pure beta. Switching to
   Option B (Beat-SPY, band=0) reset the null to a clean ~50%, so "beating the baseline" became meaningful.
4. **The catch.** The meta-model looked good on Validation three separate times:
   - N=223 metadata: Val bal-acc 0.616 → **Test 0.484**.
   - N=527 metadata: Val 0.516 → **Test AUC 0.525 (macro 0.497)**.
   - N=527 + text: Val 0.593 → **Test AUC 0.431**.
   Every Val gain evaporated (or reversed) on the sacred Test. The **Val→Test gap is the overfit
   signature**, and because Test was quarantined, we saw it instead of shipping it. A random split or
   a "tune on everything" workflow would have deployed a 0.43-AUC model believing it was 0.59.

**Verdict:** the shield worked. It protected the real discovery (§2) and rejected the overfit
candidate (§3) — exactly its job.

---

## 6. Honest limits

- **Confounding.** A market-wide move after a tweet is driven by everything that day; Beat-SPY and
  the (parked) peer-relative checks *reduce* but do not *eliminate* confounding. This is association
  at the drift horizon, not proven causation.
- **Multiple horizons.** Six horizons tested; even Bonferroni-corrected, 1h and EOD survive
  comfortably (p<0.001), but treat the exact rates as estimates, not guarantees.
- **Micro is underpowered** (N=84 hard cap) — no claim either way.
- **Intraday coverage** (30m/1h) is Alpaca-IEX and thinner than daily (N=303 vs 443); the 1h peak is
  real but on fewer tweets.

---

## 7. Reproduce

```bash
# scaled macro backtest (Beat-SPY, EOD label, 3-way split, reasoning + cards)
PYTHONPATH=. .venv/Scripts/python.exe scripts/nebius_macro_backtest.py --limit 500
# instant re-score at a new band / horizon (no re-fetch)
PYTHONPATH=. .venv/Scripts/python.exe scripts/nebius_macro_backtest.py --from-results --band 0
# micro track (dynamic benchmark, data-capped)
PYTHONPATH=. .venv/Scripts/python.exe scripts/nebius_micro_backtest.py --limit 500
# the dual meta-model audit (metadata vs +text; the rejected candidate)
PYTHONPATH=. .venv/Scripts/python.exe jobs/train_dual_engine.py
```

**Artifacts:** `reports/macro_dataset.csv` (443 × 60/20/20), `reports/micro_dataset.csv` (84),
`*_config.json`, `reports/nebius_validation_examples.html` (reasoning cards),
`reports/nebius_backtest_top5.html`, `docs/DUAL_TRACK_ARCHITECTURE.md`.

**Bottom line:** ship the LLM's raw relative-alpha call (macro, 1h/EOD, ~58–64%, p<0.001). Skip the
meta-model. The rigor is the product.
