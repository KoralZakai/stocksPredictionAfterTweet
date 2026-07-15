# Political Tweets → Sector-ETF Alpha: an honest, reproducible serverless pipeline

A **statistical ML research pipeline** for the **Nebius Serverless AI Builders
Challenge**. It asks one question rigorously: *does a raw large-language-model
read of a political tweet carry measurable, statistically significant short-term
directional information about US sector ETFs — beyond market beta?*

**The honest result: no.** Not at any horizon we can reproduce. On the held-out
chronological test set the daily signal is **a coin flip — EOD 52.2%** (n=67,
Wilson 95% CI [0.405, 0.637], p=0.40). Longer horizons are worse (3d 44.1%, 1w
44.3%, 1mo 27.1%). **`shipped_horizons` is empty**: nothing survived
Benjamini-Hochberg correction that we can reproduce without private data, so the
`/predict` endpoint serves its classification while **citing no accuracy at all**.

**We nearly shipped the opposite claim, and the bug is the story.** An earlier
version of this README reported *"EOD 61.8%, p=0.033, beats the market."* That
number was an **artifact of one line of tie-breaking logic** — see
[The tie-break artifact](#the-tie-break-artifact) below. Two independent unbiased
estimators put the true test rate at **0.507** and **0.502**. The signal is null.
(EOD is **0.522** after a later fix to the market clock — see below. Still a coin flip.)

**The 1h horizon looked like a survivor — it isn't.** After the fix, 1h read
67.8% (n=59, p_bh=0.026). Two diagnostics then explained it away: (1) **selection
bias** — 1h data exists only for tweets posted in/near market hours, and that
subset also scores 0.588 at EOD (vs 0.250 for the rest): an easier subset, not a
faster edge; (2) **wrong null** — per-asset unconditional beat-SPY base rates run
0.33–0.60 (not 0.50), and against each asset's own base rate the LLM's edge
averages ≈ 0 (ITA −0.06, XLI −0.10, LMT −0.13). The 1h effect is an artifact of
comparing a favorable subset against the wrong baseline.

**A third bug, found by chasing one tweet: the market clock ignored DST.**
`US_OPEN_UTC_HOUR` was hardcoded to **13.5** — correct only in EDT. From November to
mid-March the NYSE opens **14:30 UTC**, so every tweet posted 13:30–14:30 UTC in
winter was treated as post-open and pushed to the *next* session. Not a leak (it
anchors *later*, using *less* information — which is why none of our leak tests
caught it) but it scored the wrong day. Now resolved per-date via `zoneinfo`, with
one shared `session_phase` (a second buggy copy lived in `scripts/`).

It moved **8 of 476 rows** (EOD 0.507 → **0.522**, still null) — but one of those
rows matters: the *"no deal with Iran except UNCONDITIONAL SURRENDER"* post, our
single most-cited geopolitical event. Correctly anchored to **Friday 03-06** rather
than Monday 03-09, the model's fear trade (oil ↑, defence ↑, VIX ↑) goes from **0/3
to 3/3**. Oil gapped **+9.4%** into the open we had been skipping. **The
"obvious" trade was right — and our clock had recorded it as wrong.** One event
proves nothing (n=1, and the study still returns 0/72), but it is the fairest
version of the sceptic's best case, and it deserved a correct clock.

This is what the system was built to do. Per the project charter: *a rigorous null
result is a full success.* The deliverable is an evaluation that **refused to
confirm its own hypothesis** — and that kept finding its own errors, in both
directions, right up to the deadline.

## The "intel" artifact — a ticker rule that matched a common English word

Twin of the DJT-signature bug (Trump signs posts "President DJT"; 754 of 762 "DJT
mentions" were the sign-off). Here the word was **intel**, and in a political corpus
that word is overwhelmingly *intelligence*:

| | matches | real (the chipmaker) | false |
|---|---|---|---|
| old guard, full corpus | 55 | 4 | **51 (93%)** |
| fixed guard | 12 | 12 | 0 |

The old guard blocked the word *after* intel (`intel leak`) but not before, so
`leaking intel`, `destroy intel`, `Danish intel`, `Obama Intel Chief`, `Ex-intel
official` and every URL slug (`us-intel-has-known`) sailed through — into the
**CORPORATE cohort of a registered study**. Rescored with the fix: `CORPORATE`
**43 → 13 events** (70% of that cohort was spy tweets), 72 → 63 scored cells,
and **the verdict is unchanged: 0 survive BH** (min p_bh 0.70 → 0.76).

The mirror-image half is the one worth remembering. The dashboard kept its *own*
private Intel regex that only accepted company context **after** the word, so it scored
`The CEO of INTEL must resign`, `I met with Mr. Lip-Bu Tan, of Intel` and `I PAID ZERO
FOR INTEL` as **not Intel** — and those are exactly the mentions that contradict the
reverse-causality exhibit (Intel was *falling* before the "CEO must resign" post, then
rose +18.1%). One rule dropped 93% true positives; the other dropped the inconvenient
cases. **Detection now lives in one tested place** (`sector_mapping/entities.py`,
precision-first, regression-tested both ways in `tests/test_entities.py`), and Exhibit A
now shows all 8 mentions with their measured before/after rather than the 3 that suited
it. See `experiments/event_study/REPORT.md`.

## The tie-break artifact

`_tweet_hit` aggregates a tweet's instrument basket into one verdict (correlated
instruments within a tweet are not independent samples — aggregating is correct).
The bug was the tie rule:

```python
return int(sum(hs) / len(hs) >= 0.5)   # 1-of-2 hits -> 0.5 >= 0.5 -> counted a WIN
```

**49% of baskets hold exactly 2 instruments**, where "majority" is undefined. Every
50/50 split was silently scored as a full win. On the test split, **20 of 89 tweets
(22%) were ties auto-counted as hits** — and that gap *was* the entire result:

| test split | k/n | rate | p |
|---|---|---|---|
| original (ties win) | 55/89 | **0.618** | 0.017 |
| ties **excluded** (correct) | 35/69 | **0.507** | 0.50 |
| instrument-level (independent check) | 106/211 | **0.502** | 0.50 |

Scoring ties as *losses* (`> 0.5`) is the mirror bug — it deflates to 0.393.
A tie carries **no directional verdict**, so it is unscoreable: `_tweet_hit`
now returns `None`. Both unbiased estimators agree at ~50%.

This also explains the rejected meta-model (Val AUC 0.593 → **Test 0.431**): it
wasn't a modelling failure. **There was never any signal to learn.**

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
   **decision plane** sees *tweet text only* — the firewall is what keeps any
   measured effect attributable to the text rather than to leaked market state. The **market plane** runs *after* the
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
  "cohort_base_rate": null,   // no horizon survived BH -> no accuracy is cited
  "horizon": null,
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
| **Expected outputs** | `reports/macro_dataset.csv` (443 scored tweets) and `reports/validation_manifest.json` (`shipped_horizons: []` — nothing reproducible survived BH; per-horizon n/hit-rate/CI/p_bh, prompt+corpus hashes). |
| **Approx runtime** | Offline `--from-results`: **~10 s**. Live `--limit 10`: **~1 min** (10 Nebius calls + daily-bar fetch). Full 443-tweet live classification: **~15–25 min** (cached + resumable). Endpoint cold boot: **a few seconds** (manifest verify). |
| **Approx cost** | Offline path: **$0**. `--limit 10`: **≈$0** (a few 70B calls). Full corpus classification: a **few cents** of Nebius inference (cached, so paid once). Endpoint: pay-per-request, scale-to-zero. |

---

## Two operational modes (profiles)

The Endpoint and the Job both resolve a **profile** from `SIGNAL_PROFILE`
(default `stable`). A profile bundles **(prompt, whitelist, manifest)** together —
they can never be mixed, because the Endpoint verifies the manifest's
`prompt_template_hash` against the live prompt and refuses to boot on a mismatch.

| | **Mode A — Stable Indices (DEFAULT, shipped)** | **Mode B — Expanded Macro (EXPERIMENTAL)** |
|---|---|---|
| `SIGNAL_PROFILE` | `stable` | `macro` |
| Manifest | `reports/validation_manifest.json` | `reports/validation_manifest_macro_v1.json` |
| Universe | indices + sector ETFs + commodities | Mode A **+ TLT, UUP, FXI, GLD** (XLE was already in) |
| Prompt | frozen, hash `1eb55beb…` | whitelist-only, hash `f03db279…` |
| Status | **null** (EOD 50.7%, n=67, p=0.50; `shipped_horizons: []`) | **unvalidated** — do not quote its numbers as a result |

**Switching the deployed container** (Mode A is the default; you must opt in to B):
```bash
# Mode A (default) — nothing to set.
docker run -p 8080:8080 -e NEBIUS_API_KEY=$KEY predict

# Mode B — profile picks the macro prompt + whitelist + its own manifest.
docker run -p 8080:8080 -e NEBIUS_API_KEY=$KEY -e SIGNAL_PROFILE=macro predict
```
`GET /health` reports the active `profile`, `experimental` flag, and `manifest_path`.
`MANIFEST_PATH` still overrides the path (e.g. to read the Job's bucket output).

**Running the Mode-B Job** (writes ONLY to its own manifest; the baseline is never opened for writing):
```bash
PYTHONPATH=. python jobs/backtest/run_macro.py --limit 1000 --out-dir /data/reports
# local 10-tweet smoke (filter -> LLM -> router, no manifest written):
PYTHONPATH=. python jobs/backtest/run_macro.py --limit 10 --smoke
```

> **Honest status of Mode B:** it is a *variant*, not an improvement. Its corpus,
> splits and test rows differ from Mode A's, so **its numbers are not comparable to
> the baseline** — a different test set is a different measurement. Promotion requires
> winning on **validation** first, then a single registered test scoring. Until then
> Mode A remains the default and the shipped result.

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

## Event study: "but the Hormuz tweet moved oil 10%!" — tested, also null

The obvious objection to the null: high-impact posts (Iran blockades, direct
corporate mentions) *visibly* move markets, so the flat average must be hiding
localized signal. We tested exactly that with a pre-registered **event study**
([full report](experiments/event_study/REPORT.md)): outcome-blind cohorts
(GEO_SHOCK / CORPORATE / NOISE from text-only tags), market-model CAR + abnormal
volume over 1/3/5 sessions, permutation nulls, one BH pass over all 72 cells.

**Result: 0 of 72 cells survive** (min p_bh = 0.83). The famous moves are real
*individually* — our data surfaces "no deal with Iran except UNCONDITIONAL
SURRENDER" → USO −13.2% — but the population effect is zero. Three exhibits
explain *why the anecdotes feel true*:

**A. The Intel case is reverse causality.** The most-cited example — *"he tweeted
about Intel and it rose for a long time"* — runs backwards. Of **3 unique**
Intel-company mentions in 8,317 posts, the famous one reads *"**Intel Stock
continues to rise**… I am responsible"*. INTC in the **21 sessions before** that
post: **+114.7%** (SPY +9.4%) — an excess of **+105.3%**. In the 21 sessions
**after**: **−2.4%**. It had already doubled; he posted about it and claimed
credit. He *reflects* news rather than generating it — and memory encodes the
correlation as a sequence.

**B. "Fear then recovery" is mean reversion.** Oil's dip-and-rebound arc after
Iran tweets (−5.3% → +12.7% @42d) appears just as strongly — **stronger** (−3.1% →
**+16.7%**) — on big oil moves with **no tweet at all**. It's what volatility does.

**C. Saturation manufactures the anecdotes.** Trump posts geo/macro content on
~83% of trading sessions, so every large move has a same-day tweet to blame.
Sorting oil events by size surfaced *"The **Iran National Soccer Team** is welcome
to The World Cup"* (USO +17% @42d) and *"The Radical Left Democrats…"* (+45%
@42d) among the "strongest oil tweets."

At daily resolution there is no counterfactual; only intraday data could isolate
causality. The study also caught its own would-be artifact: a degenerate
permutation null initially flagged 22/72 cells before dedup + with-replacement
sampling corrected it to 0.

An interactive walkthrough of all three exhibits:
[`reports/dashboard.html`](reports/dashboard.html) (`make dashboard`).

## Future work: sector-relative alpha & single-stock benchmarking

**The primary development path, and the most likely reason this study found a null.**

Every instrument here is scored against **SPY** (`alpha/benchmark.py:170` —
`spy = fwd("SPY", t0)` is hardcoded). For a broad sector ETF that is defensible.
For everything else it is the wrong yardstick, and the data shows it:

| asset class | EOD beat-SPY rate | why it's suspect |
|---|---|---|
| volatility (VIXY) | **0.557** | VIXY beats SPY *mechanically* whenever SPY falls — anti-beta, not information |
| equity sector | 0.428 | the only class where SPY is a sensible factor |
| commodity | 0.491 | different asset class; SPY is not its risk factor |

Hit rate is also higher on SPY-**down** days (0.511) than SPY-**up** days (0.446) —
a fingerprint of negative correlation leaking in as apparent "skill."

**The fix: measure idiosyncratic alpha against a native benchmark.** If a tweet
targets **Intel**, judge INTC against its **sector** (SOXX/SMH), not the S&P:

- INTC **+2%**, SOXX **+5%** → INTC *underperformed* → **LOSS** (today: a "win" vs SPY)
- INTC **−1%**, SOXX **−6%** → INTC *protected value* → **WIN** (today: a "loss" vs SPY)

This strips sector beta and isolates the sentiment-driven, company-specific move —
the thing a tweet could plausibly cause. **The machinery already exists**:
[`labeling/benchmarks.py`](labeling/benchmarks.py) computes `abn_index` /
`abn_sector` / `abn_peer` via [`config/membership.py`](config/membership.py)
(`benchmarks_for`), built for the currently-parked micro track.

Wiring it in means replacing the hardcoded SPY leg in `alpha/benchmark.py:validate()`
with a per-instrument benchmark resolved from `benchmarks_for(ticker)`.

**Two cautions we would carry into that work:**
1. **Do not benchmark against a near-clone.** Measured on our own bars:
   ρ(XLK, QQQ)=**0.970**, ρ(SMH, SOXX)=**0.982**. Judging an ETF against a
   near-identical ETF drives abnormal return to microstructure noise. This applies
   to ETF-vs-ETF pairs — *not* to single-stock-vs-sector (ρ(INTC, SOXX) is far lower),
   which is the sound version.
2. **Non-equity assets (GLD, UUP, TLT) have no equity benchmark.** UUP *is* a dollar
   tracker, GLD *is* spot gold — benchmarking them against DXY/spot yields ≡0. The
   general solution is a **per-asset permutation null** (compare tweet-conditioned
   returns to that asset's own unconditional distribution);
   [`eval/baselines.py:25`](eval/baselines.py) already implements `permutation_null`.

**Honest expectation:** this could plausibly *rescue* signal that SPY-benchmarking
destroyed — or confirm the null more rigorously. Both are useful. It requires the
single-stock ingestion path and a fresh, once-only test scoring, which is why it is
future work rather than a deadline-day change.

## Limitations & threats to validity

- **Association, not causation.** With daily bars we measure 1-day drift, not the
  seconds-scale causal reaction. Macro confounding is documented, not deconfounded
  (N≈443 observational rows cannot identify causality).
- **Reproducibility boundary.** The daily path is fully public (yfinance + public
  tweet archives) and is where the null was measured. The 1h effect that survives BH
  needs a private Alpaca feed, so it is **excluded from `shipped_horizons`** and is
  not claimed.
- **The market data is fetched live, not from the committed CSV.**
  `alpha/benchmark.py:52` calls `yf.download()` at runtime; `data/real/bars.csv` is
  used only by the deprecated System-A modules. Reproducibility therefore depends on
  yfinance returning the same history, not on the committed file.
- **Small n.** The chronological test split is 89 tweets; we use Wilson intervals
  and exact binomial p-values (not the normal approximation) and BH-correct across
  the whole horizon registry.
- **Selection & survivorship.** Deleted tweets are gone and may be systematically
  the moved-then-retracted ones; the corpus is public archives of a public figure.
- **No per-tweet confidence.** By design. The meta-model that would produce one
  failed the sacred test; the response schema has no probability field.
