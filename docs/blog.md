# Building a Null-Result Machine: Testing Whether Political Tweets Move Sector ETFs, on Nebius Serverless

*#NebiusServerlessChallenge — code: https://github.com/KoralZakai/stocksPredictionAfterTweet*

## The question, and the trap

"Do Donald Trump's tweets move the stock market?" is the kind of question that
generates confident answers and terrible science. Financial time series are
noisy, non-stationary, and riddled with ways to fool yourself. Point a flexible
model at enough tweets and price bars and it *will* find a pattern — one that
evaporates the moment you test it honestly.

So I inverted the goal. This project is not a trading system and does not try to
predict prices. It is a **null-result machine**: a pipeline engineered to answer
one question rigorously —

> Do political tweets carry statistically significant short-term directional
> information about sector-ETF returns, under strict causal constraints?

— and to say **"no"** when the honest answer is no. A rigorous null result is
the *success* condition, not a failure. Everything is built on Nebius Serverless
AI Jobs (the batch research pipeline) and a Serverless AI Endpoint (inference).

## What we can and cannot measure

The data is **daily OHLCV** for ten sector ETFs (`XLK, XLE, XLF, ...`). With
daily bars you cannot see the true causal reaction to a tweet — that is priced
in seconds. What you *can* measure is a 1–3 trading-day **drift-association**
between a tweet and its mapped sector. Naming this scope honestly, up front, is
half the battle: it stops the project from overclaiming before a line of model
code exists.

## The architecture

The serverless layer is deliberately **thin**. It contains zero science; it only
marshals data in and out of pure functions. That separation is what makes the
system trustworthy.

```
Jobs (batch):  data_ingestion → dataset_build → evaluation/reporting
Endpoint:      POST /predict  ── same decide() ──┘
```

- **`data_ingestion`** validates raw tweets and bars through a DuckDB store that
  fails fast on timezone-naive timestamps or duplicate `(author, timestamp)`
  keys, then snapshots a canonical artifact.
- **`dataset_build`** produces the labeled dataset: point-in-time features
  joined to volatility-scaled labels.
- **`evaluation`** runs the statistical harness and emits the signal-or-null
  report.
- The **`/predict` endpoint** wraps the *exact same* decision function the batch
  pipeline uses.

One CPU-only Docker image runs every job and the endpoint, so batch and serve
can never drift apart.

## The five things that stop it fooling itself

**1. Point-in-time correctness.** Every feature for a tweet at time `t0` uses
only data whose trading session *closed strictly before* `t0`. A test injects
future bars and asserts the features do not move a single decimal. Leakage is
the number-one way financial ML lies; this is the guardrail.

**2. No train/serve skew.** There is exactly one pure function,
`decide(tweet, market_state)`, used by the batch jobs, the endpoint, and any
future replay harness. A test runs a sample through the batch path and the
single-event path and asserts byte-identical features. No second feature path
means no silent divergence between what you evaluated and what you serve.

**3. Leak-free labels.** The entry price is the *open of the first session that
begins strictly after the tweet* — so an off-hours, weekend, or holiday tweet
resolves cleanly to the next session with no look-ahead. Returns run forward
from there; the UP/DOWN/NEUTRAL band is scaled by *backward-only* volatility.

**4. Non-i.i.d.-aware evaluation.** Overlapping 2- and 3-day return windows
correlate nearby rows, so a random K-fold leaks by construction. Instead we use
**purged, embargoed walk-forward** cross-validation: training data is strictly
earlier than the test fold and separated by an embargo gap.

**5. Correct inference.** Every result is compared against three baselines —
always-majority, market-follow (momentum), and a **permutation null** that
shuffles labels to break any tweet↔outcome link. Significance requires both an
effect size and a permutation p-value, and *nothing* is called significant
until it survives **Benjamini–Hochberg** correction across the full registry of
tests. A **power/MDE gate** runs first and will declare the study underpowered
by construction — which, at small sample sizes, it usually is.

## The result

Run the pipeline on the current fixture and the report ends with:

```
HEADLINE: 0 of 3 cells survive BH correction (alpha=0.05).
          A null result here is a valid, expected outcome.
```

That line is the whole point. The system did not find a signal, and it *told the
truth about it* — after passing point-in-time checks, purged CV, permutation
testing, and multiple-comparison correction. A pipeline that can produce a
trustworthy "no" is far more valuable than one that always finds a "yes."

## Why serverless fits research, not just production

Reproducibility is a first-class scientific requirement, and it maps perfectly
onto Nebius Serverless AI Jobs: each stage is a deterministic, containerized,
independently runnable job with hashed inputs and a stamped run id. Another
researcher clones the repo, runs three job commands, and gets the identical
table. The endpoint demonstrates the same `decide()` live — and, honestly,
**abstains** when it has no history before `t0`, because a well-calibrated
abstention is a correct answer, not a cop-out.

## What's next

The structured-feature XGBoost model and conformal abstention slot into the
existing `decide()` seam without touching the evaluation harness — and the
deliberately-overfit transformer "canary" exists to confirm that purged CV,
permutation nulls, and BH correction correctly *refuse* to reward it. If the eval
ever crowns the canary, the eval is broken, not the model.

Until then, the pipeline stands as a small manifesto: build the machine that can
say no.
