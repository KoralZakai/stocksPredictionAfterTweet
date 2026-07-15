# experiments/context/ — System B: macro cross-referencing (EXPERIMENTAL, NOT shipped)

> Tests whether injecting **leak-safe, point-in-time public macro context** into a
> **dual-horizon** classifier (1h sentiment shock + EOD structural trend) adds real
> beat-SPY edge over the shipped tweet-only System A. **Nothing here touches the
> frozen shipped prompt, manifest, or serving path.**

## Integrity design

1. **No fabrication, no leakage.** Context = a real dated public calendar
   (`macro_calendar.py`: FOMC/CPI dates). `context_asof(t0)` returns only events
   **strictly before t0**. Enforced by `tests/test_context_leak.py`. We inject the
   *existence* of dated events, never their outcome or any hindsight sentiment.
2. **Sacred test frozen.** Comparison runs on **val** by default. Scoring `test`
   requires an explicit `--i-understand-test-is-sacred` flag and is the ONE final
   registered run — not used to pick between A and B.
3. **Shipped manifest untouched.** Outputs go to `manifest_b.json` /
   `results_b.json` here, never `reports/validation_manifest.json`.
4. **Judge is qualitative + confounded.** The LLM-judge scores rationale plausibility,
   not correctness, and `macro_alignment` is biased toward B (B was given the context).
   Only the beat-SPY metric is evidence of edge.
5. **EOD is reproducible; 1h is Alpaca-gated** (captured, not scored here).

## Run

```bash
# 1. System-B contextual backtest on val (live Nebius + daily bars):
python experiments/context/run_context.py --split val
# 2. Offline reasoning-quality judge on the produced rationales:
python experiments/context/judge.py
# 3. Comparative A-vs-B markdown report (-> REPORT.md):
python experiments/context/report.py
```

Cheap plumbing smoke (few rows, ~$0):
```bash
python experiments/context/run_context.py --split train --limit 3
```

## Pre-registered expectation

Per the meta-model precedent (Val 0.593 -> Test 0.431), adding structure to ~266
train rows tends not to generalize. A null (System B ≈ or < System A on val
beat-SPY) is a valid, honest outcome and the likely one.
