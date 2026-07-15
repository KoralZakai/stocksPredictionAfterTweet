# System A vs System B — contextual dual-horizon (EXPERIMENTAL)

**Split: `val` (validation). The sacred test stays frozen — no model was
selected on it.** EOD beat-SPY, daily bars (public/reproducible). 1h is
Alpaca-gated and not scored here.

| Metric (EOD, val) | System A (shipped, tweet-only) | System B (macro-context, dual-horizon) |
|---|---|---|
| n scored | 88 | 74 |
| Beat-SPY accuracy | 0.6477 | 0.6486 |
| Wilson 95% CI | [0.544, 0.739] | [0.535, 0.748] |
| p_raw (H1: >50%) | 0.0037 | 0.007 |
| p_BH (adj over A,B) | 0.0070 | 0.0070 |
| Judge avg (1-5, qualitative) | macro_alignment:1.932 / causal_logic:3.739 / risk_awareness:1.273 | macro_alignment:2.443 / causal_logic:3.386 / risk_awareness:1.341 |

## Read this before quoting anything

- **Validation, not test.** This compares approaches on val. The shipped EOD
  test number (61.8%) is untouched; System B would only replace the shipped prompt
  after a *single* registered test scoring that beats A under BH — not done here.
- **Judge is confounded.** System B was handed the macro context and A was not, so
  B scores higher on `macro_alignment` by construction. The judge measures prose
  plausibility, NOT predictive correctness. Only the beat-SPY row is evidence of edge.
- **Prior:** distilling/optimizing extra structure onto ~266 train rows has repeatedly
  failed to generalize (the meta-model went Val 0.593 -> Test 0.431). Expect System B
  to look comparable-or-worse once the noise is accounted for; a null here is a valid,
  honest result.
