"""Comparative A-vs-B report on the VALIDATION split. EXPERIMENTAL.

Reads manifest_b.json (+ judge_scores.json if present) and emits a Markdown table.
Everything here is on VAL — the sacred test stays frozen for the final number. The
judge column is a reasoning-quality diagnostic with a stated confound, never a
substitute for the beat-SPY metric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alpha.stats import benjamini_hochberg

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest_b.json"
JUDGE = HERE / "judge_scores.json"
OUT = HERE / "REPORT.md"


def _ci(agg: dict[str, Any]) -> str:
    lo, hi = agg.get("ci95", [0.0, 1.0])
    return f"[{lo:.3f}, {hi:.3f}]"


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} not found — run run_context.py first.")
    m = json.loads(MANIFEST.read_text())
    a, b = m["system_a"], m["system_b"]
    split = m.get("split", "val")

    # BH over the two registered EOD tests (A, B), so neither p is read raw.
    p_adj = benjamini_hochberg([a.get("p_raw", 1.0), b.get("p_raw", 1.0)])

    judge_line_a = judge_line_b = "n/a (judge not run)"
    if JUDGE.exists():
        js = json.loads(JUDGE.read_text())["summary"]
        judge_line_a = " / ".join(f"{k}:{v}" for k, v in js["system_a_avg"].items())
        judge_line_b = " / ".join(f"{k}:{v}" for k, v in js["system_b_avg"].items())

    md = f"""# System A vs System B — contextual dual-horizon (EXPERIMENTAL)

**Split: `{split}` (validation). The sacred test stays frozen — no model was
selected on it.** EOD beat-SPY, daily bars (public/reproducible). 1h is
Alpaca-gated and not scored here.

| Metric (EOD, {split}) | System A (shipped, tweet-only) | System B (macro-context, dual-horizon) |
|---|---|---|
| n scored | {a['n']} | {b['n']} |
| Beat-SPY accuracy | {a['accuracy']} | {b['accuracy']} |
| Wilson 95% CI | {_ci(a)} | {_ci(b)} |
| p_raw (H1: >50%) | {a['p_raw']} | {b['p_raw']} |
| p_BH (adj over A,B) | {p_adj[0]:.4f} | {p_adj[1]:.4f} |
| Judge avg (1-5, qualitative) | {judge_line_a} | {judge_line_b} |

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
"""
    OUT.write_text(md, encoding="utf-8")
    print(md)
    print(f"[report] -> {OUT}")


if __name__ == "__main__":
    main()
