---
name: dspy
description: Run disciplined DSPy prompt-optimization experiments as an experiment engineer — pick an optimizer, start from a LabeledFewShot baseline, change one thing per iteration, log compile/eval tokens+cost+runtime to a local results.tsv, and NEVER tune on a locked test split. Adapted for this repo (stocksPredictionAfterTweet) from github.com/SerjSmor/skills. Use when optimizing a classification/extraction prompt with DSPy.
---

# DSPy experiment engineer (repo-adapted)

Act as an experiment engineer for DSPy prompt optimization. Source methodology:
[SerjSmor/skills](https://github.com/SerjSmor/skills) (`dspy`). This copy hardwires
**this repo's integrity guardrails** — read them first; they override convenience.

## PROJECT GUARDRAILS (non-negotiable in this repo)

This repo ships a validated pipeline whose numbers are pinned in
`reports/validation_manifest.json` via `prompt_template_hash` (the exact 70B
classification prompt) + corpus sha256. The serving endpoint refuses to boot on
prompt drift. Therefore:

1. **The shipped prompt is FROZEN.** DSPy work happens ONLY under
   `experiments/dspy/`. It never edits `alpha/classify.py`, the manifest, or any
   shipped code. A candidate prompt replaces the shipped one *only after* it beats
   the shipped EOD number on the sacred test under BH correction (it likely won't —
   that is a valid finding).
2. **Train split ONLY for optimization.** The optimizer sees `split=="train"`
   (266 rows) exclusively. `val` (88) is for iteration feedback; `test` (89) is the
   sacred split — touched **once**, at the very end, for the final number. Any run
   that lets the optimizer see val/test is invalid and must be discarded.
3. **Beware the meta-model trap.** Optimizing a prompt to maximize the beat-SPY
   metric is the rejected per-tweet meta-model in prompt-space (it failed the sacred
   test 3×: text Val AUC 0.593 -> Test 0.431). Expect train/val to improve and test
   to stay flat. Report that honestly; do NOT chase test performance.
4. **Isolated deps.** `dspy` lives in `experiments/dspy/requirements.txt`, never the
   root `pyproject.toml` / `uv.lock`. The shipped CPU image stays lean.

## Workflow

1. **Interview briefly** before writing code: task type, optimization target
   (metric), Nebius budget (calls/$), model choice, and the eval protocol
   (confirm train-only optimize + test-once). Skip questions already answered.
2. **Start conservative.** First iteration is a `LabeledFewShot` (or
   `BootstrapFewShot`) baseline. Only advance to heavier optimizers (MIPROv2, etc.)
   once the baseline is logged.
3. **One bounded change per iteration.** Change exactly one of: optimizer, budget,
   signature, instructions, demo selection, or model. Encode the iteration in
   `run_name` (e.g. `lfs-baseline-iter-01`) — do not fork script copies.
4. **Keep outputs local** to `experiments/dspy/`: `train_dspy.py`, `results.tsv`,
   `runs/`, `programs/`.
5. **Log comprehensively** to `results.tsv`: run_name, optimizer, model, split
   sizes, compile+eval token counts, cost, runtime, train/val metric (and test only
   on the final registered run). W&B optional.
6. **Validate before presenting**: script runs, outputs land in the right place, a
   row was appended to `results.tsv`, and the token/cost accounting is complete.

## Metric

Reuse the shipped engine, do not reinvent it: `alpha.benchmark.relative_hit`
(beat-SPY, signed) scored against **cached** returns in
`reports/nebius_backtest_results.json` when the candidate predicts the same
instruments, else a fresh daily-bar fetch (`alpha.benchmark.forward_returns`,
public, free). The signed train label already lives in `macro_dataset.csv`
(`label_eod`).

## References
- Optimizer choice + compile/cost accounting: see the source repo's
  `references/optimizer-choice.md` and `references/compile-accounting.md`.
