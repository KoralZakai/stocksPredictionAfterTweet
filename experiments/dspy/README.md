# experiments/dspy/ — DSPy prompt optimization (EXPERIMENTAL, NOT shipped)

> **Status: experimental. Does NOT touch the frozen shipped prompt or manifest.**
> The shipped classifier is the raw 70B call in `alpha/classify.py`, pinned by
> `reports/validation_manifest.json` (`prompt_template_hash`). Nothing here changes
> that unless a candidate beats the shipped EOD number on the sacred test under BH.

## The honest hypothesis

Optimizing a prompt to maximize the beat-SPY hit rate is the **rejected per-tweet
meta-model, expressed in prompt-space** instead of weight-space. That meta-model
failed the sacred test three times (text Val AUC 0.593 → **Test 0.431**). So the
pre-registered expectation for this experiment is:

> Train/val hit-rate improves; **test does not**. Confirming that is a valid,
> reportable null — it shows the edge lives in the raw 70B read, not in a tunable
> prompt at this data scale (266 train rows).

We run it to *demonstrate* that boundary rigorously, not to chase a test number.

## Guardrails (enforced in `train_dspy.py`)

1. **Train split only** feeds the optimizer (266 rows). `val` (88) is iteration
   feedback. `test` (89) is the sacred split — scored **once**, last.
2. **Isolated deps** — `requirements.txt` here, never the root `pyproject.toml`.
3. **Shared metric** — reuses `alpha.benchmark.relative_hit` (beat-SPY, signed);
   no bespoke scoring.
4. **Replace shipped only on a test win** — and then only by regenerating the
   manifest through `jobs/backtest`, so the prompt hash stays honest.

## Layout (per the dspy skill)

```
experiments/dspy/
  train_dspy.py     # the single experiment script (run_name selects the iteration)
  requirements.txt  # dspy + Nebius LM deps, isolated
  results.tsv       # one row per run: metrics + token/cost/runtime accounting
  runs/             # per-run artifacts (created at run time)
  programs/         # compiled DSPy programs (created at run time)
```

## Run (needs `dspy` installed + `NEBIUS_API_KEY`)

```bash
pip install -r experiments/dspy/requirements.txt
# CPU self-check (no LLM, verifies split-safety only):
python experiments/dspy/train_dspy.py --selfcheck
# Baseline iteration (spends Nebius budget):
python experiments/dspy/train_dspy.py --run-name lfs-baseline-iter-01 --optimizer labeled_fewshot
```
