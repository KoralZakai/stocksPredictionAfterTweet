# archive/ — rejected experiments, kept for the record

Nothing here is on the shipped path. These files are retained so the negative
results are reproducible and auditable, not deleted and forgotten.

## `train_dual_engine.py` — the REJECTED meta-model

**What it was:** a second-stage model that tried to predict *which* raw-LLM calls
would land — i.e. a per-tweet confidence score on top of the zero-shot Nebius
classification.

**Why it's here and not in `jobs/`:** it was evaluated and **rejected on the
sacred chronological test split, three times.** The text variant is the clearest:

| Stage | AUC |
|---|---|
| Validation | 0.593 |
| **Test** | **0.431** |

Test AUC **below 0.5** means the meta-model generalized *worse than a coin flip*
— it overfit the validation fold and carried no real signal to the held-out
future. Adding model capacity (more features, fine-tuning) only widens this gap:
the train split is ~266 rows with ~28 minority-class examples.

**The conclusion we ship instead:** the **raw zero-shot LLM call** has real
relative alpha (N=443, Beat-SPY band=0: 1h 64.4%, EOD 58.2%, p<0.001, edge
front-loaded, chronological splits stable). So the product is the raw call plus a
**cohort base rate** (the historical hit-rate of calls of this type) — never a
per-tweet probability. See `reports/final_production_blueprint.md`.

This is a **valid, expected null result** for the second stage, and it is the
honest core of the submission's story.

## Note on derived CSVs

The `data/real/*.csv` and `reports/*.csv` intermediate variants were **not**
moved here: every one is still referenced by some script, so relocating them
would break the pipeline under the frozen submission scope. The canonical inputs
(`data/real/corpus_v3.csv`, `data/real/bars.csv`) and the shipped dataset
(`reports/macro_dataset.csv`) are documented in `data/PROVENANCE.md`.
