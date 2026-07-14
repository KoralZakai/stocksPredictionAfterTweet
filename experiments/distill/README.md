# experiments/distill/ — knowledge distillation (EXPERIMENTAL, NOT the shipped path)

> **Status: experimental extra track. This is NOT what the submission ships.**
> The shipped, validated pipeline is the raw **Llama-3.3-70B zero-shot call**
> (`serving/app.py`), whose numbers are certified in
> `reports/validation_manifest.json`. Nothing here changes that.

## What this is

A **response-based knowledge distillation**: fine-tune a small open model
(Qwen-2.5-7B-Instruct or Llama-3.1-8B-Instruct) to **mimic the 70B teacher's
structured classification** — `scenario`, `instruments`, `predicted_direction` —
NOT the noisy market outcome (`label_eod`).

This sidesteps the meta-model that was rejected 3× on the sacred test (which
predicted `label_eod`, text Val AUC 0.593 → **Test 0.431**). The student's target
is the teacher's *reasoning*, not the market.

## What it deliberately does NOT do (integrity guardrails)

1. **Train split only.** The student SFTs on the teacher's outputs for the
   **train-split tweets only** (chronological 60/20/20). The val/test splits are
   never seen in training. `train_distill.py` enforces this in code.
2. **No borrowed credibility.** The 64.4%/58.2% cohort rates in
   `validation_manifest.json` belong to the **70B teacher**. They do NOT
   automatically describe the 7B/8B student. `serve_distilled.py` therefore labels
   the cohort rate as the *teacher's historical rate, student pending independent
   validation* — it never presents the teacher's numbers as the student's.
3. **Re-validation required before any claim.** To honestly quote a number for the
   student, regenerate predictions with the student model and re-run the sacred
   test via the existing `jobs/backtest` path. Until then the student endpoint is
   a **compression/cost demo**, not a validated predictor.
4. **Isolated deps.** GPU/training deps live in `requirements.txt` here, never in
   the root `pyproject.toml` / `uv.lock`. The shipped CPU image stays lean and
   reproducible.

## Why distill at all

Cost/latency: a 7–8B student is far cheaper to serve than the 70B teacher. IF a
re-validation shows the student retains the edge on the sacred test, you get the
same signal at a fraction of the inference cost. IF it doesn't (likely, at ~266
train rows), that is itself an honest, reportable result — distillation did not
preserve the signal at this data scale.

## Files

| File | Role |
|---|---|
| `train_distill.py` | GPU SFT/LoRA: teacher JSON → student. Train split only. |
| `requirements.txt` | Training/serving deps (torch, transformers, peft, trl, bitsandbytes). |
| `Dockerfile.train` | CUDA image for the Nebius GPU **Job**. |
| `serve_distilled.py` | FastAPI serving the fine-tuned student (+ honest cohort caveat). |
| `Dockerfile.serve` | CUDA-runtime image for the Nebius **Endpoint**. |
| `nebius_job.yaml` | Serverless GPU training-job manifest. |
| `nebius_endpoint.yaml` | Serverless endpoint manifest. |

## Run (GPU required — untested on this CPU-only dev box)

```bash
# 1. Train the student on the teacher's train-split classifications:
python experiments/distill/train_distill.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --out experiments/distill/adapter

# 2. Serve it (still cites teacher rates, student marked unvalidated):
uvicorn experiments.distill.serve_distilled:app --host 0.0.0.0 --port 8080
```
