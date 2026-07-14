"""FastAPI serving for the distilled student. EXPERIMENTAL — see README.

The student produces the SAME structured classification schema as the 70B teacher,
so it reuses the shipped decision-plane router (`alpha.route.route_decision`) and
JSON parser — one code path, no bespoke logic.

INTEGRITY: the cohort hit-rates in validation_manifest.json belong to the 70B
TEACHER. This endpoint attaches them labelled `source_model: teacher` and
`student_validated: false`. It NEVER presents the teacher's numbers as the
student's. Re-validate the student on the sacred test before removing that flag.

Heavy deps (torch/transformers) load lazily at startup so the module imports on a
CPU-only box for linting/inspection.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from alpha.classify import _INSTRUCT, _SYSTEM, _parse_json
from alpha.route import route_decision

MANIFEST_PATH = os.environ.get("MANIFEST_PATH", "reports/validation_manifest.json")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "experiments/distill/adapter")
DISCLAIMER = "Research output. Not investment advice. Distilled student model — EXPERIMENTAL."

app = FastAPI(title="distilled-student /predict (experimental)")
_state: dict[str, Any] = {"model": None, "tokenizer": None, "manifest": None}


class PredictIn(BaseModel):
    tweet_text: str
    t0_utc: str = ""


def _load_manifest() -> dict[str, Any]:
    p = Path(MANIFEST_PATH)
    if not p.exists():
        raise RuntimeError(f"manifest not found: {MANIFEST_PATH}")
    data: dict[str, Any] = json.loads(p.read_text())
    return data


@app.on_event("startup")
def _startup() -> None:
    _state["manifest"] = _load_manifest()
    # Lazy heavy import: only needed to actually generate.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained(ADAPTER_DIR if Path(ADAPTER_DIR).exists() else base)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto")
    if Path(ADAPTER_DIR).exists():
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    _state["tokenizer"], _state["model"] = tok, model


def _student_classify(text: str) -> dict[str, Any]:
    """Generate the structured classification with the fine-tuned student."""
    import torch
    tok, model = _state["tokenizer"], _state["model"]
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"{_INSTRUCT}\n\nPOST:\n{text}"},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                             temperature=0.0, pad_token_id=tok.eos_token_id)
    completion = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse_json(completion)


def _cohort(manifest: dict[str, Any]) -> dict[str, Any] | None:
    shipped = manifest.get("shipped_horizons") or []
    if not shipped:
        return None
    h = shipped[0]
    e = manifest["horizons"][h]
    return {
        "value": e["hit_rate_test"], "ci95": e["ci95"], "n": e["n_test"], "horizon": h,
        "source_model": "teacher (Llama-3.3-70B zero-shot)",
        "student_validated": False,
        "note": ("Teacher's historical hit-rate on the held-out chronological test set. "
                 "The distilled STUDENT has NOT been independently validated; re-run the "
                 "sacred test on the student before trusting this number for it."),
    }


@app.post("/predict")
def predict(req: PredictIn) -> dict[str, Any]:
    manifest = _state["manifest"] or {}
    classified = _student_classify(req.tweet_text)      # DECISION PLANE: tweet text only
    routed = route_decision(classified)
    resp = asdict(routed)
    resp["instruments"] = [asdict(i) for i in routed.instruments]
    resp["horizon"] = (manifest.get("shipped_horizons") or [None])[0]
    resp["cohort_base_rate"] = _cohort(manifest)
    resp["manifest_version"] = manifest.get("code_rev")
    resp["model"] = "distilled-student (experimental)"
    resp["disclaimer"] = DISCLAIMER
    return resp


@app.get("/health")
def health() -> dict[str, Any]:
    manifest = _state["manifest"] or {}
    return {
        "status": "ok" if _state["model"] is not None else "loading",
        "model": "distilled-student (experimental, unvalidated)",
        "manifest_version": manifest.get("code_rev"),
        "shipped_horizons": manifest.get("shipped_horizons"),
        "warning": "Student model not independently validated on the sacred test.",
    }
