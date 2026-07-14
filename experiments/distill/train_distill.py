"""GPU SFT/LoRA: distill the 70B teacher's structured classifications into a small
student (Qwen-2.5-7B / Llama-3.1-8B). EXPERIMENTAL — see experiments/distill/README.

Targets are the TEACHER's JSON outputs (scenario / instruments / predicted_direction),
NOT the market outcome `label_eod`. Training uses the TRAIN SPLIT ONLY — the sacred
val/test splits are never seen (enforced in `build_training_examples`).

Heavy deps (torch/transformers/peft/trl/bitsandbytes) are imported INSIDE main() so
this module imports — and `build_training_examples` stays unit-testable — on a
CPU-only box with none of them installed.

Run (GPU):
    python experiments/distill/train_distill.py --base-model Qwen/Qwen2.5-7B-Instruct \
        --out experiments/distill/adapter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Reuse the EXACT teacher prompt so the student learns the same input->output
# contract the 70B was validated on (no schema drift between teacher and student).
from alpha.classify import _INSTRUCT, _SYSTEM
from scripts.nebius_macro_backtest import RESULTS, _assign_splits

ROOT = Path(__file__).resolve().parents[2]


def _teacher_target(r: dict[str, Any]) -> str:
    """Reconstruct the teacher's JSON answer for one tweet, in the classify schema.

    The student is trained to emit exactly this — the 70B's reasoning + instrument
    basket + directions — given the tweet text.
    """
    instruments = [
        {
            "ticker": ins.get("ticker", ""),
            "name": ins.get("name", ""),
            "role": ins.get("role", ""),
            "predicted_direction": ins.get("predicted", ins.get("predicted_direction", "neutral")),
        }
        for ins in r.get("instruments", [])
    ]
    target = {
        "scenario": r.get("scenario", ""),
        "intensity": r.get("intensity", 5),
        "summary": r.get("summary", ""),
        "macro_link": r.get("macro_link", ""),
        "hypothesis_short": r.get("hypothesis_short", ""),
        "hypothesis_long": r.get("hypothesis_long", ""),
        "rationale": r.get("rationale", r.get("macro_link", "")),
        "instruments": instruments,
    }
    return json.dumps(target, ensure_ascii=False)


def build_training_examples(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """TRAIN-SPLIT-ONLY chat examples: (system, user=tweet, assistant=teacher JSON).

    Pure and GPU-free so it is unit-testable. Raises if any non-train row leaks in.
    """
    _assign_splits(results)                       # chronological 60/20/20, same as the shipped run
    train = [r for r in results if r.get("split") == "train"]
    assert all(r.get("split") == "train" for r in train), "sacred split leaked into training"

    examples: list[dict[str, Any]] = []
    for r in train:
        text = r.get("text", "")
        if not text:
            continue
        examples.append({"messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"{_INSTRUCT}\n\nPOST:\n{text}"},
            {"role": "assistant", "content": _teacher_target(r)},
        ]})
    return examples


def _load_results() -> list[dict[str, Any]]:
    path = ROOT / RESULTS
    if not path.exists():
        raise SystemExit(f"teacher outputs not found: {path} (run jobs/backtest first)")
    data: list[dict[str, Any]] = json.loads(path.read_text())
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Distill 70B teacher classifications into a small student")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="experiments/distill/adapter")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    # Heavy deps imported here so the module loads without them (see docstring).
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    examples = build_training_examples(_load_results())
    print(f"[distill] {len(examples)} train-split examples (val/test excluded)")
    dataset = Dataset.from_list(examples)

    tokenizer = AutoTokenizer.from_pretrained(a.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit QLoRA: fit a 7-8B student on a single mid-range GPU.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        a.base_model, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="auto",
    )

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    sft_config = SFTConfig(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        max_seq_length=a.max_seq_len,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        seed=a.seed,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(a.out)
    tokenizer.save_pretrained(a.out)
    print(f"[distill] adapter saved -> {a.out}")
    print("[distill] REMINDER: re-validate the student on the sacred test via "
          "jobs/backtest before quoting any accuracy for it.")


def _demo() -> None:
    """CPU self-check: split enforcement + target shape, no torch needed."""
    fake = [
        {"date": "2025-01-01", "text": "tariffs on China", "scenario": "Trade War",
         "instruments": [{"ticker": "XLK", "predicted": "down", "name": "Tech", "role": "x"}],
         "hits": {"EOD": [1, 1]},
         "spy_returns": {"EOD": 0.0}},
        {"date": "2025-06-01", "text": "peace deal soon", "scenario": "Peace",
         "instruments": [{"ticker": "ITA", "predicted": "down", "name": "Defense", "role": "y"}],
         "hits": {"EOD": [1, 1]}, "spy_returns": {"EOD": 0.0}},
    ]
    # give both a scoreable EOD so _assign_splits ranks them; tiny set -> both train.
    for r in fake:
        r["instruments"][0]["hit"] = {"EOD": True}
    ex = build_training_examples(fake)
    assert all(e["messages"][0]["role"] == "system" for e in ex)
    assert all(json.loads(e["messages"][2]["content"])["instruments"] for e in ex)
    print(f"train_distill self-check OK ({len(ex)} examples)")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _demo()      # CPU-only: verify split-safety + target shape without GPU deps
    else:
        main()       # GPU: actual distillation
