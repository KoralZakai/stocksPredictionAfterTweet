"""Guardrail for the EXPERIMENTAL distill track: the student may NEVER train on the
sacred val/test splits. This test fails loudly if that invariant ever breaks.

Imports only the pure `build_training_examples` (torch is imported inside main(),
not at module load), so it runs on the CPU dev box.
"""

from __future__ import annotations

import json

from experiments.distill.train_distill import build_training_examples


def _row(date: str, direction: str = "up") -> dict:
    return {"date": date, "text": f"tweet {date}", "scenario": "Trade",
            "instruments": [{"ticker": "XLK", "predicted": direction, "name": "T", "role": "r",
                             "hit": {"EOD": True}}],
            "hits": {"EOD": [1, 1]}, "spy_returns": {"EOD": 0.0}}


def test_only_train_split_becomes_examples() -> None:
    # 10 chronological rows -> 60/20/20 => 6 train / 2 val / 2 test.
    rows = [_row(f"2025-01-{i:02d}") for i in range(1, 11)]
    examples = build_training_examples(rows)
    train_texts = {r["text"] for r in rows if r["split"] == "train"}
    used_texts = {e["messages"][1]["content"].split("POST:\n")[-1] for e in examples}
    assert used_texts == train_texts
    # the val/test rows (the 4 most recent) must be absent from training.
    excluded = {r["text"] for r in rows if r["split"] in ("val", "test")}
    assert not (used_texts & excluded)


def test_teacher_target_is_valid_json_with_directions() -> None:
    examples = build_training_examples([_row(f"2025-02-{i:02d}") for i in range(1, 11)])
    for e in examples:
        target = json.loads(e["messages"][2]["content"])
        assert target["instruments"]
        assert target["instruments"][0]["predicted_direction"] in ("up", "down", "neutral")
