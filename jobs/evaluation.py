"""Nebius Job: evaluation + reporting (§13).

Reads the labeled dataset artifact, runs the signal-or-null report (baselines,
permutation null, BH correction, power gate), writes the report as text + JSON.

Run:  python jobs/evaluation.py --dataset runs/mvp/dataset.json --out runs/mvp
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dataset.build import rows_from_json
from eval.report import format_report, run_report


def run(dataset_path: str, out_dir: str) -> None:
    rows = rows_from_json(Path(dataset_path).read_text(encoding="utf-8"))
    report, registry = run_report(rows)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = f"Dataset: {len(rows)} rows | registry: {len(registry)} tests\n\n" + format_report(report)
    (out / "report.txt").write_text(text, encoding="utf-8")
    (out / "report.json").write_text(
        json.dumps([asdict(r) for r in report], indent=2, default=str), encoding="utf-8"
    )
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.dataset, a.out)


if __name__ == "__main__":
    main()
