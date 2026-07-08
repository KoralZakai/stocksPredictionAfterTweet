"""First results table: strongest +/- abnormal moves + random samples (§ Step B).

Illustrative, NOT signal — strongest-move rows are the tails of a noisy
distribution by construction. Run after build_labeled_dataset.py.

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/results_table.py
"""

from __future__ import annotations

from typing import Any

import pandas as pd

LAB = "data/real/labeled.csv"


def show(df: pd.DataFrame, title: str) -> None:
    print(f"\n--- {title} ---")
    print(f"{'date':11}{'sector':7}{'asset':6}{'abn5d':>8}{'raw5d':>8}{'spy5d':>8} {'lab5':<8} text")
    for _, r in df.iterrows():
        def pct(x: Any) -> str:  # noqa: ANN401 — pandas cell
            return f"{float(x)*100:+.2f}" if pd.notna(x) else "  n/a"
        print(f"{str(r['timestamp_utc'])[:10]:11}{r['sector']:7}{r['asset']:6}"
              f"{pct(r['abn_5']):>8}{pct(r['raw_5']):>8}{pct(r['spy_5']):>8} "
              f"{str(r['lab_5']):<8} {str(r['text'])[:48]}")


def main() -> None:
    df = pd.read_csv(LAB)
    d = df[(df["is_spy"] == 0) & df["abn_5"].notna()].copy()
    show(d.nlargest(10, "abn_5"), "10 STRONGEST POSITIVE (abnormal 5d)")
    show(d.nsmallest(10, "abn_5"), "10 STRONGEST NEGATIVE (abnormal 5d)")
    show(d.sample(10, random_state=0), "10 RANDOM SAMPLES")


if __name__ == "__main__":
    main()
