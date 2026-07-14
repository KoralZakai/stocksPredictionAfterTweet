"""Render reports/multibench_dashboard.html from labeled_multibench.csv.

Thin: read the labeled CSV, pick the most informative cards, call
reportgen.tweet_chart.render_page. New data flows through automatically — rebuild
the CSV, rerun this.

Card selection (descriptive, no cherry-picking of outcomes): every row with a
directional label at any horizon, plus the largest |abnormal vs indices| movers,
deduped by (entity, date), capped at --top. The point is to SHOW the method on
real tweets, not to prove a result.

Run:
  PYTHONPATH=. .venv/Scripts/python.exe scripts/build_multibench_report.py
  PYTHONPATH=. .venv/Scripts/python.exe scripts/build_multibench_report.py --top 40
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

from reportgen.tweet_chart import AXIS, render_page

CSV = "data/real/labeled_multibench.csv"
OUT = "reports/multibench_dashboard.html"


def _directional(row: pd.Series) -> bool:
    return any(str(row.get(f"label_{s}", "")) in ("UP", "DOWN") for s, _ in AXIS)


def _abs_move(row: pd.Series) -> float:
    vals = [abs(row[f"abn_idx_{s}"]) for s, _ in AXIS
            if pd.notna(row.get(f"abn_idx_{s}"))]
    return max(vals, default=0.0)


def main(top: int = 30) -> None:
    df = pd.read_csv(CSV)
    df["_dir"] = df.apply(_directional, axis=1)
    df["_mv"] = df.apply(_abs_move, axis=1)
    # directional first, then biggest abnormal movers; dedup (entity, date).
    ranked = df.sort_values(["_dir", "_mv"], ascending=[False, False])
    ranked = ranked.drop_duplicates(subset=["entity", "tweet_date"]).head(top)

    rows = ranked.to_dict("records")
    run_id = hashlib.sha256(Path(CSV).read_bytes()).hexdigest()[:12]
    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_page(rows, run_id=run_id), encoding="ascii")
    print(f"rendered {len(rows)} cards ({df['_dir'].sum()} directional in full set) "
          f"-> {OUT}  (run {run_id})")


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 30
    main(n)
