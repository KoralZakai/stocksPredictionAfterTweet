"""Spot-check eden's LLM signal extractor (llm/extract.py) on a small real-tweet
sample before scaling to the full ~2,357-post corpus. Uses Haiku by default (set
LLM_MODEL to override). Reads texts straight from data/real/stock_event_dataset.csv
(bypasses jobs/llm_features.py, which targets the older fixture-shaped runs/ dir).

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/llm_sample_check.py [N]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from llm.extract import default_extractor

EVENTS = "data/real/stock_event_dataset.csv"
OUT = "reports/llm_sample_check.csv"
SEED = 7


def _load_dotenv(path: str = ".env") -> None:
    """Minimal stdlib .env loader (same pattern as scripts/fetch_alpaca_30m.py)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_load_dotenv()
os.environ.setdefault("LLM_MODEL", "claude-haiku-4-5")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    df = pd.read_csv(EVENTS)
    posts = df.drop_duplicates("post_id")[["post_id", "text"]]
    sample = posts.sample(min(n, len(posts)), random_state=SEED)

    extractor = default_extractor()
    print(f"extractor: {type(extractor).__name__}  model={os.environ.get('LLM_MODEL')}  n={len(sample)}")

    rows = []
    for _, r in sample.iterrows():
        sig = extractor.extract(str(r["text"]))
        rows.append({"post_id": r["post_id"], "text": str(r["text"])[:120], **sig.model_dump()})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    with pd.option_context("display.max_colwidth", 60, "display.width", 220):
        print(out.to_string(index=False))
    print(f"\nwrote {OUT} ({len(out)} rows)")
    print("\nevent_type distribution:")
    print(out["event_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
