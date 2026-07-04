"""Nebius Job: dataset_build (§13) — subsumes labeling + feature_engineering.

Reads the ingested snapshot, runs build_dataset() (labels + point-in-time
features via the single decide() path), writes the labeled dataset artifact.

Run:  python jobs/dataset_build.py --in runs/mvp --out runs/mvp/dataset.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data.sources.local import LocalPriceSource, LocalTweetSource
from dataset.build import build_dataset, rows_to_json

WIDE = (datetime(1990, 1, 1, tzinfo=timezone.utc), datetime(2100, 1, 1, tzinfo=timezone.utc))


def run(in_dir: str, out_path: str) -> None:
    src = Path(in_dir)
    authors = pd.read_csv(src / "tweets.csv")["author"].astype(str).unique().tolist()
    tweets = LocalTweetSource(src / "tweets.csv").get_tweets(authors, *WIDE)
    rows = build_dataset(tweets, LocalPriceSource(src / "bars.csv"))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rows_to_json(rows), encoding="utf-8")
    print(f"built {len(rows)} labeled rows -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.in_dir, a.out)


if __name__ == "__main__":
    main()
