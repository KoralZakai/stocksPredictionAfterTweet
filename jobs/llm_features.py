"""Nebius Job: llm_features (product fork, §13 shape) — OFFLINE signal extraction.

Runs the LLM over every ingested tweet ONCE and writes a signal cache. Thin CLI:
reads tweets, calls the pure llm/ modules, writes an artifact. Deterministic and
resumable — an existing cache is loaded and only fresh/changed tweets are
(re)extracted, so a rerun after adding tweets is cheap and a crash loses nothing.

The endpoint and dataset_build later READ this cache; neither calls the LLM.

Run:  python jobs/llm_features.py --in runs/mvp --out runs/mvp/llm_signals.json
      LLM_MODEL=claude-haiku-4-5 ANTHROPIC_API_KEY=... python jobs/llm_features.py ...
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data.sources.local import LocalTweetSource
from llm.cache import SignalCache
from llm.extract import DEFAULT_MODEL, default_extractor

WIDE = (datetime(1990, 1, 1, tzinfo=timezone.utc), datetime(2100, 1, 1, tzinfo=timezone.utc))


def run(in_dir: str, out_path: str, model: str | None = None) -> None:
    import os

    src = Path(in_dir)
    authors = pd.read_csv(src / "tweets.csv")["author"].astype(str).unique().tolist()
    tweets = LocalTweetSource(src / "tweets.csv").get_tweets(authors, *WIDE)

    model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    extractor = default_extractor(model)
    cache = SignalCache(out_path)

    fresh = 0
    for tw in tweets:
        if cache.get(tw.tweet_id, tw.text, model) is not None:
            continue  # unchanged -> reuse, don't pay for it again
        cache.put(tw.tweet_id, tw.text, model, extractor.extract(tw.text))
        fresh += 1
    cache.save()
    print(f"extracted {fresh} new / {len(cache)} total signals ({model}) -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    run(a.in_dir, a.out, a.model)


if __name__ == "__main__":
    main()
