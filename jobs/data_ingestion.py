"""Nebius Job 1/N: data_ingestion (§13).

Validate raw tweets + bars through the DuckDB store (fail-fast on tz-naive /
duplicate keys) and snapshot them as the canonical artifact the rest of the DAG
reads. Thin wrapper — no feature logic here.

Run:  python jobs/data_ingestion.py --tweets data/fixtures/tweets.csv \
          --bars data/fixtures/bars.csv --out runs/mvp
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data.sources.local import LocalPriceSource, LocalTweetSource
from data.storage.store import Store

WIDE = (datetime(1990, 1, 1, tzinfo=timezone.utc), datetime(2100, 1, 1, tzinfo=timezone.utc))


def run(tweets_csv: str, bars_csv: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    authors = pd.read_csv(tweets_csv)["author"].astype(str).unique().tolist()
    tickers = pd.read_csv(bars_csv)["ticker"].astype(str).unique().tolist()
    tweets = LocalTweetSource(tweets_csv).get_tweets(authors, *WIDE)
    price = LocalPriceSource(bars_csv)
    bars = [b for tk in tickers for b in price.get_daily_bars(tk, *WIDE)]

    store = Store(str(out / "store.duckdb"))  # validates on ingest, raises on bad data
    store.ingest_tweets(tweets)
    store.ingest_bars(bars)

    shutil.copyfile(tweets_csv, out / "tweets.csv")
    shutil.copyfile(bars_csv, out / "bars.csv")
    print(f"ingested {len(tweets)} tweets, {len(bars)} bars -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tweets", required=True)
    ap.add_argument("--bars", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.tweets, a.bars, a.out)


if __name__ == "__main__":
    main()
