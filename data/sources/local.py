"""Local file adapters for TweetSource / PriceSource (§2: offline only).

Reads parquet or CSV. Timestamps are normalized to tz-aware UTC on the way in
(§12) — a CSV string or a tz-naive parquet column becomes UTC here, so nothing
downstream ever sees a naive timestamp.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from data.sources.interfaces import DailyBar, PriceSource, Tweet, TweetSource


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported file (want .parquet/.csv): {path}")


class LocalTweetSource(TweetSource):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def get_tweets(self, authors: list[str], start: datetime, end: datetime) -> list[Tweet]:
        df = _read(self.path)
        ts = pd.to_datetime(df["timestamp_utc"], utc=True)
        keep = df["author"].isin(authors) & (ts >= start) & (ts < end)
        out = [
            Tweet(
                tweet_id=str(r.tweet_id),
                author=str(r.author),
                text=str(r.text),
                timestamp_utc=t.to_pydatetime(),
                is_retweet=bool(getattr(r, "is_retweet", False)),
                is_deleted=bool(getattr(r, "is_deleted", False)),
            )
            for r, t in zip(df[keep].itertuples(index=False), ts[keep], strict=True)
        ]
        out.sort(key=lambda x: x.timestamp_utc)
        return out


def load_corpus(
    path: str | Path,
    start: datetime,
    end: datetime,
    *,
    originals_only: bool = True,
    exclude_reblogs: bool = True,
    platforms: tuple[str, ...] | None = None,
) -> list[Tweet]:
    """Load the unified event corpus (§ data strategy) as Tweet objects.

    Maps the platform-agnostic schema onto the pipeline's Tweet: post_id->tweet_id,
    is_reblog->is_retweet, author constant 'trump'. The training-time filters
    (originals-only, reblog/quote exclusion, platform, date range) live here so
    the choice is one reproducible call, not scattered downstream.
    """
    df = _read(Path(path))
    ts = pd.to_datetime(df["timestamp_utc"], utc=True)
    keep = (ts >= start) & (ts < end)
    if originals_only:
        keep &= df["is_original"].astype(str) == "True"
    if exclude_reblogs:
        keep &= df["is_reblog"].astype(str) != "True"
    if platforms is not None:
        keep &= df["platform"].isin(platforms)
    # DATASEARCH §1 dedup: the same post ingested from two archives (same id, or
    # identical timestamp+text under different ids) must yield ONE event row.
    sub = df[keep].assign(ts_utc=ts[keep])
    sub = sub.drop_duplicates(subset=["post_id"]).drop_duplicates(subset=["ts_utc", "text"])
    out = [
        Tweet(
            tweet_id=str(r.post_id),
            author="trump",
            text=str(r.text),
            timestamp_utc=r.ts_utc.to_pydatetime(),
            is_retweet=str(r.is_reblog) == "True",
            is_deleted=False,
            platform=str(r.platform),
            source=str(getattr(r, "source_dataset", "")),
        )
        for r in sub.itertuples(index=False)
    ]
    out.sort(key=lambda x: x.timestamp_utc)
    return out


class LocalPriceSource(PriceSource):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def get_daily_bars(self, ticker: str, start: datetime, end: datetime) -> list[DailyBar]:
        df = _read(self.path)
        sd = pd.to_datetime(df["session_date"], utc=True)
        keep = (df["ticker"] == ticker) & (sd >= start) & (sd <= end)
        out = [
            DailyBar(
                ticker=str(r.ticker),
                session_date=d.to_pydatetime(),
                open=float(r.open),
                high=float(r.high),
                low=float(r.low),
                close=float(r.close),
                volume=int(r.volume),
            )
            for r, d in zip(df[keep].itertuples(index=False), sd[keep], strict=True)
        ]
        out.sort(key=lambda x: x.session_date)
        return out
