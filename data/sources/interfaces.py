"""TweetSource + PriceSource: the only two ways data enters the pipeline.

Both are read-only, offline, local-file-backed in V1 (§2 non-goals: no
live polling). Concrete adapters (parquet/CSV) come after these are agreed.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Tweet:
    tweet_id: str
    author: str
    text: str
    timestamp_utc: datetime  # tz-aware; §9 asserts precision is sufficient to place vs session
    is_retweet: bool = False
    is_deleted: bool = False  # observed deleted in a later crawl; survivorship flag, §9
    platform: str = "twitter"  # unified-corpus metadata (twitter|truth_social); NOT a model branch
    source: str = ""  # provenance: which archive/dataset the post came from (§9 audit)


@dataclass(frozen=True)
class DailyBar:
    ticker: str
    session_date: datetime  # tz-aware, market-local session date (no intraday, §2)
    open: float
    high: float
    low: float
    close: float
    volume: int


class TweetSource(abc.ABC):
    """Read-only tweet access. No live polling in V1 (§2)."""

    @abc.abstractmethod
    def get_tweets(self, authors: list[str], start: datetime, end: datetime) -> list[Tweet]:
        """Tweets in [start, end), tz-aware UTC timestamps, sorted by timestamp_utc ascending."""


class PriceSource(abc.ABC):
    """Read-only daily OHLCV access. Daily bars only, no intraday (§2)."""

    @abc.abstractmethod
    def get_daily_bars(self, ticker: str, start: datetime, end: datetime) -> list[DailyBar]:
        """Daily bars in [start, end], one row per regular session, sorted by session_date."""
