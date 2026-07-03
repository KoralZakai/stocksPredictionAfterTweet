from datetime import datetime, timezone

import pytest

from data.sources.interfaces import DailyBar, Tweet
from data.storage.store import Store

UTC = timezone.utc


def _tweet(tid: str, author: str = "trump", when: datetime | None = None) -> Tweet:
    return Tweet(tid, author, "x", when or datetime(2024, 1, 1, 15, tzinfo=UTC))


def test_roundtrip() -> None:
    s = Store()
    s.ingest_tweets([_tweet("1"), _tweet("2", when=datetime(2024, 1, 2, 15, tzinfo=UTC))])
    s.ingest_bars([DailyBar("XLK", datetime(2024, 1, 2, tzinfo=UTC), 1, 2, 0.5, 1.5, 100)])
    assert [t.tweet_id for t in s.load_tweets()] == ["1", "2"]
    assert s.load_bars()[0].ticker == "XLK"


def test_rejects_tz_naive() -> None:
    s = Store()
    with pytest.raises(ValueError, match="tz-naive"):
        s.ingest_tweets([_tweet("1", when=datetime(2024, 1, 1, 15))])  # naive


def test_duplicate_rolls_back() -> None:
    s = Store()
    with pytest.raises(ValueError, match="duplicate"):
        s.ingest_tweets([_tweet("1"), _tweet("2")])  # same author+timestamp
    assert s.load_tweets() == []  # atomic: nothing landed
