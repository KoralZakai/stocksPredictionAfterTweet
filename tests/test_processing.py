from datetime import datetime, timezone

from core.processing import build_tweet_record
from data.sources.interfaces import DailyBar, Tweet

UTC = timezone.utc


def test_build_tweet_record_returns_structured_payload() -> None:
    tweet = Tweet(
        tweet_id="t1",
        author="trump",
        text="drill baby drill energy",
        timestamp_utc=datetime(2024, 2, 8, 22, 0, tzinfo=UTC),
    )
    bars = [
        DailyBar("XLE", datetime(2024, 2, 5, tzinfo=UTC), 100.0, 101.0, 99.0, 100.0, 1),
        DailyBar("XLE", datetime(2024, 2, 6, tzinfo=UTC), 100.0, 102.0, 99.0, 102.0, 1),
        DailyBar("SPY", datetime(2024, 2, 5, tzinfo=UTC), 500.0, 501.0, 499.0, 500.0, 1),
        DailyBar("SPY", datetime(2024, 2, 6, tzinfo=UTC), 500.0, 502.0, 499.0, 501.0, 1),
    ]
    record = build_tweet_record(tweet, bars, future_bars=[])

    assert record["tweet_id"] == "t1"
    assert record["entities"]["sectors"] == ["XLE"]
    assert record["targets"][0]["ticker"] == "XLE"
    assert record["targets"][0]["performance"]["month_before_pct"] != 0.0
    assert "expected_direction" in record["targets"][0]
