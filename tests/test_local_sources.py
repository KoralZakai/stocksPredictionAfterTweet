from datetime import datetime, timezone
from pathlib import Path

from data.sources.local import LocalPriceSource, LocalTweetSource


def _write_tweets_csv(p: Path) -> Path:
    f = p / "tweets.csv"
    f.write_text(
        "tweet_id,author,text,timestamp_utc,is_retweet,is_deleted\n"
        "2,trump,later,2024-01-02T15:00:00Z,False,False\n"
        "1,trump,earlier,2024-01-01T15:00:00Z,False,False\n"
        "9,other,skip,2024-01-01T15:00:00Z,False,False\n"
    )
    return f


def test_tweet_source_filters_sorts_tz(tmp_path: Path) -> None:
    src = LocalTweetSource(_write_tweets_csv(tmp_path))
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, tzinfo=timezone.utc)
    got = src.get_tweets(["trump"], start, end)

    assert [t.tweet_id for t in got] == ["1", "2"]  # sorted, "other" filtered out
    assert all(t.timestamp_utc.tzinfo is not None for t in got)  # tz-aware (§12)


def test_price_source(tmp_path: Path) -> None:
    f = tmp_path / "bars.csv"
    f.write_text(
        "ticker,session_date,open,high,low,close,volume\n"
        "XLK,2024-01-02T00:00:00Z,1,2,0.5,1.5,100\n"
        "XLE,2024-01-02T00:00:00Z,9,9,9,9,9\n"
    )
    bars = LocalPriceSource(f).get_daily_bars(
        "XLK", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 3, tzinfo=timezone.utc)
    )
    assert len(bars) == 1 and bars[0].close == 1.5
