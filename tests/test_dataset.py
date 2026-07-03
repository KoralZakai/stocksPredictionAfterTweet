from datetime import datetime, timezone

from data.sources.interfaces import DailyBar, PriceSource, Tweet
from dataset.build import build_dataset

UTC = timezone.utc


class _FakePrices(PriceSource):
    def __init__(self) -> None:
        self.bars: list[DailyBar] = []
        for i, d in enumerate((5, 6, 7, 8, 9, 12, 13)):
            self.bars.append(DailyBar("XLE", datetime(2024, 2, d, tzinfo=UTC),
                                      100.0, 101.0, 99.0, 100.0 + i, 1))
            self.bars.append(DailyBar("SPY", datetime(2024, 2, d, tzinfo=UTC),
                                      500.0, 501.0, 499.0, 500.0 + i, 1))

    def get_daily_bars(self, ticker: str, start: datetime, end: datetime) -> list[DailyBar]:
        return [b for b in self.bars if b.ticker == ticker]


def test_build_dataset_joins_features_and_labels() -> None:
    tweets = [
        Tweet("1", "trump", "drill baby drill energy", datetime(2024, 2, 6, 22, tzinfo=UTC)),
        Tweet("2", "trump", "happy birthday my friend", datetime(2024, 2, 6, 22, tzinfo=UTC)),
    ]
    rows = build_dataset(tweets, _FakePrices())

    assert len(rows) == 1  # tweet 2 maps to NONE -> excluded
    r = rows[0]
    assert r.ticker == "XLE"
    assert set(r.label) == {1, 2, 3}
    assert "topic_XLE" in r.features and r.features["topic_XLE"] == 1.0
