from datetime import datetime, timezone

from data.sources.interfaces import DailyBar, PriceSource
from serving.endpoint import predict

UTC = timezone.utc


class _FakePrices(PriceSource):
    def __init__(self) -> None:
        self.bars: list[DailyBar] = []
        for i, d in enumerate((5, 6, 7, 8, 9)):
            self.bars.append(DailyBar("XLE", datetime(2024, 2, d, tzinfo=UTC),
                                      100.0, 101.0, 99.0, 100.0 + i, 1))
            self.bars.append(DailyBar("SPY", datetime(2024, 2, d, tzinfo=UTC),
                                      500.0, 501.0, 499.0, 500.0 + i, 1))

    def get_daily_bars(self, ticker: str, start: datetime, end: datetime) -> list[DailyBar]:
        return [b for b in self.bars if b.ticker == ticker]


def test_predict_maps_and_abstains_with_history() -> None:
    r = predict("drill baby drill energy", "2024-02-08T22:00:00Z", _FakePrices())
    assert r["ticker"] == "XLE" and r["abstain"] is True  # Phase 0 = abstain


def test_predict_no_sector_match() -> None:
    r = predict("happy birthday my friend", "2024-02-08T22:00:00Z", _FakePrices())
    assert r["ticker"] is None and r["reason"] == "no_sector_match"


def test_predict_abstains_without_history() -> None:
    # t0 before any stored bar -> no point-in-time features -> abstain (§2).
    r = predict("energy oil", "2020-01-01T00:00:00Z", _FakePrices())
    assert r["abstain"] is True and r["reason"] == "no_history_before_t0"
