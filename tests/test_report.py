from datetime import datetime, timezone

from data.sources.interfaces import DailyBar, PriceSource, Tweet
from dataset.build import build_dataset
from eval.report import run_report

UTC = timezone.utc


class _FakePrices(PriceSource):
    def __init__(self) -> None:
        self.bars: list[DailyBar] = []
        days = list(range(1, 28))
        for i, d in enumerate(days):
            if datetime(2024, 2, d, tzinfo=UTC).weekday() >= 5:
                continue
            self.bars.append(DailyBar("XLE", datetime(2024, 2, d, tzinfo=UTC),
                                      100.0, 101.0, 99.0, 100.0 + (i % 5), 1))
            self.bars.append(DailyBar("SPY", datetime(2024, 2, d, tzinfo=UTC),
                                      500.0, 501.0, 499.0, 500.0 + (i % 3), 1))

    def get_daily_bars(self, ticker: str, start: datetime, end: datetime) -> list[DailyBar]:
        return [b for b in self.bars if b.ticker == ticker]


def test_report_runs_and_registers() -> None:
    tweets = [
        Tweet(str(i), "trump", "energy oil drill", datetime(2024, 2, 5 + i, 22, tzinfo=UTC))
        for i in range(5)
    ]
    rows = build_dataset(tweets, _FakePrices())
    report, registry = run_report(rows, n_perm=100)
    assert len(registry) == len(report) >= 1
    for r in report:
        assert 0.0 <= r.p_value <= 1.0 and 0.0 <= r.q_value <= 1.0
