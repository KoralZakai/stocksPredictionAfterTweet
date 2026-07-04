from datetime import datetime, timezone

from core.calendar import TradingCalendar
from core.decide import decide, decide_batch
from core.market_state import market_state_as_of
from data.sources.interfaces import DailyBar, Tweet

UTC = timezone.utc


def _bar(day: int, close: float) -> DailyBar:
    return DailyBar("XLE", datetime(2024, 2, day, tzinfo=UTC), close, close + 1, close - 1, close, 1)


BARS = [_bar(d, 100.0 + i) for i, d in enumerate((5, 6, 7, 8, 9))]
SPY = [_bar(d, 500.0 + i) for i, d in enumerate((5, 6, 7, 8, 9))]
CAL = TradingCalendar([b.session_date.date() for b in BARS])
TWEET = Tweet("1", "trump", "energy drill", datetime(2024, 2, 8, 22, tzinfo=UTC))
STATE = market_state_as_of(TWEET.timestamp_utc, "XLE", BARS, SPY, CAL)


def test_no_train_serve_skew() -> None:
    # §3.2: single-event and batch paths produce identical features.
    single = decide(TWEET, STATE)
    batched = decide_batch([TWEET], [STATE])[0]
    assert single.features == batched.features


def test_phase0_abstains() -> None:
    d = decide(TWEET, STATE)
    assert d.abstain and d.direction == "ABSTAIN"


class _StubPredictor:
    def predict_direction(self, features: dict[str, float]) -> tuple[str, float, bool]:
        return "UP", 0.8, False


def test_decide_routes_through_predictor() -> None:
    # A predictor's verdict flows out, but features are still the one path's.
    d = decide(TWEET, STATE, _StubPredictor())
    assert d.direction == "UP" and d.confidence == 0.8 and d.abstain is False
    assert d.features == decide(TWEET, STATE).features  # features independent of predictor
