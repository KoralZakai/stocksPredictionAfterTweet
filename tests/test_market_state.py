from datetime import datetime, timezone

from core.calendar import TradingCalendar
from core.features import build_features
from core.market_state import market_state_as_of
from data.sources.interfaces import DailyBar, Tweet

UTC = timezone.utc


def _bar(day: int, close: float) -> DailyBar:
    return DailyBar("XLE", datetime(2024, 2, day, tzinfo=UTC), close, close + 1, close - 1, close, 1)


BARS = [_bar(d, 100.0 + i) for i, d in enumerate((5, 6, 7, 8, 9, 12, 13, 14))]
SPY = [_bar(d, 500.0 + i) for i, d in enumerate((5, 6, 7, 8, 9, 12, 13, 14))]
CAL = TradingCalendar([b.session_date.date() for b in BARS])
TWEET = Tweet("1", "trump", "energy drill", datetime(2024, 2, 9, 16, tzinfo=UTC))  # intraday Feb 9


def test_state_holds_only_bars_closed_before_t0() -> None:
    # §3.1: every bar in the point-in-time view closed strictly before t0.
    t0 = TWEET.timestamp_utc
    state = market_state_as_of(t0, "XLE", BARS, SPY, CAL)
    assert all(CAL.close_utc(b.session_date.date()) < t0 for b in state.prior_bars)
    # Feb 9 session (t0 is intraday Feb 9) has NOT closed -> excluded.
    assert all(b.session_date.date().day < 9 for b in state.prior_bars)


def test_future_injection_does_not_change_features() -> None:
    # The point-in-time canary: adding bars dated >= t0 must not move any feature.
    t0 = TWEET.timestamp_utc
    base = build_features(TWEET, market_state_as_of(t0, "XLE", BARS, SPY, CAL))

    future = BARS + [_bar(20, 999.0), _bar(21, 1234.0)]  # both after t0
    cal2 = TradingCalendar([b.session_date.date() for b in future])
    poisoned = build_features(TWEET, market_state_as_of(t0, "XLE", future, SPY, cal2))

    assert base == poisoned
