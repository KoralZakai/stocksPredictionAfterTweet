from datetime import date, datetime, timezone

from core.calendar import TradingCalendar
from data.sources.interfaces import DailyBar
from labeling.thresholds import backward_vol, class_balance, label
from labeling.windows import compute_outcome

UTC = timezone.utc


def _bar(d: int, close: float, op: float | None = None) -> DailyBar:
    return DailyBar("XLE", datetime(2024, 2, d, tzinfo=UTC), op or close, close + 1, close - 1, close, 1)


BARS = [_bar(d, 100.0 + i, op=100.0) for i, d in enumerate((5, 6, 7, 8, 9))]
CAL = TradingCalendar([b.session_date.date() for b in BARS])


def test_entry_is_open_of_s0_and_returns_reference_it() -> None:
    # after-close Feb 5 -> s0 = Feb 6, entry = open(Feb6) = 100.
    out = compute_outcome(datetime(2024, 2, 5, 22, tzinfo=UTC), BARS, CAL)
    assert out is not None
    assert out.s0_date == date(2024, 2, 6) and out.entry == 100.0
    assert out.ret[1] == 101.0 / 100.0 - 1.0  # close(Feb6)=101


def test_every_close_ts_strictly_after_t0() -> None:
    t0 = datetime(2024, 2, 5, 22, tzinfo=UTC)
    out = compute_outcome(t0, BARS, CAL)
    assert out is not None
    assert all(ts > t0 for ts in out.close_ts.values() if ts is not None)  # §3.4


def test_insufficient_forward_bars_gives_na() -> None:
    # tweet near the end -> ret_3d has no bar.
    out = compute_outcome(datetime(2024, 2, 8, 22, tzinfo=UTC), BARS, CAL)
    assert out is not None and out.ret[3] is None


def test_backward_vol_is_backward_only() -> None:
    # vol before Feb 9 must not see Feb 9's bar; needs >=3 prior closes.
    assert backward_vol(BARS, date(2024, 2, 9), window=20) is not None
    assert backward_vol(BARS, date(2024, 2, 6), window=20) is None  # only 1 prior close


def test_label_band() -> None:
    assert label(0.02, 0.01, k=0.5) == "UP"      # 0.02 > 0.005
    assert label(-0.02, 0.01, k=0.5) == "DOWN"
    assert label(0.001, 0.01, k=0.5) == "NEUTRAL"
    assert label(None, 0.01, k=0.5) == "NA"


def test_class_balance_counts() -> None:
    assert class_balance(["UP", "UP", "DOWN"]) == {"UP": 2, "DOWN": 1, "NEUTRAL": 0, "NA": 0}
