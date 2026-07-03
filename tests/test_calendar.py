from datetime import date, datetime, timezone

from core.calendar import TradingCalendar

UTC = timezone.utc
# Mon-Fri Feb 5-9 2024, then skip weekend + Presidents' Day (Feb 19), resume Feb 20.
SESSIONS = [date(2024, 2, d) for d in (5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 20, 21)]
CAL = TradingCalendar(SESSIONS)


def test_s0_before_open_is_same_session() -> None:
    # 08:00 ET = 13:00 UTC, before the 09:30 ET open -> s0 is that same session.
    assert CAL.resolve_s0(datetime(2024, 2, 5, 13, 0, tzinfo=UTC)) == date(2024, 2, 5)


def test_s0_intraday_rolls_to_next() -> None:
    # 11:00 ET = 16:00 UTC, after open -> next session.
    assert CAL.resolve_s0(datetime(2024, 2, 5, 16, 0, tzinfo=UTC)) == date(2024, 2, 6)


def test_s0_after_close_rolls_to_next() -> None:
    assert CAL.resolve_s0(datetime(2024, 2, 7, 22, 0, tzinfo=UTC)) == date(2024, 2, 8)


def test_s0_weekend_rolls_to_monday() -> None:
    assert CAL.resolve_s0(datetime(2024, 2, 10, 15, 0, tzinfo=UTC)) == date(2024, 2, 12)


def test_s0_holiday_rolls_over_gap() -> None:
    # Feb 19 (Presidents' Day) is absent -> resolves to Feb 20.
    assert CAL.resolve_s0(datetime(2024, 2, 19, 17, 0, tzinfo=UTC)) == date(2024, 2, 20)


def test_s0_past_data_is_none() -> None:
    assert CAL.resolve_s0(datetime(2024, 3, 1, tzinfo=UTC)) is None


def test_offset_spans_holiday_gap() -> None:
    assert CAL.session_at_offset(date(2024, 2, 16), 1) == date(2024, 2, 20)
    assert CAL.session_at_offset(date(2024, 2, 21), 1) is None
