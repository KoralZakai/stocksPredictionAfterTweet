"""Trading-day calendar (§5 core/calendar.py).

Owns market-local session logic: given the ordered session dates (which the
price data already encodes — a holiday is simply a missing bar), it computes
each session's open/close as tz-aware UTC and resolves

    s0 = first regular session whose OPEN is strictly after t0   (§3.3)

so off-hours / weekend / holiday tweets resolve cleanly to the next session.
DST is handled by zoneinfo; half-days get an early close.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
HALF_DAY_CLOSE = time(13, 0)


def _et_to_utc(d: date, t: time) -> datetime:
    return datetime.combine(d, t, tzinfo=ET).astimezone(timezone.utc)


class TradingCalendar:
    def __init__(self, session_dates: Iterable[date], half_days: Iterable[date] = ()) -> None:
        self._dates: list[date] = sorted(set(session_dates))
        self._half: frozenset[date] = frozenset(half_days)
        self._idx: dict[date, int] = {d: i for i, d in enumerate(self._dates)}

    @property
    def dates(self) -> list[date]:
        return list(self._dates)

    def open_utc(self, d: date) -> datetime:
        return _et_to_utc(d, REGULAR_OPEN)

    def close_utc(self, d: date) -> datetime:
        return _et_to_utc(d, HALF_DAY_CLOSE if d in self._half else REGULAR_CLOSE)

    def resolve_s0(self, t0: datetime) -> date | None:
        """First session whose open is strictly after t0, or None if past the data."""
        # ponytail: linear scan; bisect on open_utc if the session list ever gets large.
        for d in self._dates:
            if self.open_utc(d) > t0:
                return d
        return None

    def session_at_offset(self, d: date, n: int) -> date | None:
        i = self._idx[d] + n
        return self._dates[i] if 0 <= i < len(self._dates) else None
