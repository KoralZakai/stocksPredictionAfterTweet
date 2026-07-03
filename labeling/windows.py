"""Entry anchor + drift returns (§3.3-3.4, §5 labeling/windows.py).

entry = open(s0), s0 = first session open strictly after t0. Returns are
cumulative close/entry-1 at horizons 1/2/3 sessions. Every horizon's close
timestamp is asserted strictly after t0 — the point-in-time guard for labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

from core.calendar import TradingCalendar
from data.sources.interfaces import DailyBar


@dataclass(frozen=True)
class Outcome:
    s0_date: date
    entry: float
    entry_ts: datetime
    ret: dict[int, float | None]  # horizon (1/2/3) -> return, None if not enough forward bars
    close_ts: dict[int, datetime | None]


def compute_outcome(
    t0: datetime,
    bars: Sequence[DailyBar],
    cal: TradingCalendar,
    horizons: Sequence[int] = (1, 2, 3),
) -> Outcome | None:
    """`bars` = sorted daily bars for ONE ticker, aligned with `cal`."""
    s0 = cal.resolve_s0(t0)
    if s0 is None:
        return None
    by_date = {b.session_date.date(): b for b in bars}
    if s0 not in by_date:
        return None
    entry = by_date[s0].open
    entry_ts = cal.open_utc(s0)
    ret: dict[int, float | None] = {}
    close_ts: dict[int, datetime | None] = {}
    for h in horizons:
        d = cal.session_at_offset(s0, h - 1)
        if d is None or d not in by_date:
            ret[h] = None
            close_ts[h] = None
            continue
        c_ts = cal.close_utc(d)
        assert c_ts > t0, f"horizon {h} close {c_ts} not strictly after t0 {t0}"  # §3.4
        ret[h] = by_date[d].close / entry - 1.0
        close_ts[h] = c_ts
    return Outcome(s0, entry, entry_ts, ret, close_ts)
