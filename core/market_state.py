"""Point-in-time market view (§3.1, §5 core/market_state.py).

market_state_as_of(t0) exposes ONLY data knowable before t0: a daily bar
enters the view once its session has CLOSED strictly before t0. A tweet fired
intraday therefore cannot see the in-progress session's bar — only completed
history. The label side (entry = open(s0), s0 open > t0) lives in labeling/ and
never overlaps this set.

ponytail: this is a backward as-of restricted to the pre-t0 window, done in
pure Python on purpose. §3.2 forbids a second feature path, so the SAME code
serves the batch runner and the (future) per-event endpoint — no DuckDB-vs-
Python skew. DuckDB ASOF JOIN stays for bulk table ops, not per-decision state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from core.calendar import TradingCalendar
from data.sources.interfaces import DailyBar


@dataclass(frozen=True)
class MarketState:
    ticker: str
    asof: datetime
    prior_bars: tuple[DailyBar, ...]  # ticker bars closed strictly before t0
    benchmark_prior: tuple[DailyBar, ...]  # benchmark (SPY) bars closed strictly before t0


def _closed_before(bars: Sequence[DailyBar], t0: datetime, cal: TradingCalendar) -> tuple[DailyBar, ...]:
    return tuple(
        b
        for b in sorted(bars, key=lambda x: x.session_date)
        if cal.close_utc(b.session_date.date()) < t0
    )


def market_state_as_of(
    t0: datetime,
    ticker: str,
    ticker_bars: Sequence[DailyBar],
    benchmark_bars: Sequence[DailyBar],
    cal: TradingCalendar,
) -> MarketState:
    return MarketState(
        ticker=ticker,
        asof=t0,
        prior_bars=_closed_before(ticker_bars, t0, cal),
        benchmark_prior=_closed_before(benchmark_bars, t0, cal),
    )
