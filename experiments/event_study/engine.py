"""Classic event-study math on OFFLINE daily bars (data/real/bars.csv). EXPERIMENTAL.

Market model: r_asset = alpha + beta * r_SPY, OLS-fit on an ESTIMATION WINDOW of
120 trading sessions ending 6 sessions BEFORE the event session (backward-only —
CLAUDE.md 3.1). Abnormal return AR_t = r_t - (alpha + beta * r_spy_t);
CAR_w = sum(AR) over the first w event sessions. Abnormal volume = event-window
mean volume / estimation-window mean volume.

Entry anchor: s0 = first session whose OPEN is strictly after t0 (same invariant
as alpha/benchmark.daily_returns; US_OPEN_UTC_HOUR reused). Day-1 return runs
open(s0)->close(s0) so nothing before the tweet leaks in; later days close->close.

Overlap guard (CLAUDE.md 3.6): an event whose s0 is within `window` sessions of
the SAME asset's previous event is flagged overlapping=True and reported apart.
"""

from __future__ import annotations

import csv
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from alpha.benchmark import us_open_utc_hour

BARS_CSV = Path("data/real/bars.csv")
EST_LEN = 120       # estimation window, trading sessions
EST_GAP = 6         # sessions between estimation end and the event (no bleed)


@dataclass(frozen=True)
class Bar:
    date: str        # YYYY-MM-DD
    open: float
    close: float
    volume: float


def load_bars(path: Path = BARS_CSV) -> dict[str, list[Bar]]:
    """{ticker: bars sorted by date}. Offline, reproducible — no network."""
    csv.field_size_limit(10**9)
    out: dict[str, list[Bar]] = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out.setdefault(r["ticker"], []).append(
                    Bar(r["session_date"][:10], float(r["open"]), float(r["close"]),
                        float(r["volume"] or 0)))
            except ValueError:
                continue
    for bars in out.values():
        bars.sort(key=lambda b: b.date)
    return out


def s0_index(bars: list[Bar], t0: datetime) -> int | None:
    """Index of the first session whose open is strictly after t0."""
    d0 = t0.date().isoformat()
    i = bisect_left([b.date for b in bars], d0)
    if i < len(bars) and bars[i].date == d0:
        t0h = t0.hour + t0.minute / 60.0
        if us_open_utc_hour(t0.date()) > t0h:      # that date's open is still ahead of t0
            return i
        i += 1
    return i if i < len(bars) else None


def _returns(bars: list[Bar], lo: int, hi: int) -> list[float]:
    """Close-to-close daily returns for sessions (lo, hi] — needs bars[lo-1]."""
    return [bars[j].close / bars[j - 1].close - 1.0 for j in range(lo, hi)]


@dataclass(frozen=True)
class EventResult:
    ticker: str
    t0: str
    s0_date: str
    car: dict[int, float]            # window -> CAR
    abn_volume: dict[int, float]     # window -> event/estimation volume ratio
    beta: float
    alpha: float
    overlapping: bool


def study_event(bars: dict[str, list[Bar]], ticker: str, t0: datetime,
                windows: tuple[int, ...] = (1, 3, 5),
                prev_s0: dict[str, int] | None = None) -> EventResult | None:
    """CAR + abnormal volume for one (asset, t0). None if data insufficient."""
    a, m = bars.get(ticker.upper()), bars.get("SPY")
    if not a or not m:
        return None
    i0 = s0_index(a, t0)
    if i0 is None or i0 + max(windows) > len(a):
        return None
    est_hi = i0 - EST_GAP
    est_lo = est_hi - EST_LEN
    if est_lo < 1:
        return None

    # Align SPY to the asset's estimation dates (skip unmatched sessions).
    m_by_date = {b.date: j for j, b in enumerate(m)}
    ra: list[float] = []
    rm: list[float] = []
    vols: list[float] = []
    for j in range(est_lo, est_hi):
        k = m_by_date.get(a[j].date)
        if k is None or k == 0:
            continue
        ra.append(a[j].close / a[j - 1].close - 1.0)
        rm.append(m[k].close / m[k - 1].close - 1.0)
        vols.append(a[j].volume)
    if len(ra) < 60:                                   # too sparse to fit
        return None
    n = len(ra)
    mx, my = sum(rm) / n, sum(ra) / n
    if ticker.upper() == "SPY":
        # The market has no market model vs itself (beta=1 -> AR degenerates to 0).
        # Standard fallback: Brown-Warner MEAN-ADJUSTED returns, AR = r - mean(est).
        beta, alpha = 0.0, my
    else:
        vxx = sum((x - mx) ** 2 for x in rm)
        beta = (sum((x - mx) * (y - my) for x, y in zip(rm, ra)) / vxx) if vxx else 0.0
        alpha = my - beta * mx
    est_vol = (sum(vols) / len(vols)) if vols else 0.0

    # Event window: day 1 = open(s0)->close(s0); later days close->close.
    car: dict[int, float] = {}
    abn_volume: dict[int, float] = {}
    cum = 0.0
    ev_vols: list[float] = []
    for step in range(max(windows)):
        j = i0 + step
        k = m_by_date.get(a[j].date)
        if step == 0:
            r_a = a[j].close / a[j].open - 1.0
            r_m = (m[k].close / m[k].open - 1.0) if k is not None else 0.0
        else:
            r_a = a[j].close / a[j - 1].close - 1.0
            r_m = (m[k].close / m[k - 1].close - 1.0) if k not in (None, 0) else 0.0
        cum += r_a - (alpha + beta * r_m)
        ev_vols.append(a[j].volume)
        w = step + 1
        if w in windows:
            car[w] = cum
            abn_volume[w] = (sum(ev_vols) / len(ev_vols) / est_vol) if est_vol else 0.0

    overlapping = False
    if prev_s0 is not None:
        last = prev_s0.get(ticker.upper())
        overlapping = last is not None and (i0 - last) < max(windows)
        prev_s0[ticker.upper()] = i0
    return EventResult(ticker.upper(), t0.isoformat(), a[i0].date, car, abn_volume,
                       round(beta, 4), round(alpha, 6), overlapping)
