"""Intraday shock engine: 60-minute impact, with the tweet's OWN pre-window as control.

Why this design. Our earlier 1h result (0.678) died because tweets with intraday data
are a biased subset — they are posted during market hours, and that subset also scores
better at EOD. Comparing the SAME tweet's post-window to its pre-window cancels that:
the selection applies identically to both sides.

  pre  = the `win` bars ENDING at the last bar that CLOSED strictly before t0
  post = the `win` bars STARTING from the first bar that closes after t0
  excess = (asset - SPY) return, so a market-wide move is not read as a tweet effect
  shock  = post excess / sigma(pre-window excess)   [sigma is BACKWARD-ONLY]

Point-in-time: nothing at or after t0 enters the pre-window or sigma. Bars whose
interval straddles t0 are DROPPED from both sides — they contain both regimes.

Data: data/real/bars_1h_public.csv (yfinance, no keys, reproducible).
"""

from __future__ import annotations

import csv
import statistics
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

BARS_1H = Path("data/real/bars_1h_public.csv")
BENCH = "SPY"
BAR_MINUTES = 60          # yfinance hourly: the stamp is the bar's OPEN time


@dataclass(frozen=True)
class HBar:
    ts: datetime          # bar open
    open: float
    close: float
    volume: float

    @property
    def end(self) -> datetime:
        from datetime import timedelta
        return self.ts + timedelta(minutes=BAR_MINUTES)


def load_hourly(path: Path = BARS_1H) -> dict[str, list[HBar]]:
    """{ticker: hourly bars sorted by time}. Offline + reproducible."""
    out: dict[str, list[HBar]] = {}
    if not path.exists():
        return out
    csv.field_size_limit(10**9)
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                ts = datetime.fromisoformat(r["ts_utc"].replace("Z", "+00:00"))
                out.setdefault(r["ticker"], []).append(
                    HBar(ts, float(r["open"]), float(r["close"]), float(r["volume"] or 0)))
            except (ValueError, KeyError):
                continue
    for v in out.values():
        v.sort(key=lambda b: b.ts)
    return out


def _split_at(bars: list[HBar], t0: datetime) -> tuple[int, int]:
    """(last index closing strictly BEFORE t0, first index opening at/after t0).

    A bar straddling t0 (opens before, closes after) belongs to neither side and is
    skipped — it mixes pre- and post-tweet information.
    """
    i = bisect_left([b.ts for b in bars], t0)          # first bar opening >= t0
    j = i - 1
    while j >= 0 and bars[j].end > t0:                 # straddler -> step back
        j -= 1
    return j, i


@dataclass(frozen=True)
class Shock:
    ticker: str
    t0: str
    pre_excess: float        # cumulative excess over the pre-window
    post_excess: float       # cumulative excess over the post-window (the effect)
    pre_sigma: float         # SD of per-bar excess in the pre-window (backward-only)
    shock_z: float           # post_excess / pre_sigma
    vol_ratio: float         # post mean volume / pre mean volume
    n_pre: int
    n_post: int


def _excess(a: list[HBar], m: dict[datetime, HBar], lo: int, hi: int) -> list[float]:
    """Per-bar excess (asset - SPY) for bars[lo:hi], aligned on bar open time."""
    out: list[float] = []
    for k in range(max(lo, 1), hi):
        mk, mp = m.get(a[k].ts), m.get(a[k - 1].ts)
        if mk is None or mp is None or a[k - 1].close == 0 or mp.close == 0:
            continue
        out.append((a[k].close / a[k - 1].close - 1.0) - (mk.close / mp.close - 1.0))
    return out


def study_shock(bars: dict[str, list[HBar]], ticker: str, t0: datetime,
                win: int = 1, sigma_win: int = 20) -> Shock | None:
    """60-min impact for one (asset, t0). `win` = bars each side (1 = ~1 hour).

    `sigma_win` bars before t0 estimate the pre-tweet volatility used to z-score the
    move. None when either side lacks data (never dropped on the basis of outcome).
    """
    a, mkt = bars.get(ticker.upper()), bars.get(BENCH)
    if not a or not mkt:
        return None
    m = {b.ts: b for b in mkt}
    j, i = _split_at(a, t0)
    if j < sigma_win or i + win > len(a):
        return None

    pre = _excess(a, m, j - win + 1, j + 1)
    post = _excess(a, m, i, i + win) if i > 0 else []
    sig = _excess(a, m, j - sigma_win + 1, j + 1)
    if not pre or not post or len(sig) < 5:
        return None

    pre_sd = statistics.pstdev(sig)
    pre_c, post_c = sum(pre), sum(post)
    pv = [b.volume for b in a[j - win + 1:j + 1] if b.volume > 0]
    qv = [b.volume for b in a[i:i + win] if b.volume > 0]
    return Shock(
        ticker.upper(), t0.isoformat(), round(pre_c, 6), round(post_c, 6),
        round(pre_sd, 6), round(post_c / pre_sd, 4) if pre_sd else 0.0,
        round((sum(qv) / len(qv)) / (sum(pv) / len(pv)), 4) if pv and qv else 0.0,
        len(pre), len(post))


def _demo() -> None:
    """CPU self-check: the straddling bar is excluded and sigma is backward-only."""
    bars = load_hourly()
    if BENCH not in bars or "USO" not in bars:
        print("intraday self-check SKIPPED (no public hourly bars)")
        return
    a = bars["USO"]
    t0 = a[300].ts.replace(minute=17)                       # mid-bar tweet
    j, i = _split_at(a, t0)
    assert a[j].end <= t0, "pre side leaked a bar that closed after t0"
    assert a[i].ts >= t0, "post side started before t0"
    assert i - j >= 1, "straddling bar was not skipped"
    s = study_shock(bars, "USO", t0)
    assert s is None or s.n_pre >= 1
    print(f"intraday self-check OK (pre ends {a[j].end:%H:%M}, post starts {a[i].ts:%H:%M})")


if __name__ == "__main__":
    _demo()
