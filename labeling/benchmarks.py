"""Multi-benchmark abnormal return + the 'positive collection' label (v-multibench).

The core scientific object of the reframed methodology. For a tweet at t0 and a
stock, at every horizon h we ask three questions, each against the AVERAGE of the
relevant benchmarks (never one fixed benchmark — see config/membership.py):

  abn_index  = stock_ret - mean(return of every index the stock belongs to)
  abn_sector = stock_ret - mean(return of the stock's sector ETF(s))
  abn_peer   = stock_ret - median(return of the stock's sector-sibling stocks)

"Outperformed" means the stock beat ALL of index, sector, AND peers by more than
its own noise band (band = k·σ_backward·√h, σ backward-only). The label then
folds in the LLM stance:

  UP      = stance positive AND outperformed on all three
  DOWN    = stance negative AND underperformed on all three
  NEUTRAL = anything else (didn't clear the bar, or beat some-but-not-all)

Peer-beat is REQUIRED (a per-user decision): a stock that rises with its whole
sector is a macro move, not a tweet effect, so it must beat its siblings too.
This makes the label sparse on purpose — callers MUST emit the class-balance
diagnostic (labeling.thresholds.class_balance) and treat an all-NEUTRAL split as
a labeling finding, not a silent pass (§3.5).

Every return here flows through labeling.windows.compute_outcome, so the
point-in-time guard (every close strictly after t0) holds for the stock and for
every benchmark identically — no second price path.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from config.membership import BenchmarkSet, benchmarks_for
from config.settings import SETTINGS
from core.calendar import TradingCalendar
from data.sources.interfaces import DailyBar
from labeling.thresholds import backward_vol
from labeling.windows import compute_outcome

Stance = str  # "positive" | "negative" | anything-else (treated as no directional claim)


@dataclass(frozen=True)
class BenchOutcome:
    """Per-horizon abnormal returns of one stock vs its resolved benchmark set."""

    ticker: str
    benchmarks: BenchmarkSet
    raw: dict[int, float | None]         # stock cumulative return at horizon h
    abn_index: dict[int, float | None]   # raw - mean(index returns)
    abn_sector: dict[int, float | None]  # raw - mean(sector-ETF returns)
    abn_peer: dict[int, float | None]    # raw - median(peer returns)
    band: dict[int, float | None]        # k·σ_backward·√h, the "beyond noise" bar

    def outperformed(self, h: int) -> bool | None:
        """True if the stock beat index AND sector AND peers by > band at h.

        None when any required leg is missing (no bars / no peers / no vol) — an
        honest 'can't tell', never silently False.
        """
        return _beat_all(self, h, sign=+1)

    def underperformed(self, h: int) -> bool | None:
        return _beat_all(self, h, sign=-1)

    def label(self, stance: Stance, h: int) -> str:
        """UP/DOWN/NEUTRAL/NA folding stance into the outperformance test."""
        if stance == "positive":
            r = self.outperformed(h)
        elif stance == "negative":
            r = self.underperformed(h)
        else:
            return "NEUTRAL"  # no directional claim -> nothing to confirm
        if r is None:
            return "NA"
        if not r:
            return "NEUTRAL"
        return "UP" if stance == "positive" else "DOWN"


def _mean(vals: Sequence[float | None]) -> float | None:
    ok = [v for v in vals if v is not None]
    return sum(ok) / len(ok) if ok else None


def _median(vals: Sequence[float | None]) -> float | None:
    ok = [v for v in vals if v is not None]
    return statistics.median(ok) if ok else None


def _abn(raw: float | None, bench: float | None) -> float | None:
    return None if raw is None or bench is None else raw - bench


def _beat_all(o: BenchOutcome, h: int, sign: int) -> bool | None:
    band = o.band.get(h)
    legs = [o.abn_index.get(h), o.abn_sector.get(h), o.abn_peer.get(h)]
    if band is None or any(a is None for a in legs):
        return None
    # sign=+1: every leg must be > +band ; sign=-1: every leg < -band
    return all(sign * a > band for a in legs)  # type: ignore[operator]


def compute_bench_outcome(
    t0: datetime,
    ticker: str,
    bars: dict[str, Sequence[DailyBar]],
    cal: TradingCalendar,
    horizons: Sequence[int],
) -> BenchOutcome | None:
    """Resolve the stock's benchmark set and compute abn_index/sector/peer per h.

    `bars` must contain the stock and every benchmark/peer ticker it resolves to;
    missing tickers degrade that leg to None (surfaced via outperformed()==None),
    they do not crash. Returns None only if the stock's own outcome can't anchor.
    """
    bset = benchmarks_for(ticker)
    if ticker not in bars:
        return None
    stock = compute_outcome(t0, bars[ticker], cal, horizons)
    if stock is None:
        return None

    def outcomes(tks: Sequence[str]) -> list[dict[int, float | None]]:
        res = []
        for tk in tks:
            if tk in bars:
                o = compute_outcome(t0, bars[tk], cal, horizons)
                if o is not None:
                    res.append(o.ret)
        return res

    idx = outcomes(bset.indices)
    sec = outcomes(bset.sectors)
    peer = outcomes(bset.peers)

    vol = backward_vol(bars[ticker], stock.s0_date, SETTINGS.vol_window_sessions)

    raw, abn_i, abn_s, abn_p, band = {}, {}, {}, {}, {}
    for h in horizons:
        r = stock.ret[h]
        raw[h] = r
        abn_i[h] = _abn(r, _mean([o[h] for o in idx]))
        abn_s[h] = _abn(r, _mean([o[h] for o in sec]))
        abn_p[h] = _abn(r, _median([o[h] for o in peer]))
        band[h] = None if vol is None else SETTINGS.k * vol * (h ** 0.5)
    return BenchOutcome(ticker, bset, raw, abn_i, abn_s, abn_p, band)


def _demo() -> None:
    """Self-check the abn/label arithmetic with a hand-built 1-horizon case."""
    from datetime import date, timezone

    from core.calendar import TradingCalendar as _Cal

    # three sessions; stock jumps +5%, index +1%, sector +1%, one peer +1%.
    def bar(tk: str, d: str, o: float, c: float) -> DailyBar:
        return DailyBar(tk, datetime.fromisoformat(d + "T00:00:00+00:00"),
                        o, c, o, c, 1_000)

    days = ["2025-01-02", "2025-01-03", "2025-01-06"]
    cal = _Cal([date.fromisoformat(d) for d in days])
    # entry = open of s0 (2025-01-03); close(s0) gives ret_1
    bars: dict[str, Sequence[DailyBar]] = {
        "INTC": [bar("INTC", days[0], 100, 100), bar("INTC", days[1], 100, 105)],
        "SPY":  [bar("SPY", days[0], 400, 400), bar("SPY", days[1], 400, 404)],
        "QQQ":  [bar("QQQ", days[0], 400, 400), bar("QQQ", days[1], 400, 404)],
        "SMH":  [bar("SMH", days[0], 200, 200), bar("SMH", days[1], 200, 202)],
        "NVDA": [bar("NVDA", days[0], 100, 100), bar("NVDA", days[1], 100, 101)],
        "AMD":  [bar("AMD", days[0], 100, 100), bar("AMD", days[1], 100, 101)],
        "TSM":  [bar("TSM", days[0], 100, 100), bar("TSM", days[1], 100, 101)],
        "AVGO": [bar("AVGO", days[0], 100, 100), bar("AVGO", days[1], 100, 101)],
    }
    t0 = datetime(2025, 1, 2, 20, 0, tzinfo=timezone.utc)  # after day0 close -> s0=day1
    o = compute_bench_outcome(t0, "INTC", bars, cal, (1,))
    assert o is not None
    raw, ai, as_, ap = o.raw[1], o.abn_index[1], o.abn_sector[1], o.abn_peer[1]
    assert raw is not None and ai is not None and as_ is not None and ap is not None
    assert abs(raw - 0.05) < 1e-9, raw
    assert abs(ai - 0.04) < 1e-9, ai       # vs mean(SPY,QQQ)=+1%
    assert abs(as_ - 0.04) < 1e-9, as_     # vs SMH=+1%
    assert abs(ap - 0.04) < 1e-9, ap       # median peers +1%
    # too few prior sessions -> vol None -> band None -> honest "can't tell"
    assert o.band[1] is None and o.outperformed(1) is None
    print("benchmarks _demo OK:", {"abn_index_1d": round(ai, 4)})


if __name__ == "__main__":
    _demo()
