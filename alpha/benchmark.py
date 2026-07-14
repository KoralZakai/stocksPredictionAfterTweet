"""Point-in-time forward returns + beat-SPY relative-alpha scoring — MARKET PLANE.

Moved verbatim from scripts/nebius_macro_validate.py. These functions touch prices
and MUST NEVER feed the classification prompt (the leakage firewall). They score,
after the fact, whether an instrument beat SPY in the LLM's predicted direction.

Point-in-time invariant (CLAUDE.md 3.1/3.3): entry = the first session OPEN
strictly after t0. No same-day leak. Covered by tests.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

Fwd = Callable[[str, datetime], "dict[str, float] | None"]

US_OPEN_UTC_HOUR = 13.5  # ~9:30 ET; a session's open is "after t0" if this hour > t0.

# Two timescales, ONE ordered ladder:
#   30m/1h  -> Alpaca 1-min IEX bars (yfinance has no intraday history past ~60d).
#   EOD..1mo -> yfinance daily bars, in trading sessions (EOD = close of entry session).
INTRADAY_WINS: dict[str, int] = {"30m": 30, "1h": 60}       # minutes after t0 becomes actionable
DAILY_SESS: dict[str, int] = {"EOD": 1, "3d": 3, "1w": 5, "1mo": 21}
HORIZONS: list[str] = ["30m", "1h", "EOD", "3d", "1w", "1mo"]  # display / scoring order

# Option B — RELATIVE ALPHA labels. A prediction is a HIT only if the instrument
# beats the SPY benchmark in the predicted direction (abn = ret - spy_ret). This
# strips out market beta, so the null is a clean ~50% coin-flip, not the 74%
# "everything drifted up" majority. Band = 0 -> strict beat; raise for a cushion.
RELATIVE_BAND = 0.0


def _sessions(df: pd.DataFrame) -> list[tuple[datetime, float, float]]:
    """(date, open, close) rows, sorted, tz-naive dates."""
    out = []
    for ts, r in df.iterrows():
        d = pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None)
        o, c = float(r["Open"]), float(r["Close"])
        if o == o and c == c:  # drop NaN
            out.append((d, o, c))
    return sorted(out, key=lambda x: x[0])


def daily_returns(ticker: str, t0: datetime) -> dict[str, float]:
    """{EOD/3d/1w/1mo: cumulative return from the entry open}. Entry = first session
    whose open is strictly after t0 (no same-day leak)."""
    df = yf.download(ticker, start=(t0 - timedelta(days=10)).date(),
                     end=(t0 + timedelta(days=45)).date(), interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return {}
    if isinstance(df.columns, pd.MultiIndex):       # single-ticker MultiIndex -> flatten
        df.columns = df.columns.get_level_values(0)
    s = _sessions(df)
    t0h = t0.hour + t0.minute / 60.0
    i0 = next((i for i, (d, _, _) in enumerate(s)
               if d.date() > t0.date() or (d.date() == t0.date() and US_OPEN_UTC_HOUR > t0h)), None)
    if i0 is None:
        return {}
    entry = s[i0][1]
    out: dict[str, float] = {}
    for name, h in DAILY_SESS.items():
        j = i0 + h - 1
        if j < len(s):
            out[name] = s[j][2] / entry - 1.0
    return out


def session_phase(t0: datetime) -> str:
    """Market session a tweet landed in: weekend / premarket / regular / afterhours.
    Pure time function (no market data) — safe for the market plane. An identical
    copy lives in scripts/nebius_macro_backtest.py (the feature builder); kept
    separate to avoid serving/ importing from scripts/."""
    if t0.weekday() >= 5:
        return "weekend"
    h = t0.hour + t0.minute / 60.0
    if h < 13.5:
        return "premarket"
    if h >= 20.0:
        return "afterhours"
    return "regular"


def _session_anchor(t0: datetime) -> datetime:
    """When the tweet becomes tradeable: t0 if in regular hours, else the next
    session open (~13:30 UTC). Ignores holidays — fine for an intraday reaction."""
    hour = t0.hour + t0.minute / 60.0
    if t0.weekday() < 5 and US_OPEN_UTC_HOUR <= hour < 20.0:
        return t0
    if t0.weekday() < 5 and hour < US_OPEN_UTC_HOUR:
        return t0.replace(hour=13, minute=30, second=0, microsecond=0)
    d = (t0 + timedelta(days=1)).replace(hour=13, minute=30, second=0, microsecond=0)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def intraday_returns(ticker: str, t0: datetime) -> dict[str, float]:
    """{30m/1h: return over that window after the tweet is actionable}, from Alpaca
    1-min IEX bars. {} if no creds / no bars (thin IEX coverage is honest, not fatal)."""
    try:
        from scripts.fetch_alpaca_30m import KEY, SECRET, fetch_window
    except Exception:
        return {}
    if not (KEY and SECRET):
        return {}
    ref = _session_anchor(t0)
    try:
        rows = fetch_window([ticker], ref - timedelta(minutes=20),
                            ref + timedelta(minutes=max(INTRADAY_WINS.values()) + 10))
    except Exception:
        return {}
    bars = sorted(((pd.Timestamp(r["ts_utc"]).to_pydatetime(), float(r["close"]))
                   for r in rows if r.get("ticker") == ticker and r.get("close") is not None),
                  key=lambda x: x[0])
    if not bars:
        return {}

    def price_at(when: datetime) -> float | None:
        prior = [c for ts, c in bars if ts <= when]
        return prior[-1] if prior else None

    base = price_at(ref)
    if base is None or base == 0:
        return {}
    out: dict[str, float] = {}
    for name, mins in INTRADAY_WINS.items():
        p = price_at(ref + timedelta(minutes=mins))
        if p is not None:
            out[name] = p / base - 1.0
    return out


def forward_returns(ticker: str, t0: datetime) -> dict[str, float] | None:
    """All six horizons merged: intraday (Alpaca) + daily (yfinance). None if BOTH empty."""
    out = {**intraday_returns(ticker, t0), **daily_returns(ticker, t0)}
    return out or None


def _dir(x: float, band: float = 0.001) -> str:
    return "up" if x > band else "down" if x < -band else "flat"


def relative_hit(pred: str, ret: float | None, spy_ret: float | None,
                 band: float = RELATIVE_BAND) -> bool | None:
    """True if the instrument beat SPY in the predicted direction. None if unscoreable."""
    if ret is None or spy_ret is None:
        return None
    abn = ret - spy_ret
    if pred == "up":
        return abn > band
    if pred == "down":
        return abn < -band
    return None


def validate(instruments: list[dict[str, Any]], t0: datetime, fwd: Fwd = forward_returns
             ) -> tuple[list[dict[str, Any]], dict[str, list[int]], dict[str, float]]:
    """Attach returns + RELATIVE (beat-SPY) hit/miss per horizon.
    Returns (rows, hits{horizon:[hit,total]}, spy_returns).

    `fwd` = the return-fetcher (default forward_returns); the backtest injects a
    memoised version. `spy_returns` is returned so a run can be re-labelled later
    (e.g. a different band) without re-fetching any market data."""
    spy = fwd("SPY", t0) or {}
    rows: list[dict[str, Any]] = []
    hits: dict[str, list[int]] = {h: [0, 0] for h in HORIZONS}
    for ins in instruments:
        tk = str(ins.get("ticker", "")).upper()
        pred = str(ins.get("predicted_direction", "neutral")).lower()
        actual = fwd(tk, t0)
        row = {"ticker": tk, "name": ins.get("name", ""), "role": ins.get("role", ""),
               "predicted": pred, "returns": actual, "abn": {}, "hit": {}}
        if actual:
            for h in HORIZONS:
                if h in actual and h in spy:
                    row["abn"][h] = actual[h] - spy[h]     # vs S&P 500
                # SPY cannot beat itself -> the benchmark is never scored.
                if pred in ("up", "down") and tk != "SPY":
                    hit = relative_hit(pred, actual.get(h), spy.get(h))
                    if hit is not None:
                        row["hit"][h] = hit
                        hits[h][0] += int(hit)
                        hits[h][1] += 1
        rows.append(row)
    return rows, hits, spy
