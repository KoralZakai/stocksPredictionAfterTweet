"""One-tweet deep dive: "GAS PRICES COMING DOWN, FAST!" (2026-06-29T11:20:41Z).

Sector classifier put this on XLY (consumer) via a keyword misfire — "gas
prices" is an Energy statement (XLE), not Consumer Discretionary. This script
overrides the sector by hand and pulls real intraday prices (Alpaca 1-min,
same source/point-in-time rules as scripts/fetch_alpaca_30m.py) for XLE and
its 5 sector siblings: T-30m, at-tweet, session open, open+1h, end of day.

Descriptive only — one tweet, zero evidential weight (CLAUDE.md Phase-0 rule).

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/gas_tweet_deepdive.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from scripts.fetch_alpaca_30m import fetch_window, KEY, SECRET  # reuse, no new client
from data.sources.local import LocalPriceSource

T0 = datetime(2026, 6, 29, 11, 20, 41, tzinfo=timezone.utc)
TICKERS = ["XLE", "XOM", "CVX", "COP", "SLB", "EOG"]
SESSION_OPEN = datetime(2026, 6, 29, 13, 30, tzinfo=timezone.utc)  # EDT, 9:30am ET
CACHE = Path("data/real/bars_1m_gas_tweet.csv")
DAILY = "data/real/bars.csv"

# what each ticker is, for a reader unfamiliar with sector tickers.
TICKER_INFO = {
    "XLE": "Energy Select Sector SPDR ETF - basket of the whole US oil & gas sector",
    "XOM": "Exxon Mobil - integrated oil & gas major (production + refining)",
    "CVX": "Chevron - integrated oil & gas major (production + refining)",
    "COP": "ConocoPhillips - independent oil & gas producer (E&P, no refining)",
    "SLB": "SLB (Schlumberger) - oilfield services (drilling/equipment, not a producer)",
    "EOG": "EOG Resources - independent shale oil & gas producer (E&P)",
}

# tweet-content read: "prices coming down" is bearish for producer REVENUE
# (lower realized price per barrel/therm) even though the tweet's own tone is
# celebratory (politically positive framing, economically negative for XLE).
TWEET_STANCE_ON_ENERGY = "negative"  # lower prices -> lower producer revenue

CHECKPOINTS = {
    "session_open": SESSION_OPEN,
    "open+1h": SESSION_OPEN + timedelta(hours=1),
}
# T-30m / at-tweet checkpoints dropped: IEX free feed has zero pre-market
# prints for these energy names before the 13:30 UTC open (verified empty on
# fetch, not a bug) — "before" instead anchors on the prior session's close,
# same backward-only convention labeling/windows.py already uses.


def price_at(series: dict[str, tuple[list[int], list[float]]], tk: str, t: datetime) -> float | None:
    from bisect import bisect_right
    if tk not in series:
        return None
    ns, close = series[tk]
    i = bisect_right(ns, pd.Timestamp(t).as_unit("ns").value) - 1
    return None if i < 0 else float(close[i])


def main() -> None:
    assert KEY and SECRET, "no Alpaca creds in .env — see fetch_alpaca_30m.py header"

    if CACHE.exists():
        bars = pd.read_csv(CACHE)
    else:
        rows = fetch_window(TICKERS, T0 - timedelta(hours=1), SESSION_OPEN + timedelta(hours=1, minutes=5))
        bars = pd.DataFrame(rows).drop_duplicates()
        bars.to_csv(CACHE, index=False)

    ts_ns = pd.to_datetime(bars["ts_utc"], format="ISO8601", utc=True).dt.as_unit("ns").astype("int64")
    bars = bars.assign(ns=ts_ns).sort_values("ns")
    series = {tk: (list(g["ns"]), list(g["close"])) for tk, g in bars.groupby("ticker")}

    daily = LocalPriceSource(DAILY)
    prev_close: dict[str, float | None] = {}
    eod: dict[str, float | None] = {}
    for tk in TICKERS:
        b = sorted(daily.get_daily_bars(tk, T0 - timedelta(days=7), T0 + timedelta(days=1)),
                   key=lambda x: x.session_date)
        before = [x for x in b if x.session_date.date() < T0.date()]
        same_day = [x for x in b if x.session_date.date() == T0.date()]
        prev_close[tk] = before[-1].close if before else None
        eod[tk] = same_day[-1].close if same_day else None

    rows = []
    for tk in TICKERS:
        vals = {name: price_at(series, tk, t) for name, t in CHECKPOINTS.items()}
        base = prev_close[tk]
        vals["end_of_day"] = eod[tk]
        row = {"ticker": tk, "what_is_it": TICKER_INFO[tk],
               "tweet_stance_on_this_stock": TWEET_STANCE_ON_ENERGY,
               "prev_close_before_tweet": round(base, 2) if base else None,
               **{k: (round(v, 2) if v else None) for k, v in vals.items()}}
        eod_pct: float | None = None
        for k in ("session_open", "open+1h", "end_of_day"):
            v = vals[k]
            pct = None if not (base and v) else round((v / base - 1) * 100, 2)
            row[f"pct_vs_prev_close_{k}"] = pct
            if k == "end_of_day":
                eod_pct = pct

        actual_direction = "n/a" if eod_pct is None else (
            "up" if eod_pct > 0.1 else "down" if eod_pct < -0.1 else "flat")
        row["actual_direction_eod"] = actual_direction
        row["matches_tweet_stance"] = "n/a" if actual_direction == "n/a" else (
            "MATCH (moved down, as bearish-for-revenue stance implies)"
            if actual_direction == "down"
            else "OPPOSITE (moved up despite bearish-for-revenue stance)"
            if actual_direction == "up"
            else "FLAT (no meaningful move either way)")
        rows.append(row)

    out = pd.DataFrame(rows)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    out.to_csv("reports/gas_tweet_deepdive.csv", index=False, encoding="utf-8")
    print("\nwrote reports/gas_tweet_deepdive.csv")


if __name__ == "__main__":
    main()
