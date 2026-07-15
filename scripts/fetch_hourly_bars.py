"""Fetch HOURLY bars for the intraday study — public, free, reproducible.

yfinance serves ~730 days of hourly history, which covers the 2025-01 -> 2026-07
corpus window. This matters: the existing data/real/bars_1h.csv came from Alpaca
(private keys, not reproducible) and carries NO benchmark — no SPY, USO, VIXY, GLD.
Sourcing hourly bars from yfinance instead makes the intraday study reproducible by
a stranger with no keys, exactly like the daily path.

Output: data/real/bars_1h_public.csv  (ticker, ts_utc, open, high, low, close, volume)
Idempotent: re-running replaces a ticker's rows rather than duplicating them.

Run: PYTHONPATH=. python scripts/fetch_hourly_bars.py [TICKERS...]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

OUT = Path("data/real/bars_1h_public.csv")
START, END = "2025-01-01", "2026-07-15"
HEADER = ["ticker", "ts_utc", "open", "high", "low", "close", "volume"]

# The intraday universe: the benchmark + the macro assets the geo cohort needs
# (absent from the Alpaca file) + the liquid sector/index ETFs.
DEFAULT = ("SPY", "QQQ", "USO", "VIXY", "GLD", "XLE", "XLK", "XLF", "XLI", "XLY",
           "ITA", "SMH", "TLT", "UUP", "FXI")


def fetch(tickers: list[str]) -> None:
    existing: dict[str, list[list[str]]] = {}
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and row[0] != "ticker":
                    existing.setdefault(row[0], []).append(row)

    for t in tickers:
        df = yf.download(t, start=START, end=END, interval="1h",
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            print(f"  {t:5}: EMPTY — skipped")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        rows = []
        for ts, r in df.iterrows():
            u = pd.Timestamp(ts).tz_convert("UTC")
            if any(pd.isna(r[c]) for c in ("Open", "High", "Low", "Close")):
                continue
            rows.append([t, u.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         f"{float(r['Open']):.4f}", f"{float(r['High']):.4f}",
                         f"{float(r['Low']):.4f}", f"{float(r['Close']):.4f}",
                         str(int(r.get("Volume", 0) or 0))])
        existing[t] = rows                       # replace, never append twice
        print(f"  {t:5}: {len(rows):5,} bars  {rows[0][1][:16]} -> {rows[-1][1][:16]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for t in sorted(existing):
            w.writerows(existing[t])
    total = sum(len(v) for v in existing.values())
    print(f"\n-> {OUT}  ({len(existing)} tickers, {total:,} bars, public/reproducible)")


if __name__ == "__main__":
    fetch(list(sys.argv[1:]) or list(DEFAULT))
