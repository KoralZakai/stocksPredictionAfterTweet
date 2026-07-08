"""Append daily OHLCV for extra tickers to data/real/bars.csv (non-destructive).

Used to add policy-dependency names + the SOXX semis benchmark (CHIPS-Act worked
example) without regenerating the existing 47-ticker file. Same schema as
fetch_real_bars.py, so LocalPriceSource ingests it unchanged. Dedupes on
(ticker, session_date): re-running is idempotent.

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/fetch_extra_bars.py MU GFS TXN SOXX
"""

from __future__ import annotations

import sys
from pathlib import Path

import yfinance as yf

OUT = Path("data/real/bars.csv")
START, END = "2008-11-01", "2026-07-07"
HEADER = "ticker,session_date,open,high,low,close,volume"


def main(tickers: list[str]) -> None:
    existing = OUT.read_text(encoding="utf-8").splitlines() if OUT.exists() else [HEADER]
    have = {ln.split(",", 2)[0] + "|" + ln.split(",", 2)[1] for ln in existing[1:]}
    df = yf.download(tickers, start=START, end=END, interval="1d",
                     auto_adjust=True, group_by="ticker", progress=False)
    if df is None or df.empty:
        sys.exit("yfinance returned no data (network? rate limit?)")

    added = 0
    new_rows: list[str] = []
    for tk in tickers:
        sub = (df[tk] if len(tickers) > 1 else df).dropna(subset=["Open", "Close"])
        for ts, r in sub.iterrows():
            d = ts.strftime("%Y-%m-%d") + "T00:00:00Z"
            if f"{tk}|{d}" in have:
                continue
            new_rows.append(f"{tk},{d},{r.Open:.4f},{r.High:.4f},"
                            f"{r.Low:.4f},{r.Close:.4f},{int(r.Volume)}")
            added += 1
    OUT.write_text("\n".join(existing + new_rows) + "\n", encoding="utf-8")
    print(f"appended {added} bars for {tickers} -> {OUT}")
    # ponytail: fail loud if any requested ticker came back empty
    got = {r.split(",", 1)[0] for r in new_rows} | {ln.split(",", 1)[0] for ln in existing[1:]}
    missing = [t for t in tickers if t not in got]
    assert not missing, f"no bars for: {missing}"


if __name__ == "__main__":
    main(sys.argv[1:] or ["MU", "GFS", "TXN", "SOXX"])
