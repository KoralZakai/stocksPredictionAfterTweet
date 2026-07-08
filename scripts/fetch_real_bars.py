"""Fetch real daily OHLCV for the 10 sector ETFs + SPY into the fixture schema.

ponytail: writes the same columns as data/fixtures/bars.csv so the existing
LocalPriceSource adapter ingests it unchanged — no new plumbing.

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/fetch_real_bars.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yfinance as yf

from config.settings import SETTINGS
from config.universe import TRUMP_EXPOSED, all_stocks

OUT = Path("data/real/bars.csv")
# Cover ALL eras (DATASEARCH §1/§4): pre-office Twitter (2009+), in-office
# (2017-2021), Truth Social (2022-now), plus a 2008 buffer for pre-event windows.
START = "2008-11-01"
END = "2026-07-07"


def main() -> None:
    tickers = (list(SETTINGS.etfs) + [SETTINGS.benchmark] + all_stocks()
               + sorted(TRUMP_EXPOSED))  # DJT: bars exist 2024-03+ only
    df = yf.download(
        tickers, start=START, end=END, interval="1d",
        auto_adjust=True, group_by="ticker", progress=False,
    )
    if df is None or df.empty:
        sys.exit("yfinance returned no data (network? rate limit?)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = ["ticker,session_date,open,high,low,close,volume"]
    for tk in tickers:
        sub = df[tk].dropna(subset=["Open", "Close"])
        for ts, r in sub.iterrows():
            d = ts.strftime("%Y-%m-%d")
            rows.append(
                f"{tk},{d}T00:00:00Z,{r.Open:.4f},{r.High:.4f},"
                f"{r.Low:.4f},{r.Close:.4f},{int(r.Volume)}"
            )
    OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    n = len(rows) - 1
    print(f"wrote {n} bars for {len(tickers)} tickers -> {OUT}")
    # ponytail: one runnable check — fail loud if a ticker came back empty
    got = {r.split(",", 1)[0] for r in rows[1:]}
    missing = set(tickers) - got
    assert not missing, f"no bars for: {missing}"


if __name__ == "__main__":
    main()
