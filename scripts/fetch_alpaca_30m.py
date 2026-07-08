"""Fill the 30-minute reaction columns from Alpaca 1-minute bars (free IEX feed).

Populates raw_30m / abn_30m in data/real/intraday_reactions.csv — the columns
scripts/intraday_reactions.py reserves (hourly yfinance can't reach 30m history).

Scope-lean by design: we do NOT bulk-download 18 months of minute bars. For each
of the ~139 events we fetch ONE small window (baseline .. anchor+35m) for the
entity + its sector ETF — ~139 REST calls, inside the free tier's 200 req/min.
Raw bars are cached to data/real/bars_1m_events.csv so reruns are offline.

Semantics (same point-in-time rules as the hourly panel):
  baseline_30m = last 1-min bar fully CLOSED strictly before t0 (IEX prints may
                 be extended-hours — closer to the tweet than the hourly panel's
                 baseline; the two columns are therefore self-consistent, not
                 mixed across sources).
  price @ T    = close of last 1-min bar with start+1min <= T, T = anchor+30m
                 (anchor = t0 in-session, else next open — same anchor CSV col).
  abn_30m      = entity move - sector-ETF move over the identical window.

Setup (one-time, free): create a key at https://app.alpaca.markets (paper is
fine), then set env vars APCA_API_KEY_ID and APCA_API_SECRET_KEY.

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/fetch_alpaca_30m.py
"""

from __future__ import annotations

import os
import sys
import time
from bisect import bisect_right
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

PANEL = Path("data/real/intraday_reactions.csv")
CACHE = Path("data/real/bars_1m_events.csv")
URL = "https://data.alpaca.markets/v2/stocks/bars"
BAR = timedelta(minutes=1)
WINDOW = timedelta(minutes=30)

def _load_dotenv(path: str = ".env") -> None:
    """Minimal stdlib .env loader (no python-dotenv dep). Shell env wins."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_load_dotenv()
# accept the common spellings — this repo's .env uses ALPACA_API_KEY/ALPACA_API_SECRET
KEY = (os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
       or os.environ.get("ALPACA_API_KEY_ID"))
SECRET = (os.environ.get("ALPACA_API_SECRET") or os.environ.get("ALPACA_SECRET_KEY")
          or os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET_KEY"))


def fetch_window(symbols: list[str], start: datetime, end: datetime) -> list[dict]:
    """1-min IEX bars for symbols in [start, end]; follows pagination; retries 429."""
    assert KEY and SECRET
    rows: list[dict] = []
    params: dict[str, str] = {
        "symbols": ",".join(symbols), "timeframe": "1Min", "feed": "iex",
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": "10000",
    }
    while True:
        r = requests.get(URL, params=params,
                         headers={"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET},
                         timeout=30)
        if r.status_code == 429:           # free tier rate limit — back off once
            time.sleep(3)
            continue
        if r.status_code == 401:
            sys.exit("401 Unauthorized — Alpaca rejected the credentials.\n"
                     f"  KEY in use starts {KEY[:5]!r} (len {len(KEY or '')}).\n"
                     "  A valid Alpaca key looks like 'PKB7VG23...' (uppercase PK, no dash).\n"
                     "  Fix .env:  ALPACA_API_KEY=PK...   ALPACA_API_SECRET=<40-char secret>\n"
                     "  (no quotes/spaces), then rerun.")
        r.raise_for_status()
        data = r.json()
        for sym, bars in (data.get("bars") or {}).items():
            rows += [{"ticker": sym, "ts_utc": b["t"], "close": b["c"]} for b in bars]
        tok = data.get("next_page_token")
        if not tok:
            return rows
        params["page_token"] = tok


def main() -> None:
    if not (KEY and SECRET):
        sys.exit("No Alpaca credentials. Create a free key at https://app.alpaca.markets\n"
                 "then set ALPACA_API_KEY and ALPACA_API_SECRET (in .env) and rerun.")
    for nm, v in [("ALPACA_API_KEY", KEY), ("ALPACA_API_SECRET", SECRET)]:
        try:
            v.encode("ascii")
        except UnicodeEncodeError:
            bad = next(c for c in v if ord(c) > 127)
            sys.exit(f"{nm} contains a non-ASCII character ({bad!r}, {hex(ord(bad))}) — "
                     "a bad paste (wrong keyboard layout / smart quote). Re-copy the raw "
                     "value from Alpaca into .env (no quotes, no spaces).")
    panel = pd.read_csv(PANEL)
    t0s = pd.to_datetime(panel["t0_utc"], format="ISO8601", utc=True)
    anchors = pd.to_datetime(panel["anchor_utc"], format="ISO8601", utc=True)

    # ---- fetch (or reuse cache) ------------------------------------------
    if CACHE.exists():
        bars = pd.read_csv(CACHE)
    else:
        all_rows: list[dict] = []
        for i, (idx, e) in enumerate(panel.iterrows()):
            t0, anc = t0s[idx].to_pydatetime(), anchors[idx].to_pydatetime()
            syms = sorted({str(e["entity"]), str(e["sector_etf"])})
            # baseline may sit hours before an overnight anchor: pull from t0-2h
            all_rows += fetch_window(syms, t0 - timedelta(hours=2), anc + timedelta(minutes=35))
            if (i + 1) % 25 == 0:
                print(f"  fetched {i + 1}/{len(panel)} event windows...")
        bars = pd.DataFrame(all_rows).drop_duplicates()
        bars.to_csv(CACHE, index=False)
        print(f"cached {len(bars)} minute bars -> {CACHE}")

    # ---- index: int-ns arrays per ticker (same trick as the hourly panel) --
    series: dict[str, tuple[list[int], list[float]]] = {}
    ts_ns = (pd.to_datetime(bars["ts_utc"], format="ISO8601", utc=True)
             .dt.as_unit("ns").astype("int64"))  # force ns to match lookup keys
    bars = bars.assign(ns=ts_ns).sort_values("ns")
    for tk, g in bars.groupby("ticker"):
        series[tk] = (list(g["ns"]), list(g["close"]))

    def price_at(tk: str, t: datetime) -> float | None:
        if tk not in series:
            return None
        ns, close = series[tk]
        i = bisect_right(ns, pd.Timestamp(t - BAR).as_unit("ns").value) - 1
        return None if i < 0 else float(close[i])

    # ---- fill raw_30m / abn_30m ------------------------------------------
    filled = 0
    for idx, e in panel.iterrows():
        t0, anc = t0s[idx].to_pydatetime(), anchors[idx].to_pydatetime()
        tk, etf = str(e["entity"]), str(e["sector_etf"])
        base, ebase = price_at(tk, t0), price_at(etf, t0)
        p, ep = price_at(tk, anc + WINDOW), price_at(etf, anc + WINDOW)
        if not base or not ebase or p is None:
            continue
        raw = p / base - 1
        panel.loc[idx, "raw_30m"] = round(raw, 5)
        if ep is not None:
            panel.loc[idx, "abn_30m"] = round(raw - (ep / ebase - 1), 5)
        filled += 1

    panel.to_csv(PANEL, index=False, encoding="utf-8")
    print(f"\nfilled 30m columns for {filled}/{len(panel)} events -> {PANEL}")
    print("\nmedian |abnormal| move by window (30m now real):")
    for col, lab in [("abn_30m", "+30m"), ("abn_1h", "+1h "),
                     ("abn_2h", "+2h "), ("abn_3h", "+3h ")]:
        s = pd.to_numeric(panel[col], errors="coerce").dropna().abs()
        if len(s):
            print(f"  {lab}: {s.median() * 100:5.2f}%   (n={len(s)})")

    # ponytail: self-check — 30m magnitude should sit below the 1h magnitude
    # in the median (monotone noise accumulation); warn loud if inverted
    m30 = pd.to_numeric(panel["abn_30m"], errors="coerce").dropna().abs().median()
    m1h = pd.to_numeric(panel["abn_1h"], errors="coerce").dropna().abs().median()
    if m30 and m1h and m30 > m1h * 1.5:
        print(f"\nWARNING: median |abn_30m| ({m30:.4f}) >> |abn_1h| ({m1h:.4f}) — "
              "check IEX sparsity / baseline mismatch before trusting.")


if __name__ == "__main__":
    main()
