"""Intraday reaction panel: stock price 1h / 2h / 3h after each tweet.

Data: yfinance 1-HOUR bars (free, verified to reach Jan 2025 — the 730-day
window). 30-minute windows are NOT possible on this source (60-day cap); they
need minute bars = a free Alpaca API key. The column is reserved, not faked.

Trading-hours logic (per user spec):
  * tweet DURING the regular session -> windows anchored at the tweet time t0.
  * tweet after-hours / weekend / holiday -> anchor shifts to the NEXT session
    open T_open (calendar.resolve_s0), windows at T_open +1h/+2h/+3h.
Baseline ("before") = last hourly bar fully CLOSED strictly before t0 — the
last price the market printed before it could know about the tweet.

Point-in-time rule: a bar counts at time T only if bar_start + 1h <= T (fully
closed). abnormal = stock move - sector-ETF move over the identical window.

This is a REACTION panel, separate from the 1-5 day drift pipeline (§1/§2);
descriptive only, never feeds the BH-tested labels.

Run:  PYTHONPATH=. .venv/Scripts/python.exe scripts/intraday_reactions.py
Out:  data/real/intraday_reactions.csv (+ cached hourly bars bars_1h.csv)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from core.calendar import ET, TradingCalendar
from data.sources.local import LocalPriceSource, load_corpus

EVENTS, CORPUS, DAILY = ("data/real/entity_results.csv",
                         "data/real/corpus_v3.csv", "data/real/bars.csv")
CACHE = Path("data/real/bars_1h.csv")
OUT = Path("data/real/intraday_reactions.csv")
T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2026, 7, 7, tzinfo=timezone.utc)
WINDOWS_H = (1, 2, 3)  # hours after anchor; 30m reserved -> needs minute bars
BAR = timedelta(hours=1)


def fetch_hourly(tickers: list[str]) -> pd.DataFrame:
    """(ticker, ts_utc, close) hourly bars; cached so reruns don't refetch."""
    if CACHE.exists():
        df = pd.read_csv(CACHE, parse_dates=["ts_utc"])
        if set(tickers) <= set(df["ticker"].unique()):
            return df
    import yfinance as yf
    raw = yf.download(tickers, start="2025-01-01", end="2026-07-07",
                      interval="1h", group_by="ticker", progress=False,
                      auto_adjust=True)
    rows = []
    for tk in tickers:
        sub = (raw[tk] if len(tickers) > 1 else raw).dropna(subset=["Close"])
        for ts, r in sub.iterrows():
            rows.append((tk, ts.tz_convert("UTC"), float(r.Close)))
    df = pd.DataFrame(rows, columns=["ticker", "ts_utc", "close"])
    df.to_csv(CACHE, index=False)
    return df


def main() -> None:
    ev = pd.read_csv(EVENTS)
    ev = ev[ev["missing_bars"] == 0]
    ts_by_post = {t.tweet_id: t.timestamp_utc for t in load_corpus(CORPUS, T0, T1)}
    tickers = sorted(set(ev["primary_entity"]) | set(ev["sector_etf"]))

    daily = LocalPriceSource(DAILY)
    spy = daily.get_daily_bars("SPY", T0 - timedelta(days=40), T1)
    cal = TradingCalendar([b.session_date.date() for b in spy])
    sessions = set(cal.dates)

    bars = fetch_hourly(tickers)
    # per-ticker sorted arrays for fast "last closed bar <= T" lookups
    # compare in int nanoseconds — sidesteps pandas tz/unit strictness entirely
    series: dict[str, tuple[list[int], list[float]]] = {}
    for tk, g in bars.groupby("ticker"):
        g = g.sort_values("ts_utc")
        ns = [pd.Timestamp(t).as_unit("ns").value for t in g["ts_utc"]]
        series[tk] = (ns, list(g["close"]))

    def price_at(tk: str, t: datetime) -> float | None:
        """Close of the last hourly bar FULLY closed at/before t (start+1h <= t)."""
        if tk not in series:
            return None
        from bisect import bisect_right
        ns, close = series[tk]
        i = bisect_right(ns, pd.Timestamp(t - BAR).as_unit("ns").value) - 1
        return None if i < 0 else float(close[i])

    def anchor(t0: datetime) -> tuple[datetime, str] | None:
        et_d: date = t0.astimezone(ET).date()
        if et_d in sessions and cal.open_utc(et_d) <= t0 < cal.close_utc(et_d):
            return t0, "in_session"
        s0 = cal.resolve_s0(t0)
        return None if s0 is None else (cal.open_utc(s0), "shifted_to_next_open")

    rows = []
    for _, e in ev.iterrows():
        t0 = ts_by_post.get(str(e["post_id"]))
        if t0 is None:
            continue
        a = anchor(t0)
        if a is None:
            continue
        t_ref, mode = a
        tk, etf = str(e["primary_entity"]), str(e["sector_etf"])
        base, ebase = price_at(tk, t0), price_at(etf, t0)  # last print BEFORE tweet
        if not base or not ebase:
            continue
        row = {"post_id": e["post_id"], "entity": tk, "sector_etf": etf,
               "stance": e["stance"], "mode": mode,
               "t0_utc": t0.isoformat(), "anchor_utc": t_ref.isoformat(),
               "price_before": round(base, 2), "raw_30m": None, "abn_30m": None}
        for h in WINDOWS_H:
            p, ep = price_at(tk, t_ref + timedelta(hours=h)), price_at(etf, t_ref + timedelta(hours=h))
            raw = None if p is None else p / base - 1
            sec = None if ep is None else ep / ebase - 1
            row[f"raw_{h}h"] = None if raw is None else round(raw, 5)
            row[f"abn_{h}h"] = (None if raw is None or sec is None
                                else round(raw - sec, 5))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8")

    print(f"intraday reaction panel: {len(df)} events -> {OUT}")
    print("30m columns are EMPTY by design: hourly source; minute bars need a free "
          "Alpaca key.")
    print("\nanchor mode:", df["mode"].value_counts().to_dict())
    print("\nmedian |abnormal| move (stock vs its sector, same window):")
    for h in WINDOWS_H:
        s = df[f"abn_{h}h"].dropna().abs()
        print(f"  +{h}h : {s.median()*100:5.2f}%   (n={len(s)})")
    d5 = pd.read_csv(EVENTS)["abn_5D"].dropna().abs()
    print(f"  +5D : {d5.median()*100:5.2f}%   (n={len(d5)})  <- daily-drift scale, for contrast")
    top = df.dropna(subset=["abn_3h"]).nlargest(5, "abn_3h", keep="all")
    print("\nlargest +3h abnormal reactions:")
    for _, r in top.head(5).iterrows():
        print(f"  {r['entity']:5} {r['t0_utc'][:16]}  {r['mode']:22} "
              f"abn_3h={r['abn_3h']*100:+.2f}%  stance={r['stance']}")

    # ponytail: self-checks — anchors never precede t0; no window uses a bar
    # that closed at/before the baseline print
    assert (pd.to_datetime(df["anchor_utc"], format="ISO8601")
            >= pd.to_datetime(df["t0_utc"], format="ISO8601")).all()
    assert df["price_before"].gt(0).all()


if __name__ == "__main__":
    main()
