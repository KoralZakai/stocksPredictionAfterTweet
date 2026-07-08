"""Compact pre-modeling data audit (§9): timestamps, coverage, leakage.

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/audit.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import pandas as pd

from config.settings import SETTINGS
from config.universe import all_stocks
from core.calendar import TradingCalendar
from core.market_state import market_state_as_of
from data.sources.local import LocalPriceSource, load_corpus
from labeling.windows import compute_outcome

CORPUS = "data/real/corpus.csv"
BARS = "data/real/bars.csv"


def hdr(s: str) -> None:
    print(f"\n{'='*70}\n{s}\n{'='*70}")


def audit_timestamps() -> None:
    hdr("A. TIMESTAMP CONSISTENCY")
    df = pd.read_csv(CORPUS)
    ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    n, n_tz = len(df), int(ts.notna().sum())
    print(f"rows: {n}   parsed as tz-aware UTC: {n_tz}   missing/unparseable ts: {n - n_tz}")
    print("low-confidence (second-precision, Truth Social):",
          int((df["ts_confidence"] == "s").sum()))
    print("\nposts per year x platform:")
    yr = ts.dt.year
    for (y, plat), cnt in df.assign(y=yr).groupby(["y", "platform"]).size().items():
        print(f"  {int(y)}  {plat:13} {cnt:6}")
    gap = int(((ts >= pd.Timestamp("2021-01-09", tz="UTC")) &
               (ts < pd.Timestamp("2022-02-14", tz="UTC"))).sum())
    print(f"\n2021-01-09 .. 2022-02-14 off-platform gap: {gap} posts (expect ~0)")


def audit_coverage() -> None:
    hdr("B. MARKET DATA COVERAGE")
    c = duckdb.connect()
    tickers = list(SETTINGS.etfs) + [SETTINGS.benchmark] + all_stocks()
    rng = {t: (mn, mx, n) for t, mn, mx, n in c.execute(
        f"SELECT ticker, min(session_date), max(session_date), count(*) "
        f"FROM '{BARS}' GROUP BY 1"
    ).fetchall()}
    train_ok: list[str] = []
    val_ok: list[str] = []
    missing: list[str] = []
    for t in tickers:
        if t not in rng:
            missing.append(t)
            continue
        mn, mx, _ = rng[t]
        covers_train = mn.year <= 2017 and mx.year >= 2020
        covers_val = mx.year >= 2025
        (train_ok if covers_train else missing).append(t)
        if covers_val:
            val_ok.append(t)
        if not covers_train:
            print(f"  PARTIAL {t}: {mn.date()} .. {mx.date()} (train window incomplete)")
    print(f"tickers: {len(tickers)}   full train 2017-2021: {len(train_ok)}"
          f"   reach validation >=2025: {len(val_ok)}")
    if missing:
        print(f"  MISSING / PARTIAL train coverage: {missing}")
    else:
        print("  all tickers cover the 2017-2021 train window.")


def audit_leakage() -> None:
    hdr("C. LEAKAGE CHECK (point-in-time)")
    tweets = load_corpus(
        CORPUS, datetime(2017, 1, 1, tzinfo=timezone.utc),
        datetime(2021, 1, 9, tzinfo=timezone.utc))
    price = LocalPriceSource(BARS)
    lo = datetime(2016, 1, 1, tzinfo=timezone.utc)
    hi = datetime(2021, 3, 1, tzinfo=timezone.utc)
    spy = price.get_daily_bars("SPY", lo, hi)
    xle = price.get_daily_bars("XLE", lo, hi)
    cal = TradingCalendar([b.session_date.date() for b in spy])

    checked = feat_ok = label_ok = 0
    for t in tweets[:500]:
        ms = market_state_as_of(t.timestamp_utc, "XLE", xle, spy, cal)
        # every prior bar must have CLOSED strictly before t0
        if all(cal.close_utc(b.session_date.date()) < t.timestamp_utc for b in ms.prior_bars):
            feat_ok += 1
        out = compute_outcome(t.timestamp_utc, xle, cal, (1, 3, 5))
        if out is None:
            continue
        checked += 1
        # entry open strictly after t0; every label close strictly after t0
        if out.entry_ts > t.timestamp_utc and all(
            ct is None or ct > t.timestamp_utc for ct in out.close_ts.values()
        ):
            label_ok += 1
    print(f"sampled events: {len(tweets[:500])}   with computable outcome: {checked}")
    print(f"pre-event features strictly < t0:  {feat_ok}/{len(tweets[:500])}")
    print(f"labeling windows strictly  > t0:   {label_ok}/{checked}")
    assert feat_ok == len(tweets[:500]) and label_ok == checked, "LEAKAGE DETECTED"
    print("PASS: no forward-looking data in features or labels.")


def main() -> None:
    audit_timestamps()
    audit_coverage()
    audit_leakage()


if __name__ == "__main__":
    main()
