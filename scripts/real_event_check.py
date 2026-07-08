"""Spot-check the abnormal-return spine on REAL data (plumbing validation, §10).

For real 2019 Trump tariff tweets -> map to Industrials (XLI) & Semis (SMH),
compute abnormal_ret_h = ETF_ret_h - SPY_ret_h over horizons 1/3/5 sessions,
reusing the real compute_outcome() + calendar. Prints the events with the
largest |abnormal 1d| move.

NOT evidence of signal — a handful of hand-picked events carry zero evidential
weight (§10). This validates tweet -> s0 -> abnormal-return alignment on real
bars before the full batch build.

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/real_event_check.py
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.calendar import TradingCalendar
from data.sources.local import LocalPriceSource, LocalTweetSource
from labeling.windows import compute_outcome

HORIZONS = (1, 3, 5)
START = datetime(2019, 1, 1, tzinfo=timezone.utc)
END = datetime(2019, 12, 31, tzinfo=timezone.utc)
# Tariff/China tweets map to Industrials + Semiconductors (rule-based, §6).
KEYWORDS = ("tariff", "china")
CANDIDATE_ETFS = ("XLI", "SMH")


def main() -> None:
    tweets = LocalTweetSource("data/real/tweets.csv").get_tweets(["trump"], START, END)
    events = [
        t for t in tweets
        if not t.is_retweet and any(k in t.text.lower() for k in KEYWORDS)
    ]
    price = LocalPriceSource("data/real/bars.csv")
    spy = price.get_daily_bars("SPY", datetime(2016, 1, 1, tzinfo=timezone.utc), END)
    cal = TradingCalendar([b.session_date.date() for b in spy])
    etf_bars = {e: price.get_daily_bars(e, datetime(2016, 1, 1, tzinfo=timezone.utc), END)
                for e in CANDIDATE_ETFS}

    rows = []
    for t in events:
        spy_out = compute_outcome(t.timestamp_utc, spy, cal, HORIZONS)
        if spy_out is None:
            continue
        for etf in CANDIDATE_ETFS:
            out = compute_outcome(t.timestamp_utc, etf_bars[etf], cal, HORIZONS)
            if out is None:
                continue
            abn: dict[int, float | None] = {}
            for h in HORIZONS:
                a, s = out.ret[h], spy_out.ret[h]
                abn[h] = a - s if a is not None and s is not None else None
            if abn[1] is None:
                continue
            rows.append((abs(abn[1]), t, etf, out.s0_date, abn, out.ret, spy_out.ret))

    rows.sort(key=lambda r: r[0], reverse=True)
    print(f"{len(events)} tariff/China tweets in 2019 -> {len(rows)} (tweet,ETF) events\n")
    print(f"{'date':10} {'ETF':4} {'raw1d':>7} {'spy1d':>7} {'abn1d':>7} {'abn3d':>7} {'abn5d':>7}  tweet")
    for _, t, etf, s0, abn, raw, spyr in rows[:10]:
        def pct(x: float | None) -> str:
            return f"{x*100:+.2f}" if x is not None else "  n/a"
        print(f"{str(s0):10} {etf:4} {pct(raw[1]):>7} {pct(spyr[1]):>7} "
              f"{pct(abn[1]):>7} {pct(abn[3]):>7} {pct(abn[5]):>7}  {t.text[:60]}")

    # ponytail: one runnable check — abnormal return must equal raw minus benchmark
    _, t, etf, _, abn, raw, spyr = rows[0]
    a1, r1, s1 = abn[1], raw[1], spyr[1]
    assert a1 is not None and r1 is not None and s1 is not None
    assert abs(a1 - (r1 - s1)) < 1e-12, "abnormal != raw - spy"


if __name__ == "__main__":
    main()
