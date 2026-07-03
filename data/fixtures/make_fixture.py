"""Generate the MVP-10 fixture: 10 tweets + daily bars for the ETF universe.

SYNTHETIC, deterministic (seeded). Phase 0 is plumbing correctness only — the
prices are a seeded random walk, NOT real market data, and outcomes carry zero
evidential weight (§10). Swap in real OHLCV + real tweets for Phase 1.

Run:  python data/fixtures/make_fixture.py   (writes tweets.csv, bars.csv here)

The 10 tweets deliberately span every s0-resolution case the Phase-0 gate
checks: before-open, intraday, after-close, weekend, and holiday.
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

from config.settings import SETTINGS

HERE = Path(__file__).parent
HOLIDAYS = {date(2024, 1, 15), date(2024, 2, 19)}  # MLK, Presidents' Day


def _sessions(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in HOLIDAYS:
            out.append(d)
        d += timedelta(days=1)
    return out


def write_bars() -> None:
    rng = random.Random(SETTINGS.seed)
    sessions = _sessions(date(2024, 1, 2), date(2024, 2, 29))
    with (HERE / "bars.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "session_date", "open", "high", "low", "close", "volume"])
        for i, tk in enumerate((*SETTINGS.etfs, SETTINGS.benchmark)):
            price = 50.0 + 10.0 * i
            for d in sessions:
                gap = 1.0 + rng.gauss(0, 0.003)
                op = round(price * gap, 2)
                price = round(op * (1.0 + rng.gauss(0, 0.01)), 2)
                hi = round(max(op, price) * 1.004, 2)
                lo = round(min(op, price) * 0.996, 2)
                w.writerow([tk, f"{d.isoformat()}T00:00:00Z", op, hi, lo, price,
                            rng.randint(1_000_000, 5_000_000)])


# (timestamp UTC, text) — timestamps chosen to hit each session-placement case.
TWEETS = [
    ("2024-02-05T13:00:00Z", "We will drill baby drill — American energy dominance!"),   # before-open
    ("2024-02-06T16:00:00Z", "Massive tariffs on China, our manufacturing comes home"),  # intraday
    ("2024-02-07T22:00:00Z", "Our brave military needs more funding for defense"),       # after-close
    ("2024-02-10T15:00:00Z", "The big banks and the Fed are ripping off Americans"),     # weekend (Sat)
    ("2024-02-12T13:00:00Z", "American chip makers and semiconductors will lead"),       # before-open
    ("2024-02-13T16:00:00Z", "Lower drug prices now — big pharma is on notice"),         # intraday
    ("2024-02-14T22:00:00Z", "Buy American cars, the auto industry is booming"),         # after-close
    ("2024-02-19T17:00:00Z", "Tech and technology are the future of America"),           # holiday
    ("2024-02-21T16:00:00Z", "Oil prices too high, OPEC must increase supply"),          # intraday
    ("2024-02-22T13:00:00Z", "Trade war with China escalates, factory jobs hit"),        # before-open
]


def write_tweets() -> None:
    with (HERE / "tweets.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "author", "text", "timestamp_utc", "is_retweet", "is_deleted"])
        for i, (ts, text) in enumerate(TWEETS, 1):
            w.writerow([str(i), "trump", text, ts, False, False])


if __name__ == "__main__":
    write_bars()
    write_tweets()
    print(f"wrote {HERE/'bars.csv'} and {HERE/'tweets.csv'}")
