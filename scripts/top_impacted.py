"""EXPLORATORY: empirical 'top impacted stocks' per tweet by abnormal return.

*** NOT A LABEL, NOT A FEATURE, NOT A TRAINING FILTER. ***
Ranking stocks by how much they moved AFTER a tweet is selecting on the outcome
(the future). Using it to build the modeling set would leak future info (violates
§3.1) and manufacture false attribution. This view exists ONLY to eyeball how
often the biggest empirical mover disagrees with the rule-based text link — i.e.
to SHOW the false-attribution problem, not to exploit it.

For a sample of rule-mapped events it prints the top-3 |abnormal 5d| movers across
the whole 45-asset universe, and flags whether any rule-linked asset made the list.

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/top_impacted.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config.settings import SETTINGS
from config.universe import SECTOR_STOCKS
from core.calendar import TradingCalendar
from data.sources.local import LocalPriceSource, load_corpus
from labeling.windows import compute_outcome
from sector_mapping.entities import asset_relevance

CORPUS, BARS = "data/real/corpus.csv", "data/real/bars.csv"
LO = datetime(2016, 1, 1, tzinfo=timezone.utc)
T0, T1 = datetime(2017, 1, 1, tzinfo=timezone.utc), datetime(2021, 1, 9, tzinfo=timezone.utc)
N_EVENTS = 15


def main() -> None:
    tweets = [t for t in load_corpus(CORPUS, T0, T1, platforms=("twitter",))
              if asset_relevance(t.text)]
    price = LocalPriceSource(BARS)
    universe = sorted({*SETTINGS.etfs, *(s for ss in SECTOR_STOCKS.values() for s in ss)})
    bars = {t: price.get_daily_bars(t, LO, T1 + timedelta(days=40)) for t in [*universe, "SPY"]}
    cal = TradingCalendar([b.session_date.date() for b in bars["SPY"]])

    sample = tweets[:: max(1, len(tweets) // N_EVENTS)][:N_EVENTS]
    print("EXPLORATORY empirical attribution (leakage-fenced) — rule link vs biggest mover\n")
    agree = 0
    for tw in sample:
        spy = compute_outcome(tw.timestamp_utc, bars["SPY"], cal, (5,))
        if spy is None or spy.ret[5] is None:
            continue
        moves = []
        for a in universe:
            o = compute_outcome(tw.timestamp_utc, bars[a], cal, (5,))
            if o and o.ret[5] is not None:
                moves.append((a, o.ret[5] - spy.ret[5]))
        moves.sort(key=lambda kv: abs(kv[1]), reverse=True)
        linked = set(asset_relevance(tw.text))
        top3 = moves[:3]
        hit = any(a in linked for a, _ in top3)
        agree += hit
        top_s = ", ".join(f"{a}{v*100:+.1f}%" for a, v in top3)
        print(f"{str(tw.timestamp_utc)[:10]}  rule-linked={sorted(linked)}")
        print(f"    empirical top-3 movers: {top_s}   rule-linked in top3? {'YES' if hit else 'no'}")
        print(f"    \"{tw.text[:70]}\"\n")
    print(f"rule-linked asset was the biggest empirical mover in {agree}/{len(sample)} sampled events")
    print("(low overlap = market moves are mostly NOT about the tweet — the whole point.)")


if __name__ == "__main__":
    main()
