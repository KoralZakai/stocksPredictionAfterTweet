"""DESCRIPTIVE case study: what actually happened to the price after the strongest
tweets? EXPLORATORY — hindsight-selected, no hypothesis test, no p-values claimed.

Answers "was there a fear effect, a dip then a recovery, and how long did it last?"
by tracing the abnormal-return PATH out to ~2 months (1/3/5/10/21/42 sessions).

The control is the whole point: big moves mean-revert regardless of cause. So every
tweet-day path is shown against (a) random days for the same asset and (b) days with
an equally large day-1 move but NO qualifying tweet. If the tweet-day shape is not
distinguishable from the big-move-no-tweet shape, the "Trump effect" is just what
volatility does.

Run: PYTHONPATH=. python experiments/event_study/case_paths.py
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.event_study.engine import load_bars, study_event

WINDOWS = (1, 3, 5, 10, 21, 42)      # day, ~week, ~2wk, ~month, ~2 months
SEED = 20260715
RESULTS = Path("reports/nebius_backtest_results.json")

# Text selectors for the "strong tweet" families the user named.
THEMES: dict[str, tuple[re.Pattern[str], str]] = {
    "IRAN/HORMUZ/OIL": (re.compile(
        r"\b(iran|hormuz|strait|opec|oil|crude|drill|refiner|tanker)\b", re.I), "USO"),
    "CHINA/TARIFF": (re.compile(r"\b(china|tariff|beijing|xi\b|trade war)\b", re.I), "FXI"),
    "INTEL/SEMIS": (re.compile(r"\b(intel|semiconductor|chips?|nvidia|tsmc)\b", re.I), "SMH"),
}


def _t0(row: dict[str, Any]) -> datetime:
    h = float(row.get("hour_utc", 14.0))
    return datetime.fromisoformat(row["date"]).replace(
        hour=int(h), minute=int((h % 1) * 60), tzinfo=timezone.utc)


def _fmt_path(car: dict[int, float]) -> str:
    return "  ".join(f"{w}d:{car.get(w, float('nan')) * 100:+6.2f}%" for w in WINDOWS)


def _mean_path(paths: list[dict[int, float]]) -> dict[int, float]:
    out = {}
    for w in WINDOWS:
        vals = [p[w] for p in paths if w in p]
        out[w] = sum(vals) / len(vals) if vals else float("nan")
    return out


def main() -> None:
    rng = random.Random(SEED)
    bars = load_bars()
    rows = json.loads(RESULTS.read_text())

    for theme, (rx, asset) in THEMES.items():
        hits = [r for r in rows if rx.search(r.get("text", ""))]
        # One event per session (same-day tweets share one s0 and one path).
        paths: list[tuple[float, str, dict[int, float], str]] = []
        seen: set[str] = set()
        for r in sorted(hits, key=lambda x: x["date"]):
            er = study_event(bars, asset, _t0(r), WINDOWS)
            if er is None or er.s0_date in seen:
                continue
            seen.add(er.s0_date)
            paths.append((abs(er.car[1]), er.s0_date, er.car, r.get("text", "")[:64]))

        if not paths:
            print(f"\n### {theme} -> {asset}: no scoreable events\n")
            continue
        paths.sort(key=lambda x: -x[0])

        print(f"\n{'='*100}\n### {theme}  ->  {asset}   ({len(paths)} tweet-days)\n{'='*100}")
        print(f"{'s0':12}{'abnormal-return path (cumulative, vs market model)':52}")
        print("-" * 100)
        for _mag, s0, car, txt in paths[:5]:
            print(f"{s0:12}{_fmt_path(car)}")
            print(f"{'':12}{txt!r}")

        # --- controls ---
        all_pool = [b.date for b in bars.get(asset, [])
                    if "2025-01-01" <= b.date <= "2026-04-01"]
        rnd_paths = []
        for d in rng.sample(all_pool, min(120, len(all_pool))):
            er = study_event(bars, asset, datetime.fromisoformat(d).replace(
                hour=14, tzinfo=timezone.utc), WINDOWS)
            if er:
                rnd_paths.append(er.car)

        # Big-move days WITHOUT a qualifying tweet, matched on |day-1| magnitude.
        tweet_days = {s0 for _m, s0, _c, _t in paths}
        big_no_tweet = [c for c in rnd_paths if abs(c[1]) >= 0.03]
        top5_mean = _mean_path([c for _m, _s, c, _t in paths[:5]])
        all_tweet_mean = _mean_path([c for _m, _s, c, _t in paths])

        print("-" * 100)
        print(f"{'MEAN top-5':12}{_fmt_path(top5_mean)}   <- the headline anecdotes")
        print(f"{'MEAN all':12}{_fmt_path(all_tweet_mean)}   <- ALL {len(paths)} tweet-days")
        print(f"{'MEAN random':12}{_fmt_path(_mean_path(rnd_paths))}   <- random days (n={len(rnd_paths)})")
        if big_no_tweet:
            print(f"{'MEAN big-move':12}{_fmt_path(_mean_path(big_no_tweet))}   "
                  f"<- |1d|>=3% days, NO qualifying tweet (n={len(big_no_tweet)})")
        print(f"(tweet-days covered {len(tweet_days)} of {len(all_pool)} sessions "
              f"= {len(tweet_days) / max(len(all_pool), 1):.0%} of the calendar)")


if __name__ == "__main__":
    main()
