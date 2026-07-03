"""Phase-0 MVP-10 harness (§10, build-order step 6).

Runs the full plumbing for the 10 fixture tweets via the SAME build_dataset()
that modeling/eval will use, and prints one row per tweet:
  tweet -> sector ETF -> s0 -> ret_1d/2d/3d -> vol-scaled label per horizon,
plus the class-balance diagnostic (§3.5).

Correctness inspection ONLY. Do NOT read these outcomes for signal — 10
synthetic outcomes carry zero evidential weight (§10).

Run:  PYTHONPATH=. python scripts/phase0_mvp.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from config.settings import SETTINGS
from data.sources.local import LocalPriceSource, LocalTweetSource
from dataset.build import build_dataset
from labeling.thresholds import class_balance

FIX = Path(__file__).resolve().parent.parent / "data" / "fixtures"
WIDE = (datetime(2000, 1, 1, tzinfo=timezone.utc), datetime(2100, 1, 1, tzinfo=timezone.utc))


def main() -> None:
    tweets = LocalTweetSource(FIX / "tweets.csv").get_tweets(["trump"], *WIDE)
    rows = build_dataset(tweets, LocalPriceSource(FIX / "bars.csv"))

    print(f"{'id':>2}  {'t0 (UTC)':<16} {'ETF':<4} {'conf':>4}  {'s0':<10} "
          f"{'ret_1d':>8} {'ret_2d':>8} {'ret_3d':>8}   labels(1d/2d/3d)")
    print("-" * 96)
    per_horizon: dict[int, list[str]] = {h: [] for h in SETTINGS.horizon_days}
    for r in rows:
        for h in SETTINGS.horizon_days:
            per_horizon[h].append(r.label[h])
        rets = " ".join(
            f"{r.ret[h]:+.4f}" if r.ret[h] is not None else "   NA  "
            for h in SETTINGS.horizon_days
        )
        lab = "/".join(r.label[h] for h in SETTINGS.horizon_days)
        print(f"{r.tweet_id:>2}  {r.t0:%Y-%m-%d %H:%M} {r.ticker:<4} "
              f"{r.map_confidence:>4.2f}  {r.s0_date}  {rets}   {lab}")

    dropped = len(tweets) - len(rows)
    print(f"\n{len(rows)}/{len(tweets)} tweets mapped to a sector ({dropped} -> NONE, excluded).")
    print("Class balance (sec 3.5 diagnostic - degenerate split = labeling failure):")
    for h in SETTINGS.horizon_days:
        print(f"  ret_{h}d: {class_balance(per_horizon[h])}")


if __name__ == "__main__":
    main()
