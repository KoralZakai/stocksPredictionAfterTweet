"""Signal-or-null report over the fixture (§7). The reporting_job entrypoint.

Run:  PYTHONPATH=. python scripts/phase1_report.py

At N=10 the honest verdict is "underpowered / no signal survives correction" —
which is exactly the point of the exercise (§1).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from data.sources.local import LocalPriceSource, LocalTweetSource
from dataset.build import build_dataset
from eval.report import format_report, run_report

FIX = Path(__file__).resolve().parent.parent / "data" / "fixtures"
WIDE = (datetime(2000, 1, 1, tzinfo=timezone.utc), datetime(2100, 1, 1, tzinfo=timezone.utc))


def main() -> None:
    tweets = LocalTweetSource(FIX / "tweets.csv").get_tweets(["trump"], *WIDE)
    rows = build_dataset(tweets, LocalPriceSource(FIX / "bars.csv"))
    report, registry = run_report(rows)
    print(f"Dataset: {len(rows)} rows | registry: {len(registry)} tests\n")
    print(format_report(report))


if __name__ == "__main__":
    main()
