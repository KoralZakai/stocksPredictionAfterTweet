"""Re-fetch market returns for rows whose ENTRY ANCHOR moved under the DST fix.

The cached results JSON stores `returns` computed with a hardcoded 13:30-UTC open.
That is only correct in EDT; from November to mid-March the NYSE opens 14:30 UTC, so
any tweet posted 13:30-14:30 UTC was treated as post-open and pushed to the NEXT
session — scoring the wrong day. `alpha.benchmark.us_open_utc_hour` now resolves the
open per-date; this script re-scores ONLY the rows that anchor differently, leaving
every other row's cached returns byte-identical.

The LLM classifications are NOT re-run: the decision plane never saw a price, so the
fix cannot change what the model predicted — only which session we score it against.

Run: PYTHONPATH=. python scripts/rescore_dst_rows.py [--apply]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from alpha.benchmark import forward_returns, us_open_utc_hour, validate

RESULTS = Path("reports/nebius_backtest_results.json")
_NY = ZoneInfo("America/New_York")


def _t0(row: dict[str, Any]) -> datetime:
    h = float(row.get("hour_utc", 14.0))
    return datetime.fromisoformat(row["date"]).replace(
        hour=int(h), minute=int(round((h % 1) * 60)), tzinfo=timezone.utc)


def affected(rows: list[dict[str, Any]]) -> list[int]:
    """Indices whose anchor moves: the old code used 13.5, the real open is later."""
    out = []
    for i, r in enumerate(rows):
        h = float(r.get("hour_utc", 0.0))
        real = us_open_utc_hour(datetime.fromisoformat(r["date"]).date())
        if 13.5 <= h < real:            # old: "open passed" (wrong). new: still pre-open.
            out.append(i)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="re-score DST-mis-anchored rows")
    ap.add_argument("--apply", action="store_true", help="write the corrected cache")
    a = ap.parse_args()

    rows: list[dict[str, Any]] = json.loads(RESULTS.read_text())
    idx = affected(rows)
    print(f"rows: {len(rows)}   anchor moves under the DST fix: {len(idx)}")

    memo: dict[str, Any] = {}

    def fwd(ticker: str, t0: datetime) -> Any:
        k = f"{ticker}|{t0.date()}"
        if k not in memo:
            memo[k] = forward_returns(ticker, t0)
        return memo[k]

    for i in idx:
        r = rows[i]
        t0 = _t0(r)
        et = t0.astimezone(_NY)
        instruments = [{"ticker": x["ticker"], "predicted_direction": x.get("predicted", "")}
                       for x in r.get("instruments", [])]
        new_rows, new_hits, new_spy = validate(instruments, t0, fwd=fwd)
        old_eod = [x.get("hit", {}).get("EOD") for x in r["instruments"]]
        # Merge scoring back onto the cached classification (predictions untouched).
        for old, new in zip(r["instruments"], new_rows):
            old["returns"], old["abn"], old["hit"] = new["returns"], new["abn"], new["hit"]
        r["hits"], r["spy_returns"] = new_hits, new_spy
        new_eod = [x.get("hit", {}).get("EOD") for x in r["instruments"]]
        print(f"  {r['date']} {r['hour_utc']:5.2f}UTC ({et.strftime('%H:%M %Z')}) "
              f"EOD legs {old_eod} -> {new_eod}   {r['text'][:44]!r}")

    if a.apply:
        RESULTS.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n-> rewrote {RESULTS} ({len(idx)} rows re-scored; the rest untouched)")
    else:
        print("\n(dry run — pass --apply to write)")


if __name__ == "__main__":
    main()
