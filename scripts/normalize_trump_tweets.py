"""Normalize the raw Trump Twitter archives into the project's tweet schema.

Inputs (same schema: ID, Time, Tweet URL, Tweet Text — MarkHershey archive):
  data/real/trump_in_office_raw.csv   # 2017-01-20 .. 2021-01-08
  data/real/trump_bf_office_raw.csv   # 2009-05-04 .. 2017-01-19 (DATASEARCH §1)
Output:
  data/real/tweets.csv                # + ts_confidence column (ms|min)

t0 primarily comes from the tweet's numeric ID (a Twitter *snowflake* encoding
exact UTC ms) — NOT the archive's tz-less `Time` column. Tweets BEFORE the
snowflake cutover (2010-11-04, sequential ids < ~3e10) have no snowflake: for
those we use the `Time` column, whose timezone we INFER empirically by
comparing it against snowflake-decoded UTC on rows that have both, then
localize properly (zoneinfo, DST-aware, if it turns out to be US/Eastern).
Those rows get ts_confidence='min' (minute precision — still enough to place
a tweet before-open/intraday/after-close, §9).

ponytail: text kept verbatim (utf-8); cosmetic mojibake logged as a §9 note.

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/normalize_trump_tweets.py
Downstream: scripts/build_corpus.py (reads tweets.csv incl. ts_confidence).
"""

from __future__ import annotations

import csv
import re
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

RAWS = (Path("data/real/trump_in_office_raw.csv"),
        Path("data/real/trump_bf_office_raw.csv"))
OUT = Path("data/real/tweets.csv")
TWITTER_EPOCH_MS = 1288834974657
SNOWFLAKE_MIN_ID = 10**11  # sequential ids topped out ~3e10; snowflakes pass 1e11 in <1min
STATUS_RE = re.compile(r"/status/(\d+)")


def snowflake_utc(status_id: int) -> datetime:
    """Exact UTC post time encoded in a Twitter snowflake id."""
    ms = (status_id >> 22) + TWITTER_EPOCH_MS
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def read_raw(path: Path) -> list[tuple[int, str, str]]:
    """(status_id, time_str, text) rows; skips rows without a /status/ id."""
    rows: list[tuple[int, str, str]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 4:
                continue
            m = STATUS_RE.search(row[2])
            if not m:
                continue
            rows.append((int(m.group(1)), row[1].strip(), row[3].strip().strip('"')))
    return rows


def infer_archive_tz(rows: list[tuple[int, str, str]]) -> ZoneInfo | timezone:
    """Median (snowflake UTC - naive Time) over snowflake rows -> archive tz.

    Empirically the MarkHershey archive's Time column is UTC+8 (SGT, no DST):
    median comes out ~-8h. A stable near-integer offset -> fixed timezone;
    a 4-5h offset -> US/Eastern (DST-aware); anything unstable fails loudly.
    """
    diffs: list[float] = []
    for tid, tstr, _ in rows:
        if tid < SNOWFLAKE_MIN_ID or len(diffs) >= 500:
            continue
        try:
            naive = datetime.strptime(tstr, "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        diffs.append((snowflake_utc(tid).replace(tzinfo=None) - naive).total_seconds() / 3600)
    med = statistics.median(diffs)
    if 3.25 < med < 5.75:  # ET is UTC-4/-5 -> snowflake UTC is AHEAD of naive ET
        return ZoneInfo("America/New_York")
    whole = round(med)
    if abs(med - whole) < 0.25:  # stable fixed offset (0 = UTC, -8 = SGT, ...)
        return timezone(timedelta(hours=-whole))  # Time = UTC + (-med)
    raise AssertionError(f"archive Time column has unexpected offset: median {med:+.2f}h")


def main() -> None:
    seen: set[int] = set()
    out_rows: list[tuple[str, str, str, str, str, str, str]] = []
    n_min = 0

    for raw in RAWS:
        rows = read_raw(raw)
        tz = infer_archive_tz(rows)
        print(f"{raw.name}: {len(rows)} rows, Time column tz = {tz}")
        for tid, tstr, text in rows:
            if tid in seen:
                continue
            seen.add(tid)
            if tid >= SNOWFLAKE_MIN_ID:
                ts, conf = snowflake_utc(tid), "ms"
            else:  # pre-snowflake (2009-2010): archive Time column, inferred tz
                naive = datetime.strptime(tstr, "%Y-%m-%d %H:%M")
                ts, conf = naive.replace(tzinfo=tz).astimezone(timezone.utc), "min"
                n_min += 1
            is_rt = "True" if text.startswith("RT @") else "False"
            out_rows.append((str(tid), "trump", text,
                             ts.isoformat(timespec="microseconds"), is_rt, "False", conf))

    out_rows.sort(key=lambda r: r[3])  # chronological
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "author", "text", "timestamp_utc",
                    "is_retweet", "is_deleted", "ts_confidence"])
        w.writerows(out_rows)

    n_rt = sum(1 for r in out_rows if r[4] == "True")
    print(f"wrote {len(out_rows)} tweets -> {OUT}")
    print(f"  retweets flagged: {n_rt}   minute-precision (pre-snowflake): {n_min}")
    print(f"  date range: {out_rows[0][3]}  ..  {out_rows[-1][3]}")

    # ponytail: runnable checks — snowflake anchor + chronological + pre-2011-only 'min'
    assert snowflake_utc(869766994899468288).isoformat().startswith(
        "2017-05-31T04:06"), "snowflake decode drifted (covfefe anchor)"
    assert all(r[3] < "2011" for r in out_rows if r[6] == "min"), "min-conf leaked past 2010"


if __name__ == "__main__":
    main()
