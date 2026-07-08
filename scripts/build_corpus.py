"""Build the unified Trump event corpus from Twitter + Truth Social (data strategy).

Inputs:
  data/real/tweets.csv               # Twitter 2017-2021, already snowflake-normalized
  data/real/truth_social_raw.parquet # Truth Social 2022-2026 (read via DuckDB, no pyarrow)
Output:
  data/real/corpus.csv               # unified schema, deduped, reblog/quote flagged

Unified schema (platform-agnostic; the ML pipeline reads only text + timestamp):
  post_id, platform, timestamp_utc, text, is_original, is_quote, is_reblog,
  source_dataset, ts_confidence

Reblog/quote *exclusion* is NOT done here — it's a reversible filter at training
time. This corpus keeps every row, flagged. Dedup (exact text, earliest-wins) IS
done here, and its report is a first-class diagnostic.

ponytail: DuckDB (already a dep) reads the parquet — no pyarrow/fastparquet.
Text mojibake left as-is (cosmetic, ASCII keywords unaffected; §9 audit note).

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/build_corpus.py
"""

from __future__ import annotations

import csv
import re
from datetime import timezone
from pathlib import Path

import duckdb

TWEETS = Path("data/real/tweets.csv")
TRUTH = Path("data/real/truth_social_raw.parquet")
OUT = Path("data/real/corpus.csv")
COLS = ["post_id", "platform", "timestamp_utc", "text",
        "is_original", "is_quote", "is_reblog", "source_dataset", "ts_confidence"]

_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")


def norm_text(t: str) -> str:
    """Normalization used ONLY for dedup keys — the stored text stays verbatim."""
    return _WS.sub(" ", _URL.sub("", t.lower())).strip()


Row = dict[str, str]


def load_twitter() -> list[Row]:
    rows: list[Row] = []
    with TWEETS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            is_rt = r["is_retweet"] == "True"
            rows.append({
                "post_id": r["tweet_id"], "platform": "twitter",
                "timestamp_utc": r["timestamp_utc"], "text": r["text"],
                "is_original": str(not is_rt), "is_quote": "False",
                "is_reblog": str(is_rt), "source_dataset": "twitter_archive",
                # snowflake-exact 'ms', except pre-snowflake 2009-2010 rows ('min')
                "ts_confidence": r.get("ts_confidence", "ms"),
            })
    return rows


def load_truth() -> list[Row]:
    q = f"""SELECT truth_id, created_at_utc, content_text, post_type,
                   is_quote, is_retruth FROM '{TRUTH.as_posix()}'"""
    rows: list[Row] = []
    for tid, ts, text, ptype, is_q, is_rt in duckdb.connect().execute(q).fetchall():
        if ts is None or not text:
            continue
        ts_utc = ts.replace(tzinfo=timezone.utc).isoformat(timespec="microseconds")
        reblog = ptype == "reblog" or bool(is_rt)
        quote = ptype == "quote" or bool(is_q)
        rows.append({
            "post_id": str(tid), "platform": "truth_social",
            "timestamp_utc": ts_utc, "text": str(text),
            "is_original": str(ptype == "original"),
            "is_quote": str(quote), "is_reblog": str(reblog),
            "source_dataset": "truth_social_dataset",
            "ts_confidence": "s",  # dataset field, second precision
        })
    return rows


def main() -> None:
    tw, ts = load_twitter(), load_truth()
    all_rows = tw + ts

    # dedup: exact normalized text, keep earliest timestamp (cross-source too).
    best: dict[str, Row] = {}
    removed = 0
    for r in sorted(all_rows, key=lambda x: x["timestamp_utc"]):
        key = norm_text(r["text"])
        if not key:
            continue
        if key in best:
            removed += 1
            continue
        best[key] = r
    corpus = sorted(best.values(), key=lambda x: x["timestamp_utc"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(corpus)

    def count(rows: list[Row], **cond: str) -> int:
        return sum(all(r[k] == v for k, v in cond.items()) for r in rows)

    print(f"twitter rows: {len(tw)}   truth_social rows: {len(ts)}")
    print(f"exact-text duplicates removed: {removed}")
    print(f"unified corpus: {len(corpus)} rows -> {OUT}")
    print(f"  originals: {count(corpus, is_original='True')}"
          f"   reblogs: {count(corpus, is_reblog='True')}"
          f"   quotes: {count(corpus, is_quote='True')}")
    print(f"  twitter: {count(corpus, platform='twitter')}"
          f"   truth_social: {count(corpus, platform='truth_social')}")
    print(f"  range: {corpus[0]['timestamp_utc'][:10]} .. {corpus[-1]['timestamp_utc'][:10]}"
          f"   (13-month off-platform gap 2021-01..2022-02 is expected)")

    # ponytail: one runnable check — schema width + no dup text keys survive
    assert len(best) == len({norm_text(r["text"]) for r in corpus}), "dup keys leaked"
    assert all(len(r) == len(COLS) for r in corpus), "schema width drift"


if __name__ == "__main__":
    main()
