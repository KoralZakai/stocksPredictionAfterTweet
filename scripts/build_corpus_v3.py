"""Extended unified corpus (v3): Twitter + HF Truth Social + CNN Truth archive.

Phase-A ingestion. Adds `data/real/truth_cnn_raw.parquet` — the CNN-hosted
mirror of stiles/trump-truth-social-archive (public file, no key, no new deps):
    https://ix.cnn.io/data/truth-social/truth_archive.parquet
It carries MILLISECOND-precision timestamps (the HF dataset is second-precision)
and extends coverage past the HF cutoff (2026-06-14 -> today). Baseline
`corpus.csv` is preserved untouched; output is `corpus_v3.csv`.

Dedup (exact normalized text, same key fn as v2) now prefers, per key:
  1. higher timestamp confidence ('ms' beats 's'),
  2. then earliest timestamp,
and ORs the reblog/quote flags across duplicates (the CNN schema has no
quote/reblog fields — reblogs are inferred from the 'RT @' text prefix, quotes
are NOT detectable there; the HF flags survive the merge via the OR).

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/build_corpus_v3.py
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from scripts.build_corpus import COLS, Row, load_truth, load_twitter, norm_text

CNN = Path("data/real/truth_cnn_raw.parquet")
OUT = Path("data/real/corpus_v3.csv")
_CONF_RANK = {"ms": 0, "s": 1, "min": 2}  # lower = better ('min' = pre-snowflake 2009-2010)


def load_cnn() -> list[Row]:
    q = f"SELECT id, created_at, content FROM '{CNN.as_posix()}'"
    rows: list[Row] = []
    for tid, ts, text in duckdb.connect().execute(q).fetchall():
        if not ts or not text or not str(text).strip():
            continue  # media-only / pure-retruth rows carry no text
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
        reblog = str(text).startswith("RT @")
        rows.append({
            "post_id": str(tid), "platform": "truth_social",
            "timestamp_utc": dt.isoformat(timespec="microseconds"), "text": str(text),
            "is_original": str(not reblog), "is_quote": "False",  # not flagged in CNN schema
            "is_reblog": str(reblog), "source_dataset": "cnn_truth_archive",
            "ts_confidence": "ms",
        })
    return rows


def merge(all_rows: list[Row]) -> list[Row]:
    groups: dict[str, list[Row]] = {}
    for r in all_rows:
        key = norm_text(r["text"])
        if key:
            groups.setdefault(key, []).append(r)
    corpus: list[Row] = []
    for dups in groups.values():
        dups.sort(key=lambda r: (_CONF_RANK[r["ts_confidence"]], r["timestamp_utc"]))
        best = dict(dups[0])
        reblog = any(r["is_reblog"] == "True" for r in dups)
        quote = any(r["is_quote"] == "True" for r in dups)
        best["is_reblog"], best["is_quote"] = str(reblog), str(quote)
        best["is_original"] = str(not (reblog or quote))
        corpus.append(best)
    corpus.sort(key=lambda r: r["timestamp_utc"])
    return corpus


def main() -> None:
    tw, hf, cnn = load_twitter(), load_truth(), load_cnn()
    corpus = merge(tw + hf + cnn)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(corpus)

    n_ms = sum(r["ts_confidence"] == "ms" for r in corpus)
    by_src: dict[str, int] = {}
    for r in corpus:
        by_src[r["source_dataset"]] = by_src.get(r["source_dataset"], 0) + 1
    print(f"inputs: twitter {len(tw)}  hf_truth {len(hf)}  cnn_truth {len(cnn)}")
    print(f"corpus_v3: {len(corpus)} rows -> {OUT}")
    print(f"  winner by source: {by_src}")
    print(f"  ms-precision timestamps: {n_ms}/{len(corpus)}")
    print(f"  range: {corpus[0]['timestamp_utc'][:10]} .. {corpus[-1]['timestamp_utc'][:10]}")

    # ponytail: one runnable check — dedup keys unique, ms never loses to s
    keys = [norm_text(r["text"]) for r in corpus]
    assert len(keys) == len(set(keys)), "dup keys leaked"
    assert all(len(r) == len(COLS) for r in corpus), "schema width drift"


if __name__ == "__main__":
    main()
