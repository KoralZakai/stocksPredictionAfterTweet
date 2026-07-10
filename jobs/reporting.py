"""Nebius Job: reporting (§13, step 7) — regenerate the eden dashboard.

Thin CLI, zero science: read the event dataset, top up the signal cache for any
post it has not seen, call reporting.build_report() + render_html(), write HTML.

New data flows through automatically: drop new rows into the events CSV, rerun.
Posts already in the cache are NOT re-extracted (content-addressed), so a rerun
after appending a week of posts costs only the new ones.

Run:  python jobs/reporting.py \
        --events data/real/stock_event_dataset.csv \
        --signals runs/real/llm_signals.json \
        --out reports/eden_dashboard.html
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from llm.cache import SignalCache
from llm.extract import DEFAULT_MODEL, HeuristicExtractor, default_extractor
from llm.schema import SCHEMA_VERSION, TweetSignal
from reportgen.render import render_html
from reportgen.stats import build_report


def _run_id(events: Path, model: str) -> str:
    """Deterministic stamp: same inputs -> same id (§12 reproducibility)."""
    h = hashlib.sha256()
    h.update(events.read_bytes())
    h.update(f"|{model}|{SCHEMA_VERSION}".encode())
    return h.hexdigest()[:12]


def run(events_path: str, signals_path: str, out_path: str, model: str | None = None) -> None:
    events = Path(events_path)
    d = pd.read_csv(events, low_memory=False)
    d["post_id"] = d["post_id"].astype(str)

    extractor = default_extractor(model)
    model = model or DEFAULT_MODEL
    label = ("HeuristicExtractor (deterministic offline fallback -- no API key; Claude did not run)"
             if isinstance(extractor, HeuristicExtractor) else f"Claude {model}")

    cache = SignalCache(signals_path)
    texts = d.drop_duplicates("post_id").set_index("post_id")["text"].astype(str)
    signals: dict[str, TweetSignal] = {}
    fresh = 0
    for post_id, text in texts.items():
        pid = str(post_id)
        s = cache.get(pid, text, model)
        if s is None:
            s = extractor.extract(text)
            cache.put(pid, text, model, s)
            fresh += 1
        signals[pid] = s
    cache.save()

    report = build_report(d, signals, label)
    rid = _run_id(events, model)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(report, rid), encoding="ascii")

    print(f"signals: {fresh} new / {len(signals)} total ({label})")
    print(f"report:  {report.n_posts:,} posts, {report.n_rows:,} rows, run {rid} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/real/stock_event_dataset.csv")
    ap.add_argument("--signals", default="runs/real/llm_signals.json")
    ap.add_argument("--out", default="reports/eden_dashboard.html")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    run(a.events, a.signals, a.out, a.model)


if __name__ == "__main__":
    main()
