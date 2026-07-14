"""EDA: does a real signal exist? Real-Claude sentiment on prominent tech tweets.

Pre-training reality check (NOT a training step). Picks tweets that DIRECTLY name
a major tech stock (INTC, NVDA, AMD, AAPL, MSFT, TSM, AVGO, TSLA, AMZN), asks the
REAL Claude extractor whether the text is bullish/bearish (falls back to the
keyword heuristic only if ANTHROPIC_API_KEY is unset — and says so loudly), then
prints the stock's abnormal return vs THREE averaged benchmarks (indices SPY/QQQ/
DIA, sector ETF, peer median) across 30m -> 1mo, and renders the Phase-B SVG cards
for exactly these events.

Cheap + cached: signals are content-addressed in runs/real/llm_signals.json, so a
rerun costs zero API calls. A handful of tweets => a handful of Claude calls.

Run (real Claude):
  export ANTHROPIC_API_KEY=sk-ant-...            # bash   (see README / step 1)
  PYTHONPATH=. .venv/Scripts/python.exe scripts/eda_tech_tweets.py
Options:
  --entities INTC NVDA AMD ...   # override the tech set
  --limit 14                     # max events (deduped by entity+date)
  --model claude-haiku-4-5       # cheaper than the opus default
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

from config.membership import benchmarks_for, name_of
from data.sources.local import load_corpus
from labeling.benchmarks import compute_bench_outcome
from llm.cache import SignalCache
from llm.extract import DEFAULT_MODEL, HeuristicExtractor, default_extractor
from reportgen.tweet_chart import render_page
from scripts.build_multibench import (
    CORPUS, HORIZONS, SUF, T0, T1, _intraday_index, _load_bars,
)
from sector_mapping.entities import entity_matches


def _load_dotenv(path: str = ".env") -> None:
    """Minimal stdlib .env loader (same pattern as scripts/llm_sample_check.py).

    Lets ANTHROPIC_API_KEY live in the gitignored .env so the real-Claude path
    activates with no per-session export. `setdefault` -> a shell export wins.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_load_dotenv()

INTRADAY = "data/real/intraday_reactions.csv"
OUT_HTML = "reports/eda_tech_cards.html"
OUT_CSV = "reports/eda_tech_events.csv"

TECH = ["INTC", "NVDA", "AMD", "AAPL", "MSFT", "TSM", "AVGO", "TSLA", "AMZN"]
_STANCE = {"bullish": "positive", "bearish": "negative", "neutral": "neutral"}

# display columns: intraday (sector-only) then daily. (label key, header)
COLS = [("30m", "30m"), ("1h", "1h"), ("eod", "EOD"),
        ("3d", "3d"), ("1w", "1w"), ("2w", "2w"), ("1mo", "1mo")]


def _pct(v: float | None) -> str:
    return "  n/a  " if v is None else f"{v * 100:+6.2f}%"


def _pick(entities: list[str], limit: int, per_entity: int) -> list:
    """Direct-mention tweets per ticker, deduped by (entity, date).

    Caps each requested ticker at `per_entity` events so the sample doesn't skew
    to whichever name Trump tweeted about earliest/most (INTC would otherwise get
    one slot behind a wall of NVDA). Within a ticker we keep the LATEST events —
    the notable 2025 INTC posts (CEO meeting / resignation call) are mid-2025, not
    January. Overall output still truncated to `limit`.
    """
    want = set(entities)
    tweets = load_corpus(CORPUS, T0, T1)
    seen: set[tuple[str, str]] = set()
    by_tk: dict[str, list] = {tk: [] for tk in entities}
    for tw in tweets:
        for tk, m in entity_matches(tw.text).items():
            if m.tier != "direct" or tk not in want:
                continue
            key = (tk, tw.timestamp_utc.date().isoformat())
            if key in seen:
                continue
            seen.add(key)
            by_tk[tk].append((tw, tk, m))
    out = []
    for tk in entities:
        evs = sorted(by_tk[tk], key=lambda x: x[0].timestamp_utc)
        out.extend(evs[-per_entity:])  # latest per_entity for this ticker
    out.sort(key=lambda x: x[0].timestamp_utc)
    return out[:limit]


def run(entities: list[str], limit: int, model: str, per_entity: int,
        allow_heuristic: bool = False) -> None:
    # Windows consoles default to cp1252; tweets carry emoji/smart quotes.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    extractor = default_extractor(model)
    live = not isinstance(extractor, HeuristicExtractor)

    # This is an EDA pass to judge REAL sentiment vs drift — the keyword heuristic
    # would defeat the point. Refuse it unless explicitly opted-in.
    if not live and not allow_heuristic:
        sys.exit(
            "ERROR: no ANTHROPIC_API_KEY -> would fall back to the keyword heuristic,\n"
            "which is exactly the coarse sentiment this EDA is meant to avoid.\n"
            "Set the key so the real Claude extractor runs:\n"
            "  - persistent: add  ANTHROPIC_API_KEY=sk-ant-...  to the gitignored .env\n"
            "  - one-off:    PowerShell  $env:ANTHROPIC_API_KEY = \"sk-ant-...\"\n"
            "               bash        export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Or pass --allow-heuristic for an offline dry run (coarse sentiment)."
        )

    bars, cal = _load_bars()
    intraday = _intraday_index(INTRADAY)
    cache = SignalCache("runs/real/llm_signals.json")

    print("=" * 96)
    print(f"EDA: real-Claude sentiment vs multi-benchmark drift  |  extractor = "
          f"{'Claude ' + model if live else 'HEURISTIC (--allow-heuristic; coarse sentiment)'}")
    print("=" * 96)
    if not live:
        print("!! Running the keyword heuristic (opted in). Numbers are real; only the "
              "sentiment is coarse.\n")

    rows: list[dict] = []
    for tw, tk, m in _pick(entities, limit, per_entity):
        sig = cache.get(tw.tweet_id, tw.text, model)
        if sig is None:
            try:
                sig = extractor.extract(tw.text)
            except Exception as e:  # noqa: BLE001 — surface API/billing errors cleanly, not as a traceback
                cache.save()  # keep any signals extracted before the failure
                sys.exit(
                    f"\nLLM extraction failed: {type(e).__name__}: {e}\n"
                    "Common cause: an out-of-credit or invalid ANTHROPIC_API_KEY. "
                    "Check console.anthropic.com -> Plans & Billing, then rerun "
                    "(cached events are kept, so you only pay for the rest)."
                )
            cache.put(tw.tweet_id, tw.text, model, sig)
        stance = _STANCE.get(sig.direction_of_intent, "neutral")

        o = compute_bench_outcome(tw.timestamp_utc, tk, bars, cal, HORIZONS)
        if o is None:
            continue
        bset = benchmarks_for(tk)
        row: dict[str, object] = {
            "post_id": tw.tweet_id, "tweet_date": tw.timestamp_utc.date().isoformat(),
            "entity": tk, "entity_name": name_of(tk), "stance": stance,
            "indices_used": " ".join(bset.indices), "sectors_used": " ".join(bset.sectors),
            "peers_used": " ".join(bset.peers), "used_fallback": int(bset.used_fallback),
            "text": tw.text[:200],
        }
        for h in HORIZONS:
            s = SUF[h]
            row[f"raw_{s}"] = _r(o.raw[h])
            row[f"abn_idx_{s}"] = _r(o.abn_index[h])
            row[f"abn_sec_{s}"] = _r(o.abn_sector[h])
            row[f"abn_peer_{s}"] = _r(o.abn_peer[h])
            row[f"band_{s}"] = _r(o.band[h])
            row[f"label_{s}"] = o.label(stance, h)
        ir = intraday.get((tw.tweet_id, tk))
        for win in ("30m", "1h"):
            abn = None
            if ir is not None and f"abn_{win}" in ir and not pd.isna(ir[f"abn_{win}"]):
                abn = float(ir[f"abn_{win}"])
            row[f"abn_sec_{win}"] = _r(abn)
            row[f"label_{win}"] = ""  # coarse intraday label omitted in EDA view
        rows.append(row)
        _print_event(row, sig)

    if not rows:
        print("no direct-mention events found for", entities)
        return

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8")
    run_id = hashlib.sha256("".join(r["post_id"] for r in rows).encode()).hexdigest()[:12]  # type: ignore[arg-type]
    Path(OUT_HTML).write_text(
        render_page(rows, run_id=run_id, title="EDA: prominent tech tweets -> multi-benchmark reactions"),
        encoding="ascii")
    print("=" * 96)
    print(f"{len(rows)} events -> {OUT_CSV}  +  {OUT_HTML}  (open the HTML for the SVG cards)")
    print("Read: outperformance needs the stock to beat ALL THREE benchmark columns "
          "beyond its noise band, and to persist across horizons — not one lucky cell.")


def _print_event(row: dict, sig) -> None:
    st = str(row["stance"]).upper()
    print(f"\n{'-' * 96}")
    print(f"{row['entity']} ({row['entity_name']})  {row['tweet_date']}   "
          f"LLM sentiment: {st}   [event={sig.event_type}, mag={sig.magnitude}, cert={sig.certainty}]")
    print(f'  "{str(row["text"])[:150]}"')
    print(f"  benchmarks -> indices: {row['indices_used'] or '(none)'} | "
          f"sector: {row['sectors_used'] or '(none)'} | peers: {row['peers_used'] or '(none)'}")
    hdr = "  " + f"{'':16}" + "".join(f"{h:>9}" for _, h in COLS)
    print(hdr)
    for label, keyfn in (("stock raw", "raw"), ("vs indices", "abn_idx"),
                         ("vs sector", "abn_sec"), ("vs peers", "abn_peer")):
        cells = []
        for key, _ in COLS:
            if key in ("30m", "1h"):
                # intraday: only sector-relative available
                v = row.get(f"abn_sec_{key}") if keyfn in ("abn_sec",) else None
            else:
                v = row.get(f"{keyfn}_{key}")
            cells.append(_pct(v if isinstance(v, (int, float)) else None))
        print(f"  {label:16}" + "".join(f"{c:>9}" for c in cells))
    verdicts = "  ".join(f"{h}:{row.get(f'label_{k}', '')}" for k, h in COLS if row.get(f"label_{k}"))
    print(f"  folded verdict:  {verdicts}")


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", nargs="+", default=TECH)
    ap.add_argument("--limit", type=int, default=27)
    ap.add_argument("--per-entity", type=int, default=3,
                    help="max events per ticker (keeps INTC from being crowded out)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--allow-heuristic", action="store_true",
                    help="offline dry run with the keyword extractor (no API key)")
    a = ap.parse_args()
    run(a.entities, a.limit, a.model, a.per_entity, a.allow_heuristic)


if __name__ == "__main__":
    main()
