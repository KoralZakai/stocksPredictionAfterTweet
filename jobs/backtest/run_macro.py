"""Mode B (expanded-macro) backtest runner — writes to its OWN manifest.

Never opens the baseline files for writing. Outputs:
  reports/results_macro_v1.json          (scored rows, Mode-B)
  reports/validation_manifest_macro_v1.json

Differences from Mode A (all bundled in alpha.profiles):
  - expanded whitelist  (+TLT/UUP/FXI/GLD; XLE was already in)
  - whitelist-only prompt (its own prompt_template_hash)
  - expanded pre-filter (adds rates/energy/currency/gold/debt tokens to the geo regex)

The stable geo filter already matches china/tariff/sanction/war/peace/opec/deal, so
the expansion adds the genuinely-missing macro tokens only.

Run (local smoke):
  PYTHONPATH=. python jobs/backtest/run_macro.py --limit 10 --smoke
Run (full):
  PYTHONPATH=. python jobs/backtest/run_macro.py --limit 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from alpha.benchmark import forward_returns, validate
from alpha.env import env, load_dotenv
from alpha.profiles import MACRO_EXTRA_RX, MEASUREMENT_WINDOW, PROFILES
from alpha.route import route_decision
from jobs.backtest.entrypoint import _print_manifest, build_manifest
from scripts.nebius_macro_backtest import (
    CORPUS, GEO_RX, GEO_THEMES, _assign_splits, _categorize, _relabel, load_corpus,
)
from sector_mapping.themes import active_themes

OUT_RESULTS = Path("reports/results_macro_v1.json")
OUT_MANIFEST = Path("reports/validation_manifest_macro_v1.json")
PROFILE = PROFILES["macro"]


def macro_candidates(start: datetime, end: datetime, limit: int) -> list[Any]:
    """Stable geo filter OR the new macro tokens. Stride-sampled to `limit` so the
    corpus stays spread across the window rather than clustered on one news week."""
    out = [tw for tw in load_corpus(CORPUS, start, end)
           if (set(active_themes(tw.text)) & GEO_THEMES)
           or GEO_RX.search(tw.text) or MACRO_EXTRA_RX.search(tw.text)]
    out.sort(key=lambda t: t.timestamp_utc)
    if len(out) > limit:
        step = len(out) / limit
        out = [out[int(i * step)] for i in range(limit)]
    return out


def _memo_fwd() -> Any:
    cache: dict[str, Any] = {}

    def f(ticker: str, t0: datetime) -> Any:
        key = f"{ticker}|{t0.date()}"
        if key not in cache:
            cache[key] = forward_returns(ticker, t0)
        return cache[key]
    return f


def score_tweet(tw: Any, pred: dict[str, Any], fwd: Any) -> dict[str, Any]:
    """Classify output + market scoring -> a row in the shared results shape."""
    t0 = tw.timestamp_utc
    instruments = [{"ticker": str(i.get("ticker", "")).upper(),
                    "name": i.get("name", ""), "role": i.get("role", ""),
                    "predicted_direction": str(i.get("predicted_direction", "neutral")).lower()}
                   for i in pred.get("instruments", [])]
    rows, hits, spy = validate(instruments, t0, fwd=fwd)
    for r in rows:                       # results shape uses "predicted"
        r["predicted"] = r.pop("predicted", r.get("predicted", ""))
    category = pred.get("category", "") or pred.get("scenario", "")
    return {
        "date": t0.date().isoformat(), "hour_utc": t0.hour + t0.minute / 60.0,
        "text": tw.text,
        # `scenario` kept for shape-compatibility with the shared results tooling.
        "scenario": category, "category": category,
        "intensity": pred.get("intensity", 5), "summary": pred.get("summary", ""),
        "trump_interpretation": pred.get("trump_interpretation", ""),
        "macro_link": pred.get("macro_link", ""),
        "direction_rationale": pred.get("direction_rationale", ""),
        "rationale": pred.get("rationale", ""),
        # Fixed protocol fact, attached in code — never model-generated.
        "measurement_window": MEASUREMENT_WINDOW,
        "instruments": rows, "hits": hits, "spy_returns": spy,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Mode B (expanded-macro) backtest")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-05-13")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out-dir", default="reports",
                    help="where to write Mode-B outputs (use /data/reports on the bucket)")
    ap.add_argument("--smoke", action="store_true",
                    help="print per-tweet routing (filter/LLM/router) and skip the manifest")
    a = ap.parse_args()
    load_dotenv()
    out_dir = Path(a.out_dir)
    out_results = out_dir / OUT_RESULTS.name
    out_manifest = out_dir / OUT_MANIFEST.name

    api_key = env("NEBIUS_API_KEY", "EXPO_PUBLIC_NEBIUS_API_KEY")
    if not api_key:
        sys.exit("No NEBIUS_API_KEY. Put it in .env (gitignored) or export it.")
    model = env("NEBIUS_MODEL", "EXPO_PUBLIC_NEBIUS_MODEL", default="meta-llama/Llama-3.3-70B-Instruct")
    base = env("NEBIUS_BASE_URL", "EXPO_PUBLIC_NEBIUS_BASE_URL", default="https://api.studio.nebius.ai/v1")

    start = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc) + timedelta(days=1)
    cands = macro_candidates(start, end, a.limit)
    print(f"[macro] profile={PROFILE.name} prompt={PROFILE.prompt_hash()[:12]} "
          f"candidates={len(cands)} model={model}")

    fwd = _memo_fwd()
    results: list[dict[str, Any]] = []
    for i, tw in enumerate(cands, 1):
        try:
            pred = PROFILE.classify(tw.text, base_url=base, api_key=api_key, model=model)
        except Exception as exc:
            print(f"  [{i}/{len(cands)}] classify failed: {exc}")
            continue
        if a.smoke:
            routed = route_decision(pred, whitelist=PROFILE.whitelist)
            tick = ",".join(f"{x.ticker}:{x.direction[:1]}" for x in routed.instruments) or "-"
            off = [str(x.get("ticker", "")).upper() for x in pred.get("instruments", [])
                   if str(x.get("ticker", "")).upper() not in PROFILE.whitelist]
            cat = str(pred.get("category", ""))[:34]
            print(f"  [{i:2}] {routed.decision:8} {tick:22} {cat:34} off-menu={off or '-'}  "
                  f"{tw.text[:40]!r}")
            continue
        results.append(score_tweet(tw, pred, fwd))
        print(f"  [{i}/{len(cands)}] scored  {tw.timestamp_utc.date()}  {tw.text[:40]!r}")

    if a.smoke:
        print("\n[macro] smoke only — no manifest written, baseline untouched.")
        return

    _relabel(results, 0.0)
    for r in results:
        r["tag"] = _categorize(r)
    _assign_splits(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_results.write_text(json.dumps(results, indent=2), encoding="utf-8")

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest = build_manifest(results, alpha=a.alpha, corpus_file=CORPUS, now_utc=now,
                              prompt_hash=PROFILE.prompt_hash(), profile=PROFILE.name)
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    _print_manifest(manifest)
    print(f"\n-> results  -> {out_results}")
    print(f"-> manifest -> {out_manifest}   (baseline validation_manifest.json UNTOUCHED)")


if __name__ == "__main__":
    main()
