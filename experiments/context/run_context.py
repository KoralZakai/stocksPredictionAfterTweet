"""System-B contextual backtest runner. EXPERIMENTAL — shipped path untouched.

For each row of a split (DEFAULT val — NOT test), inject the leak-safe macro
context, classify with the System-B dual-horizon prompt, and score EOD beat-SPY
against SPY (daily bars, public/reproducible; 1h needs Alpaca so it is captured
but not scored here). System A is the cached shipped result on the same rows.

Writes ONLY under experiments/context/ (results_b.json, manifest_b.json). It NEVER
touches reports/validation_manifest.json or the shipped prompt. `--split test` is
allowed but gated behind --i-understand-test-is-sacred for the ONE final scoring.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpha.benchmark import forward_returns, validate
from alpha.env import env, load_dotenv
from alpha.stats import binom_p_greater, wilson_ci
from scripts.nebius_macro_backtest import RESULTS, _assign_splits, _tweet_hit

from experiments.context.macro_calendar import context_asof
from experiments.context.prompt_b import classify_b

HERE = Path(__file__).resolve().parent
OUT_RESULTS = HERE / "results_b.json"
OUT_MANIFEST = HERE / "manifest_b.json"


def _t0(row: dict[str, Any]) -> datetime:
    h = float(row.get("hour_utc", 14.0))
    return datetime.fromisoformat(row["date"]).replace(
        hour=int(h), minute=int((h % 1) * 60), tzinfo=timezone.utc)


def _memo_fwd() -> Any:
    cache: dict[tuple[str, str], Any] = {}

    def fwd(ticker: str, t0: datetime) -> Any:
        key = (ticker.upper(), t0.date().isoformat())
        if key not in cache:
            cache[key] = forward_returns(ticker, t0)
        return cache[key]
    return fwd


def _tweet_hit_b(instruments_b: list[dict[str, Any]], t0: datetime, fwd: Any) -> int | None:
    """EOD beat-SPY at tweet level for System B: majority vote of scoreable instruments."""
    rows, _hits, _spy = validate(instruments_b, t0, fwd=fwd)
    votes = [r["hit"]["EOD"] for r in rows if isinstance(r["hit"].get("EOD"), bool)]
    if not votes:
        return None
    return int(sum(votes) / len(votes) >= 0.5)


def _agg(hits: list[int]) -> dict[str, Any]:
    n, k = len(hits), sum(hits)
    lo, hi = wilson_ci(k, n)
    return {"n": n, "hits": k, "accuracy": round(k / n, 4) if n else None,
            "ci95": [round(lo, 4), round(hi, 4)], "p_raw": round(binom_p_greater(k, n), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description="System-B contextual backtest (val by default)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap rows (0 = all)")
    ap.add_argument("--i-understand-test-is-sacred", action="store_true",
                    help="required to score --split test (the ONE final run)")
    ap.add_argument("--model", default=env("NEBIUS_MODEL", "EXPO_PUBLIC_NEBIUS_MODEL",
                                           default="meta-llama/Llama-3.3-70B-Instruct"))
    a = ap.parse_args()
    load_dotenv()
    if a.split == "test" and not a.i_understand_test_is_sacred:
        raise SystemExit("Refusing to score the sacred test split without "
                         "--i-understand-test-is-sacred. Compare A vs B on val.")

    api_key = env("NEBIUS_API_KEY", "EXPO_PUBLIC_NEBIUS_API_KEY")
    base = env("NEBIUS_BASE_URL", "EXPO_PUBLIC_NEBIUS_BASE_URL",
               default="https://api.studio.nebius.ai/v1")
    if not api_key:
        raise SystemExit("No NEBIUS_API_KEY (.env) for the live System-B run.")

    data: list[dict[str, Any]] = json.loads((Path.cwd() / RESULTS).read_text())
    _assign_splits(data)
    rows = [r for r in data if r.get("split") == a.split]
    if a.limit:
        rows = rows[:a.limit]
    print(f"[context] System-B run on split={a.split}, {len(rows)} rows, model={a.model}")

    fwd = _memo_fwd()
    out: list[dict[str, Any]] = []
    a_hits: list[int] = []
    b_hits: list[int] = []
    for i, r in enumerate(rows, 1):
        t0 = _t0(r)
        ctx = context_asof(t0)
        try:
            pred_b = classify_b(r["text"], ctx, base_url=base, api_key=api_key, model=a.model)
        except Exception as exc:                       # a bad row must not kill the run
            print(f"  [{i}/{len(rows)}] classify_b failed: {exc}")
            continue
        instruments_b = [{"ticker": ins.get("ticker", ""),
                          "predicted_direction": str(ins.get("direction_eod", "neutral")).lower()}
                         for ins in pred_b.get("instruments", [])]
        b_hit = _tweet_hit_b(instruments_b, t0, fwd)
        a_hit = _tweet_hit(r, "EOD")                   # cached System-A EOD beat-SPY
        rec = {"text": r["text"][:120], "split": a.split, "macro_context": ctx,
               "a_hit_eod": a_hit, "b_hit_eod": b_hit,
               "a_rationale": r.get("rationale", ""), "b_rationale": pred_b.get("rationale", ""),
               "b_scenario": pred_b.get("scenario", ""), "b_instruments": pred_b.get("instruments", [])}
        out.append(rec)
        if a_hit is not None:
            a_hits.append(a_hit)
        if b_hit is not None:
            b_hits.append(b_hit)
        print(f"  [{i}/{len(rows)}] A={a_hit} B={b_hit}  ctx={'yes' if ctx else 'none'}")

    manifest_b = {
        "experiment": "system-b-contextual-dual-horizon",
        "split": a.split, "horizon": "EOD", "note_1h": "1h needs Alpaca; captured, not scored here",
        "system_a": _agg(a_hits), "system_b": _agg(b_hits),
        "shipped_manifest_untouched": True,
    }
    OUT_RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    OUT_MANIFEST.write_text(json.dumps(manifest_b, indent=2), encoding="utf-8")
    print(f"\n[context] A EOD beat-SPY: {manifest_b['system_a']}")
    print(f"[context] B EOD beat-SPY: {manifest_b['system_b']}")
    print(f"[context] -> {OUT_RESULTS.name}, {OUT_MANIFEST.name}")


if __name__ == "__main__":
    main()
