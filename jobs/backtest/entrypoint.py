"""Nebius Serverless AI Job: `backtest-and-validate` — the batch workload.

Run-to-completion, CPU-only, idempotent. This is a THIN CLI: it reuses the proven
scoring in scripts/nebius_macro_backtest.py (which wraps alpha/) and adds only the
statistical GATE — Wilson CIs, exact binomial p-values, and a Benjamini-Hochberg
correction across the full horizon registry — then writes the two artifacts the
Endpoint consumes:

    reports/macro_dataset.csv          (the per-tweet modelling frame)
    reports/validation_manifest.json   (the CONTRACT: which horizons may be served)

The manifest records `prompt_template_hash` so the Endpoint refuses to boot if its
live classification prompt has drifted from the one that produced these numbers.

REPRODUCIBILITY: intraday horizons (30m/1h) need Alpaca keys. Without them the run
still succeeds — those horizons are marked `requires_alpaca` and excluded from
`shipped_horizons`, so the shipped decision horizon is always daily (EOD) and
regenerable by a stranger with no private keys. See data/PROVENANCE.md.

Modes:
    --from-results   score from the cached predictions JSON (no Nebius, no market
                     fetch; instant, $0). Default when the cache exists.
    --limit N        smoke mode: classify only N tweets live (~$0 for 10).
    (full)           classify every corpus tweet live via Nebius, then score.

Run (gate, $0):
    MSYS_NO_PATHCONV=1 PYTHONPATH=. .venv/Scripts/python.exe jobs/backtest/entrypoint.py --from-results
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from alpha.classify import prompt_template_hash
from alpha.stats import benjamini_hochberg, binom_p_greater, wilson_ci

# The backtest owns the scoring/split logic; we only orchestrate + add stats.
from scripts.nebius_macro_backtest import (
    CORPUS, RESULTS, _assign_splits, _categorize, _relabel, _tweet_hit, write_dataset,
)

SCHEMA_VERSION = "1.0"
METRIC = "beat_spy_signed_band0"
SPLIT = "chronological_60_20_20"
DAILY_HORIZONS = ["EOD", "3d", "1w", "1mo"]      # regenerable from public daily bars
INTRADAY_HORIZONS = ["30m", "1h"]                # need Alpaca (private) -> diagnostic only
MANIFEST = "reports/validation_manifest.json"
DISCLAIMER = "Research output. Not investment advice."


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _sha256_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    return sha256(p.read_bytes()).hexdigest()


def _horizon_stats(test: list[dict[str, Any]], h: str) -> tuple[int, int]:
    """(hits, scored) at horizon h over the test split, aggregated to ONE call per
    tweet (correlated instruments within a tweet are not independent samples)."""
    k = n = 0
    for r in test:
        v = _tweet_hit(r, h)
        if v is not None:
            n += 1
            k += v
    return k, n


def build_manifest(results: list[dict[str, Any]], *, alpha: float, corpus_file: str,
                   now_utc: str) -> dict[str, Any]:
    """Pure manifest assembly from already-scored results. Testable in isolation.

    A horizon is in the registry only if it has scoreable test data; intraday
    horizons that are empty (no Alpaca) are recorded as skipped, not shipped.
    """
    test = [r for r in results if r.get("split") == "test"]

    registry: list[str] = []
    per_h: dict[str, dict[str, Any]] = {}
    for h in DAILY_HORIZONS + INTRADAY_HORIZONS:
        k, n = _horizon_stats(test, h)
        requires_alpaca = h in INTRADAY_HORIZONS
        if n == 0:
            per_h[h] = {"n_test": 0, "skipped": True, "requires_alpaca": requires_alpaca,
                        "reason": "no scoreable test data (Alpaca keys absent?)"
                                  if requires_alpaca else "no scoreable test data"}
            continue
        registry.append(h)
        lo, hi = wilson_ci(k, n)
        per_h[h] = {
            "n_test": n, "hits_test": k, "hit_rate_test": round(k / n, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "p_raw": round(binom_p_greater(k, n, 0.5), 6),
            "requires_alpaca": requires_alpaca,
        }

    # BH across the WHOLE registry (daily + any available intraday).
    praw = [per_h[h]["p_raw"] for h in registry]
    padj = benjamini_hochberg(praw)
    for h, pb in zip(registry, padj):
        per_h[h]["p_bh"] = round(pb, 6)
        per_h[h]["survives_bh"] = bool(pb < alpha)

    # Shipped = survives BH AND regenerable without private keys (daily only).
    shipped = [h for h in registry
               if per_h[h].get("survives_bh") and not per_h[h]["requires_alpaca"]]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": now_utc,
        "code_rev": _git_sha(),
        "corpus": {"file": corpus_file, "sha256": _sha256_file(corpus_file), "n": len(results)},
        "metric": METRIC,
        "split": SPLIT,
        "alpha": alpha,
        "prompt_template_hash": prompt_template_hash(),
        "horizons": per_h,
        "registry": registry,
        "shipped_horizons": shipped,
        "intraday_note": ("Intraday horizons (30m/1h) require Alpaca keys and are "
                          "excluded from shipped_horizons for reproducibility."),
        "disclaimer": DISCLAIMER,
    }


def _print_manifest(m: dict[str, Any]) -> None:
    print("=" * 78)
    print(f"VALIDATION MANIFEST   rev={m['code_rev']}   metric={m['metric']}")
    print(f"corpus={m['corpus']['file']}  n={m['corpus']['n']}  "
          f"sha256={m['corpus']['sha256'][:12]}...")
    print(f"prompt_template_hash={m['prompt_template_hash'][:12]}...  alpha={m['alpha']}")
    print("=" * 78)
    hdr = f"{'horizon':8}{'n_test':>7}{'hit_rate':>10}{'ci95':>18}{'p_raw':>10}{'p_bh':>10}  survives"
    print(hdr)
    print("-" * len(hdr))
    for h in DAILY_HORIZONS + INTRADAY_HORIZONS:
        e = m["horizons"].get(h, {})
        if e.get("skipped"):
            print(f"{h:8}{'--':>7}   (skipped: {e.get('reason', '')})")
            continue
        ci = f"[{e['ci95'][0]:.3f},{e['ci95'][1]:.3f}]"
        mark = "YES" if e.get("survives_bh") else "no"
        alp = " (needs Alpaca)" if e.get("requires_alpaca") else ""
        print(f"{h:8}{e['n_test']:>7}{e['hit_rate_test']:>10.3f}{ci:>18}"
              f"{e['p_raw']:>10.4f}{e.get('p_bh', float('nan')):>10.4f}  {mark}{alp}")
    print("-" * len(hdr))
    print(f"SHIPPED HORIZONS (survive BH, reproducible without private keys): "
          f"{m['shipped_horizons'] or '(none)'}")


def _load_or_run(a: argparse.Namespace) -> list[dict[str, Any]]:
    """Get scored results: from the cache (default/$0) or by a live Nebius run."""
    cache = Path(RESULTS)
    if a.from_results or (cache.exists() and not a.live):
        if not cache.exists():
            sys.exit(f"--from-results needs {RESULTS}; run a live pass first.")
        results: list[dict[str, Any]] = json.loads(cache.read_text())
        return results
    # Live path: reuse the backtest's run() to classify + fetch + score, then reload.
    from datetime import timedelta

    from scripts.nebius_macro_backtest import run
    from alpha.env import _env, _load_dotenv
    _load_dotenv()
    api_key = _env("NEBIUS_API_KEY", "EXPO_PUBLIC_NEBIUS_API_KEY")
    if not api_key:
        sys.exit("No NEBIUS_API_KEY for a live run. Put it in .env or use --from-results.")
    model = _env("NEBIUS_MODEL", "EXPO_PUBLIC_NEBIUS_MODEL",
                 default="meta-llama/Llama-3.3-70B-Instruct")
    base = _env("NEBIUS_BASE_URL", "EXPO_PUBLIC_NEBIUS_BASE_URL",
                default="https://api.studio.nebius.ai/v1")
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc) + timedelta(days=1)
    run(start, end, a.limit, model, base, api_key, top=5)
    live: list[dict[str, Any]] = json.loads(cache.read_text())
    return live


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="backtest-and-validate: emit macro_dataset + manifest")
    ap.add_argument("--from-results", action="store_true",
                    help="score from cached predictions JSON (no Nebius/market; $0)")
    ap.add_argument("--live", action="store_true", help="force a live Nebius classification run")
    ap.add_argument("--limit", type=int, default=10, help="smoke: max tweets in a live run")
    ap.add_argument("--band", type=float, default=0.0, help="beat-SPY noise band (0 = strict)")
    ap.add_argument("--alpha", type=float, default=0.05, help="BH significance level")
    ap.add_argument("--manifest", default=MANIFEST)
    a = ap.parse_args()

    results = _load_or_run(a)
    _relabel(results, a.band)                       # re-score beat-SPY at this band
    for r in results:
        r["tag"] = _categorize(r)
    _assign_splits(results)
    write_dataset(results)                          # -> reports/macro_dataset.csv

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest = build_manifest(results, alpha=a.alpha, corpus_file=CORPUS, now_utc=now)
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(a.manifest).write_text(json.dumps(manifest, indent=2), encoding="ascii")
    _print_manifest(manifest)
    print(f"\n-> manifest -> {a.manifest}")
    print("-> dataset  -> reports/macro_dataset.csv")


if __name__ == "__main__":
    main()
