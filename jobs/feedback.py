"""Nebius Serverless AI Job: `feedback` — closes the loop.

    /predict -> prediction_log.jsonl (bucket) -> THIS JOB -> prospective_replication.json

Reads what the Endpoint actually served, waits for each call's horizon to CLOSE, labels
it with the SAME functions the backtest uses, and scores the accumulated live calls
against the SAME pre-registered registry. Run-to-completion, CPU-only, idempotent.

This is the honest version of "feed the predictions back in". It is worth being precise
about why, because the naive version of this job is the single fastest way to destroy
everything this project has established.

WHAT THIS JOB DOES NOT DO, AND WHY
----------------------------------
1. It does NOT merge live calls into the training corpus and retrain.
   The Endpoint sees whatever tweets a CALLER chose to send it. That is not a sample of
   Trump's posts; it is a sample of what somebody found interesting enough to ask about
   — which correlates with exactly the big market moves we are trying to test against.
   Training on it would manufacture the signal the whole study concluded is absent.
   Live calls therefore form a REPLICATION set, never a training set.

2. It does NOT re-run the gate until something passes.
   Re-scoring a growing dataset and shipping the first time p < alpha is optional
   stopping: it reaches "significance" with probability 1 given enough looks, on pure
   noise. So every look is counted (`n_looks`) and the BH correction is applied over
   registry x looks. The correction gets STRICTER as the log grows. That is the point:
   you must not be able to wait your way to a finding.

3. It does NOT auto-ship a horizon.
   `shipped_horizons` stays a human decision made against the pre-registered schedule.
   This job writes a report; it never writes the manifest the Endpoint boots from.

WHAT IT IS FOR: a prospective, out-of-sample, pre-registered replication — genuinely
stronger evidence than the retrospective study, precisely because the calls were made
and logged BEFORE the outcome existed. If the null is real, this is where it keeps
being real, on data nobody could have fitted to.

NO SECOND SCORING PATH (CLAUDE.md 3.2): returns come from alpha.benchmark.forward_returns,
hits from _relabel/relative_hit, tweet verdicts from _tweet_hit — the exact
functions the backtest calls. This job only marshals and counts.

Run (offline, no network needed if --no-fetch):
    MSYS_NO_PATHCONV=1 PYTHONPATH=. .venv/Scripts/python.exe jobs/feedback.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpha.stats import benjamini_hochberg, binom_p_greater, wilson_ci
from serving.observe import log_path

OUT = Path("reports/prospective_replication.json")
# Daily-only. 30m/1h need Alpaca keys, so a stranger re-running this job cannot
# reproduce them (see data/PROVENANCE.md) — they stay out of the shippable registry.
DAILY_HORIZONS = ("EOD", "3d", "1w", "1mo")


def load_log(path: Path) -> list[dict[str, Any]]:
    """Read the append-only log, newest-wins dedupe by (tweet, t0).

    The dedupe is load-bearing, not hygiene. A caller replaying the same post 50 times
    would otherwise become 50 'independent' observations of one event and crush every
    p-value — the same-bar duplicate artifact that fabricated 9 intraday cells at
    p=0.0018, arriving through a new door.
    """
    if not path.exists():
        return []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue                      # a torn last line from a killed container
        seen[(r.get("tweet_sha256", ""), r.get("t0_utc", ""))] = r
    return list(seen.values())


def mature(entries: list[dict[str, Any]], fetch: bool = True) -> list[dict[str, Any]]:
    """Attach realized returns to each logged call, for the horizons that have CLOSED.

    Maturity needs no calendar arithmetic: forward_returns only returns a horizon whose
    bar EXISTS. A call made an hour ago yields nothing at 1mo and is simply not scored
    at 1mo yet — it stays in the log and matures on a later run. That is why the job is
    idempotent and safe to run on a schedule.
    """
    if not fetch:
        return []
    from alpha.benchmark import forward_returns
    from scripts.nebius_macro_backtest import _relabel

    rows: list[dict[str, Any]] = []
    for e in entries:
        t0 = _parse(e.get("t0_utc", ""))
        if t0 is None or not e.get("instruments"):
            continue
        spy = forward_returns("SPY", t0)
        if not spy:
            continue                      # nothing has closed yet for this t0
        ins = []
        for i in e["instruments"]:
            tk, pred = i.get("ticker"), i.get("predicted")
            if not tk or pred not in ("up", "down"):
                continue
            ret = forward_returns(tk, t0)
            if ret:
                ins.append({"ticker": tk, "predicted": pred, "returns": ret})
        if ins:
            rows.append({"text": e.get("tweet_text", ""), "date": e.get("t0_utc", "")[:10],
                         "spy_returns": spy, "instruments": ins,
                         "manifest_version": e.get("manifest_version")})
    _relabel(rows)                       # sets ins["hit"][h] — the SAME scorer as the backtest
    return rows


def _parse(s: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


def score(rows: list[dict[str, Any]], n_looks: int) -> dict[str, Any]:
    """Per-horizon replication scorecard, corrected for how many times we have looked.

    BH denominator = len(registry) * n_looks. Crude (Bonferroni-flavoured over looks)
    and deliberately conservative: it means the bar RISES every time this job runs, so
    re-running cannot be a strategy for eventually passing.
    """
    from scripts.nebius_macro_backtest import _tweet_hit

    cells: list[dict[str, Any]] = []
    pvals: list[float] = []
    for h in DAILY_HORIZONS:
        verdicts = [v for v in (_tweet_hit(r, h) for r in rows) if v is not None]
        n, k = len(verdicts), sum(verdicts)
        p = binom_p_greater(k, n) if n else 1.0
        lo, hi = wilson_ci(k, n)
        cells.append({"horizon": h, "n": n, "hits": k,
                      "hit_rate": round(k / n, 4) if n else None,
                      "ci95": [round(lo, 4), round(hi, 4)], "p_raw": round(p, 6)})
        pvals.append(p)
    # Pad the BH input to registry x looks so the correction knows how many chances
    # we have actually given ourselves. Unused slots are p=1.0 (no free significance).
    padded = pvals + [1.0] * (len(DAILY_HORIZONS) * max(n_looks - 1, 0))
    adj = benjamini_hochberg(padded)[:len(cells)]
    for c, a in zip(cells, adj, strict=True):
        c["p_bh"] = round(a, 6)
        c["survives_bh"] = bool(a < 0.05 and (c["hit_rate"] or 0) > 0.5)
    return {"cells": cells, "n_looks": n_looks,
            "bh_denominator": len(DAILY_HORIZONS) * max(n_looks, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the Endpoint's served calls, offline.")
    ap.add_argument("--log", default=str(log_path()))
    ap.add_argument("--no-fetch", action="store_true",
                    help="parse + dedupe only; do not resolve prices (offline smoke)")
    a = ap.parse_args()

    entries = load_log(Path(a.log))
    prior = json.loads(OUT.read_text()) if OUT.exists() else {}
    n_looks = int(prior.get("n_looks", 0)) + 1

    rows = mature(entries, fetch=not a.no_fetch)
    res = score(rows, n_looks)
    res.update({
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "n_logged": len(entries), "n_matured": len(rows),
        "log": a.log,
        "is_training_data": False,
        "note": ("PROSPECTIVE REPLICATION over calls the Endpoint actually served. NOT "
                 "training data: callers choose which tweets to send, so this set is "
                 "selected, not sampled. NOT a licence to ship: shipped_horizons stays a "
                 "human decision. p_bh is corrected over registry x n_looks, so the bar "
                 "rises every run — you cannot wait your way to a finding."),
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"[feedback] logged={len(entries)} matured={len(rows)} look #{n_looks} "
          f"(BH denominator {res['bh_denominator']})")
    for c in res["cells"]:
        r = f"{c['hit_rate']:.3f}" if c["hit_rate"] is not None else "  -  "
        print(f"[feedback]   {c['horizon']:>4}  n={c['n']:<4} hit={r}  "
              f"p_raw={c['p_raw']:.3f}  p_bh={c['p_bh']:.3f}  "
              f"{'SURVIVES' if c['survives_bh'] else 'no'}")
    print(f"[feedback] -> {OUT}")


if __name__ == "__main__":
    main()
