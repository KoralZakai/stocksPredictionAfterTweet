"""Pre-registered event study runner: registry -> score -> permutation null -> BH
-> REPORT.md. EXPERIMENTAL — touches nothing in the shipped path.

Discipline (the whole point):
  1. The FULL grid of cells is enumerated and written to registry.json BEFORE any
     scoring; the scorer refuses a cell not in the registry.
  2. Permutation null per cell: the same statistic on N_PERM random event dates
     drawn from the SAME asset's tradable sessions in the corpus window. An event
     is excluded only for missing data, never for its outcome.
  3. ONE Benjamini-Hochberg pass across the entire registry (both families +
     volume). The BH denominator is every question we asked.
  4. Overlapping events are scored but ALSO reported separately.

Run:  PYTHONPATH=. python experiments/event_study/run_study.py [--results PATH]
      (offline: cached LLM tags + committed daily bars; no API, no network)
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpha.stats import benjamini_hochberg

from experiments.event_study.cohorts import EVENT_WINDOWS, FAMILIES, tag_cohort
from experiments.event_study.engine import EventResult, load_bars, study_event

HERE = Path(__file__).resolve().parent
N_PERM = 1000
SEED = 20260715
CORPUS_LO, CORPUS_HI = "2025-01-01", "2026-05-12"     # event-date span of the corpus


def _t0(row: dict[str, Any]) -> datetime:
    h = float(row.get("hour_utc", 14.0))
    return datetime.fromisoformat(row["date"]).replace(
        hour=int(h), minute=int((h % 1) * 60), tzinfo=timezone.utc)


def build_registry(events: dict[str, dict[str, list[Any]]]) -> list[dict[str, str | int]]:
    """Every (cohort, asset, window, family) cell + volume cells — BEFORE scoring."""
    cells: list[dict[str, str | int]] = []
    for cohort, per_asset in sorted(events.items()):
        for asset in sorted(per_asset):
            for w in EVENT_WINDOWS:
                for fam in FAMILIES:
                    cells.append({"cohort": cohort, "asset": asset, "window": w, "family": fam})
                cells.append({"cohort": cohort, "asset": asset, "window": w, "family": "volume"})
    return cells


def _stat(vals: list[float], family: str) -> float:
    if family == "abs":
        return sum(abs(v) for v in vals) / len(vals)
    return sum(vals) / len(vals)                       # signed drift / volume ratio


def _perm_pool(bars_of_asset: list[Any]) -> list[datetime]:
    """Random-anchor pool: every tradable session of the asset inside the corpus
    window, as a 14:00 UTC pseudo-t0 (outcome-blind by construction)."""
    return [datetime.fromisoformat(b.date).replace(hour=14, tzinfo=timezone.utc)
            for b in bars_of_asset if CORPUS_LO <= b.date <= CORPUS_HI]


def main() -> None:
    ap = argparse.ArgumentParser(description="pre-registered event study (offline)")
    ap.add_argument("--results", default="reports/nebius_backtest_results.json")
    ap.add_argument("--label", default="discovery-476")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    a = ap.parse_args()
    rng = random.Random(SEED)

    rows = json.loads(Path(a.results).read_text())
    bars = load_bars()

    # ---- 1. tag cohorts (text-only) and collect events per (cohort, asset)
    events: dict[str, dict[str, list[datetime]]] = {}
    texts: list[tuple[str, str, str]] = []            # (cohort, iso_t0, text) for case studies
    for r in rows:
        cohort, assets = tag_cohort(r)
        t0 = _t0(r)
        texts.append((cohort, t0.isoformat(), r.get("text", "")[:200]))
        for asset in assets:
            events.setdefault(cohort, {}).setdefault(asset.upper(), []).append(t0)

    # ---- 2. registry FIRST
    registry = build_registry(events)
    (HERE / "registry.json").write_text(json.dumps(
        {"label": a.label, "seed": SEED, "n_perm": a.n_perm, "cells": registry}, indent=2))
    registered = {(c["cohort"], c["asset"], c["window"], c["family"]) for c in registry}
    print(f"[study] registry: {len(registry)} cells (written before scoring)")

    # ---- 3. score real events. DEDUPE to unique event-days per (cohort, asset):
    # several tweets on one day share one s0 and would double-count the identical
    # CAR, silently shrinking the apparent variance (a cousin of the tie bug).
    scored: dict[tuple[str, str], list[EventResult]] = {}
    prev_s0: dict[str, dict[str, int]] = {}
    n_dropped = n_dup = 0
    for cohort, per_asset in events.items():
        for asset, t0s in per_asset.items():
            track = prev_s0.setdefault(cohort, {})
            res = []
            seen_s0: set[str] = set()
            for t0 in sorted(t0s):
                er = study_event(bars, asset, t0, EVENT_WINDOWS, prev_s0=track)
                if er is None:
                    n_dropped += 1                     # missing data ONLY — never outcome
                    continue
                if er.s0_date in seen_s0:
                    n_dup += 1                         # same event-day, same CAR
                    continue
                seen_s0.add(er.s0_date)
                res.append(er)
            scored[(cohort, asset)] = res
    print(f"[study] events scored; dropped-for-missing-data={n_dropped}, same-day dupes={n_dup}")

    # ---- 4a. precompute the null pool ONCE per asset (every tradable session in
    # the corpus window scored as a pseudo-event) — permutation draws then sample
    # these cached values instead of re-running the engine millions of times.
    pool_cache: dict[str, list[EventResult]] = {}
    for asset in sorted({asset for (_c, assets) in
                         ((c, per) for c, per in events.items()) for asset in assets}):
        cached = []
        for t0 in _perm_pool(bars.get(asset, [])):
            er = study_event(bars, asset, t0, EVENT_WINDOWS)
            if er is not None:
                cached.append(er)
        pool_cache[asset] = cached
    print("[study] null pools: " +
          ", ".join(f"{k}={len(v)}" for k, v in sorted(pool_cache.items())))

    # ---- 4b. per-cell stat + permutation null
    out_cells: list[dict[str, Any]] = []
    pvals: list[float] = []
    for cell in registry:
        cohort, asset, w, fam = cell["cohort"], cell["asset"], int(cell["window"]), cell["family"]
        assert (cohort, asset, w, fam) in registered   # scorer refuses unregistered cells
        res = scored.get((cohort, str(asset)), [])
        vals = ([r.abn_volume[w] for r in res] if fam == "volume"
                else [r.car[w] for r in res])
        if len(vals) < 5:
            out_cells.append({**cell, "n": len(vals), "skipped": "n<5"})
            continue
        obs = _stat(vals, str(fam))
        pool_vals = [(r.abn_volume[w] if fam == "volume" else r.car[w])
                     for r in pool_cache.get(str(asset), [])]
        if len(pool_vals) < 30:
            out_cells.append({**cell, "n": len(vals), "skipped": "pool too small"})
            continue
        # Null WITH replacement. Geo tweets cover ~83% of sessions, so sampling
        # n~=pool WITHOUT replacement collapses the null's variance (finite-
        # population effect) and floors every p — a manufactured result. With
        # replacement the null keeps its honest sigma^2/n.
        null: list[float] = []
        for _ in range(a.n_perm):
            null.append(_stat(rng.choices(pool_vals, k=len(vals)), str(fam)))
        if fam == "signed":
            p = (sum(1 for x in null if abs(x) >= abs(obs)) + 1) / (len(null) + 1)
        else:                                          # abs / volume: one-sided greater
            p = (sum(1 for x in null if x >= obs) + 1) / (len(null) + 1)
        n_overlap = sum(1 for r in res if r.overlapping)
        out_cells.append({**cell, "n": len(vals), "n_overlapping": n_overlap,
                          "observed": round(obs, 5),
                          "null_mean": round(sum(null) / len(null), 5),
                          "p_raw": round(p, 4)})
        pvals.append(p)

    # ---- 5. ONE BH pass over everything that produced a p
    adj = benjamini_hochberg(pvals)
    it = iter(adj)
    for c in out_cells:
        if "p_raw" in c:
            c["p_bh"] = round(next(it), 4)
            c["survives_bh"] = c["p_bh"] < 0.05

    # ---- 6. case studies surfaced by DATA (top |CAR| 1d), not by memory
    text_by_key = {(c, t0): txt for c, t0, txt in texts}
    flat = [(abs(r.car[1]), cohort, asset, r)
            for (cohort, asset), res in scored.items() for r in res]
    flat.sort(key=lambda x: -x[0])
    cases = [{"cohort": c, "asset": t, "s0": r.s0_date, "car_1d": round(r.car[1], 4),
              "overlapping": r.overlapping,
              "tweet": text_by_key.get((c, r.t0), "")} for _, c, t, r in flat[:10]]

    result = {"label": a.label, "seed": SEED, "n_perm": a.n_perm,
              "n_events_by_cohort": {c: len(v) for c, v in
                                     ((c, [t for (cc, _), rs in scored.items() if cc == c
                                           for t in rs]) for c in events)},
              "cells": out_cells, "top_case_studies": cases}
    (HERE / "study_results.json").write_text(json.dumps(result, indent=2))

    survivors = [c for c in out_cells if c.get("survives_bh")]
    print(f"\n[study] cells with p: {len(pvals)}; SURVIVE BH: {len(survivors)}")
    for c in survivors:
        print(f"  {c['cohort']}/{c['asset']} w={c['window']} {c['family']}: "
              f"obs={c['observed']} null={c['null_mean']} p_bh={c['p_bh']}")
    print("[study] -> registry.json, study_results.json")


if __name__ == "__main__":
    main()
