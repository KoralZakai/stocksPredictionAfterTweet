"""Pre-registered INTRADAY shock study: does a tweet category move markets in 60min?

Design (the whole point):
  * WITHIN-EVENT CONTROL. Each tweet's own 60min PRE-window is its control. Tweets
    with intraday data are a biased subset (in-session only, 28.5% of posts) — that
    bias hits both windows equally, so comparing them cancels it. This is what our
    earlier 1h result (0.678, later explained as selection bias) failed to do.
  * COHORT CONTRAST. NOISE is the control cohort. "GEO moves oil" only means
    something if GEO moves oil MORE than a random in-session post does.
  * REGISTRY FIRST. Every (cohort, asset, metric) cell is written before scoring;
    one BH pass over the whole grid. MDE is reported next to every cell, so a null
    at MDE 0.20% reads differently from a null at MDE 5%.
  * Tags are deterministic + text-only (no LLM, $0, cannot drift).

Run: PYTHONPATH=. python experiments/event_study/run_intraday.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpha.benchmark import session_phase
from alpha.stats import benjamini_hochberg
from sector_mapping.entities import entity_matches

from experiments.event_study.intraday import load_hourly, study_shock

HERE = Path(__file__).resolve().parent
CORPUS = Path("data/real/corpus_v3.csv")
LO, HI = "2025-01-01", "2026-07-06"
SEED, N_PERM = 20260715, 2000

# --- taxonomy: deterministic, text-only. Precedence CORPORATE > GEO > MACRO > NOISE.
GEO_RX = re.compile(r"\b(iran|hormuz|israel|russia|ukraine|war|missile|strike|sanction|"
                    r"nato|opec|nuclear|ceasefire|invasion|troops|military)\b", re.I)
MACRO_RX = re.compile(r"\b(tariff|fed|federal reserve|interest rate|rates|inflation|"
                      r"deficit|debt|dollar|tax(?:es)?|trade deal|economy|jobs report)\b", re.I)

# Assets watched per cohort (pre-registered). All are in the public hourly file.
COHORT_ASSETS: dict[str, tuple[str, ...]] = {
    "GEOPOLITIC": ("USO", "VIXY", "GLD", "XLE", "ITA"),
    "MACRO": ("TLT", "UUP", "GLD", "XLF", "VIXY"),
    "CORPORATE": (),          # filled per-event with the named company's sector proxy
    "NOISE": ("USO", "VIXY", "GLD", "XLE", "ITA", "TLT", "UUP", "XLF"),   # control
}
# Single names have no public hourly bars here, so a corporate post is watched via
# its sector proxy — the closest reproducible stand-in.
CORP_PROXY = {"AAPL": "XLK", "MSFT": "XLK", "NVDA": "SMH", "INTC": "SMH", "AMD": "SMH",
              "AMZN": "XLY", "TSLA": "XLY", "MCD": "XLY", "HD": "XLY", "NKE": "XLY",
              "BA": "ITA", "LMT": "ITA", "NOC": "ITA", "RTX": "ITA", "GD": "ITA",
              "CAT": "XLI", "GE": "XLI", "UNP": "XLI",
              "JPM": "XLF", "GS": "XLF", "MS": "XLF", "BAC": "XLF", "WFC": "XLF",
              "XOM": "XLE", "CVX": "XLE", "COP": "XLE"}
METRICS = ("post_excess", "abs_excess", "vol_ratio")


def tag(text: str) -> tuple[str, tuple[str, ...]]:
    direct = [t for t, m in entity_matches(text).items() if m.tier == "direct" and t != "DJT"]
    if direct:
        proxies = tuple({CORP_PROXY[t] for t in direct if t in CORP_PROXY})
        return "CORPORATE", proxies
    if GEO_RX.search(text):
        return "GEOPOLITIC", COHORT_ASSETS["GEOPOLITIC"]
    if MACRO_RX.search(text):
        return "MACRO", COHORT_ASSETS["MACRO"]
    return "NOISE", COHORT_ASSETS["NOISE"]


def _events() -> list[tuple[str, datetime, tuple[str, ...], str]]:
    import csv
    csv.field_size_limit(10**9)
    out = []
    with CORPUS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = r["timestamp_utc"][:10]
            if not (LO <= d <= HI):
                continue
            t0 = datetime.fromisoformat(r["timestamp_utc"][:19]).replace(tzinfo=timezone.utc)
            if session_phase(t0) != "regular":       # need a pre AND post window
                continue
            cohort, assets = tag(r["text"])
            if assets:
                out.append((cohort, t0, assets, r["text"][:160]))
    return out


def _val(s: Any, metric: str) -> float:
    if metric == "post_excess":
        return s.post_excess
    if metric == "abs_excess":
        return abs(s.post_excess)
    return s.vol_ratio


def main() -> None:
    ap = argparse.ArgumentParser(description="pre-registered intraday shock study")
    ap.add_argument("--sigma", type=float, default=2.0, help="Heavy Hitter threshold (z)")
    a = ap.parse_args()
    rng = random.Random(SEED)
    bars = load_hourly()
    if not bars:
        raise SystemExit("no public hourly bars — run scripts/fetch_hourly_bars.py")

    evs = _events()
    print(f"[intraday] in-session events: {len(evs)}")

    # ---- DEDUPE TO ONE OBSERVATION PER (asset, hourly bar).
    # He posts in bursts: several tweets inside one hour all resolve to the SAME bar
    # and would otherwise be counted as independent evidence of the same price move
    # (two posts a minute apart both scored GLD at z=-11.16). That inflates n and
    # breaks the permutation test's exchangeability. One bar = one observation; when
    # a burst mixes cohorts the bar is tagged by precedence CORPORATE>GEO>MACRO>NOISE.
    PREC = {"CORPORATE": 0, "GEOPOLITIC": 1, "MACRO": 2, "NOISE": 3}
    slots: dict[tuple[str, str], tuple[str, datetime, str]] = {}
    n_dup = 0
    for cohort, t0, assets, text in evs:
        for asset in assets:
            s = study_shock(bars, asset, t0)
            if s is None:
                continue                                  # missing data only
            key = (asset, s.t0[:13])                      # (asset, post-bar hour)
            prev = slots.get(key)
            if prev is None:
                slots[key] = (cohort, t0, text)
            else:
                n_dup += 1
                if PREC[cohort] < PREC[prev[0]]:          # more specific cohort wins
                    slots[key] = (cohort, t0, text)

    scored: dict[tuple[str, str], list[Any]] = {}
    hitters: list[dict[str, Any]] = []
    for (asset, _hr), (cohort, t0, text) in slots.items():
        s = study_shock(bars, asset, t0)
        if s is None:
            continue
        scored.setdefault((cohort, asset), []).append(s)
        if abs(s.shock_z) >= a.sigma:
            hitters.append({"cohort": cohort, "asset": asset, "t0": s.t0,
                            "z": s.shock_z, "post_excess": s.post_excess,
                            "vol_ratio": s.vol_ratio, "text": text})
    print(f"[intraday] same-bar duplicates collapsed: {n_dup} "
          f"-> {sum(len(v) for v in scored.values())} unique (asset, bar) observations")

    # ---- registry BEFORE stats
    registry = [{"cohort": c, "asset": t, "metric": m}
                for (c, t) in sorted(scored) for m in METRICS]
    (HERE / "intraday_registry.json").write_text(json.dumps(
        {"seed": SEED, "n_perm": N_PERM, "sigma": a.sigma, "cells": registry}, indent=2))
    print(f"[intraday] registry: {len(registry)} cells (written before scoring)")

    # ---- per cell: cohort vs the NOISE control on the same asset, permutation test
    cells: list[dict[str, Any]] = []
    pvals: list[float] = []
    for cell in registry:
        c, t, m = cell["cohort"], cell["asset"], cell["metric"]
        obs_l = [_val(s, m) for s in scored[(c, t)]]
        ctl_l = [_val(s, m) for s in scored.get(("NOISE", t), [])]
        n, sd = len(obs_l), statistics.pstdev([_val(s, "post_excess") for s in scored[(c, t)]])
        mde = 2.80 * sd / (n ** 0.5) if n else float("nan")
        if c == "NOISE" or n < 10 or len(ctl_l) < 10:
            cells.append({**cell, "n": n, "skipped": "control cohort" if c == "NOISE"
                          else "n<10", "mde": round(mde, 5)})
            continue
        obs = sum(obs_l) / n - sum(ctl_l) / len(ctl_l)     # cohort minus control
        pool = obs_l + ctl_l
        null = []
        for _ in range(N_PERM):                            # label permutation
            rng.shuffle(pool)
            null.append(sum(pool[:n]) / n - sum(pool[n:]) / len(ctl_l))
        p = (sum(1 for x in null if abs(x) >= abs(obs)) + 1) / (N_PERM + 1)
        cells.append({**cell, "n": n, "n_control": len(ctl_l),
                      "cohort_mean": round(sum(obs_l) / n, 5),
                      "control_mean": round(sum(ctl_l) / len(ctl_l), 5),
                      "diff": round(obs, 5), "mde": round(mde, 5), "p_raw": round(p, 4)})
        pvals.append(p)

    adj = benjamini_hochberg(pvals)
    it = iter(adj)
    for c in cells:
        if "p_raw" in c:
            c["p_bh"] = round(next(it), 4)
            c["survives_bh"] = c["p_bh"] < 0.05

    hitters.sort(key=lambda h: -abs(h["z"]))
    out = {"seed": SEED, "n_perm": N_PERM, "sigma_threshold": a.sigma,
           "n_events": len(evs),
           "n_by_cohort": {c: len({s.t0 for s in v})
                           for (c, _t), v in scored.items() for c in [c]},
           "cells": cells, "heavy_hitters": hitters[:15]}
    (HERE / "intraday_results.json").write_text(json.dumps(out, indent=2))

    surv = [c for c in cells if c.get("survives_bh")]
    print(f"[intraday] cells with p: {len(pvals)}   SURVIVE BH: {len(surv)}")
    for c in surv:
        print(f"   {c['cohort']}/{c['asset']} {c['metric']}: cohort={c['cohort_mean']} "
              f"control={c['control_mean']} diff={c['diff']} p_bh={c['p_bh']}")
    print(f"[intraday] Heavy Hitters (|z|>={a.sigma}): {len(hitters)}")
    for h in hitters[:5]:
        print(f"   z={h['z']:+6.2f} {h['asset']:5} {h['t0'][:16]} {h['cohort']:11} {h['text'][:44]!r}")
    print("[intraday] -> intraday_registry.json, intraday_results.json")


if __name__ == "__main__":
    main()
