"""REGISTERED tone study: does anger/emphasis moderate the market response?

The user hypothesis, stated honestly before scoring:

    "Grumpy/angry posts (exclamation marks, ALL CAPS, attack vocabulary) should
     be weighted as MORE influential."

We do not assume it — we test it. The tone score is frozen (tone.py). The unit of
observation is the deduped company-mention EPISODE from mirror_test.py (one per
ticker x 21-session window; episode tone = max tone across its posts, because the
hypothesis is about the angriest form the message took).

REGISTRY — exactly four cells, enumerated here before any scoring, one BH pass:

  F1  spearman( tone , |abnormal move  t0 .. +1 session | )   forward, immediate
  F21 spearman( tone , |abnormal move  t0 .. +21 sessions| )  forward, drift
  M21 spearman( tone , |abnormal move  -21 .. t0 |         )  the MIRROR: is he
                                                              angrier when the move
                                                              already happened?
  D1  high-tone (top tercile) vs low-tone (bottom tercile) mean |1d abnormal| gap

Null: permutation of tone across episodes (5000 draws, seed 20260715). A cell
"finds something" only if p_bh < 0.05. If F-cells are null and M21 is not, the
correct reading is: anger is a symptom of the move, not a cause — consistent with
everything else this project found.

Deterministic, offline, committed data only.
Run: PYTHONPATH=. python experiments/event_study/tone_study.py
"""

from __future__ import annotations

import json
import random
import statistics as st
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpha.stats import benjamini_hochberg

from experiments.event_study.engine import load_bars, s0_index
from experiments.event_study.mirror_test import _events, _spearman
from experiments.event_study.tone import tone

SEED = 20260715
N_PERM = 5000
OUT = Path("experiments/event_study/tone_study_results.json")


def _post_abn(bars: dict[str, list[Any]], md: dict[str, int],
              tk: str, t0: datetime, fwd: int) -> float | None:
    """Abnormal (vs SPY) return from the first open after t0 to the close fwd
    sessions later — the same leak-free anchor as everything else."""
    a, m = bars.get(tk), bars["SPY"]
    i0 = s0_index(a, t0) if a else None
    if a is None or i0 is None or i0 + fwd >= len(a):
        return None
    d = a[i0 + fwd].date
    if d not in md or a[i0].date not in md:
        return None
    return (a[i0 + fwd].close / a[i0].open - 1) - (
        m[md[d]].close / m[md[a[i0].date]].open - 1)


def _perm_p(stat: float, xs: list[float], ys: list[float],
            rng: random.Random) -> float:
    cnt, ys2 = 0, ys[:]
    for _ in range(N_PERM):
        rng.shuffle(ys2)
        if abs(_spearman(xs, ys2)) >= abs(stat):
            cnt += 1
    return cnt / N_PERM


def main() -> None:
    rng = random.Random(SEED)
    bars = load_bars()
    md = {b.date: j for j, b in enumerate(bars["SPY"])}

    # Episodes with their tone. mirror_test._events gives (tweets, episodes); the
    # episode's tone = max over the posts inside its window, resolved by re-scanning
    # the corpus once (episodes carry date + ticker).
    import csv
    csv.field_size_limit(10**9)
    rows = [r for r in csv.DictReader(open("data/real/corpus_v3.csv", encoding="utf-8"))
            if "2025-01-01" <= r["timestamp_utc"][:10] <= "2026-07-06"]
    _, episodes = _events(bars, md)
    from sector_mapping.entities import entity_matches
    by_tk_day: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        for tk, m_ in entity_matches(r["text"]).items():
            if m_.tier == "direct" and tk != "DJT":
                by_tk_day.setdefault(tk, []).append(
                    (r["timestamp_utc"][:10], tone(r["text"]).score))

    ep = []
    for e in episodes:
        t0 = datetime.fromisoformat(e["d"]).replace(hour=5, tzinfo=timezone.utc)
        # max tone of this ticker's posts within [episode day, +30 calendar days)
        from datetime import timedelta
        hi = (t0 + timedelta(days=30)).strftime("%Y-%m-%d")
        tones = [s for d, s in by_tk_day.get(e["tk"], []) if e["d"] <= d < hi]
        if not tones:
            continue
        f1 = _post_abn(bars, md, e["tk"], t0, 1)
        f21 = _post_abn(bars, md, e["tk"], t0, 21)
        if f1 is None or f21 is None or e["mv"] is None:
            continue
        ep.append({"tk": e["tk"], "d": e["d"], "tone": max(tones),
                   "f1": abs(f1), "f21": abs(f21), "m21": abs(e["mv"])})

    tones = [x["tone"] for x in ep]
    cells = []
    for key, label in (("f1", "F1  forward |move| t0..+1"),
                       ("f21", "F21 forward |move| t0..+21"),
                       ("m21", "M21 MIRROR |move| -21..t0")):
        ys = [x[key] for x in ep]
        rho = _spearman(tones, ys)
        p = _perm_p(rho, tones, ys, rng)
        cells.append({"cell": key, "label": label, "n": len(ep),
                      "spearman": round(rho, 4), "p_raw": round(p, 5)})

    # D1: top vs bottom tercile of tone, gap in mean |1d| move
    srt = sorted(ep, key=lambda x: x["tone"])
    k = len(srt) // 3
    lo, hi = srt[:k], srt[-k:]
    gap = st.mean(x["f1"] for x in hi) - st.mean(x["f1"] for x in lo)
    pool = [x["f1"] for x in lo + hi]
    cnt = 0
    for _ in range(N_PERM):
        rng.shuffle(pool)
        if abs(st.mean(pool[:len(hi)]) - st.mean(pool[len(hi):])) >= abs(gap):
            cnt += 1
    cells.append({"cell": "D1", "label": "D1  angry vs calm tercile, |1d| gap",
                  "n": len(lo) + len(hi), "gap_pp": round(gap * 100, 3),
                  "p_raw": round(cnt / N_PERM, 5)})

    adj = benjamini_hochberg([c["p_raw"] for c in cells])
    for c, a in zip(adj, cells) if False else zip(cells, adj):
        c["p_bh"] = round(a, 5)
        c["survives_bh"] = bool(a < 0.05)

    res = {"seed": SEED, "n_perm": N_PERM, "n_episodes": len(ep),
           "registry": [c["cell"] for c in cells], "cells": cells,
           "note": ("Registered BEFORE scoring (module docstring). Tone weights are "
                    "frozen in tone.py. A null here means anger does NOT earn a "
                    "weight; M21-only signal means anger is a symptom of the move.")}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[tone] episodes={len(ep)}  (registry fixed at 4 cells, BH over 4)")
    for c in cells:
        extra = f"rho={c['spearman']:+.3f}" if "spearman" in c else f"gap={c['gap_pp']:+.2f}pp"
        print(f"[tone]  {c['label']:<38} n={c['n']:<4} {extra:<14} "
              f"p_raw={c['p_raw']:.4f}  p_bh={c['p_bh']:.4f}  "
              f"{'SURVIVES' if c['survives_bh'] else 'no'}")
    print(f"[tone] -> {OUT}")


if __name__ == "__main__":
    main()
