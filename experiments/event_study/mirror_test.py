"""The MIRROR TEST — which way does the tweet-market arrow point?

Forward (tweets -> prices) is the registered null: 0 of 63 cells. This script
measures the REVERSE arrow on the same committed data: does his posting behaviour
track what the market just did?

Three measurements, in order of what survived:

1. TWEET-level: |prior-21d abnormal move| of a mentioned company vs random days.
   -> gap +2.27pp, perm p=0.002 ... and it is FAKE. Burst duplicates again
   (ten TSLA tweets about the same -26%). Kept here as the demonstration.
2. EPISODE-level (dedupe: one observation per ticker x 21-session window):
   -> gap +0.85pp, perm p=0.21. Dead. The dedupe kills the tweet-level "finding".
3. INTENSITY: per episode, does the NUMBER of mentions track the size of the
   move behind it? spearman(count, |move|), permutation p.
   -> rho=+0.23, p=0.037 (n=65). Single-mention episodes sit on ~4% moves;
   3+-mention episodes sit on ~8.4%. He does not tweet about more companies
   when markets move -- he tweets MORE TIMES about the same one.

STATUS: EXPLORATORY. One registered-in-code test, raw p, no BH family. It is
reported as "suggestive" in the blog, never as a survivor. The honest headline
stays the registered forward null.

Run: PYTHONPATH=. python experiments/event_study/mirror_test.py
Deterministic (seed 20260715), offline, committed data only.
"""

from __future__ import annotations

import csv
import random
import re
import statistics as st
from datetime import datetime, timezone
from typing import Any

from sector_mapping.entities import entity_matches

from experiments.event_study.engine import load_bars, s0_index

SEED = 20260715
LO, HI = "2025-01-01", "2026-07-06"
BACK = 21
N_PERM = 5000


def _prior_abn(bars: dict[str, list[Any]], md: dict[str, int],
               tk: str, t0: datetime, back: int = BACK) -> float | None:
    a, m = bars.get(tk), bars["SPY"]
    i0 = s0_index(a, t0) if a else None
    if a is None or i0 is None or i0 - back <= 0:
        return None
    d1, d0 = a[i0 - 1].date, a[i0 - back].date
    if d1 not in md or d0 not in md:
        return None
    return (a[i0 - 1].close / a[i0 - back].close - 1) - (
        m[md[d1]].close / m[md[d0]].close - 1)


def _events(bars: dict[str, list[Any]], md: dict[str, int]) -> tuple[list[dict[str, Any]],
                                                                     list[dict[str, Any]]]:
    """(all tweet-level mentions, deduped episodes). Episode = ticker x 21-session
    window; repeat mentions inside the window increment the episode's count instead
    of becoming new observations — the burst-duplicate lesson, applied up front."""
    csv.field_size_limit(10**9)
    with open("data/real/corpus_v3.csv", encoding="utf-8") as f:
        rows = sorted((r for r in csv.DictReader(f)
                       if LO <= r["timestamp_utc"][:10] <= HI),
                      key=lambda r: r["timestamp_utc"])
    tweets: list[dict[str, Any]] = []
    cur: dict[str, dict[str, Any]] = {}
    episodes: list[dict[str, Any]] = []
    seen_txt: set[str] = set()
    for r in rows:
        key = re.sub(r"[^a-z0-9]", "", r["text"][:90].lower())
        for tk, mm in entity_matches(r["text"]).items():
            if mm.tier != "direct" or tk == "DJT" or tk not in bars:
                continue
            t0 = datetime.fromisoformat(r["timestamp_utc"].replace("Z", "+00:00"))
            if t0.tzinfo is None:
                t0 = t0.replace(tzinfo=timezone.utc)
            i0 = s0_index(bars[tk], t0)
            if i0 is None:
                continue
            if key not in seen_txt:
                seen_txt.add(key)
                v = _prior_abn(bars, md, tk, t0)
                if v is not None:
                    tweets.append({"tk": tk, "d": r["timestamp_utc"][:10], "mv": v})
            if tk in cur and i0 - cur[tk]["i0"] < BACK:
                cur[tk]["n"] += 1
                continue
            if tk in cur:
                episodes.append(cur[tk])
            cur[tk] = {"tk": tk, "i0": i0, "n": 1, "d": r["timestamp_utc"][:10],
                       "mv": _prior_abn(bars, md, tk, t0)}
    episodes += list(cur.values())
    return tweets, [e for e in episodes if e["mv"] is not None]


def _baseline(bars: dict[str, list[Any]], md: dict[str, int], rng: random.Random,
              tickers: list[str], draws: int) -> list[float]:
    out: list[float] = []
    for tk in tickers:
        a = bars[tk]
        pool = [b for j, b in enumerate(a) if 30 < j < len(a) - 1 and LO <= b.date <= HI]
        for _ in range(draws):
            b = rng.choice(pool)
            t0 = datetime.fromisoformat(b.date).replace(hour=5, tzinfo=timezone.utc)
            v = _prior_abn(bars, md, tk, t0)
            if v is not None:
                out.append(v)
    return out


def _perm_gap(a: list[float], b: list[float], rng: random.Random) -> float:
    obs = st.mean(a) - st.mean(b)
    pool, k, cnt = a + b, len(a), 0
    for _ in range(N_PERM):
        rng.shuffle(pool)
        if st.mean(pool[:k]) - st.mean(pool[k:]) >= obs:
            cnt += 1
    return cnt / N_PERM


def _spearman(x: list[float], y: list[float]) -> float:
    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        rk = [0.0] * len(xs)
        for pos, i in enumerate(order):
            rk[i] = float(pos)
        return rk
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry, strict=True))
    den = (sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry)) ** 0.5
    return num / den if den else 0.0


def main() -> None:
    rng = random.Random(SEED)
    bars = load_bars()
    md = {b.date: j for j, b in enumerate(bars["SPY"])}
    tweets, eps = _events(bars, md)

    t_abs = [abs(t["mv"]) for t in tweets]
    base_t = [abs(v) for v in _baseline(bars, md, rng, [t["tk"] for t in tweets], 40)]
    p1 = _perm_gap(t_abs, base_t, rng)
    print(f"[mirror] 1 TWEET level    n={len(t_abs):<4} mention {st.mean(t_abs)*100:5.2f}% "
          f"vs random {st.mean(base_t)*100:5.2f}%  p={p1:.4f}   <- FAKE (burst duplicates)")

    e_abs = [abs(e["mv"]) for e in eps]
    base_e = [abs(v) for v in _baseline(bars, md, rng, [e["tk"] for e in eps], 80)]
    p2 = _perm_gap(e_abs, base_e, rng)
    print(f"[mirror] 2 EPISODE level  n={len(e_abs):<4} episode {st.mean(e_abs)*100:5.2f}% "
          f"vs random {st.mean(base_e)*100:5.2f}%  p={p2:.4f}   <- the dedupe kills it")

    counts = [float(e["n"]) for e in eps]
    rho = _spearman(counts, e_abs)
    ys, cnt = e_abs[:], 0
    for _ in range(N_PERM):
        rng.shuffle(ys)
        if _spearman(counts, ys) >= rho:
            cnt += 1
    one = [abs(e["mv"]) for e in eps if e["n"] == 1]
    multi = [abs(e["mv"]) for e in eps if e["n"] >= 3]
    print(f"[mirror] 3 INTENSITY      n={len(eps):<4} spearman(count,|move|)={rho:+.3f}  "
          f"p={cnt / N_PERM:.4f}   <- exploratory, suggestive")
    print(f"[mirror]   1-mention episodes: {st.mean(one)*100:.2f}% behind them (n={len(one)}) | "
          f"3+: {st.mean(multi)*100:.2f}% (n={len(multi)})")


if __name__ == "__main__":
    main()
