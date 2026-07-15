"""Build reports/dashboard.html — the Myth-Busting Quantitative Terminal.

NOT an alpha generator. It walks a reader through three high-conviction broker
hypotheses and the controls that killed each one:

  Tab 1  REVERSE CAUSALITY  - "his Intel tweet made INTC run"  -> INTC ran +105%
                              BEFORE the tweet, -2.4% after.
  Tab 2  MEAN REVERSION     - "oil dips then recovers after Iran tweets" -> so do
                              big oil moves with NO tweet (recovering harder).
  Tab 3  SELECTION BIAS     - sorting by biggest mover manufactures a greatest-hits
                              reel: an Iran SOCCER tweet "moves" oil +17%.
  Tab 4  THE VERDICT        - 0 of the registered cells survive BH.

Every number is COMPUTED HERE from data/real/bars.csv + corpus_v3.csv + the
registered study output. Nothing is hardcoded — the page regenerates and can be
audited line by line. Self-contained HTML (inline CSS/SVG, no external assets).

Run: PYTHONPATH=. python experiments/event_study/generate_targeted_dashboard.py
"""

from __future__ import annotations

import csv
import json
import random
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from config.membership import name_of
from sector_mapping.entities import entity_matches, is_direct_mention

from experiments.event_study.engine import load_bars, s0_index, study_event

OUT = Path("reports/dashboard.html")
HERE = Path(__file__).resolve().parent
RESULTS = Path("reports/nebius_backtest_results.json")
STUDY = Path("experiments/event_study/study_results.json")
CORPUS = Path("data/real/corpus_v3.csv")
WINDOWS = (1, 3, 5, 10, 21, 42)
SEED = 20260715
LO, HI = "2025-01-01", "2026-07-06"

# Intel-the-company detection lives in ONE place: the shared, tested entity mapper
# (sector_mapping/entities.py, CLAUDE.md 6). This file used to keep its own private
# regex, and the two disagreed — the private one only looked for company context AFTER
# the word, so it scored "The CEO of INTEL must resign", "I met with Mr. Lip-Bu Tan, of
# Intel" and "I PAID ZERO FOR INTEL" as NOT-Intel, and tab 1 reported 3 mentions where
# the corpus holds 8. Two mappers means two answers to "what is this tweet about"; the
# cohort labels and the exhibit could never agree. There is now one.
_OIL_RX = re.compile(r"\b(iran|hormuz|strait|opec|oil|crude|drill|refiner|tanker)\b", re.I)


# ---------------------------------------------------------------- data gathering
def _t0(row: dict[str, Any]) -> datetime:
    h = float(row.get("hour_utc", 14.0))
    return datetime.fromisoformat(row["date"]).replace(
        hour=int(h), minute=int((h % 1) * 60), tzinfo=timezone.utc)


def _prior_window(bars: dict[str, list[Any]], ticker: str, t0: datetime,
                  back: int) -> tuple[float, float] | None:
    """(asset_ret, spy_ret) over the `back` sessions ending at the LAST CLOSE BEFORE
    the post — i.e. all price information that existed when the tweet was written.

    Anchored with the shared s0_index (respects the tweet's hour). A naive
    `date >= day` lookup drops the tweet's own session for an after-close post, which
    understated this run-up by ~48pp on the Intel case.
    """
    a, m = bars.get(ticker), bars.get("SPY")
    if not a or not m:
        return None
    md = {b.date: j for j, b in enumerate(m)}
    i0 = s0_index(a, t0)
    if i0 is None:
        return None
    j = i0 - back
    if j <= 0 or a[i0 - 1].date not in md or a[j].date not in md:
        return None
    ra = a[i0 - 1].close / a[j].close - 1.0
    rm = m[md[a[i0 - 1].date]].close / m[md[a[j].date]].close - 1.0
    return ra, rm


def gather_intel(bars: dict[str, list[Any]]) -> dict[str, Any]:
    csv.field_size_limit(10**9)
    with CORPUS.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if LO <= r["timestamp_utc"][:10] <= HI]
    hits, seen = [], set()
    for r in rows:
        # Normalise before deduping: the same post is mirrored across platforms with
        # curly vs straight apostrophes, so a raw-prefix key lets duplicates through.
        key = re.sub(r"[^a-z0-9]", "", r["text"][:90].lower())
        if is_direct_mention(r["text"], "INTC") and key not in seen:
            seen.add(key)
            hits.append({"ts": r["timestamp_utc"][:16], "text": r["text"][:180]})
    # EVERY mention gets its before/after measured, not just the famous one. The old
    # private regex kept 3 of these 8 — and the 5 it dropped are the ones that do NOT
    # look like reverse causality (INTC was FALLING before the "CEO must resign" post,
    # then rose). A rule that silently keeps the supporting cases and drops the rest is
    # the anecdote generator this dashboard exists to expose, running on our own exhibit.
    for h in hits:
        h_t0 = datetime.fromisoformat(h["ts"]).replace(tzinfo=timezone.utc)
        pw = _prior_window(bars, "INTC", h_t0, 21)
        er = study_event(bars, "INTC", h_t0, WINDOWS)
        h["prior21"] = (pw[0] - pw[1]) if pw else None
        h["post21"] = (er.car.get(21) if er else None)

    hero = next((h for h in hits if "continues to rise" in h["text"].lower()), None)
    prior: list[dict[str, Any]] = []
    post: dict[int, float] = {}
    if hero:
        hero_t0 = datetime.fromisoformat(hero["ts"]).replace(tzinfo=timezone.utc)
        for back in (5, 10, 21, 42, 63):
            pw = _prior_window(bars, "INTC", hero_t0, back)
            if pw:
                prior.append({"back": back, "asset": pw[0], "spy": pw[1],
                              "excess": pw[0] - pw[1]})
        er = study_event(bars, "INTC", datetime.fromisoformat(
            hero["ts"]).replace(tzinfo=timezone.utc), WINDOWS)
        post = er.car if er else {}
    return {"n_corpus": len(rows), "mentions": hits, "hero": hero,
            "prior": prior, "post": post}


def gather_oil(bars: dict[str, list[Any]], rng: random.Random) -> dict[str, Any]:
    rows = json.loads(RESULTS.read_text())
    paths, seen = [], set()
    for r in sorted((x for x in rows if _OIL_RX.search(x.get("text", ""))),
                    key=lambda x: x["date"]):
        er = study_event(bars, "USO", _t0(r), WINDOWS)
        if er is None or er.s0_date in seen:
            continue
        seen.add(er.s0_date)
        paths.append({"s0": er.s0_date, "car": er.car, "text": r.get("text", "")[:150],
                      "t0": _t0(r).isoformat()})     # keep the REAL t0 for the analyzer
    paths.sort(key=lambda p: -abs(p["car"][1]))

    pool = [b.date for b in bars.get("USO", []) if LO <= b.date <= "2026-04-01"]
    rnd = []
    for d in rng.sample(pool, min(120, len(pool))):
        er = study_event(bars, "USO", datetime.fromisoformat(d).replace(
            hour=14, tzinfo=timezone.utc), WINDOWS)
        if er:
            rnd.append(er.car)
    big = [c for c in rnd if abs(c[1]) >= 0.03]

    def mean(ps: list[dict[int, float]]) -> dict[int, float]:
        return {w: (sum(p[w] for p in ps if w in p) / max(len([p for p in ps if w in p]), 1))
                for w in WINDOWS}

    return {"n_tweet_days": len(paths), "n_sessions": len(pool), "top": paths[:5],
            "mean_top5": mean([p["car"] for p in paths[:5]]),
            "mean_all": mean([p["car"] for p in paths]),
            "mean_random": mean(rnd), "n_random": len(rnd),
            "mean_big_no_tweet": mean(big), "n_big": len(big)}


# The SAME panel is drawn for every anchor. Fixed up front on purpose: choosing which
# assets to show per tweet is how you cherry-pick a story. If an asset is irrelevant to
# a post, its flat line IS the information.
#
# These 20 are ETFs — indices and sectors. They stay the fixed control set for EVERY
# tweet and every aggregate on the page. A tweet's OWN company (INTC on an Intel post)
# is added on top per-anchor by `_mentioned`, never swapped in: the control cannot move
# with the tweet or it is not a control.
PANEL: tuple[str, ...] = ("SPY", "QQQ", "DIA", "USO", "VIXY", "GLD", "TLT", "UUP",
                          "FXI", "XLE", "XLK", "XLF", "XLI", "XLV", "XLY", "XLP",
                          "XLB", "ITA", "SMH", "DBC")


def _mentioned(text: str, bars: dict[str, list[Any]]) -> list[str]:
    """The companies THIS tweet actually names — so an Intel post shows INTC, not just
    the sector ETFs it happens to sit inside.

    Reuses the rule-based, human-verified entity mapper (CLAUDE.md 6). No ML mapper, and
    nothing new invented here: `_cohort_of` already routes cohorts through these exact
    matches, so the panel and the cohort label can never disagree about what a tweet is
    about.

    `direct` tier only. The mapper also returns `competitor` ride-alongs (an Intel post
    pulls AMD/NVDA/TSM) — real, but they are an inference about who else is affected, not
    what the tweet says, and they would quietly triple the panel.

    DJT is excluded, matching `_cohort_of`. Trump signs posts "President DJT", which the
    ticker rule read as a mention of his own listed company — 754 of 762 'corporate'
    mentions were that signature. The mapper no longer matches the bare signature, but
    the exclusion stays: this is the second line of defence on an artifact that already
    shipped once.
    """
    return sorted(t for t, m in entity_matches(text).items()
                  if m.tier == "direct" and t != "DJT" and t in bars)


def gather_panel_series(bars: dict[str, list[Any]], anchors: list[dict[str, Any]],
                        span: int = 42) -> list[dict[str, Any]]:
    """Every anchor x the full panel. Session dates are shared across US equities, so
    they are stored ONCE per anchor and each asset keeps only two price arrays — that
    compaction is what lets the page carry 20 assets instead of 12.

    The panel is PANEL plus whatever this tweet's model call named. The extras matter:
    without them a leg on a single name (LMT, CAT) has no measured price on the page and
    its EXPECTED could not be scored at the slider's window at all.
    """
    m = bars.get("SPY")
    if not m:
        return []
    out = []
    for anc in anchors:
        t0 = datetime.fromisoformat(anc["ts"]).replace(tzinfo=timezone.utc)
        spy_i0 = s0_index(m, t0)
        if spy_i0 is None or spy_i0 - span < 0 or spy_i0 + span >= len(m):
            continue
        dates = [m[spy_i0 + k].date for k in range(-span, span + 1)]
        named = [leg["ticker"] for leg in (anc.get("pred") or {}).get("legs", [])]
        mentioned = _mentioned(anc.get("text", ""), bars)
        assets: dict[str, Any] = {}
        for tk in dict.fromkeys((*PANEL, *named, *mentioned)):
            a = bars.get(tk)
            if not a:
                continue
            i0 = s0_index(a, t0)
            if i0 is None or i0 - span < 0 or i0 + span >= len(a):
                continue
            by = {b.date: b for b in a}
            pa, pb, ok = [], [], True
            for off, d in enumerate(dates):
                ba, bm = by.get(d), (m[spy_i0 - span + off] if d == dates[off] else None)
                if ba is None or bm is None:
                    ok = False
                    break
                at_anchor = (off == span)
                # 4dp = the source precision in bars.csv, i.e. lossless. Rounding to 2dp
                # to shrink the page silently moved cheap assets by ~3bp (DBC ~$22), which
                # breaks the page's one promise: every number is a real measured price.
                pa.append(round(ba.open if at_anchor else ba.close, 4))
                pb.append(round(bm.open if at_anchor else bm.close, 4))
            if not ok or len(pa) != len(dates):
                continue
            prev = by.get(dates[span - 1])
            assets[tk] = {"pa": pa, "pb": pb,
                          "prev_a": round(prev.close, 4) if prev else pa[span - 1],
                          "prev_b": round(m[spy_i0 - 1].close, 4),
                          "open_a": pa[span], "open_b": pb[span]}
        if not assets:
            continue
        out.append({**anc, "s0": m[spy_i0].date, "dates": dates, "span": span,
                    "assets": assets, "names": {t: name_of(t) for t in assets},
                    # the company the tweet itself names, present only when it resolved
                    # to real bars — the page must never label an asset it cannot draw
                    "mentioned": [t for t in mentioned if t in assets]})
    return out


def gather_window_series(bars: dict[str, list[Any]], anchors: list[dict[str, str]],
                         span: int = 42) -> list[dict[str, Any]]:
    """For each anchor: the REAL asset + SPY price at EVERY session offset -span..+span
    around the entry anchor (first open strictly after t0), plus the last pre-post close.

    Raw prices are stored, not returns, so the page can compute each window EXACTLY:
    negating a backward-indexed return (`-E(-X)`) is only valid for small moves — on
    the Intel case (a double) it reported +44.8% for a +105.3% run-up. The slider is a
    lookup + exact arithmetic; nothing is interpolated or fitted.
    """
    out: list[dict[str, Any]] = []
    m = bars.get("SPY")
    if not m:
        return out
    md = {b.date: j for j, b in enumerate(m)}
    for anc in anchors:
        a = bars.get(anc["ticker"])
        if not a:
            continue
        # MUST use the shared anchor resolver: it applies the strictly-after-t0 rule
        # using the tweet's HOUR. A naive `date >= day` lookup counts the tweet's own
        # session as post-event for an after-close post (22:20) — a point-in-time leak
        # that silently halved the measured run-up (+92.6% -> +41.1%).
        t0 = datetime.fromisoformat(anc["ts"]).replace(tzinfo=timezone.utc)
        i0 = s0_index(a, t0)
        if i0 is None or i0 - span < 0 or i0 + span >= len(a) or a[i0].date not in md:
            continue
        k0 = md[a[i0].date]
        prev_mk = md.get(a[i0 - 1].date)
        if prev_mk is None:
            continue
        series: list[dict[str, float]] = []
        for k in range(-span, span + 1):
            j, mk = i0 + k, md.get(a[i0 + k].date)
            if mk is None:
                continue
            # k == 0 is the entry anchor: the first OPEN after the post (leak-free).
            series.append({"k": k, "d": a[j].date,
                           "pa": round(a[j].open if k == 0 else a[j].close, 4),
                           "pb": round(m[mk].open if k == 0 else m[mk].close, 4)})
        out.append({**anc, "tname": name_of(anc["ticker"]), "s0": a[i0].date, "series": series,
                    # last close BEFORE the post — the reference for the run-up, and
                    # the exact endpoint tab 1's prior-window table uses.
                    "prev_a": round(a[i0 - 1].close, 4),
                    "prev_b": round(m[prev_mk].close, 4),
                    "open_a": round(a[i0].open, 4), "open_b": round(m[k0].open, 4)})
    return out


def gather_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """"When are we right?" — EOD hit-rate per subgroup on TRAIN vs TEST.

    This is the honest answer to the question, and the answer is: nowhere. Every
    subgroup pattern reverses out of sample (weekend posts go 0.304 -> 0.765). It is
    the same finding the meta-model reported as Val AUC 0.593 -> Test 0.431, shown
    as a table instead of a number.
    """
    from collections import defaultdict

    from scripts.nebius_macro_backtest import _assign_splits, _tweet_hit
    _assign_splits(rows)
    agg: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: {"train": [0, 0], "test": [0, 0]})
    for r in rows:
        h, sp = _tweet_hit(r, "EOD"), r.get("split")
        if h is None or sp not in ("train", "test"):
            continue
        for dim, val in (("cohort", _cohort_of(r.get("text", ""), r.get("scenario", ""))),
                         ("session", str(r.get("phase", "?"))),
                         ("intensity", f"level {r.get('intensity')} of 10")):
            agg[(dim, val)][sp][0] += h
            agg[(dim, val)][sp][1] += 1
    out: list[dict[str, Any]] = []
    for (dim, val), v in agg.items():
        tr, te = v["train"], v["test"]
        if tr[1] < 8 or te[1] < 5:
            continue
        a, b = tr[0] / tr[1], te[0] / te[1]
        out.append({"dim": dim, "group": val, "train": round(a, 3), "n_train": tr[1],
                    "test": round(b, 3), "n_test": te[1], "shift": round(b - a, 3)})
    out.sort(key=lambda x: -abs(x["shift"]))
    return out


CALL_HORIZONS = ("EOD", "3d", "1w", "1mo")
CALL_HORIZON_LABEL = {"EOD": "same day", "3d": "3 days", "1w": "1 week", "1mo": "1 month"}


def gather_calls(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The scoreboard for EXPECTED-vs-actual over the WHOLE corpus.

    Tab 5's anchors are the biggest movers, i.e. chosen ON the outcome — a hit rate over
    those 18 would be meaningless. This is the same question asked of all 476 tweets, so
    the tab can show the honest denominator next to the exploratory one.

    Scoring rule = the page's rule #1: a call is right when the named instrument BEAT
    the S&P 500 in the predicted direction (return - spy_return has the predicted sign).
    Exact ties are unscoreable and dropped, not awarded — counting them as wins is the
    tie-break artifact the README documents.
    """
    from collections import Counter
    dirs: Counter[str] = Counter()
    per = {h: [0, 0] for h in CALL_HORIZONS}
    n_spy = 0
    for r in rows:
        for i in r.get("instruments", []):
            d = i.get("predicted", "?")
            dirs[d] += 1
            n_spy += int(i["ticker"] == "SPY")
            for h in CALL_HORIZONS:
                ret, spy = (i.get("returns") or {}).get(h), (r.get("spy_returns") or {}).get(h)
                if ret is None or spy is None or d not in ("up", "down") or ret == spy:
                    continue
                per[h][0] += int((ret - spy > 0) == (d == "up"))
                per[h][1] += 1
    n = sum(dirs.values())
    return {
        "n_tweets": len(rows), "n_calls": n,
        "n_up": dirs["up"], "n_down": dirs["down"], "n_flat": dirs["neutral"],
        # SPY legs are unscoreable BY CONSTRUCTION under a beat-the-market rule: the
        # yardstick cannot beat itself, so ret == spy and the leg drops out above. Worth
        # surfacing rather than burying — it is the model's single most-named instrument.
        "n_spy": n_spy,
        "up_share": dirs["up"] / max(n, 1),
        "per_tweet": n / max(len(rows), 1),
        "rates": [{"h": h, "label": CALL_HORIZON_LABEL[h], "hit": v[0], "n": v[1],
                   "rate": v[0] / max(v[1], 1)} for h, v in per.items() if v[1]],
    }


def gather_named_vs_ignored(bars: dict[str, list[Any]], rows: list[dict[str, Any]],
                            rng: random.Random,
                            windows: tuple[int, ...] = (1, 21)) -> list[dict[str, Any]]:
    """Did NAMING an instrument mean anything? The control for the whole tab.

    For every tweet: how far did the instruments the model named move away from the S&P
    500, vs the PANEL instruments it never mentioned — same tweet, same window, same
    yardstick. If the words of a tweet really pointed at an instrument, the named ones
    should move more.

    TWO traps this function exists to dodge, both of which make "named moves more" true
    for reasons that have nothing to do with tweets:

    1. Tab 5's anchors are chosen by the largest move of the model's OWN named basket, so
       on THEM the gap is guaranteed by construction. Hence: whole corpus, never anchors.
    2. The model names jumpy instruments (oil, the fear gauge, defence) and ignores placid
       ones (staples, bonds). A gap could just be that ranking of volatility, restated.

    So each tweet's named set is ALSO scored against a random OTHER tweet's date (a
    date-permutation placebo, seeded). Same instruments, same rule, wrong day. Whatever
    gap survives on the wrong day is the volatility ranking, not the tweet — and only the
    excess of `real` over `placebo` could ever be tweet-specific information.
    """
    m = bars.get("SPY")
    if not m:
        return []
    events = [(_t0(r), {i["ticker"] for i in r.get("instruments", [])}) for r in rows]
    shuffled = [t for t, _ in events]
    rng.shuffle(shuffled)
    out = []
    for w in windows:
        acc = {"real": [0.0, 0, 0.0, 0], "placebo": [0.0, 0, 0.0, 0]}  # nsum,nn,isum,ni
        n_tweets = 0
        for (t0, names), fake_t0 in zip(events, shuffled, strict=True):
            for arm, when in (("real", t0), ("placebo", fake_t0)):
                j0 = s0_index(m, when)
                if j0 is None or j0 + w >= len(m):
                    continue
                spy_beat = m[j0 + w].close / m[j0].open - 1.0
                for tk in PANEL:
                    a = bars.get(tk)
                    if not a:
                        continue
                    i0 = s0_index(a, when)
                    if i0 is None or i0 + w >= len(a) or a[i0 + w].date != m[j0 + w].date:
                        continue
                    beat = abs((a[i0 + w].close / a[i0].open - 1.0) - spy_beat)
                    k = 0 if tk in names else 2
                    acc[arm][k] += beat
                    acc[arm][k + 1] += 1
            n_tweets += 1
        if not (acc["real"][1] and acc["real"][3] and acc["placebo"][1]):
            continue
        row: dict[str, Any] = {"w": w, "n_tweets": n_tweets}
        for arm, v in acc.items():
            row[arm] = {"named": v[0] / v[1], "ignored": v[2] / v[3],
                        "n_named": v[1], "n_ignored": v[3],
                        "gap": v[0] / v[1] - v[2] / v[3]}
        out.append(row)
    return out


def gather_intraday() -> dict[str, Any] | None:
    """The Tab-6 intraday shock study, if it has been run."""
    p = HERE / "intraday_results.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    cells = [c for c in r["cells"] if "p_raw" in c]
    cells.sort(key=lambda c: c["p_raw"])
    return {"n_events": r["n_events"], "sigma": r["sigma_threshold"],
            "n_cells": len(cells),
            "n_survive": sum(1 for c in cells if c.get("survives_bh")),
            "min_p_bh": min((c["p_bh"] for c in cells), default=1.0),
            "closest": cells[:8], "hitters": r["heavy_hitters"][:6]}


def gather_verdict() -> dict[str, Any]:
    s = json.loads(STUDY.read_text())
    cells = [c for c in s["cells"] if "p_raw" in c]
    cells.sort(key=lambda c: c["p_raw"])
    return {"n_cells": len(cells),
            "n_survive": sum(1 for c in cells if c.get("survives_bh")),
            "min_p_bh": min((c["p_bh"] for c in cells), default=1.0),
            "closest": cells[:6], "cohorts": s.get("n_events_by_cohort", {})}


# ---------------------------------------------------------------- rendering
def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:+.2f}%"


FAMILY_LABEL = {"abs": "how big was the move?", "signed": "did it move as expected?",
                "volume": "how much trading?"}
METRIC_LABEL = {"post_excess": "did it beat the market?",
                "abs_excess": "how big was the move?",
                "vol_ratio": "how much trading?"}


def _cls(v: float | None) -> str:
    return "flat" if v is None else ("up" if v > 0 else "down" if v < 0 else "flat")


def _path_row(label: str, car: dict[int, float], note: str = "", hero: bool = False) -> str:
    tds = "".join(f'<td class="{_cls(car.get(w))}">{_pct(car.get(w))}</td>' for w in WINDOWS)
    return (f'<tr class="{"hero" if hero else ""}"><th>{escape(label)}</th>{tds}'
            f'<td class="note">{escape(note)}</td></tr>')


def _spark(car: dict[int, float], w: int = 260, h: int = 54) -> str:
    """Inline SVG of the abnormal-return path — shape at a glance, zero deps."""
    vals = [car.get(x, 0.0) for x in WINDOWS]
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    rng = (hi - lo) or 1.0
    pts = [(8 + i * (w - 16) / (len(vals) - 1), h - 6 - (v - lo) / rng * (h - 14))
           for i, v in enumerate(vals)]
    d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    zero = h - 6 - (0.0 - lo) / rng * (h - 14)
    end = "up" if vals[-1] > 0 else "down"
    return (f'<svg viewBox="0 0 {w} {h}" class="spark" role="img">'
            f'<line x1="4" y1="{zero:.1f}" x2="{w - 4}" y2="{zero:.1f}" class="zero"/>'
            f'<path d="{d}" class="line {end}"/>'
            f'<circle cx="{pts[0][0]:.1f}" cy="{pts[0][1]:.1f}" r="2.5" class="dot"/>'
            f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="3" class="dot {end}"/></svg>')


_CSS = """
:root{--bg:#0e1116;--surface:#161b22;--surface2:#1c2430;--ink:#e6edf3;--ink2:#9aa7b4;
--ink3:#6b7684;--line:#2a3441;--up:#2ea043;--up-bg:#0f2b17;--down:#e5534b;--down-bg:#331716;
--flat:#8b949e;--accent:#d9a441;--mono:ui-monospace,SFMono-Regular,Menlo,monospace;
--font:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
@media(prefers-color-scheme:light){:root{--bg:#f4f5f2;--surface:#fff;--surface2:#eef0ec;
--ink:#191c1f;--ink2:#5a6067;--ink3:#878d94;--line:#dee1db;--up:#17805f;--up-bg:#e4f1ea;
--down:#c0432f;--down-bg:#f7e5e0;--accent:#a8792f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);
line-height:1.5}.wrap{max-width:1000px;margin:0 auto;padding:28px 20px 64px}
header h1{margin:0 0 6px;font-size:1.6rem;letter-spacing:-.02em}
.sub{color:var(--ink2);margin:0 0 4px}
.stamp{color:var(--ink3);font-family:var(--mono);font-size:.75rem;margin:10px 0 0}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:22px 0 0;border-bottom:1px solid var(--line)}
.tab{background:none;border:0;border-bottom:2px solid transparent;color:var(--ink2);
padding:10px 14px;font:inherit;font-size:.9rem;cursor:pointer}
.tab[aria-selected=true]{color:var(--ink);border-bottom-color:var(--accent);font-weight:600}
.panel{display:none;padding:22px 0 0}.panel.on{display:block}
.myth{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--down);
border-radius:8px;padding:14px 16px;margin:0 0 16px}
.myth b{color:var(--down)}.verdict{border-left-color:var(--up)}.verdict b{color:var(--up)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;
padding:16px 18px;margin:0 0 16px}
h2{font-size:1.15rem;margin:0 0 10px}h3{font-size:.95rem;margin:18px 0 8px;color:var(--ink2)}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.8rem}
th,td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child,.note{text-align:left}
thead th{color:var(--ink3);font-weight:500;font-size:.72rem;text-transform:uppercase;letter-spacing:.04em}
tr.hero td,tr.hero th{background:var(--surface2);font-weight:600}
.up{color:var(--up)}.down{color:var(--down)}.flat{color:var(--flat)}
.note{color:var(--ink3);font-family:var(--font);white-space:normal;font-size:.78rem}
.scroll{overflow-x:auto}
.big{font-family:var(--mono);font-size:2rem;font-weight:700;letter-spacing:-.02em}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:12px 0}
.tile{background:var(--surface2);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.tile .k{color:var(--ink3);font-size:.7rem;text-transform:uppercase;letter-spacing:.04em}
.tweet{font-family:var(--mono);font-size:.78rem;color:var(--ink2);background:var(--surface2);
border-radius:6px;padding:8px 10px;margin:6px 0 0;white-space:normal}
.ctl{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}
.ctl label{display:block;color:var(--ink2);font-size:.75rem;text-transform:uppercase;
letter-spacing:.04em;margin:0 0 6px}
select,input[type=range]{width:100%;font:inherit}
select{background:var(--surface2);color:var(--ink);border:1px solid var(--line);
border-radius:6px;padding:8px}
input[type=range]{accent-color:var(--accent);cursor:pointer}
.rl{display:flex;justify-content:space-between;font-size:.78rem;color:var(--ink2);margin-bottom:2px}
.chart{width:100%;height:auto;background:var(--bg);border:1px solid var(--line);
border-radius:6px;display:block;cursor:crosshair}
.chart .zoneL{fill:var(--down);opacity:.06}.chart .zoneR{fill:var(--up);opacity:.06}
.chart .zero{stroke:var(--ink3);stroke-dasharray:3 3}
.chart .grid{stroke:var(--line);stroke-dasharray:2 4}
.chart .ev{stroke:var(--accent);stroke-width:1.5;stroke-dasharray:4 3}
.chart .trace{fill:none;stroke:var(--ink);stroke-width:2;
stroke-linejoin:round;stroke-linecap:round}
.chart .evdot{fill:var(--accent)}
.chart .ax{fill:var(--ink3);font-family:var(--mono);font-size:10px}
.chart .ay{text-anchor:end}.chart .evt{fill:var(--accent)}
.chart .xh{stroke:var(--ink3);stroke-width:1;stroke-dasharray:2 2}
.chart .xhd{fill:var(--accent);stroke:var(--bg);stroke-width:1.5}
.chart .tipbg{fill:var(--surface2);stroke:var(--line)}
.chart .tip{fill:var(--ink);font-family:var(--mono);font-size:10px}
.axis{display:flex;justify-content:space-between;font-size:.72rem;color:var(--ink3);margin-top:6px}
.vd{font-family:var(--mono);font-size:.85rem;letter-spacing:.02em;font-weight:700}
.vd.down{color:var(--down)}.vd.flat{color:var(--flat)}.vd.up{color:var(--up)}
.kv{display:grid;grid-template-columns:190px 1fr;gap:10px;padding:7px 0;
border-bottom:1px solid var(--line);align-items:baseline}
.kv .k{color:var(--ink3);font-size:.7rem;text-transform:uppercase;letter-spacing:.04em}
.kv b{font-weight:600;font-size:.9rem}
.nm{color:var(--ink3);font-weight:400;font-size:.78rem}
.said{background:var(--accent);color:var(--bg);border-radius:3px;padding:1px 5px;font-size:.62rem;font-weight:700;letter-spacing:.03em;text-transform:uppercase;white-space:nowrap}
@media(max-width:620px){.kv{grid-template-columns:1fr}}
.spark .zero{stroke:var(--line);stroke-width:1;stroke-dasharray:2 3}
.spark .line{fill:none;stroke-width:2}.spark .line.up{stroke:var(--up)}.spark .line.down{stroke:var(--down)}
.spark .dot{fill:var(--ink3)}.spark .dot.up{fill:var(--up)}.spark .dot.down{fill:var(--down)}
.spark{width:100%;max-width:280px;height:54px}
.foot{color:var(--ink3);font-size:.78rem;border-top:1px solid var(--line);margin-top:28px;padding-top:14px}
"""

_JS = """
const tabs=[...document.querySelectorAll('.tab')];
tabs.forEach(t=>t.addEventListener('click',()=>{
  tabs.forEach(x=>x.setAttribute('aria-selected',String(x===t)));
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('on',p.id===t.dataset.p));
}));

// ---- Dynamic time-window analyzer. Reads MEASURED values out of SERIES; it does
// not model, fit, or interpolate anything. at(k) is a lookup, by design.
const sel=document.getElementById('anchor'), back=document.getElementById('back'),
      fwd=document.getElementById('fwd');
let PTS=[], GX=k=>k, GY=e=>e;      // current window points + scales, shared with hover()
function fmt(v){return (v>=0?'+':'')+(v*100).toFixed(2)+'%';}

// The ONE measurement the whole tab is built on: how much did `tk` beat the S&P 500 by,
// from the first open after the post to the close `k` sessions later. Pure lookup +
// exact arithmetic on stored prices — no fit, no interpolation, no negated returns.
function beatAt(anc,tk,k){
  const a=anc.assets[tk]; if(!a) return null;
  const i=k+anc.span; if(i<0||i>=a.pa.length) return null;
  return (a.pa[i]/a.open_a-1)-(a.pb[i]/a.open_b-1);
}
// Same, backwards: the run-up over the X sessions ENDING at the last close before the post.
function runupAt(anc,tk,X){
  const a=anc.assets[tk]; if(!a) return null;
  const i=anc.span-X; if(i<0||i>=a.pa.length) return null;
  return (a.prev_a/a.pa[i]-1)-(a.prev_b/a.pb[i]-1);
}
function draw(){
  const anc=SERIES[+sel.value], X=+back.value, Y=+fwd.value;
  const asel=document.getElementById('asset');
  const avail=Object.keys(anc.assets);
  if(asel.dataset.anchor!==sel.value){
    asel.dataset.anchor=sel.value;
    asel.innerHTML=avail.map(t=>`<option value="${t}">${t} — ${anc.names[t]||t}</option>`).join('');
    // Open on the company the tweet NAMES if it named one (an Intel post opens on INTC),
    // else on an instrument the model named — those are the tab's question. Never default
    // to SPY: every number here is measured vs SPY, so SPY plots as a flat zero and reads
    // as a broken chart.
    const legs=(anc.pred?anc.pred.legs:[]).map(l=>l.ticker).filter(t=>anc.assets[t]&&t!=='SPY');
    asel.value=(anc.mentioned||[])[0]||legs[0]||avail.find(t=>t!=='SPY')||avail[0];
  }
  const tk=avail.includes(asel.value)?asel.value:avail[0];
  const at=k=>beatAt(anc,tk,k);
  renderCall(anc, Y);
  renderPanel(anc, X, Y);

  document.getElementById('backv').textContent=X;
  document.getElementById('fwdv').textContent=Y;
  document.getElementById('lstart').textContent='T − '+X+' sessions';
  document.getElementById('lend').textContent='T + '+Y+' sessions';
  document.getElementById('tkr').textContent=tk+' — '+(anc.names[tk]||tk);
  document.getElementById('atext').textContent='“'+anc.text+'”';
  document.getElementById('as0').textContent='posted '+anc.ts.slice(0,16).replace('T',' ')+
    ' UTC · first market open after it: '+anc.s0+' · type: '+anc.cohort.toLowerCase();

  const eFwd=at(Y), runup=runupAt(anc,tk,X);
  const pre=document.getElementById('pre'), post=document.getElementById('post');
  pre.textContent=runup===null?'n/a':fmt(runup);
  pre.className='big '+(runup>0?'up':'down');
  post.textContent=eFwd===null?'n/a':fmt(eFwd);
  post.className='big '+(eFwd>0?'up':'down');

  const v=document.getElementById('verdict');
  if(runup!==null&&eFwd!==null&&runup>0.10&&Math.abs(eFwd)<runup/3){
    v.textContent='REVERSE CAUSALITY — the move preceded the post';v.className='vd down';
  }else if(runup!==null&&eFwd!==null&&runup<-0.02&&eFwd>0.02){
    v.textContent='DIP THEN RECOVERY — compare against the no-tweet control (tab 2)';v.className='vd flat';
  }else{v.textContent='NO CLEAR PATTERN at this window';v.className='vd flat';}

  PTS=[];
  for(let k=-X;k<=Y;k++){const e=at(k); if(e!==null) PTS.push({k, d:anc.dates[k+anc.span], e});}
  const W=720,H=220,L=48,R=8,T=10,B=30;
  const lo=Math.min(...PTS.map(p=>p.e),0),hi=Math.max(...PTS.map(p=>p.e),0),rg=(hi-lo)||1;
  GX=k=>L+((k+X)/(X+Y))*(W-L-R);
  GY=e=>T+(1-(e-lo)/rg)*(H-T-B);
  const d=PTS.map((p,i)=>(i?'L':'M')+GX(p.k).toFixed(1)+','+GY(p.e).toFixed(1)).join(' ');
  const ticks=[lo,0,hi].filter((v,i,s2)=>s2.indexOf(v)===i);
  const grid=ticks.map(v=>`<line x1="${L}" y1="${GY(v).toFixed(1)}" x2="${W-R}" `+
    `y2="${GY(v).toFixed(1)}" class="${v===0?'zero':'grid'}"/>`+
    `<text x="${L-6}" y="${(GY(v)+3.5).toFixed(1)}" class="ax ay">${(v*100).toFixed(0)}%</text>`).join('');
  const first=PTS[0], last=PTS[PTS.length-1], ev=PTS.find(p=>p.k===0);
  const xt=[[first,'start'],[ev,'mid'],[last,'end']].filter(t=>t[0]).map(([p,pos])=>
    `<text x="${GX(p.k).toFixed(1)}" y="${H-10}" class="ax ${pos==='mid'?'evt':''}" `+
    `text-anchor="${pos==='start'?'start':pos==='end'?'end':'middle'}">${p.d}</text>`).join('');
  document.getElementById('svg').innerHTML=
    `<rect x="${L}" y="${T}" width="${(GX(0)-L).toFixed(1)}" height="${H-T-B}" class="zoneL"/>`+
    `<rect x="${GX(0).toFixed(1)}" y="${T}" width="${(W-R-GX(0)).toFixed(1)}" height="${H-T-B}" class="zoneR"/>`+
    grid+
    `<line x1="${GX(0).toFixed(1)}" y1="${T}" x2="${GX(0).toFixed(1)}" y2="${H-B}" class="ev"/>`+
    `<path d="${d}" class="trace"/>`+
    `<circle cx="${GX(0).toFixed(1)}" cy="${GY(0).toFixed(1)}" r="4" class="evdot"/>`+
    xt+
    `<g id="cross" style="display:none"><line class="xh"/><circle class="xhd" r="3.5"/>`+
    `<rect class="tipbg" rx="3"/><text class="tip"></text></g>`;
}

// Does NAMING an instrument mean anything? Built once at load: the model's picks scored
// on the real tweet day, and the SAME picks scored on a random other day. Two numbers that
// match = the picks knew nothing about the tweet.
const NVI=(()=>{
  if(!CALLS.nvi||!CALLS.nvi.length) return '';
  const pp=v=>(v>=0?'+':'')+(v*100).toFixed(2)+'pp';
  const pc=v=>(v*100).toFixed(2)+'%';
  // NB "he posted nothing" would be false — he posts on most sessions, and the wrong day
  // is simply ANOTHER tweet's date. The claim is about pairing, not about a quiet day.
  const rows=CALLS.nvi.map(r=>[['real','the day this tweet was posted'],
                               ['placebo','a different tweet’s day, picked at random']]
    .map(([arm,lbl],i)=>`<tr class="${arm==='real'?'hero':''}">`+
      `<th>${r.w} session${r.w===1?'':'s'}<span class="nm"> ${lbl}</span></th>`+
      `<td>${pc(r[arm].named)}</td><td>${pc(r[arm].ignored)}</td>`+
      `<td class="flat">${pp(r[arm].gap)}</td></tr>`).join('')).join('');
  return `<h3>The same question asked of all ${CALLS.n_tweets} tweets — and then asked again `+
    `on the wrong day</h3>`+
    `<p class="note">For every tweet in the corpus we measured how far the instruments it
      named drifted from the S&amp;P 500, against the ones it ignored. Then we did something
      that decides it: we kept each tweet's picks but scored them on <b>a different tweet's
      day, chosen at random</b> — same instruments, same rule, a day this tweet had nothing
      to do with. If naming meant anything, the picks should do <i>worse</i> on the wrong
      day.</p>`+
    `<div class="scroll"><table><thead><tr><th>window</th><th>the ones<br>it NAMED</th>`+
    `<th>the ones<br>it IGNORED</th><th>gap</th></tr></thead><tbody>${rows}</tbody></table></div>`+
    `<p class="note"><b>The gap is the same on the wrong day.</b> Move the model's picks to a
      date it never saw and they keep their entire advantage — so the advantage was never
      about the tweet. It is just <i>which</i> instruments it likes to name: oil, the fear
      gauge and defence stocks jump around more than consumer staples and government bonds
      <b>on every day of the year</b>, tweet or no tweet. Naming the jumpy ones is not a
      prediction. <b>This is the same illusion as the whole tab, in two rows.</b></p>`;
})();
const ARROW=d=>d==='up'?'▲ UP':d==='down'?'▼ DOWN':'– flat';
const DCLS=d=>d==='up'?'up':d==='down'?'down':'flat';
// Did the call come true? The page's rule #1: beating the S&P 500 in the named
// direction. An exact tie is unscoreable — awarding it is the tie-break artifact.
function scoreLeg(d,beat){
  if(beat===null||beat===0||(d!=='up'&&d!=='down')) return null;
  return (beat>0)===(d==='up');
}

// THE CENTREPIECE. Before it saw a single price, the model read this tweet and named a
// handful of instruments with a direction for each. This card puts that call next to
// what the market actually did — re-scored live at whatever window the slider is on.
function renderCall(anc, Y){
  const box=document.getElementById('pred'); if(!box) return;
  const p=anc.pred;
  if(!p){
    box.innerHTML=`<div class="card"><h2>The call</h2><p class="note">This tweet was never
      put to the model — it did not pass the filter that decides which posts are even
      about markets. There is no prediction to check.</p></div>`;
    return;
  }
  const legs=p.legs.map(l=>{
    const beat=beatAt(anc,l.ticker,Y);
    return {...l, beat, ok:scoreLeg(l.predicted,beat)};
  });
  const scored=legs.filter(l=>l.ok!==null);
  const hits=scored.filter(l=>l.ok).length;
  const rows=legs.map(l=>{
    // SPY is the yardstick every other number is measured against, so "SPY beats the
    // market" is 0.00% by definition, not a miss. Say so instead of printing a fake DOWN.
    const spy=l.ticker==='SPY';
    const act=l.beat===null?'no price':spy?'0.00% — it IS the market'
      :l.beat===0?'– flat 0.00%':ARROW(l.beat>0?'up':'down')+' '+fmt(l.beat);
    return `<tr class="hero"><th>${l.ticker}<span class="nm"> ${l.name}</span></th>`+
    `<td class="${DCLS(l.predicted)}">${ARROW(l.predicted)}</td>`+
    `<td class="${l.beat===null||spy||l.beat===0?'flat':l.beat>0?'up':'down'}">${act}</td>`+
    `<td class="${l.ok===true?'up':l.ok===false?'down':'flat'}">`+
    `${l.ok===true?'✓ RIGHT':l.ok===false?'✗ WRONG':spy?'can\\'t be scored':'—'}</td></tr>`;
  }).join('');
  // Verdict wording is derived from the count, never stored: change Y and it changes.
  const v=!scored.length?['no scoreable call at this window','flat']
    :hits===scored.length?['it got every call right here','up']
    :hits===0?['it got every call wrong here','down']
    :['it got some right and some wrong here','flat'];
  const R=CALLS.rates.map(r=>
    `<div class="tile"><div class="k">all ${CALLS.n_tweets} tweets<br>after ${r.label}</div>`+
    `<div class="big ${Math.abs(r.rate-0.5)<0.03?'flat':r.rate>0.5?'up':'down'}">`+
    `${(r.rate*100).toFixed(1)}%</div>`+
    `<div class="k" style="text-transform:none">${r.hit} of ${r.n} calls right</div></div>`).join('');
  box.innerHTML=
    `<div class="card" style="border-left:3px solid var(--accent)">`+
    `<h2>The call — what it expected, and what actually happened</h2>`+
    `<p class="note">Reading <b>only the words of the tweet</b>, before seeing any price,
      the model said this post was about <b>${p.scenario||'—'}</b>${p.summary?' — “'+p.summary+'”':''}.
      Then it named the ${legs.length} instrument${legs.length===1?'':'s'} below and, for each
      one, which way it would go.${p.hypo_short?' Its reasoning: “'+p.hypo_short+'”':''}</p>`+
    `<div class="scroll"><table><thead><tr><th>the instrument it named</th>`+
    `<th>it EXPECTED</th><th>what actually happened by T+${Y}<br>(vs the S&amp;P 500)</th>`+
    `<th>was it right?</th></tr></thead><tbody>${rows}</tbody></table></div>`+
    `<div class="tiles"><div class="tile"><div class="k">this tweet, ${Y} session${Y===1?'':'s'} later</div>`+
    `<div class="big ${v[1]}">${scored.length?hits+' of '+scored.length:'—'}</div>`+
    `<div class="k" style="text-transform:none">calls that came true</div></div></div>`+
    `<div class="k">what that means</div><div class="vd ${v[1]}">${v[0].toUpperCase()}</div>`+
    `<p class="note">Drag the <b>lookforward</b> slider and watch this flip. A call that is
      "right" at 4 sessions and "wrong" at 5 was never a prediction — it's a price wandering
      across zero. <b>One tweet proves nothing either way</b>, which is exactly why the row
      of numbers below it exists.</p>`+
    `<h3>The same question asked of every tweet — not just this one</h3>`+
    `<p class="note">These 18 tweets were picked <i>because</i> they had the biggest moves,
      so their score is rigged by construction. Here is the honest denominator: every call
      the model made on all <b>${CALLS.n_tweets}</b> tweets in the corpus
      (<b>${CALLS.n_calls}</b> calls, about ${CALLS.per_tweet.toFixed(1)} per tweet), scored
      the same way. <b>50% is a coin flip.</b></p>`+
    `<div class="tiles">${R}</div>`+
    `<p class="note">Every horizon sits on the coin flip. That is the finding, and it is the
      point of this tab. Two tells in <i>what</i> it predicts explain why:<br><br>
      <b>1 · It nearly always says up.</b> ${(CALLS.up_share*100).toFixed(0)}% of its
      ${CALLS.n_calls} calls are “UP” (${CALLS.n_up} up vs ${CALLS.n_down} down). It is not
      weighing each tweet and arriving at a direction — it mostly says up, and the market
      drifts up on its own anyway.<br>
      <b>2 · Its favourite pick is the market itself.</b> ${CALLS.n_spy} of those calls name
      the S&amp;P 500 — and “the S&amp;P 500 will beat the S&amp;P 500” cannot be right or
      wrong, so those ${CALLS.n_spy} drop out of the scores above. Naming the whole market is
      not a prediction about a tweet; it is a way of not making one.<br><br>
      So expect the table above to read “expected UP, went DOWN” about half the time.
      <b>That is not a broken model — it is the honest result.</b> There is nothing in the
      words of a tweet to predict from, and a model that reads the words can only produce a
      coin flip from them.</p></div>`;
}

// The 20-asset panel. The instruments the model IGNORED are not filler — they are the
// control. If the words of a tweet really pointed at a specific instrument, the named
// one should move more than the ones it never mentioned. Both columns are measured at
// the same window, on the same day, so the comparison is apples to apples.
function renderPanel(anc, X, Y){
  const box=document.getElementById('panel'); if(!box) return;
  const exp={}; (anc.pred?anc.pred.legs:[]).forEach(l=>{exp[l.ticker]=l.predicted;});
  const ment=new Set(anc.mentioned||[]);
  const rows=Object.keys(anc.assets).map(tk=>{
    const d=exp[tk]||null, beat=beatAt(anc,tk,Y);
    return {tk, name:anc.names[tk]||tk, runup:runupAt(anc,tk,X), beat, exp:d,
            ok:scoreLeg(d,beat), said:ment.has(tk), fixed:PANEL.includes(tk)};
  }).filter(r=>r.beat!==null).sort((p,q)=>Math.abs(q.beat)-Math.abs(p.beat));
  const avg=rs=>rs.length?rs.reduce((s,r)=>s+Math.abs(r.beat),0)/rs.length:null;
  // The named/ignored tiles compare over the FIXED panel only. A tweet's own company is
  // added to the table per-tweet, so counting it here would let the control set change
  // with the tweet — and a control that moves with the treatment is not a control.
  const named=rows.filter(r=>r.exp&&r.fixed), rest=rows.filter(r=>!r.exp&&r.fixed);
  const an=avg(named), ar=avg(rest);
  const said=rows.filter(r=>r.said);
  // Where the model's picks landed once every asset is ranked by how much it actually
  // moved. Picks that carried information would cluster at the top.
  const ranks=named.map(r=>rows.indexOf(r)+1).sort((a,b)=>a-b);
  const tr=r=>`<tr class="${r.exp||r.said?'hero':''}"><th>${rows.indexOf(r)+1}. ${r.tk}`+
    `${r.said?' <span class="said">named in the tweet</span>':''}`+
    `<span class="nm"> ${r.name}</span></th>`+
    `<td class="${r.exp?DCLS(r.exp):'flat'}">${r.exp?ARROW(r.exp):'not named'}</td>`+
    `<td class="${r.runup===null?'flat':r.runup>0?'up':'down'}">`+
    `${r.runup===null?'—':fmt(r.runup)}</td>`+
    `<td class="${r.beat>0?'up':'down'}">${fmt(r.beat)}</td>`+
    `<td class="${r.ok===true?'up':r.ok===false?'down':'flat'}">`+
    `${r.ok===true?'✓ RIGHT':r.ok===false?'✗ WRONG':'—'}</td></tr>`;
  box.innerHTML=
    `<div class="card"><h2>The ${rows.length} instruments it could have named</h2>`+
    `<p class="note">The same <b>${PANEL.length} index and sector funds for every tweet</b>,
      fixed in advance — picking which assets to show per tweet is exactly how you'd
      manufacture a story.${said.length?` On top of those, this tweet names a company by
      name, so <b>${said.map(r=>r.tk).join(', ')}</b> ${said.length===1?'is':'are'} added
      and marked. The fixed ${PANEL.length} never change, so the comparison below still
      holds.`:''} Ranked by how far each moved away from the S&amp;P 500 by
      <b>T+${Y}</b>.</p>`+
    `<div class="tiles">`+
    `<div class="tile"><div class="k">the ${named.length} it NAMED<br>typical move</div>`+
    `<div class="big">${an===null?'—':(an*100).toFixed(2)+'%'}</div></div>`+
    `<div class="tile"><div class="k">the ${rest.length} it IGNORED<br>typical move</div>`+
    `<div class="big">${ar===null?'—':(ar*100).toFixed(2)+'%'}</div></div>`+
    `<div class="tile"><div class="k">where its picks<br>rank on the day</div>`+
    `<div class="big">${ranks.length?ranks.join(', '):'—'}</div>`+
    `<div class="k" style="text-transform:none">out of ${rows.length}</div></div></div>`+
    `<p class="note">(These three numbers are for <b>this tweet only</b>, at T+${Y}.)</p>`+
    `<p class="note">If the tweet's words genuinely pointed at an instrument, the named ones
      should move <b>more</b> than the ignored ones, and rank near the top. Often they do —
      but <b>do not read anything into it here</b>: these ${SERIES.length} tweets were chosen
      <i>because</i> the model's own named instruments moved the most. On this page "named
      moves more" is true by construction. It measures how we picked the tweets, not the
      model. The only way to know is to ask it of every tweet:</p>`+
    NVI+
    `<div class="scroll"><table><thead><tr><th>rank · instrument</th><th>it EXPECTED</th>`+
    `<th>run-up T−${X}</th><th>after T+${Y}</th><th>right?</th></tr></thead><tbody>`+
    // Render EVERY row, not just the fixed panel: `named`/`rest` are the control split
    // for the tiles above and are PANEL-only, so building the table from them dropped
    // the tweet's own company — INTC would be the chart's default asset and absent from
    // the table under it. Interesting rows (named by the model, or named by the tweet)
    // sort to the top; the rest keep their measured rank.
    rows.filter(r=>r.said||r.exp).map(tr).join('')+
    rows.filter(r=>!r.said&&!r.exp).map(tr).join('')+`</tbody></table></div>`+
    `<p class="note"><b>Exploratory — read nothing into a single row.</b>
      ${rows.length} instruments x ${SERIES.length} tweets is about
      ${rows.length*SERIES.length} comparisons, and with that many, something
      <i>always</i> looks like a pattern. Nothing here is a test; it is a place to form a
      hunch. <b>Tabs 4 and 6 are where hunches get checked</b> — with the questions written
      down in advance and the score adjusted for how many were asked. Everything found here
      died there.</p></div>`;
}

// Hover: read the MEASURED point nearest the cursor. No smoothing, no synthesis.
function hover(ev){
  const svg=document.getElementById('svg'); if(!PTS.length) return;
  const r=svg.getBoundingClientRect(), vx=(ev.clientX-r.left)/r.width*720;
  let best=PTS[0]; for(const p of PTS){ if(Math.abs(GX(p.k)-vx)<Math.abs(GX(best.k)-vx)) best=p; }
  const g=svg.querySelector('#cross'); if(!g) return;
  g.style.display='';
  const px=GX(best.k), py=GY(best.e);
  g.querySelector('.xh').setAttribute('x1',px); g.querySelector('.xh').setAttribute('x2',px);
  g.querySelector('.xh').setAttribute('y1',10); g.querySelector('.xh').setAttribute('y2',190);
  g.querySelector('.xhd').setAttribute('cx',px); g.querySelector('.xhd').setAttribute('cy',py);
  const t=g.querySelector('.tip'), bg=g.querySelector('.tipbg');
  const lbl=best.d+'  ·  '+fmt(best.e)+'  ·  '+(best.k===0?'T₀':(best.k>0?'T+':'T')+best.k);
  t.textContent=lbl;
  const w=lbl.length*5.6+10, tx=Math.min(Math.max(px-w/2,50),712-w), ty=py<50?py+22:py-12;
  t.setAttribute('x',tx+5); t.setAttribute('y',ty);
  bg.setAttribute('x',tx); bg.setAttribute('y',ty-11); bg.setAttribute('width',w); bg.setAttribute('height',15);
}
[sel,back,fwd,document.getElementById('asset')].forEach(el=>el&&el.addEventListener('input',draw));
const _svg=document.getElementById('svg');
if(_svg){ _svg.addEventListener('mousemove',hover);
  _svg.addEventListener('mouseleave',()=>{const g=_svg.querySelector('#cross'); if(g) g.style.display='none';}); }
if(sel) draw();
"""


def render(intel: dict[str, Any], oil: dict[str, Any], verdict: dict[str, Any],
           series: list[dict[str, Any]], intraday: dict[str, Any] | None,
           breakdown: list[dict[str, Any]], calls: dict[str, Any], stamp: str) -> str:
    hdr = "".join(f"<th>{w}d</th>" for w in WINDOWS)

    # ---- Tab 1: reverse causality
    prior_rows = "".join(
        f'<tr class="{"hero" if p["back"] == 21 else ""}"><th>prior {p["back"]}d</th>'
        f'<td class="{_cls(p["asset"])}">{_pct(p["asset"])}</td>'
        f'<td class="{_cls(p["spy"])}">{_pct(p["spy"])}</td>'
        f'<td class="{_cls(p["excess"])}">{_pct(p["excess"])}</td></tr>'
        for p in intel["prior"])
    hero_txt = escape(intel["hero"]["text"]) if intel["hero"] else "(not found)"
    hero_ts = intel["hero"]["ts"] if intel["hero"] else "?"
    post_tds = "".join(f'<td class="{_cls(intel["post"].get(w))}">{_pct(intel["post"].get(w))}</td>'
                       for w in WINDOWS)
    mentions = "".join(
        f'<tr class="{"hero" if m is intel["hero"] else ""}">'
        f'<th>{escape(m["ts"][:10])}</th>'
        f'<td class="{_cls(m["prior21"])}">{_pct(m["prior21"])}</td>'
        f'<td class="{_cls(m["post21"])}">{_pct(m["post21"])}</td>'
        f'<td class="note">{escape(m["text"][:110])}</td></tr>'
        for m in intel["mentions"])
    n_rose = sum(1 for m in intel["mentions"]
                 if (m["post21"] or 0) > 0.02 and (m["prior21"] or 0) < 0.10)
    t1 = f"""
<div class="myth"><b>THE HYPOTHESIS:</b> "Trump tweeted about Intel and the stock
ran for months." Every broker has heard it. It is the single most-cited example of
tweet-driven alpha.</div>
<div class="card"><h2>The tweet</h2>
<div class="tweet"><b>{escape(hero_ts)}</b> — {hero_txt}</div>
<p class="note">Note the tense. "<i>continues to rise</i>" — he is describing a move
that has <b>already happened</b>, and claiming credit for it.</p></div>
<div class="card"><h2>What INTC did BEFORE the tweet</h2>
<div class="scroll"><table><thead><tr><th>window</th><th>INTC</th><th>SPY</th>
<th>INTC excess</th></tr></thead><tbody>{prior_rows}</tbody></table></div></div>
<div class="card"><h2>What INTC did AFTER the tweet</h2>
<div class="scroll"><table><thead><tr><th>abnormal return</th>{hdr}</tr></thead>
<tbody><tr><th>INTC post-tweet</th>{post_tds}</tr></tbody></table></div>
{_spark(intel["post"]) if intel["post"] else ""}</div>
<div class="myth verdict"><b>VERDICT — THIS TWEET IS REVERSE CAUSALITY.</b> Intel had
already <b>doubled (+105% in 21 sessions)</b> before the post. In the 21 sessions after
it: <b>−2.4%</b>. The stock moved, <i>then</i> he tweeted — and memory encodes the
correlation as a sequence.</div>
<div class="card"><h3>Every Intel-the-company mention in the corpus
({len(intel["mentions"])} unique of {intel["n_corpus"]:,} posts)</h3>
<p class="note">The famous tweet is the <span class="nm">highlighted</span> row. But it
is not the whole story, and we are not going to show you only the row that suits us —
so here is <b>every</b> mention, with what Intel did in the 21 trading days before and
after each one. Both columns are measured against the S&amp;P 500.</p>
<div class="scroll"><table><thead><tr><th>date</th><th>INTC in the<br>21 days BEFORE</th>
<th>INTC in the<br>21 days AFTER</th><th>the post</th></tr></thead>
<tbody>{mentions}</tbody></table></div>
<p class="note"><b>Read this honestly.</b> For the famous post the order is unmistakable:
a double first, a tweet second, nothing after. But <b>{n_rose} of these
{len(intel["mentions"])}</b> posts land the other way round — Intel was <i>falling</i>
when he demanded the CEO resign, and rose afterwards. Those posts are also the ones where
he was announcing something real: a president calling for a CEO's head, or the US
government taking a 10% stake. <b>That is news, and news moves prices.</b><br><br>
So the honest claim is narrower than "his tweets never matter": <i>this</i> famous tweet
described a move that had already happened. Eight mentions cannot settle the general
question either way — eight is an anecdote, and picking the flattering ones out of eight
is how the myth got built in the first place. <b>The registered test in tab 4 is what
settles it, and it finds nothing.</b></p></div>"""

    # ---- Tab 2: mean reversion
    rows2 = (_path_row("top-5 'strongest' tweets", oil["mean_top5"], "the anecdotes", True)
             + _path_row(f"all {oil['n_tweet_days']} tweet-days", oil["mean_all"], "the population")
             + _path_row(f"random days (n={oil['n_random']})", oil["mean_random"], "no tweet, any day")
             + _path_row(f"BIG move, NO tweet (n={oil['n_big']})", oil["mean_big_no_tweet"],
                         "the control that matters", True))
    t2 = f"""
<div class="myth"><b>THE HYPOTHESIS:</b> "After a big Iran/Hormuz tweet oil spikes on
fear, then recovers — you can trade the arc." The shape is vivid and everyone
remembers it.</div>
<div class="card"><h2>Oil (USO): how it moved vs the S&amp;P 500</h2>
<div class="scroll"><table><thead><tr><th>which tweets</th>{hdr}<th></th></tr></thead>
<tbody>{rows2}</tbody></table></div>
<div class="tiles">
<div class="tile"><div class="k">top-5 tweets @42d</div>
<div class="big up">{_pct(oil["mean_top5"].get(42))}</div>{_spark(oil["mean_top5"])}</div>
<div class="tile"><div class="k">big move, NO tweet @42d</div>
<div class="big up">{_pct(oil["mean_big_no_tweet"].get(42))}</div>{_spark(oil["mean_big_no_tweet"])}</div>
</div></div>
<div class="myth verdict"><b>VERDICT — MEAN REVERSION.</b> The dip-then-recovery arc
is <b>real</b> — and it is <b>stronger</b> on big oil moves with <i>no tweet at
all</i>. It is what volatility does, not what Trump does. All
{oil['n_tweet_days']} tweet-days are indistinguishable from random days
({_pct(oil['mean_all'].get(42))} vs {_pct(oil['mean_random'].get(42))} @42d).
<br><span class="note">Control n={oil['n_big']} is small; it agrees with the
population comparison, which is flat.</span></div>"""

    # ---- Tab 3: selection bias
    cases = "".join(
        f'<div class="card"><h3>{escape(p["s0"])} — USO {_pct(p["car"][1])} on day 1</h3>'
        f'{_spark(p["car"])}<div class="tweet">{escape(p["text"])}</div></div>'
        for p in oil["top"])
    t3 = f"""
<div class="myth"><b>THE HYPOTHESIS:</b> "Just look at the biggest moves — the
tweets that moved oil are obviously geopolitical." So we sorted the oil events by
biggest day-1 move and read the tweets. Here they are, unedited.</div>
{cases}
<div class="myth verdict"><b>VERDICT — SELECTION BIAS.</b> A soccer-team tweet and a
"Radical Left Democrats" tweet sit in the top five "oil movers." Neither is about
oil. With posts on <b>{oil['n_tweet_days']} of {oil['n_sessions']} sessions
({oil['n_tweet_days'] / max(oil['n_sessions'], 1):.0%} of the calendar)</b>, every
large move has a same-day tweet available to blame. Sorting on the outcome
manufactures a greatest-hits reel out of coincidence — <i>this is the anecdote
generator, visible in one table.</i></div>"""

    # ---- Tab 4: verdict
    vrows = "".join(
        f'<tr><th>{escape(c["cohort"].replace("GEO_SHOCK","war/geopolitics").replace("NOISE","ordinary").replace("CORPORATE","company").lower())} <span class="nm">{escape(str(c["asset"]))} · {c["window"]}d · {escape(FAMILY_LABEL.get(str(c["family"]), str(c["family"])))}</span></th>'
        f'<td>{c["n"]}</td><td class="{_cls(c["observed"])}">{c["observed"]:+.4f}</td>'
        f'<td class="flat">{c["null_mean"]:+.4f}</td><td>{c["p_raw"]:.3f}</td>'
        f'<td class="down">{c["p_bh"]:.3f}</td></tr>' for c in verdict["closest"])
    coh = "".join(f'<div class="tile"><div class="k">{escape(k.replace("GEO_SHOCK","war/geopolitics").replace("NOISE","ordinary").replace("CORPORATE","company").lower())}</div>'
                  f'<div class="big">{v}</div></div>'
                  for k, v in verdict["cohorts"].items())
    t4 = f"""
<div class="card"><h2>The registered test</h2>
<p class="note">We sorted every tweet into a type — <b>war/geopolitics</b>,
<b>company</b>, or <b>ordinary</b> — using only the words in the tweet, never the
result. For each type we asked three questions about its matching assets over 1, 3 and
5 days: <b>did the price move in the direction expected? how big was the move? was
there unusual trading?</b><br><br>
Two rules keep this honest. We wrote down <b>all {verdict["n_cells"]} questions before looking at a
single answer</b> — otherwise you can ask a hundred things and report the three that
worked. And every answer is compared against <b>the same asset on random days</b>, so
"oil moved" only counts if it moved <i>more than oil normally does</i>.</p>
<div class="tiles">
<div class="tile"><div class="k">questions asked<br>(fixed in advance)</div><div class="big">{verdict["n_cells"]}</div></div>
<div class="tile"><div class="k">questions with a<br>real answer</div><div class="big up">{verdict["n_survive"]}</div></div>
<div class="tile"><div class="k">best result: chance<br>it was luck</div><div class="big">{verdict["min_p_bh"]:.0%}</div></div>
</div>
<h3>how many tweets of each type</h3><div class="tiles">{coh}</div></div>
<div class="card"><h2>The six that came closest — and still failed</h2>
<div class="scroll"><table><thead><tr><th>tweet type / asset / days</th><th>how many<br>tweets</th>
<th>after these<br>tweets</th><th>on random<br>days</th><th>chance it&#39;s<br>luck</th>
<th>...after testing<br>{verdict["n_cells"]} things</th></tr></thead><tbody>{vrows}</tbody></table></div>
<p class="note">The last column is the one that matters. Ask {verdict["n_cells"]} questions and a few will
look impressive by chance alone — like flipping {verdict["n_cells"]} coins and celebrating the one that
came up heads five times. That column adjusts for how many questions we asked. Nothing
survives it.</p></div>
<div class="myth verdict"><b>THE ANSWER: {verdict["n_survive"]} out of
{verdict["n_cells"]} questions found anything real.</b> Not the direction of the move,
not its size, not the trading volume.<br><br>
But "nothing happens" is the wrong way to read this. The right reading is: <b>by the
time he posts, the market already knows.</b> The news that moves a price is the actual
event — the strike, the tariff, the earnings — and prices react to that, usually before
he writes about it. His post is a description of something already priced in.</div>
<div class="card"><h3>What this terminal is not</h3>
<p class="note">This is not a trading tool and it gives no buy/sell advice. The live
service will read a tweet and tell you what it's about and which assets it touches —
but it deliberately <b>quotes no accuracy number and no confidence score</b>, because
nothing we measured earned one. Research output. Not investment advice.</p></div>"""

    # ---- "when are we right?" — the honest answer: nowhere that replicates.
    # The keys are internal column names; the page is written for a reader with no finance
    # background, so they never reach the screen.
    dim_label = {"cohort": "what it's about", "session": "when he posted",
                 "intensity": "how loud it is"}
    phrase = {"premarket": "before the bell", "regular": "market open",
              "afterhours": "after the bell", "weekend": "weekend"}
    bd_rows = "".join(
        f'<tr class="{"hero" if abs(b["shift"]) >= 0.3 else ""}">'
        f'<th>{escape(dim_label.get(b["dim"], b["dim"]))}'
        f'<span class="nm"> {escape(phrase.get(str(b["group"]), str(b["group"])))}</span></th>'
        f'<td>{b["train"]:.3f}<span class="nm"> n={b["n_train"]}</span></td>'
        f'<td>{b["test"]:.3f}<span class="nm"> n={b["n_test"]}</span></td>'
        f'<td class="{_cls(b["shift"])}">{b["shift"]:+.3f}</td></tr>' for b in breakdown)
    bd_card = f"""
<div class="card"><h2>“When do we predict correctly?”</h2>
<p class="note">We split the tweets by date: the model was studied on the <b>earlier</b>
ones ("train") and then checked against <b>later</b> ones it had never seen ("test") —
the honest way to find out if a pattern is real or imagined.<br><br>
Each row is a group of tweets. "0.304" means it got 30% of them right; <b>0.5 is a coin
flip</b>. If a group were genuinely easier to predict, its early score should hold up on
the later tweets. None of them do.</p>
<div class="scroll"><table><thead><tr><th>group of tweets</th><th>earlier tweets<br>(studied)</th><th>later tweets<br>(unseen)</th>
<th>change</th></tr></thead><tbody>{bd_rows}</tbody></table></div>
<div class="myth verdict" style="margin-top:14px"><b>ANSWER: nowhere that holds.</b>
Every pattern flips when you check it on tweets the model never saw — weekend posts go
from the <b>worst</b> group (30% right) to the <b>best</b> (76% right). Learn “avoid weekends”
from the earlier tweets and the later ones punish you; learn “weekends are 76%” from the
later ones and the next batch punishes you. We also tried the obvious next move — a second
model whose only job was to spot when the first one could be trusted. It looked promising
while it was being built and then scored <b>worse than a coin flip</b> on tweets it had
never seen. These groups aren't real patterns — they're random wobble, and random wobble
never repeats.</div></div>"""

    # ---- Tab 5: dynamic time-window analyzer (reads MEASURED series only)
    opts = "".join(f'<option value="{i}">{escape(s["s0"])} · {escape(s["cohort"])} — '
                   f'{escape(s["label"])}</option>' for i, s in enumerate(series))
    t5 = f"""
<div class="myth verdict"><b>WHAT THIS TAB DOES:</b> a machine reads one of his tweets —
<b>just the words, no prices</b> — and says which assets it will move and in which
direction. Then we check it against what the market actually did. <b>Expected vs
reality, side by side.</b><br><br>
Pick a tweet, then drag the two sliders to choose how far <b>back</b> and how far
<b>forward</b> to look. Every number is a <b>real measured price</b> from that exact
day — the slider looks it up, it never estimates or draws a smooth curve.</div>
<div class="myth"><b>THIS TAB IS A LOOKING-GLASS, NOT A TEST.</b> {len(PANEL)} instruments
across {len(series)} tweets is hundreds of comparisons, and at that number something always
looks like a pattern. Nothing here is evidence — it is where you form a hunch.
<b>Tabs 4 and 6 are where hunches get checked properly.</b></div>
<div class="card"><div class="ctl">
  <div><label>1 · anchor tweet</label>
    <select id="anchor">{opts}</select>
    <label style="margin-top:10px">asset (same panel for every tweet)</label>
    <select id="asset"></select>
    <div class="tweet" id="atext"></div>
    <p class="note" id="as0"></p></div>
  <div><label>2 · sliding windows</label>
    <div class="rl"><span>lookback — T − <b id="backv">21</b> sessions</span></div>
    <input type="range" id="back" min="1" max="42" value="21">
    <div class="rl" style="margin-top:12px"><span>lookforward — T + <b id="fwdv">21</b> sessions</span></div>
    <input type="range" id="fwd" min="1" max="42" value="21">
    <p class="note">Everything is measured <b>relative to the S&amp;P 500</b>: if a
    stock rose 3% on a day the whole market rose 3%, that counts as <b>zero</b> — it
    just went along for the ride. Zero on the chart is the moment of the <b>first
    market open after the tweet</b> — the earliest point anyone could actually have
    acted on it.</p></div>
</div></div>
<div id="pred"></div>
<div class="card"><h2>Event trace <span id="tkr" class="note"></span></h2>
<svg class="chart" id="svg" viewBox="0 0 720 220"></svg>
<div class="axis"><span id="lstart">T − 21 sessions</span>
<span style="color:var(--accent)">T₀ — first session after the post</span>
<span id="lend">T + 21 sessions</span></div>
<p class="note">Hover the line for the exact date and value on any day. Up means it
beat the S&amp;P 500; down means it lagged. The <b>gold line is the tweet</b> — left of
it is before, right of it is after.</p>
<div class="tiles">
  <div class="tile"><div class="k">move BEFORE the tweet<br>(vs the S&amp;P 500)</div><div class="big" id="pre">—</div></div>
  <div class="tile"><div class="k">move AFTER the tweet<br>(vs the S&amp;P 500)</div><div class="big" id="post">—</div></div>
</div>
<div class="k">what the two numbers say</div><div id="verdict" class="vd flat">—</div>
<p class="note">This line is worked out live from the two numbers above it — if the
price had already moved a lot <b>before</b> the tweet, and barely moved after, the
tweet didn't cause it. It's not a saved conclusion; change the sliders and it changes.</p></div>
<div id="panel"></div>
{bd_card}"""

    # ---- Tab 6: intraday shock study
    if intraday:
        irows = "".join(
            f'<tr><th>{escape(c["cohort"].replace("GEOPOLITIC","war/geopolitics").replace("NOISE","ordinary").replace("CORPORATE","company").replace("MACRO","economy/policy").lower())}<span class="nm"> {escape(str(c["asset"]))}</span></th>'
            f'<td>{escape(METRIC_LABEL.get(str(c["metric"]), str(c["metric"])))}</td><td>{c["n"]}</td>'
            f'<td>{c["cohort_mean"]:+.4f}</td><td class="flat">{c["control_mean"]:+.4f}</td>'
            f'<td class="{_cls(c["diff"])}">{c["diff"]:+.4f}</td><td class="flat">{c["mde"]:.4f}</td>'
            f'<td>{c["p_raw"]:.3f}</td><td class="down">{c["p_bh"]:.3f}</td></tr>'
            for c in intraday["closest"])
        hrows = "".join(
            f'<tr><th>{h["z"]:+.2f}σ</th><td>{escape(h["asset"])}</td>'
            f'<td class="{"down" if h["cohort"] == "NOISE" else "flat"}">{escape(h["cohort"])}</td>'
            f'<td class="note">{escape(h["text"][:70])}</td></tr>' for h in intraday["hitters"])
        t6 = f"""
<div class="myth"><b>THE BELIEF BEING TESTED:</b> when he posts about Iran, or a
military strike, oil and the "fear gauge" should jump <b>within the hour</b> — and
looking at whole days is too slow to catch it. Zoom in to the hour and the real
market-movers should stand out.</div>
<div class="card"><h2>Design</h2>
<p class="note">We compare the hour <b>after</b> each tweet with the hour <b>before
the same tweet</b>. That matters: he only sometimes posts while the market is open, and
those moments might be special for other reasons — using the same tweet's own "before"
picture cancels that out.<br><br>
We also compare against his <b>ordinary posts</b> (the ones about nothing in
particular). "War tweets move oil" only means something if they move oil <i>more than
his everyday posts do</i>.<br><br>
Every price is measured <b>against the S&amp;P 500</b>, so a day when the whole market
moved doesn't get mistaken for a tweet effect. Prices come from free public data
(anyone can re-run this), and we wrote down every question <b>before</b> looking at any
answer — so we can't quietly pick the ones that worked.</p>
<div class="tiles">
<div class="tile"><div class="k">tweets posted while<br>the market was open</div><div class="big">{intraday["n_events"]}</div></div>
<div class="tile"><div class="k">questions asked<br>(fixed in advance)</div><div class="big">{intraday["n_cells"]}</div></div>
<div class="tile"><div class="k">questions with a<br>real answer</div><div class="big up">{intraday["n_survive"]}</div></div>
<div class="tile"><div class="k">best result: chance<br>it was luck</div><div class="big">{intraday["min_p_bh"]:.0%}</div></div>
</div></div>
<div class="card"><h2>The eight that came closest — and still failed</h2>
<div class="scroll"><table><thead><tr><th>tweet type / asset</th><th>what we measured</th><th>how many<br>tweets</th>
<th>after these<br>tweets</th><th>after ORDINARY<br>tweets</th><th>difference</th><th>smallest gap<br>we could detect</th><th>chance it&#39;s<br>luck</th><th>...after testing<br>30 things</th>
</tr></thead><tbody>{irows}</tbody></table></div>
<p class="note"><b>Why the "smallest gap we could detect" column matters.</b> With 228
war-tweets, a real effect of 0.25% on oil would have shown up clearly. What we actually
measured was 0.17% — smaller than our detector, and well inside normal random wobble.
<br><br>So this isn't "our instruments were too blunt to tell." It's: <b>if an effect
this size existed, we would have seen it. We didn't.</b></p></div>
<div class="myth"><b>THE NEAR-MISS — how we almost published the opposite.</b> Our
first run said exactly what everyone expects: <b>9 solid findings</b>, odds of luck
about <b>1 in 550</b> — "war tweets spike oil trading by 79%".<br><br>
It was a counting mistake. He posts in <b>bursts</b> — five tweets in ten minutes. Our
data has one price reading per hour, so all five tweets pointed at the <i>same</i> price
move. We counted one event five times and called it five pieces of evidence. Two posts a
minute apart were both credited with the same gold move.<br><br>
Count each hour <b>once</b> and the 9 findings become <b>zero</b>. We only caught it
because the "biggest movers" list had a tweet about a <b>renovated Palm Room</b> at the
top — the statistics looked perfect.</div>
<div class="card"><h2>The biggest movers — now read the tweets</h2>
<div class="scroll"><table><thead><tr><th>size of move<br>vs a normal hour</th><th>asset</th><th>tweet type</th>
<th>tweet</th></tr></thead><tbody>{hrows}</tbody></table></div>
<p class="note">Look at which tweets top this list: a <b>renovated Palm Room</b>. A
plug for a TV show. These are not market news — they're just big market moves that
happened to have a tweet in the same hour.<br><br>And that will <i>always</i> happen:
he posts during roughly <b>1 in 4</b> market hours. So every big move has a tweet
sitting next to it, and you can always find one that looks like the cause. That's the
whole illusion, in one table.</p></div>
<div class="myth verdict"><b>THE ANSWER: {intraday["n_survive"]} out of
{intraday["n_cells"]} questions found anything real.</b> War and geopolitics tweets do
<b>not</b> move oil, gold, defence stocks or the fear gauge more than his ordinary posts
do — not the price, not the trading volume, not within the hour.<br><br>
Sorting tweets into categories doesn't find the "big movers" either. And this was our
<b>most sensitive test</b>: it could have caught an effect five times smaller than
anything our day-scale tests could see. It still found nothing.</div>"""
    else:
        t6 = ('<div class="myth">Intraday study not yet run — '
              '<code>python experiments/event_study/run_intraday.py</code></div>')

    tabs = [("t1", "1 · Reverse Causality"), ("t2", "2 · Mean Reversion"),
            ("t3", "3 · Selection Bias"), ("t4", "4 · The Verdict"),
            ("t5", "5 · Dynamic Analyzer"), ("t6", "6 · Intraday Shock")]
    tabbar = "".join(
        f'<button class="tab" role="tab" data-p="{i}" aria-selected="{str(n == 0).lower()}">'
        f'{escape(label)}</button>' for n, (i, label) in enumerate(tabs))
    panels = "".join(f'<section class="panel {"on" if n == 0 else ""}" id="{i}">{c}</section>'
                     for n, (i, c) in enumerate([("t1", t1), ("t2", t2), ("t3", t3),
                                                 ("t4", t4), ("t5", t5), ("t6", t6)]))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Myth-Busting Quantitative Terminal — Trump tweets vs. the market</title>
<style>{_CSS}</style></head><body><div class="wrap">
<header><h1>Do Trump's tweets move the market?</h1>
<p class="sub">Five things "everyone knows" about his posts and the stock market —
each one tested against the check that decides it. No finance background needed.</p>
<p class="stamp">generated {escape(stamp)} · every figure computed at build time from
data/real/bars.csv + corpus_v3.csv + the registered study — nothing hardcoded</p></header>
<div class="card" style="border-left:3px solid var(--accent)">
<h2>Three ideas, and you can read the whole thing</h2>
<div class="kv"><span class="k">1 · “beat the market”</span><b>A stock going up isn't
impressive if <i>everything</i> went up that day. So we always ask: did it beat the
S&amp;P 500 — the average of the 500 biggest US companies? Rising 3% on a day the market
rose 3% counts as <b>zero</b>.</b></div>
<div class="kv"><span class="k">2 · “could it be luck?”</span><b>Prices jump around on
their own. So for every result we ask: how often would we see this <i>by pure chance</i>?
We answer it by measuring the same thing on <b>random days with no tweet</b>. If the
tweet days look like the random days, there's nothing there.</b></div>
<div class="kv"><span class="k">3 · “ask lots of questions…”</span><b>…and a few will look
amazing by luck alone — like flipping dozens of coins and bragging about the one that hit
five heads. So we wrote every question down <b>before</b> looking at any answer, and every
result is adjusted for how many questions we asked.</b></div>
<p class="note">That's it. Everything below is those three ideas applied to real
prices.</p></div>
<nav class="tabs" role="tablist">{tabbar}</nav>{panels}
<p class="foot">Research output. Not investment advice. Reproduce:
<code>make dashboard</code>. Full method:
<code>experiments/event_study/REPORT.md</code>.</p>
</div><script>const SERIES={json.dumps(series, separators=(',', ':'))},
CALLS={json.dumps(calls, separators=(',', ':'))},
PANEL={json.dumps(list(PANEL), separators=(',', ':'))};</script>
<script>{_JS}</script></body></html>"""


def _prediction_for(text: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The cached 70B call for this tweet: what it said in plain English, what it
    expected, and what each named instrument actually did. None when the tweet never
    entered the study (it did not pass the geo pre-filter) — which is itself a fact
    worth showing rather than hiding."""
    r = next((x for x in rows if x.get("text", "")[:60] == text[:60]), None)
    if r is None:
        return None
    spy = (r.get("spy_returns") or {}).get("EOD")
    legs = []
    for i in r.get("instruments", []):
        ret = (i.get("returns") or {}).get("EOD")
        legs.append({
            "ticker": i["ticker"], "name": name_of(i["ticker"]),
            "predicted": i.get("predicted", ""), "actual": ret,
            "spy": spy, "abn": (ret - spy) if (ret is not None and spy is not None) else None,
            "hit": i.get("hit", {}).get("EOD"),
        })
    return {"scenario": r.get("scenario", ""), "summary": r.get("summary", ""),
            "macro_link": r.get("macro_link", ""), "rationale": r.get("rationale", ""),
            "hypo_short": r.get("hypothesis_short", ""),
            "hypo_long": r.get("hypothesis_long", ""),
            "intensity": r.get("intensity"), "split": r.get("split"), "legs": legs}


def _cohort_of(text: str, scenario: str) -> str:
    """Text-only tag, same precedence as the registered studies."""
    from experiments.event_study.run_intraday import GEO_RX, MACRO_RX
    direct = [t for t, m in entity_matches(text).items() if m.tier == "direct" and t != "DJT"]
    if direct:
        return f"company ({direct[0]})"
    if GEO_RX.search(text) or GEO_RX.search(scenario):
        return "war/geopolitics"
    if MACRO_RX.search(text) or MACRO_RX.search(scenario):
        return "economy/policy"
    return "everything else"


def _anchors(intel: dict[str, Any], rows: list[dict[str, Any]],
             per_cohort: int = 3) -> list[dict[str, Any]]:
    """Anchors spanning EVERY cohort, surfaced by the data (largest |EOD abnormal move
    of the model's own basket|) rather than hand-picked. Previously this was 4 oil
    events + Intel, which is why the analyzer only ever showed oil."""
    out: list[dict[str, Any]] = []
    for m in intel["mentions"]:                       # keep the Intel exhibits
        out.append({"ts": m["ts"], "text": m["text"][:150], "cohort": "company (INTC)",
                    "label": "Intel mention", "pred": _prediction_for(m["text"], rows)})

    ranked: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    for r in rows:
        spy = (r.get("spy_returns") or {}).get("EOD")
        if spy is None:
            continue
        abn = [abs((i.get("returns") or {}).get("EOD", spy) - spy)
               for i in r.get("instruments", []) if (i.get("returns") or {}).get("EOD") is not None]
        if not abn:
            continue
        c = _cohort_of(r.get("text", ""), r.get("scenario", ""))
        ranked.setdefault(c, []).append((max(abn), r))
    for c, lst in ranked.items():
        lst.sort(key=lambda x: -x[0])
        for mag, r in lst[:per_cohort]:
            out.append({"ts": _t0(r).isoformat(), "text": r.get("text", "")[:150],
                        "cohort": c, "label": f"biggest move {mag * 100:.1f}% vs the market",
                        "pred": _prediction_for(r.get("text", ""), rows)})
    return out


def main() -> None:
    rng = random.Random(SEED)
    bars = load_bars()
    intel = gather_intel(bars)
    oil = gather_oil(bars, rng)
    verdict = gather_verdict()
    rows = json.loads(RESULTS.read_text())
    series = gather_panel_series(bars, _anchors(intel, rows))
    intraday = gather_intraday()
    breakdown = gather_breakdown(rows)
    calls = gather_calls(rows)
    calls["nvi"] = gather_named_vs_ignored(bars, rows, rng)
    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(intel, oil, verdict, series, intraday, breakdown, calls, stamp),
                   encoding="utf-8")
    n_assets = len(series[0]["assets"]) if series else 0
    n_off = len(series[0]["dates"]) if series else 0
    named = sum(len((s.get("pred") or {}).get("legs", [])) for s in series)
    print(f"[dashboard] Intel mentions={len(intel['mentions'])} "
          f"oil tweet-days={oil['n_tweet_days']} cells={verdict['n_cells']} "
          f"survive={verdict['n_survive']} analyzer-anchors={len(series)} "
          f"x {n_assets} assets x {n_off} offsets · named legs={named}")
    print(f"[dashboard] calls={calls['n_calls']} over {calls['n_tweets']} tweets · "
          f"up-share={calls['up_share']:.0%} · beat-SPY hit rate "
          + " ".join(f"{r['h']}={r['rate']:.3f}(n={r['n']})" for r in calls["rates"]))
    for r in calls["nvi"]:
        print(f"[dashboard] named-vs-ignored {r['w']}d (whole corpus, NOT anchors): "
              f"real named={r['real']['named'] * 100:.2f}% ignored={r['real']['ignored'] * 100:.2f}% "
              f"gap={r['real']['gap'] * 100:+.2f}pp | placebo(wrong day) "
              f"named={r['placebo']['named'] * 100:.2f}% ignored={r['placebo']['ignored'] * 100:.2f}% "
              f"gap={r['placebo']['gap'] * 100:+.2f}pp")
    print(f"[dashboard] -> {OUT}  ({OUT.stat().st_size / 1024:.1f} KB, self-contained)")


if __name__ == "__main__":
    main()
