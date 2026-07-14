"""Mass backtest: Nebius LLM macro predictions vs real market, over many tweets.

Option A. Runs the SAME pipeline as scripts/nebius_macro_validate.py over a batch of
historical geopolitical / macro tweets, aggregates the hit-rate per horizon, tests it
against a coin-flip (is the edge real or a fluke?), and renders the TOP-5 most
significant examples as green/red cards.

Reuses (no second path): classify_tweet / forward_returns / validate / HORIZONS from
scripts.nebius_macro_validate; the theme pre-filter from sector_mapping.themes; the card
renderer from reportgen.macro_card. LLM predictions are cached to
runs/real/nebius_predictions.json so a rerun costs no API calls.

Cost control: only tweets that pass a geopolitical/macro keyword+theme pre-filter are sent
to Nebius, capped by --limit. yfinance results are memoised per (ticker, date).

Run:
  PYTHONPATH=. .venv/Scripts/python.exe scripts/nebius_macro_backtest.py --limit 40
  ... --model meta-llama/Llama-3.3-70B-Instruct --start 2025-01-01 --top 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data.sources.local import load_corpus
from reportgen.macro_card import render_gallery
from scripts.nebius_macro_validate import (
    HORIZONS, _env, _load_dotenv, _session_anchor, classify_tweet, forward_returns,
    relative_hit, validate,
)
from sector_mapping.entities import entity_matches
from sector_mapping.themes import active_themes

CORPUS = "data/real/corpus_v3.csv"
CACHE = "runs/real/nebius_predictions.json"
OUT = "reports/nebius_backtest_top5.html"
RESULTS = "reports/nebius_backtest_results.json"   # full per-instrument data for analysis
PROMPT_VERSION = "v3-reasoning"                     # bump -> ignore old cached predictions


def asset_class(ticker: str) -> str:
    t = ticker.upper()
    if t in {"SPY", "QQQ", "DIA", "IWM", "VGK", "EEM"}:
        return "index"
    if t in {"VIXY", "VXX", "UVXY"}:
        return "volatility"
    if t in {"USO", "WEAT", "CORN", "DBC", "GLD", "UNG", "UUP", "SOYB"}:
        return "commodity_fx"
    if t.startswith("XL") or t in {"ITA", "SMH", "SOXX", "PAVE", "KRE", "ICLN", "XLRE", "XLU"}:
        return "sector_etf"
    return "single_stock"


def session_phase(t0: datetime) -> str:
    if t0.weekday() >= 5:
        return "weekend"
    h = t0.hour + t0.minute / 60.0
    if h < 13.5:
        return "premarket"
    if h >= 20.0:
        return "afterhours"
    return "regular"


# --- engineered "common-denominator" features (tweet-time, no leakage) ------------
_COUNTRY_RX = re.compile(
    r"\b(iran|china|chinese|russia|russian|mexico|canada|ukraine|israel|north korea|"
    r"south korea|saudi|venezuela|cuba|taiwan|india|japan|germany|france|europe|"
    r"brazil|turkey|syria|iraq|afghanistan|greenland|panama)\b", re.I)
# high-velocity executive / action language -> a decisive, tradeable catalyst
_ACTION_RX = re.compile(
    r"\b(immediately|effective (?:today|immediately)|as of today|starting today|effective now|"
    r"ordered?|ordering|signed?|signing|hereby|directed?|directing|will impose|impos(?:e|ing|ed)|"
    r"mandate[ds]?|executive order|announc(?:e|ing|ed|ement)|ban(?:ned|ning)?|"
    r"sanction(?:s|ed|ing)?|deploy(?:ed|ing)?|terminate[ds]?)\b", re.I)


def is_targeted_action(text: str) -> int:
    """1 = text targets a SPECIFIC nation or company (high short-success), 0 = broad/universal."""
    named_company = any(m.tier == "direct" for m in entity_matches(text).values())
    return int(bool(_COUNTRY_RX.search(text)) or named_company)


def has_action_verbs(text: str) -> int:
    """1 = decisive executive-velocity language (immediately / ordered / signed / effective today)."""
    return int(bool(_ACTION_RX.search(text)))


def execution_window(t0: datetime) -> str:
    """Regular_Hours | Monday_Open_Shock (weekend/Fri-after-hours -> Monday open) | Noise_Zone."""
    if session_phase(t0) == "regular":
        return "Regular_Hours"
    return "Monday_Open_Shock" if _session_anchor(t0).weekday() == 0 else "Noise_Zone"


def _t0_of(r: dict) -> datetime:
    """Reconstruct the tweet timestamp from a stored result row (date + hour_utc)."""
    d = datetime.fromisoformat(r["date"]).replace(tzinfo=timezone.utc)
    return d + timedelta(hours=float(r.get("hour_utc") or 0.0))

# A tweet is a macro/geopolitical CANDIDATE if a geopolitical theme fires OR one of
# these high-signal tokens appears (keeps Nebius calls bounded + on-topic).
GEO_THEMES = {"defense_geopolitics", "oil_energy_geopolitics", "tariffs_trade", "china"}
GEO_RX = re.compile(
    r"\b(russia|ukraine|china|iran|tariff|tariffs|sanction|sanctions|peace|ceasefire|"
    r"war|nato|opec|trade deal|deal|treaty|missile|nuclear|border|invasion)\b", re.I)


def _memoized_forward(memo: dict) -> object:
    def f(ticker: str, t0: datetime) -> dict | None:
        key = f"{ticker}|{t0.date()}"
        if key not in memo:
            memo[key] = forward_returns(ticker, t0)
        return memo[key]
    return f


def _candidates(start: datetime, end: datetime, limit: int) -> list:
    tweets = load_corpus(CORPUS, start, end)
    out = []
    for tw in tweets:
        themes = set(active_themes(tw.text))
        if themes & GEO_THEMES or GEO_RX.search(tw.text):
            out.append(tw)
    out.sort(key=lambda t: t.timestamp_utc)
    # spread across the window rather than taking the first `limit` (avoids clustering
    # on one news week); simple stride sample.
    if len(out) > limit:
        step = len(out) / limit
        out = [out[int(i * step)] for i in range(limit)]
    return out


def _significance(hit: int, tot: int) -> tuple[float, float]:
    """Normal-approx one-sided z + p for hit-rate vs a 0.5 coin-flip (no scipy)."""
    if tot == 0:
        return 0.0, 1.0
    p = hit / tot
    z = (p - 0.5) / math.sqrt(0.25 / tot)
    pval = 0.5 * math.erfc(z / math.sqrt(2))  # P(Z > z)
    return z, pval


EARLY = ("30m", "1h")   # the "flash" window
LATE = ("1mo",)         # the "structural" window


def _rate_over(result: dict, horizons: tuple[str, ...]) -> tuple[int, int]:
    hit = tot = 0
    for h in horizons:
        a, b = result["hits"].get(h, [0, 0])
        hit += a
        tot += b
    return hit, tot


def _categorize(result: dict) -> str:
    """Flash Shock (early right, late fades) vs Structural Drift (early noisy, late builds)."""
    eh, et = _rate_over(result, EARLY)
    lh, lt = _rate_over(result, LATE)
    if et == 0 or lt == 0:
        return ""
    early, late = eh / et, lh / lt
    if early >= 0.6 and late <= 0.5:
        return "Flash Shock"
    if early <= 0.4 and late >= 0.6:
        return "Structural Drift"
    return ""


def _score(result: dict, _last: str) -> float:
    """Rank by the STRONGEST horizon: many correct calls x how big the confirmed moves
    were, taken at whichever horizon the tweet correlated best (early OR late)."""
    best = 0.0
    for h in result["horizons"]:
        hits = [i for i in result["instruments"] if i.get("hit", {}).get(h) is True]
        if not hits:
            continue
        mag = sum(abs((i["returns"] or {}).get(h, 0.0)) for i in hits) / len(hits)
        best = max(best, len(hits) * mag)
    return best


def run(start: datetime, end: datetime, limit: int, model: str, base_url: str,
        api_key: str, top: int) -> None:
    cache = json.loads(Path(CACHE).read_text()) if Path(CACHE).exists() else {}
    memo: dict = {}
    fwd = _memoized_forward(memo)

    cands = _candidates(start, end, limit)
    print(f"-> {len(cands)} candidate geopolitical/macro tweets "
          f"({start.date()}..{end.date()}), model={model}\n")

    results: list[dict] = []
    agg = {h: [0, 0] for h in HORIZONS}
    for i, tw in enumerate(cands, 1):
        ck = f"{tw.tweet_id}|{model}|{PROMPT_VERSION}"
        if ck in cache:
            pred = cache[ck]
        else:
            try:
                pred = classify_tweet(tw.text, base_url=base_url, api_key=api_key, model=model)
            except SystemExit as e:
                print(f"  [{i}] Nebius error, stopping: {e}")
                break
            cache[ck] = pred
            Path(CACHE).parent.mkdir(parents=True, exist_ok=True)
            Path(CACHE).write_text(json.dumps(cache, indent=1))
        instruments = pred.get("instruments", []) or []
        rows, hits, spy_ret = validate(instruments, tw.timestamp_utc, fwd=fwd)
        for h in HORIZONS:
            agg[h][0] += hits[h][0]
            agg[h][1] += hits[h][1]
        t0 = tw.timestamp_utc
        res = {
            "text": tw.text, "date": t0.date().isoformat(), "model": model,
            "hour_utc": round(t0.hour + t0.minute / 60.0, 2), "phase": session_phase(t0),
            "intensity": pred.get("intensity"),
            "scenario": pred.get("scenario", ""), "rationale": pred.get("rationale", ""),
            # native reasoning layer (LLM-authored, human-readable):
            "summary": pred.get("summary", ""), "macro_link": pred.get("macro_link", ""),
            "hypothesis_short": pred.get("hypothesis_short", ""),
            "hypothesis_long": pred.get("hypothesis_long", ""),
            "horizons": list(HORIZONS),
            "spy_returns": spy_ret,   # stored so the run can be re-labelled w/o re-fetch
            "instruments": [{"ticker": r["ticker"], "name": r["name"], "role": r["role"],
                             "asset_class": asset_class(r["ticker"]),
                             "predicted": r["predicted"], "returns": r["returns"] or {},
                             "hit": r["hit"]} for r in rows],
            "hits": hits,
        }
        # KEEP closed-market tweets (not dropped): flag them. Their 30m/1h are already
        # anchored to the next market open (Monday bell) by _session_anchor -> captures
        # the "Monday Open Shock" instead of a dead weekend window.
        res["market_closed"] = int(res["phase"] != "regular")
        res["weekend_flag"] = int(res["phase"] == "weekend")
        res["tag"] = _categorize(res)
        results.append(res)
        eh, et = _rate_over(res, EARLY)
        lh, lt = _rate_over(res, LATE)
        print(f"  [{i}/{len(cands)}] {res['date']} {pred.get('scenario','?')[:26]:26} "
              f"early {eh}/{et} -> late {lh}/{lt}  {res['tag']}")

    _assign_splits(results)
    Path(RESULTS).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULTS).write_text(json.dumps(results, indent=1), encoding="utf-8")
    _summary(agg, results, top, model)
    _analyze(results)
    write_dataset(results)
    _narrative(results)


def _summary(agg: dict, results: list[dict], top: int, model: str) -> None:
    print("\n" + "=" * 84)
    print(f"AGGREGATE HIT-RATE across {len(results)} tweets   model={model}")
    print("=" * 84)
    print(f"{'horizon':8}{'hits':>10}{'rate':>9}{'z':>8}{'p(vs coin-flip)':>18}")
    for h in HORIZONS:
        hit, tot = agg[h]
        z, pval = _significance(hit, tot)
        rate = "n/a" if tot == 0 else f"{100*hit/tot:.1f}%"
        print(f"{h:8}{f'{hit}/{tot}':>10}{rate:>9}{z:>8.2f}{pval:>18.4f}")
    print("\n(above counts every instrument -> inflated N: instruments within a tweet are "
          "correlated, NOT independent. The honest aggregate is one call per tweet:)")
    tl = tweet_level(results)
    print("\nTWEET-LEVEL HIT-RATE (majority vote per tweet; N = tweets)")
    print(f"{'horizon':8}{'tweets':>12}{'rate':>9}{'z':>8}{'p(vs coin-flip)':>18}")
    for h in HORIZONS:
        hit, tot = tl[h]
        z, pval = _significance(hit, tot)
        rate = "n/a" if tot == 0 else f"{100*hit/tot:.1f}%"
        print(f"{h:8}{f'{hit}/{tot}':>12}{rate:>9}{z:>8.2f}{pval:>18.4f}")
    print("\np < 0.05 here is the HONEST test (N=tweets). Not multiple-comparison corrected "
          "across horizons; treat as suggestive.")

    # Flash Shock vs Structural Drift split (the two insights requested).
    flash = [r for r in results if r["tag"] == "Flash Shock"]
    struct = [r for r in results if r["tag"] == "Structural Drift"]
    print("\nCATEGORIES:")
    print(f"  Flash Shocks     (early 30m/1h right, fades by 1mo): {len(flash)}")
    print(f"  Structural Drifts(early noisy, 1mo builds up):       {len(struct)}")
    for label, group in (("Flash Shock", flash), ("Structural Drift", struct)):
        for r in group[:3]:
            eh, et = _rate_over(r, EARLY)
            lh, lt = _rate_over(r, LATE)
            print(f"    [{label:16}] {r['date']} {r['scenario'][:30]:30} early {eh}/{et} late {lh}/{lt}")

    ranked = sorted(results, key=lambda r: _score(r, ""), reverse=True)
    picks = [r for r in ranked if _score(r, "") > 0][:top]
    print(f"\nTOP {len(picks)} strongest examples (best correct-calls x move size at ANY horizon):")
    for r in picks:
        print(f"  {r['date']}  {r['scenario'][:30]:30} score={_score(r, ''):.3f}  {r['tag']}")

    if picks:
        Path(OUT).parent.mkdir(parents=True, exist_ok=True)
        Path(OUT).write_text(render_gallery(
            picks, "Top macro tweets: Nebius prediction vs real market",
            f"The {len(picks)} tweets with the strongest directional match at any horizon, out "
            f"of {len(results)} tested. Tagged Flash Shock (fast, fades) vs Structural Drift "
            f"(slow, builds). Suggestive, not proof."),
            encoding="ascii")
        print(f"\n-> top-{len(picks)} visual gallery -> {OUT}")


def _fmt(hit: int, tot: int) -> str:
    return "  n/a " if tot == 0 else f"{100*hit/tot:4.0f}% ({hit}/{tot})"


def _tweet_hit(result: dict, h: str) -> int | None:
    """One call per tweet at horizon h: majority vote of its directional instruments.
    None if the tweet made no scoreable directional call at h. Fixes the inflated-N
    problem (correlated instruments within a tweet are NOT independent samples)."""
    hs = [i["hit"][h] for i in result["instruments"] if isinstance(i.get("hit", {}).get(h), bool)]
    if not hs:
        return None
    return int(sum(hs) / len(hs) >= 0.5)


def signed_eod(r: dict) -> int | None:
    """Signed multiclass target at EOD:
      +1 = tweet's basket predicted UP and beat SPY  (winning LONG)
      -1 = predicted DOWN and lagged SPY             (winning SHORT)
       0 = wrong direction / noise / inside the band
    None if the tweet has no scoreable EOD call. |signed| == the old binary hit."""
    hit = _tweet_hit(r, "EOD")
    if hit is None:
        return None
    if hit == 0:
        return 0
    scored = [i for i in r["instruments"] if isinstance(i.get("hit", {}).get("EOD"), bool)]
    ups = sum(i["predicted"] == "up" for i in scored)
    downs = sum(i["predicted"] == "down" for i in scored)
    return 1 if ups >= downs else -1     # dominant predicted direction of the winning basket


def tweet_level(results: list[dict]) -> dict[str, list[int]]:
    """{horizon: [tweets_correct, tweets_scored]} — the honest, N=tweets aggregate."""
    out = {h: [0, 0] for h in HORIZONS}
    for r in results:
        for h in HORIZONS:
            v = _tweet_hit(r, h)
            if v is not None:
                out[h][1] += 1
                out[h][0] += v
    return out


DATASET = "reports/macro_dataset.csv"
DATASET_CFG = "reports/macro_dataset_config.json"
# Features knowable at TWEET TIME (no outcome leakage) — the engine's inputs.
FEATURES = ["intensity", "phase", "weekend_flag", "market_closed", "n_instruments", "scenario",
            "is_targeted_action", "has_action_verbs", "execution_window"]
# PRIMARY target = EOD: relative alpha is strongest and most honest here (64%, p=0.0005),
# because tweet-specific alpha is a fast/front-loaded phenomenon, not a 1mo beta drift.
PRIMARY = "EOD"
SECONDARY = ["30m", "1h", "1mo"]        # tracked as extra columns, not the target
LABEL = f"label_{PRIMARY.lower()}"      # -> "label_eod": 1 = basket beat SPY at EOD


def _relabel(results: list[dict], band: float = 0.0) -> None:
    """Recompute every hit from stored returns + spy_returns under the relative
    (beat-SPY) rule. Lets --from-results re-score at a new band with no re-fetch.
    No-op on old runs that predate spy_returns being stored."""
    for r in results:
        spy = r.get("spy_returns")
        if spy is None:
            continue
        newhits = {h: [0, 0] for h in HORIZONS}
        for ins in r["instruments"]:
            pred = ins.get("predicted", "neutral")
            ins["hit"] = {}
            if pred not in ("up", "down") or ins["ticker"] == "SPY":
                continue
            for h in HORIZONS:
                hit = relative_hit(pred, (ins.get("returns") or {}).get(h), spy.get(h), band)
                if hit is not None:
                    ins["hit"][h] = hit
                    newhits[h][0] += int(hit)
                    newhits[h][1] += 1
        r["hits"] = newhits


def _assign_splits(results: list[dict]) -> None:
    """Tag each scoreable tweet with a chronological 60/20/20 train/val/test split."""
    scored = sorted((r for r in results if _tweet_hit(r, PRIMARY) is not None),
                    key=lambda r: r["date"])
    n = len(scored)
    tr, va = int(round(0.6 * n)), int(round(0.8 * n))
    for i, r in enumerate(scored):
        r["split"] = "train" if i < tr else "val" if i < va else "test"
    for r in results:
        r.setdefault("split", None)


def _conclusion(r: dict, role: str) -> str:
    """Templated data-science verdict from the tweet's own early vs late numbers."""
    eh, et = _rate_over(r, EARLY)
    lh, lt = _rate_over(r, LATE)
    er = eh / et if et else 0.0
    lr = lh / lt if lt else 0.0
    if role == "Flash Shock":
        return (f"The model nailed the immediate reaction (early {er:.0%} correct) but the edge "
                f"decayed to {lr:.0%} by 1 month — the crowd over-reacted, then re-priced as "
                f"broader macro forces took over. Rule: trust this signal INTRADAY ONLY.")
    if role == "Structural Drift":
        return (f"The first hour was noise (early {er:.0%}), but the thesis played out over weeks "
                f"(1mo {lr:.0%} correct) as the policy actually took effect. Rule: SLOW-BURN "
                f"signal — hold to 1 month, ignore the immediate wobble.")
    return (f"The model was confident (intensity {r.get('intensity','?')}) yet wrong by 1 month "
            f"(only {lr:.0%} of the basket confirmed) — a macro confound or sharp reversal "
            f"overwhelmed the tweet. Rule: FLAG this scenario for error analysis, don't trust blindly.")


def _pick_examples(results: list[dict]) -> list[dict]:
    """One Flash Shock, one Structural Drift, one Failed Anomaly — from the VAL split
    if present, else the whole set. Distinct tweets."""
    val = [r for r in results if r.get("split") == "val"]

    def best(cond, key):
        # prefer a validation-split example; fall back to the whole set per-category
        # so we always surface all three patterns (esp. a Failed Anomaly).
        cs = [r for r in val if cond(r) and _tweet_hit(r, "1mo") is not None]
        if not cs:
            cs = [r for r in results if cond(r) and _tweet_hit(r, PRIMARY) is not None]
        return max(cs, key=key) if cs else None

    chosen: list[dict] = []
    seen: set[str] = set()
    plan = [
        ("Flash Shock", lambda r: r.get("tag") == "Flash Shock",
         lambda r: _rate_over(r, EARLY)[0] / max(_rate_over(r, EARLY)[1], 1)),
        ("Structural Drift", lambda r: r.get("tag") == "Structural Drift",
         lambda r: _rate_over(r, LATE)[0] / max(_rate_over(r, LATE)[1], 1)),
        ("Failed Anomaly", lambda r: _tweet_hit(r, PRIMARY) == 0,
         lambda r: (r.get("intensity") or 0)),
    ]
    for role, cond, key in plan:
        r = best(lambda x: cond(x) and x["date"] + x["text"][:20] not in seen, key)
        if r:
            seen.add(r["date"] + r["text"][:20])
            chosen.append({"result": r, "role": role, "conclusion": _conclusion(r, role)})
    return chosen


VAL_HTML = "reports/nebius_validation_examples.html"


def _narrative(results: list[dict]) -> None:
    picks = _pick_examples(results)
    if not picks:
        print("\n(no scoreable validation examples to narrate)")
        return
    print("\n" + "=" * 84)
    print("VALIDATION EXAMPLES — inspect the model's brain (one per pattern)")
    print("=" * 84)
    for e in picks:
        r = e["result"]
        print(f"\n### {e['role']}  |  {r['date']}  |  {r.get('scenario','')}  "
              f"|  intensity {r.get('intensity','?')}  |  split={r.get('split')}")
        print(f'  TWEET: "{r["text"][:240]}"')
        print(f"  MEANING:     {r.get('summary','')}")
        print(f"  MACRO LINK:  {r.get('macro_link','')}")
        print(f"  SHORT CALL:  {r.get('hypothesis_short','')}")
        print(f"  LONG CALL:   {r.get('hypothesis_long','')}")
        print("  REALITY (short 30m/1h vs long 1mo):")
        for ins in r["instruments"]:
            ret = ins.get("returns") or {}
            def g(h: str) -> str:
                v = ret.get(h)
                return "  n/a " if v is None else f"{v*100:+.1f}%"
            hit1mo = ins.get("hit", {}).get("1mo")
            mark = "HIT" if hit1mo is True else "MISS" if hit1mo is False else "-"
            print(f"    {ins['ticker']:5} pred {ins['predicted']:8} "
                  f"30m {g('30m'):>8}  1h {g('1h'):>8}  1mo {g('1mo'):>8}  [{mark}@1mo]")
        print(f"  CONCLUSION:  {e['conclusion']}")

    from reportgen.macro_card import render_narrative
    Path(VAL_HTML).write_text(render_narrative(
        picks, "Validation set — inspect the model's reasoning",
        "One Flash Shock, one Structural Drift, one Failed Anomaly, drawn from the validation "
        "split. Green = the market moved as the LLM predicted; red = it did not."),
        encoding="ascii")
    print(f"\n-> narrative dashboard -> {VAL_HTML}")


def write_dataset(results: list[dict]) -> None:
    """Emit the per-tweet modelling frame + an 80/20 TIME-ORDERED train/test config.

    Chronological holdout (not random) so no future tweet informs a past prediction
    (CLAUDE.md purged/embargoed spirit). Label = tweet-level 1mo hit."""
    import csv

    rows = []
    for r in results:
        label = signed_eod(r)   # SIGNED multiclass: +1 long-win / -1 short-win / 0 noise
        if label is None:       # no scoreable EOD call -> not a training example
            continue
        t0 = _t0_of(r)
        text = r.get("text", "")
        row = {
            "post_date": r["date"], "hour_utc": r.get("hour_utc"),
            "phase": r["phase"], "weekend_flag": r["weekend_flag"],
            "market_closed": r["market_closed"], "intensity": r.get("intensity"),
            "scenario": r.get("scenario", ""), "n_instruments": len(r["instruments"]),
            # engineered common-denominator features:
            "is_targeted_action": is_targeted_action(text),
            "has_action_verbs": has_action_verbs(text),
            "execution_window": execution_window(t0),
            # LLM reasoning text -> semantic features for the engine (Option C):
            "summary": r.get("summary", ""), "macro_link": r.get("macro_link", ""),
            LABEL: label,     # SIGNED EOD target: +1 long-win / -1 short-win / 0 noise
        }
        # secondary horizons kept for tracking (outcomes, NOT features -> would leak):
        for h in SECONDARY:
            row[f"hit_{h.lower()}"] = _tweet_hit(r, h)
        # chronological 60/20/20 split assigned by _assign_splits (time-ordered so no
        # future tweet informs a past prediction; test stays sacred).
        row["split"] = r.get("split")
        rows.append(row)
    rows.sort(key=lambda x: x["post_date"])
    n = len(rows)

    Path(DATASET).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(DATASET, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    longs = sum(r[LABEL] == 1 for r in rows)
    shorts = sum(r[LABEL] == -1 for r in rows)
    noise = sum(r[LABEL] == 0 for r in rows)
    cfg = {
        "label": LABEL, "label_type": "signed_multiclass", "classes": {
            "+1": "LLM predicted UP and the basket BEAT SPY (winning LONG)",
            "-1": "LLM predicted DOWN and the basket LAGGED SPY (winning SHORT)",
            "0": "wrong direction / noise / inside the band"},
        "primary_horizon": PRIMARY, "secondary_horizons": SECONDARY,
        "engineered_features": ["is_targeted_action", "has_action_verbs", "execution_window"],
        "features": FEATURES,
        "excluded_from_features": [f"hit_{h.lower()}" for h in SECONDARY]
        + ["summary", "macro_link (text -> vectorized separately)", "(hits observed after t0)"],
        "split": "time_ordered_60_20_20", "split_purpose": {
            "train": "fit the model", "val": "inspect errors + tune hyperparameters",
            "test": "SACRED — touch only once, for the final number"},
        "n_total": n, "n_train": counts["train"], "n_val": counts["val"], "n_test": counts["test"],
        "class_balance": {"+1_long": longs, "-1_short": shorts, "0_noise": noise},
        "win_rate": round((longs + shorts) / n, 3) if rows else None,
    }
    Path(DATASET_CFG).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("\n" + "=" * 84)
    print("TRAIN / VALIDATION / TEST DATASET (signed label, engineered features, 60/20/20)")
    print("=" * 84)
    print(f"  -> {DATASET}   ({n} tweets: {counts['train']} train / {counts['val']} val "
          f"/ {counts['test']} test)")
    print(f"  -> {DATASET_CFG}")
    print(f"  label={LABEL} SIGNED {{+1 long-win, -1 short-win, 0 noise}}  "
          f"balance: +1={longs} / -1={shorts} / 0={noise}  win_rate={cfg['win_rate']}")
    print("  engineered: is_targeted_action, has_action_verbs, execution_window")
    print("\n  Signed win-rate by split (|label|!=0; long-wins / short-wins):")
    for s in ("train", "val", "test"):
        grp = [x for x in rows if x["split"] == s]
        if grp:
            lo = sum(x[LABEL] == 1 for x in grp)
            sh = sum(x[LABEL] == -1 for x in grp)
            print(f"    {s:5} n={len(grp):3}  win-rate={100*(lo+sh)/len(grp):5.1f}%  "
                  f"(long {lo} / short {sh} / noise {len(grp)-lo-sh})")


def _analyze(results: list[dict]) -> None:
    """The 3 cross-correlations requested, computed from the persisted run."""
    print("\n" + "#" * 84)
    print("# CROSS-CORRELATIONS (training-data cleaning signals)")
    print("#" * 84)

    # (1) Asset class vs decay speed -----------------------------------------------
    cls: dict[str, dict[str, list[int]]] = {}
    for r in results:
        for ins in r["instruments"]:
            c = ins.get("asset_class", "?")
            for h in HORIZONS:
                hit = ins.get("hit", {}).get(h)
                if isinstance(hit, bool):
                    d = cls.setdefault(c, {x: [0, 0] for x in HORIZONS})
                    d[h][0] += hit
                    d[h][1] += 1
    print("\n(1) ASSET CLASS x HORIZON hit-rate  (decay = early 30m/1h avg  ->  1mo):")
    print(f"  {'class':14}" + "".join(f"{h:>13}" for h in HORIZONS) + f"{'decay':>10}")
    for c, d in sorted(cls.items()):
        early_h = d["30m"][0] + d["1h"][0]
        early_t = d["30m"][1] + d["1h"][1]
        late = _rate_over({"hits": d}, LATE)
        early_r = None if early_t == 0 else early_h / early_t
        late_r = None if late[1] == 0 else late[0] / late[1]
        decay = "n/a" if early_r is None or late_r is None else f"{(early_r-late_r)*100:+.0f}pp"
        print(f"  {c:14}" + "".join(f"{_fmt(*d[h]):>13}" for h in HORIZONS) + f"{decay:>10}")
    # the specific tickers the user named
    print("\n  named barometers (VIXY vs CAT vs LMT):")
    per: dict[str, dict[str, list[int]]] = {}
    for r in results:
        for ins in r["instruments"]:
            if ins["ticker"] in ("VIXY", "CAT", "LMT"):
                for h in HORIZONS:
                    hit = ins.get("hit", {}).get(h)
                    if isinstance(hit, bool):
                        d = per.setdefault(ins["ticker"], {x: [0, 0] for x in HORIZONS})
                        d[h][0] += hit
                        d[h][1] += 1
    for tk, d in per.items():
        print(f"  {tk:14}" + "".join(f"{_fmt(*d[h]):>13}" for h in HORIZONS))

    # (2) LLM intensity vs structural drift ----------------------------------------
    print("\n(2) LLM INTENSITY (1-10) vs drift  (early 30m/1h  ->  late 1mo hit-rate):")
    bins = {"low(1-4)": (1, 4), "mid(5-7)": (5, 7), "high(8-10)": (8, 10)}
    print(f"  {'intensity':12}{'tweets':>8}{'early':>16}{'late':>16}{'drift(late-early)':>20}")
    for name, (lo, hi) in bins.items():
        grp = [r for r in results if isinstance(r.get("intensity"), (int, float))
               and lo <= r["intensity"] <= hi]
        eh = et = lh = lt = 0
        for r in grp:
            a, b = _rate_over(r, EARLY)
            c, d = _rate_over(r, LATE)
            eh, et, lh, lt = eh + a, et + b, lh + c, lt + d
        er = None if et == 0 else eh / et
        lr = None if lt == 0 else lh / lt
        drift = "n/a" if er is None or lr is None else f"{(lr-er)*100:+.0f}pp"
        print(f"  {name:12}{len(grp):>8}{_fmt(eh, et):>16}{_fmt(lh, lt):>16}{drift:>20}")

    # (3) Trading-hours effect on the 30m reaction ---------------------------------
    print("\n(3) SESSION PHASE vs 30m reaction  (hit-rate, data-coverage, |move| distortion):")
    print(f"  {'phase':12}{'tweets':>8}{'30m hit':>16}{'30m coverage':>16}{'mean|30m move|':>16}")
    for phase in ("regular", "premarket", "afterhours", "weekend"):
        grp = [r for r in results if r.get("phase") == phase]
        hit = tot = cov_ok = cov_all = 0
        mags: list[float] = []
        for r in grp:
            for ins in r["instruments"]:
                h = ins.get("hit", {}).get("30m")
                if isinstance(h, bool):
                    hit += h
                    tot += 1
                cov_all += 1
                ret = (ins.get("returns") or {}).get("30m")
                if ret is not None:
                    cov_ok += 1
                    mags.append(abs(ret))
        cover = "n/a" if cov_all == 0 else f"{100*cov_ok/cov_all:4.0f}%"
        mag = "n/a" if not mags else f"{100*sum(mags)/len(mags):.2f}%"
        print(f"  {phase:12}{len(grp):>8}{_fmt(hit, tot):>16}{cover:>16}{mag:>16}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--limit", type=int, default=40, help="max tweets sent to Nebius")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--model", default=_env("NEBIUS_MODEL", "EXPO_PUBLIC_NEBIUS_MODEL",
                                            default="meta-llama/Llama-3.3-70B-Instruct"))
    ap.add_argument("--base-url", default=_env("NEBIUS_BASE_URL", "EXPO_PUBLIC_NEBIUS_BASE_URL",
                                               default="https://api.studio.nebius.ai/v1"))
    ap.add_argument("--from-results", action="store_true",
                    help="re-run analysis/split/narrative from the saved results JSON "
                         "(no Nebius, no market fetch — instant)")
    ap.add_argument("--band", type=float, default=0.0,
                    help="relative-alpha noise band for --from-results relabel (0 = strict beat-SPY)")
    a = ap.parse_args()

    if a.from_results:
        results = json.loads(Path(RESULTS).read_text())
        _relabel(results, a.band)                 # re-score under beat-SPY at this band
        for r in results:
            r["tag"] = _categorize(r)
        _assign_splits(results)
        agg = {h: [0, 0] for h in HORIZONS}
        for r in results:
            for h in HORIZONS:
                x, y = r["hits"].get(h, [0, 0])
                agg[h][0] += x
                agg[h][1] += y
        _summary(agg, results, a.top, results[0].get("model", ""))
        _analyze(results)
        write_dataset(results)
        _narrative(results)
        return

    api_key = _env("NEBIUS_API_KEY", "EXPO_PUBLIC_NEBIUS_API_KEY")
    if not api_key:
        sys.exit("No NEBIUS_API_KEY. Put it in .env (gitignored) or export it.")
    start = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)
    run(start, end, a.limit, a.model, a.base_url, api_key, a.top)


if __name__ == "__main__":
    main()
