"""Stock-level event-study dataset (DATASEARCH.md end-to-end; v3 builder).

Inputs : data/real/corpus_v3.csv  (78k posts 2009-2026, all platforms)
         data/real/bars.csv       (daily OHLCV 2008-2026, 46 tickers + DJT)
Outputs: data/real/market_events.csv        one row per market-RELEVANT post
         data/real/stock_event_dataset.csv  one row per (post x asset) event

Upgrades over the first v3 cut:
  (i)   FULL era: pre-office Twitter 2009+ through Truth Social 2026 (was
        2017-2021 only); all platforms; provenance (platform/source/date/hour)
        on every row.
  (ii)  market-relevance FILTER (sector keywords | entity mention | policy
        theme — sector_mapping/themes.py) with an auditable funnel report,
        emitted as market_events.csv (DATASEARCH §2).
  (iii) three relevance layers combined via combined_relevance(); every link
        is pre-t0 text-only (§3.1, §6) + human-readable `explanation` per row.
  (iv)  SPY is always in the universe (DATASEARCH §3): one SPY row per
        relevant post; for SPY the benchmark degenerates (spy_h=0, abn_h=raw)
        so the raw-spy-abn identity holds and labels measure the MARKET move.
  (v)   sector-adjusted return sec_h = raw_h - primary_sector_ETF_raw_h for
        stock rows (DATASEARCH §6); None for ETFs/SPY.
  (vi)  event windows per (post x asset): PRE -1/-3/-5/-10/-20/-21 sessions
        (raw + abnormal-vs-SPY), POST +1/+2/+3/+5/+10/+21/+42 sessions
        (raw, SPY, abnormal, sector-adjusted), vol-scaled labels at
        +1/+3/+5/+10/+21 (±k·σ_backward·√h).

Leakage fence: asset linking uses text only; compute_outcome asserts every
label close strictly > t0; pre-trend uses bars CLOSED strictly before t0 via
market_state_as_of. No modeling here.

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/build_stock_dataset_v3.py
Downstream: scripts/final_results.py (analysis + examples + conclusion).
"""

from __future__ import annotations

import statistics
from collections import Counter
from datetime import datetime, timedelta, timezone

import pandas as pd

from config.settings import SETTINGS
from config.universe import SECTOR_STOCKS, TRUMP_EXPOSED
from core.calendar import TradingCalendar
from core.market_state import market_state_as_of
from data.sources.interfaces import DailyBar
from data.sources.local import LocalPriceSource, load_corpus
from labeling.windows import Outcome, compute_outcome
from sector_mapping.entities import asset_relevance, is_direct_mention
from sector_mapping.themes import active_themes, combined_relevance, market_relevance

CORPUS, BARS = "data/real/corpus_v3.csv", "data/real/bars.csv"
OUT = "data/real/stock_event_dataset.csv"
EVENTS_OUT = "data/real/market_events.csv"
PRE_H = (1, 3, 5, 10, 20, 21)          # sessions back (incl. ~1 month = 21)
POST_H = (1, 2, 3, 5, 10, 21, 42)      # sessions forward (1wk=5, 2wk=10, 1mo=21, 2mo=42)
LABEL_H = (1, 3, 5, 10, 21)
VOLW = SETTINGS.vol_window_sessions
LO = datetime(2008, 1, 1, tzinfo=timezone.utc)
# DATASEARCH comment: Twitter archive "not relevant for now" + user: 2025-2026 only.
# T0=2025 keeps Truth Social era and drops the 2009-2021 Twitter archive in one cut.
T0, T1 = datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2026, 7, 7, tzinfo=timezone.utc)

# macro themes that make SPY itself the natural target asset
MACRO_THEMES = frozenset({"stock_market_macro", "fed_rates", "taxes", "jobs_economy"})

STOCK_SECTORS: dict[str, list[str]] = {}
for _e, _ns in SECTOR_STOCKS.items():
    for _n in _ns:
        STOCK_SECTORS.setdefault(_n, []).append(_e)


def pre_ctx(closes: list[float]) -> dict[str, float] | None:
    """Backward-only trend + vol from bars CLOSED before t0. None if too short."""
    need = max(max(PRE_H), VOLW) + 1
    if len(closes) < need:
        return None
    ctx = {f"pre_ret_{p}": closes[-1] / closes[-1 - p] - 1.0 for p in PRE_H}
    dr = [closes[i] / closes[i - 1] - 1.0 for i in range(len(closes) - VOLW, len(closes))]
    ctx["pre_vol"] = statistics.pstdev(dr)
    return ctx


def label(abn: float | None, vol: float, h: int) -> str | None:
    if abn is None:
        return None
    thr = SETTINGS.k * vol * (h**0.5)
    return "UP" if abn > thr else "DOWN" if abn < -thr else "NEUTRAL"


def rnd(x: float | None) -> float | None:
    return None if x is None else round(x, 6)


def explain(src: str, asset: str, sectors: str, themes: list[str]) -> str:
    """Human-readable, pre-t0-only reason the asset was linked (DATASEARCH §7)."""
    if src == "direct":
        return f"post names {asset} (or its exec/brand) directly"
    if src == "sector":
        return f"sector keywords in post -> {sectors or asset}"
    if src == "theme":
        return f"policy themes [{'/'.join(themes)}] -> {sectors or asset}"
    return f"macro post (themes [{'/'.join(themes) or 'keywords'}]) -> market benchmark"


def score_reason(src: str, r: float, is_etf: bool) -> str:
    """Why relevance == r (DATASEARCH §7 'add reason of scoring')."""
    if src == "direct":
        return "direct mention => 1.0"
    if src == "macro":
        return f"SPY benchmark row: {'macro theme active => 0.5' if r == 0.5 else 'default => 0.2'}"
    layer = "sector-keyword share" if src == "sector" else "policy-theme strength"
    return f"{layer} = {r}" if is_etf else f"0.5 x ETF {layer} (member stock) = {r}"


def main() -> None:
    tweets = load_corpus(CORPUS, T0, T1)  # all platforms, originals only, no reblogs
    price = LocalPriceSource(BARS)
    universe = sorted({*SETTINGS.etfs, *TRUMP_EXPOSED,
                       *(s for ss in SECTOR_STOCKS.values() for s in ss)})
    bars = {t: price.get_daily_bars(t, LO, T1 + timedelta(days=100))
            for t in [*universe, "SPY"]}
    cal = TradingCalendar([b.session_date.date() for b in bars["SPY"]])

    funnel = Counter({"posts_total": len(tweets)})
    theme_posts: Counter[str] = Counter()
    reason_posts: Counter[str] = Counter()
    asset_skips: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    events: list[dict[str, object]] = []

    for tw in tweets:
        ok, reasons = market_relevance(tw.text)
        if not ok:
            funnel["dropped_not_market_relevant"] += 1
            continue
        funnel["market_relevant"] += 1
        for reason in reasons:
            reason_posts["theme" if reason.startswith("theme:") else reason] += 1
        themes = active_themes(tw.text)
        for th in themes:
            theme_posts[th] += 1

        spy_out = compute_outcome(tw.timestamp_utc, bars["SPY"], cal, POST_H)
        spy_pre = pre_ctx([b.close for b in market_state_as_of(
            tw.timestamp_utc, "SPY", bars["SPY"], bars["SPY"], cal).prior_bars])
        if spy_out is None or spy_pre is None:
            funnel["dropped_no_spy_window"] += 1
            continue

        rel = combined_relevance(tw.text)
        ts = tw.timestamp_utc
        base: dict[str, object] = {  # per-post provenance (who/where/when — §9)
            "post_id": tw.tweet_id, "timestamp_utc": ts.isoformat(),
            "date_utc": ts.date().isoformat(), "hour_utc": ts.hour,
            "platform": tw.platform, "source": tw.source,
        }
        etf_out: dict[str, Outcome | None] = {}  # per-post cache for sector adjustment

        def emit(asset: str, r: float, src: str, a_bars: list[DailyBar],
                 out: Outcome, pf: dict[str, float], spy_degenerate: bool) -> bool:
            abn: dict[int, float | None] = {}
            for h in POST_H:
                a, s = out.ret[h], (0.0 if spy_degenerate else spy_out.ret[h])
                abn[h] = a - s if a is not None and s is not None else None
            labs = {h: label(abn[h], pf["pre_vol"], h) for h in LABEL_H}
            if labs[5] is None:
                return False
            is_etf = asset in SETTINGS.etfs or asset == "SPY"
            sectors = ("market" if asset == "SPY"
                       else asset if is_etf else "/".join(STOCK_SECTORS.get(asset, [])))
            # sector-adjusted return (stocks only): raw - primary sector ETF raw
            sec: dict[int, float | None] = dict.fromkeys(POST_H)
            prim = STOCK_SECTORS.get(asset, [None])[0]
            if prim is not None and not is_etf:
                if prim not in etf_out:
                    etf_out[prim] = compute_outcome(ts, bars[prim], cal, POST_H)
                po = etf_out[prim]
                if po is not None:
                    for h in POST_H:
                        a, p = out.ret[h], po.ret[h]
                        sec[h] = a - p if a is not None and p is not None else None
            row: dict[str, object] = {
                **base,
                "asset": asset, "is_etf": int(is_etf), "sectors": sectors,
                "relevance": r, "rel_source": src,
                "rel_reason": score_reason(src, r, is_etf),
                "is_direct": int(src == "direct"),
                "themes": "/".join(themes),
                "explanation": explain(src, asset, sectors, themes),
                "trump_exposed": int(asset in TRUMP_EXPOSED),
                "weekday": ts.weekday(),
                "after_hours": int(ts.hour < 14 or ts.hour >= 21),
                "pre_vol": round(pf["pre_vol"], 6),
            }
            for p in PRE_H:
                row[f"pre_ret_{p}"] = round(pf[f"pre_ret_{p}"], 6)
                row[f"pre_abn_{p}"] = round(pf[f"pre_ret_{p}"] - spy_pre[f"pre_ret_{p}"], 6)
            for h in POST_H:
                row[f"raw_{h}"] = rnd(out.ret[h])
                row[f"spy_{h}"] = rnd(0.0 if spy_degenerate else spy_out.ret[h])
                row[f"abn_{h}"] = rnd(abn[h])
                row[f"sec_{h}"] = rnd(sec[h])
            for h in LABEL_H:
                row[f"lab_{h}"] = labs[h]
            row["text"] = tw.text[:200]
            rows.append(row)
            return True

        emitted = 0
        # SPY always in the universe (DATASEARCH §3): the market itself is a row.
        spy_rel = 0.5 if MACRO_THEMES & set(themes) else 0.2
        emitted += emit("SPY", spy_rel, "macro", bars["SPY"], spy_out, spy_pre,
                        spy_degenerate=True)

        ent_rel = asset_relevance(tw.text)
        for asset, r in sorted(rel.items()):
            pf = pre_ctx([b.close for b in market_state_as_of(
                ts, asset, bars[asset], bars["SPY"], cal).prior_bars])
            out = compute_outcome(ts, bars[asset], cal, POST_H)
            if pf is None or out is None:
                asset_skips[asset] += 1
                continue
            direct = is_direct_mention(tw.text, asset)
            src = ("direct" if direct
                   else "sector" if ent_rel.get(asset, 0.0) >= r else "theme")
            if emit(asset, r, src, bars[asset], out, pf, spy_degenerate=False):
                emitted += 1
            else:
                asset_skips[asset] += 1

        events.append({**base, "reasons": "|".join(reasons), "themes": "/".join(themes),
                       "n_assets_linked": len(rel), "n_event_rows": emitted,
                       "text": tw.text[:280]})
        if emitted:
            funnel["posts_with_rows"] += 1

    pd.DataFrame(events).to_csv(EVENTS_OUT, index=False, encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8")

    print(f"== market_events: {len(events)} relevant posts -> {EVENTS_OUT} ==")
    print(f"== stock_event_dataset: {len(df)} rows -> {OUT} ==")
    print("universe:", " ".join([*universe, "SPY"]))
    print("\nfilter funnel:")
    for k, v in funnel.items():
        print(f"  {k:38} {v}")
    print("\nfilter reasons (posts, non-exclusive):", dict(reason_posts))
    print("\nposts per active theme:")
    for th, n in theme_posts.most_common():
        print(f"  {th:26} {n}")
    print(f"\nunique posts: {df['post_id'].nunique()}   assets: {df['asset'].nunique()}"
          f"   stock rows: {int((df['is_etf'] == 0).sum())}"
          f"   ETF rows: {int((df['is_etf'] == 1).sum())}")
    print("platform rows:", df["platform"].value_counts().to_dict())
    print("rel_source rows:", df["rel_source"].value_counts().to_dict())
    for h in LABEL_H:
        print(f"label balance +{h}s:", df[f"lab_{h}"].value_counts(dropna=False).to_dict())
    if asset_skips:
        print("\nrows skipped for missing history/outcome (ticker: n):",
              dict(asset_skips.most_common()))

    # ponytail: runnable self-checks — bounds, abnormal identity (incl. SPY rows),
    # sector identity, provenance present
    assert df["relevance"].between(0, 1).all()
    chk = df.dropna(subset=["raw_5", "spy_5", "abn_5"]).head(1000)
    assert ((chk["raw_5"] - chk["spy_5"] - chk["abn_5"]).abs() < 2e-6).all(), "abn != raw-spy"
    spy_chk = df[df["asset"] == "SPY"].dropna(subset=["abn_5"]).head(200)
    assert ((spy_chk["abn_5"] - spy_chk["raw_5"]).abs() < 2e-6).all(), "SPY abn != raw"
    assert set(df["rel_source"]) <= {"direct", "sector", "theme", "macro"}
    assert (df["source"] != "").all() and (df["platform"] != "").all(), "provenance missing"


if __name__ == "__main__":
    main()
