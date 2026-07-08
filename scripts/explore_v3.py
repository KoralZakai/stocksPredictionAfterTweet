"""EXPLORATORY analysis of stock_events_v3.csv (Phase C — no modeling).

Everything here is descriptive. Ranking events by post-move size is selecting
on the OUTCOME — these tables are illustrations of the noise structure, never
labels, features, or filters for a modeling set (leakage fence, §3.1).

Sections:
  1. theme / sector / stock link coverage (pre-t0 text layer only)
  2. top strongest +/- abnormal 5d reactions among strongly-linked rows
  3. pre-trend vs post-trend: already-moving / reversal / fresh-move split
  4. empirical top-3 movers vs the rule-linked set (false-attribution view)
  5. honest noise stats (|abn| by link layer)

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/explore_v3.py
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pandas as pd

from config.settings import SETTINGS
from config.universe import SECTOR_STOCKS
from core.calendar import TradingCalendar
from data.sources.local import LocalPriceSource, load_corpus
from labeling.windows import compute_outcome
from sector_mapping.themes import combined_relevance

EVENTS, CORPUS, BARS = "data/real/stock_events_v3.csv", "data/real/corpus_v3.csv", "data/real/bars.csv"
STRONG_REL = 0.3     # "strongly text-linked" row threshold for the example tables
MOVE = 0.02          # "strong post move" = |abnormal 5d| >= 2%
PRE_MOVE = 0.01      # "already moving" = |pre abnormal 5d| >= 1%
LO = datetime(2016, 1, 1, tzinfo=timezone.utc)
T0, T1 = datetime(2017, 1, 1, tzinfo=timezone.utc), datetime(2021, 1, 9, tzinfo=timezone.utc)


_NON_ASCII = re.compile(r"[^\x20-\x7e]")


def clean(s: object) -> str:
    """cp1252 console cannot print the corpus mojibake — ASCII-sanitize."""
    return _NON_ASCII.sub("?", str(s))


def hdr(s: str) -> None:
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


def pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def link_summary(df: pd.DataFrame) -> dict[object, str]:
    """post_id -> 'sectors=... stocks~...' (text-linked assets, pre-t0 only)."""
    out: dict[object, str] = {}
    for pid, g in df.groupby("post_id"):
        etfs = sorted(g[g["is_etf"] == 1]["asset"])
        stocks = list(g[g["is_etf"] == 0].sort_values(
            ["relevance", "asset"], ascending=[False, True])["asset"].head(5))
        out[pid] = (f"sectors={','.join(etfs) if etfs else '-'}"
                    f"  top-stocks={','.join(stocks) if stocks else '-'}")
    return out


def show_events(df: pd.DataFrame, title: str, links: dict[object, str]) -> None:
    print(f"\n--- {title} ---")
    for _, r in df.iterrows():
        print(f"{str(r['timestamp_utc'])[:16]}  {r['asset']:5} rel={r['relevance']:.2f}"
              f" ({r['rel_source']})  themes={r['themes'] if pd.notna(r['themes']) else '-'}")
        print(f"    {links.get(r['post_id'], '')}")
        print(f"    pre: 5d_abn {pct(r['pre_abn_5'])}  21d_abn {pct(r['pre_abn_21'])}"
              f"   post: 1d {pct(r['abn_1'])}  5d {pct(r['abn_5'])}  21d {pct(r['abn_21'])}")
        print(f"    \"{clean(r['text'])[:90]}\"")


def main() -> None:
    df = pd.read_csv(EVENTS)

    hdr("1. LINK COVERAGE (pre-t0 text layers only)")
    themes = df.drop_duplicates("post_id")["themes"].dropna().str.split("/").explode()
    print("top themes by unique market-relevant posts:")
    print(themes.value_counts().head(12).to_string())
    etf_rows = df[df["is_etf"] == 1]
    print("\nsectors most-linked (ETF rows / unique posts):")
    g = etf_rows.groupby("asset").agg(rows=("post_id", "size"), posts=("post_id", "nunique"))
    print(g.sort_values("rows", ascending=False).to_string())
    st = df[df["is_etf"] == 0].groupby("asset").agg(
        rows=("post_id", "size"), posts=("post_id", "nunique"),
        direct=("is_direct", "sum"))
    print("\ntop 12 stocks most-linked:")
    print(st.sort_values("rows", ascending=False).head(12).to_string())

    links = link_summary(df)
    strong = df[(df["relevance"] >= STRONG_REL) & df["abn_5"].notna()].copy()
    # one row per post (max |abn_5| among its strongly-linked assets), then collapse
    # posts sharing the exact same (asset, outcome window) — tweetstorm clusters
    # would otherwise fill the table with one event. EXPLORATORY only.
    strong["a5"] = strong["abn_5"].abs()
    per_post = (strong.sort_values("a5", ascending=False)
                .drop_duplicates("post_id").drop_duplicates(["asset", "abn_5"]))

    hdr(f"2. STRONGEST ABNORMAL 5d REACTIONS (rows with relevance >= {STRONG_REL}; "
        "outcome-ranked = EXPLORATORY ONLY)")
    show_events(per_post.nlargest(10, "abn_5"), "top 10 POSITIVE abnormal 5d", links)
    show_events(per_post.nsmallest(10, "abn_5"), "top 10 NEGATIVE abnormal 5d", links)

    hdr("3. PRE-TREND vs POST-TREND (was the move already underway before t0?)")
    mov = strong[strong["a5"] >= MOVE].drop_duplicates("post_id")
    mov_shown = mov.drop_duplicates(["asset", "abn_5"])  # display dedup only
    cont = mov[(mov["pre_abn_5"].abs() >= PRE_MOVE)
               & ((mov["pre_abn_5"] > 0) == (mov["abn_5"] > 0))]
    rev = mov[(mov["pre_abn_5"].abs() >= PRE_MOVE)
              & ((mov["pre_abn_5"] > 0) != (mov["abn_5"] > 0))]
    flat = mov[mov["pre_abn_5"].abs() < PRE_MOVE]
    n = len(mov)
    print(f"strong post-movers (|abn_5|>={MOVE:.0%}, rel>={STRONG_REL}): {n} posts")
    print(f"  already moving same direction pre-t0 (|pre_abn_5|>={PRE_MOVE:.0%}): "
          f"{len(cont)} ({len(cont) / n:.0%})")
    print(f"  reversal (pre-trend opposite sign):                        "
          f"{len(rev)} ({len(rev) / n:.0%})")
    print(f"  flat before (|pre_abn_5|<{PRE_MOVE:.0%}):                          "
          f"{len(flat)} ({len(flat) / n:.0%})")
    dd = mov_shown  # display dedup of tweetstorm clusters
    show_events(dd[dd.index.isin(cont.index)].nlargest(4, "a5"),
                "examples: move ALREADY UNDERWAY before the post", links)
    show_events(dd[dd.index.isin(rev.index)].nlargest(4, "a5"),
                "examples: post coincided with a REVERSAL", links)
    show_events(dd[dd.index.isin(flat.index)].nlargest(4, "a5"),
                "examples: flat before, moved after (best 'reaction' look)", links)

    hdr("4. EMPIRICAL TOP MOVERS vs RULE LINK (leakage-fenced false-attribution view)")
    tweets = [t for t in load_corpus(CORPUS, T0, T1, platforms=("twitter",))
              if combined_relevance(t.text)]
    price = LocalPriceSource(BARS)
    universe = sorted({*SETTINGS.etfs, *(s for ss in SECTOR_STOCKS.values() for s in ss)})
    bars = {t: price.get_daily_bars(t, LO, T1 + timedelta(days=40))
            for t in [*universe, "SPY"]}
    cal = TradingCalendar([b.session_date.date() for b in bars["SPY"]])
    sample = tweets[:: max(1, len(tweets) // 20)][:20]
    agree = shown = 0
    for tw in sample:
        spy = compute_outcome(tw.timestamp_utc, bars["SPY"], cal, (5,))
        if spy is None or spy.ret[5] is None:
            continue
        moves: list[tuple[str, float]] = []
        for a in universe:
            o = compute_outcome(tw.timestamp_utc, bars[a], cal, (5,))
            if o and o.ret[5] is not None and spy.ret[5] is not None:
                moves.append((a, o.ret[5] - spy.ret[5]))
        moves.sort(key=lambda kv: abs(kv[1]), reverse=True)
        linked = set(combined_relevance(tw.text))
        hit = any(a in linked for a, _ in moves[:3])
        agree += hit
        shown += 1
        if shown <= 6:
            top_s = ", ".join(f"{a}{v * 100:+.1f}%" for a, v in moves[:3])
            print(f"{str(tw.timestamp_utc)[:10]}  linked={sorted(linked)[:6]}"
                  f"{'...' if len(linked) > 6 else ''}")
            print(f"    top-3 movers: {top_s}   linked-in-top3? {'YES' if hit else 'no'}")
            print(f"    \"{clean(tw.text)[:70]}\"")
    print(f"\nrule-linked asset in empirical top-3 movers: {agree}/{shown} sampled events")

    hdr("5. NOISE STATS (honest aggregates)")
    ok = df[df["abn_5"].notna()]
    for src in ("direct", "sector", "theme"):
        sub = ok[ok["rel_source"] == src]
        print(f"  {src:7} rows={len(sub):6}  median|abn_5|={sub['abn_5'].abs().median():.4f}"
              f"  mean|abn_5|={sub['abn_5'].abs().mean():.4f}")
    print(f"  ALL     rows={len(ok):6}  median|abn_5|={ok['abn_5'].abs().median():.4f}"
          f"  vs median threshold(k*vol*sqrt5)={(SETTINGS.k * ok['pre_vol'] * 5**0.5).median():.4f}")
    hi = ok[ok["relevance"] >= STRONG_REL]
    lo_ = ok[ok["relevance"] < STRONG_REL]
    print(f"  strong-link rows median|abn_5|={hi['abn_5'].abs().median():.4f}"
          f"   weak-link rows median|abn_5|={lo_['abn_5'].abs().median():.4f}")
    print("  (if strong-link ~= weak-link, text linkage adds no visible move size — noise dominates)")

    # ponytail: runnable self-check — the three pre/post buckets partition the movers
    assert len(cont) + len(rev) + len(flat) == n


if __name__ == "__main__":
    main()
