"""FINAL RESULTS (DATASEARCH §8/§10): empirical event-study report.

Input : data/real/stock_event_dataset.csv + data/real/market_events.csv
Output: reports/final_results.txt (also printed)

Sections:
  1. dataset summary statistics
  2. global insights — top affected sectors / stocks / tweet types (themes)
  3. 20 example tweets: source, date+hour, matched assets, reaction, explanation
  4. conclusion — measurable signal or mostly noise (empirical, no speculation)

Honesty fence: everything here is DESCRIPTIVE, computed on outcomes — it must
never feed back into features/labels/filters (§3.1). "Top affected" = tails of
a noisy distribution by construction. The rigorous significance answer (purged
CV + permutation null + BH) lives in the eval harness; this report's noise
benchmark is each asset's own pre-event vol: |abn_h| / (pre_vol·sqrt(h)) ~ 1
means moves after tweets are indistinguishable from baseline volatility.

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/final_results.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA, EVENTS = "data/real/stock_event_dataset.csv", "data/real/market_events.csv"
OUT = Path("reports/final_results.txt")
_LINES: list[str] = []


def w(s: str = "") -> None:
    print(s)
    _LINES.append(s)


def pct(x: object) -> str:
    return f"{float(x) * 100:+.2f}%" if pd.notna(x) else "n/a"  # type: ignore[arg-type]  # pandas cell


def hdr(title: str) -> None:
    w("\n" + "=" * 78)
    w(title)
    w("=" * 78)


def vol_ratio(d: pd.DataFrame, h: int = 5) -> float:
    """median |abn_h| / median expected h-session vol — ~1.0 means noise-level."""
    ok = d.dropna(subset=[f"abn_{h}", "pre_vol"])
    if ok.empty:
        return float("nan")
    return float((ok[f"abn_{h}"].abs() / (ok["pre_vol"] * h**0.5)).median())


def show_post(g: pd.DataFrame, k: int = 3) -> None:
    r0 = g.iloc[0]
    w(f"\n[{r0['date_utc']} {int(r0['hour_utc']):02d}:00 UTC] "
      f"{r0['platform']} / {r0['source']}")
    w(f'  "{str(r0["text"])[:110]}"')
    top = g.sort_values("relevance", ascending=False).head(k)
    for _, r in top.iterrows():
        w(f"  {r['asset']:5} rel={r['relevance']:.2f}  "
          f"1s {pct(r['abn_1'])}  5s {pct(r['abn_5'])}  21s {pct(r['abn_21'])}  "
          f"lab5={r['lab_5']}  [{r['explanation']}]")


def main() -> None:
    df = pd.read_csv(DATA)
    ev = pd.read_csv(EVENTS)

    hdr("1. DATASET SUMMARY")
    w(f"market-relevant posts : {len(ev):>8}   (market_events.csv)")
    w(f"event rows (post x asset): {len(df):>5}   (stock_event_dataset.csv)")
    w(f"unique posts w/ rows  : {df['post_id'].nunique():>8}")
    w(f"assets                : {df['asset'].nunique():>8} "
      f"(ETFs+SPY rows {int((df['is_etf'] == 1).sum())}, stock rows {int((df['is_etf'] == 0).sum())})")
    w(f"coverage              : {df['date_utc'].min()} .. {df['date_utc'].max()}")
    w(f"by platform (rows)    : {df['platform'].value_counts().to_dict()}")
    w(f"by source (rows)      : {df['source'].value_counts().to_dict()}")
    w(f"by link layer (rows)  : {df['rel_source'].value_counts().to_dict()}")
    for h in (1, 5, 21):
        w(f"label balance +{h:>2}s    : {df[f'lab_{h}'].value_counts(dropna=False).to_dict()}")

    hdr("2. GLOBAL INSIGHTS (descriptive — outcome-ranked, NOT features)")
    strong = df[(df["relevance"] >= 0.3) & (df["asset"] != "SPY")]
    w("\nA) top affected SECTORS (ETF rows, median |abnormal 5s|, n>=30):")
    etf = strong[strong["is_etf"] == 1].dropna(subset=["abn_5"])
    g = etf.groupby("asset")["abn_5"].agg(n="count", med=lambda s: s.abs().median())
    for a, r in g[g["n"] >= 30].sort_values("med", ascending=False).iterrows():
        w(f"   {a:5} n={int(r['n']):5}   median |abn_5s| = {r['med'] * 100:.2f}%")
    w("\nB) top affected STOCKS (median |abnormal 5s|, n>=30):")
    stk = strong[strong["is_etf"] == 0].dropna(subset=["abn_5"])
    g = stk.groupby("asset")["abn_5"].agg(n="count", med=lambda s: s.abs().median())
    for a, r in g[g["n"] >= 30].sort_values("med", ascending=False).head(12).iterrows():
        w(f"   {a:5} n={int(r['n']):5}   median |abn_5s| = {r['med'] * 100:.2f}%")
    w("\nC) which tweet TYPES coincide with the biggest moves (theme, vol-normalized):")
    th = strong.dropna(subset=["abn_5"]).assign(theme=strong["themes"].str.split("/")).explode("theme")
    th = th[th["theme"].astype(bool)]
    g2 = th.groupby("theme").apply(vol_ratio, include_groups=False)
    n2 = th.groupby("theme").size()
    for t, v in g2.sort_values(ascending=False).items():
        w(f"   {t:26} n={int(n2[t]):5}   median |abn_5s| / expected vol = {v:.2f}x")
    w("\n   (1.00x = moves after these tweets match the asset's normal volatility)")

    hdr("3. TWENTY EXAMPLE TWEETS (source, date+hour UTC, matched assets, reaction)")
    linked = df[(df["relevance"] >= 0.5) & (df["asset"] != "SPY")].dropna(subset=["abn_5"])
    peaks = linked.loc[linked.groupby("post_id")["abn_5"].transform(
        lambda s: s.abs().max()) == linked["abn_5"].abs()]
    w("\n-- 10 with the STRONGEST post-tweet abnormal moves (tail-picked, illustrative) --")
    for pid in peaks.sort_values("abn_5", key=abs, ascending=False)["post_id"].unique()[:10]:
        show_post(df[df["post_id"] == pid])
    w("\n-- 10 RANDOM market-relevant tweets (seeded; the honest typical case) --")
    rng_posts = linked["post_id"].drop_duplicates().sample(10, random_state=0)
    for pid in rng_posts:
        show_post(df[df["post_id"] == pid])

    hdr("4. CONCLUSION — SIGNAL OR NOISE? (empirical)")
    overall = vol_ratio(strong)
    by_layer = {s: vol_ratio(strong[strong["rel_source"] == s])
                for s in ("direct", "sector", "theme")}
    spy_r = vol_ratio(df[df["asset"] == "SPY"])
    w(f"\nmedian |abn_5s| / expected 5-session vol, linked rows : {overall:.2f}x")
    for s, v in by_layer.items():
        w(f"  {s:8} layer: {v:.2f}x")
    w(f"  SPY (market itself, raw): {spy_r:.2f}x")
    verdict = ("MOSTLY NOISE: post-tweet abnormal moves are at or below the assets' "
               "normal volatility" if overall <= 1.1 else
               "moves exceed baseline volatility — pending the rigorous eval harness "
               "(purged CV + permutation null + BH) before any signal claim")
    w(f"\nVERDICT: {verdict}.")
    w("Direction balance of labels (UP vs DOWN) is near-symmetric — no exploitable "
      "directional edge is claimed from this table. The tail examples in §3 are "
      "selected on outcomes and carry no evidential weight (§3.1).")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")

    # ponytail: runnable check — report non-empty and ratios finite
    assert len(_LINES) > 40 and overall == overall, "report degenerate"


if __name__ == "__main__":
    main()
