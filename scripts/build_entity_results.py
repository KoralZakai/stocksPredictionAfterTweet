"""STEP 3 — full entity-level result set (tweet x resolved-entity), real values.

Schema per row (one per tweet x primary matched entity; competitors are a listed
field, not their own rows):

  tweet_text | tweet_date | primary_entity | match_tier | rel_reason | stance |
  competitor_tickers | sector_etf |
  raw_return[1D,2D,3D,5D,10D,21D] |
  abnormal_return[...]  (raw - sector_etf raw, same window) |
  verdict[...]          (match/opposite/flat/no_call/no_claim)

verdict is stance-driven: only computed when stance is directional (positive/
negative); neutral/no_clear_stance -> "no_claim"; missing bars -> "no_call".
match  = abnormal-return sign agrees with stance direction (beyond the flat band)
opposite = disagrees ; flat = |abnormal| within the vol-scaled band.

Reuses the SAME pure modules as everything else (no second feature path, §3.2):
entity_matches (tiered gazetteer), entity_stance, market_state_as_of,
compute_outcome. Text-only linking; every label close asserted > t0 upstream.

Run (worked example + stance audit, no full-corpus write):
  PYTHONPATH=. .venv/Scripts/python.exe scripts/build_entity_results.py --example
Run (FULL corpus -> data/real/entity_results.csv):
  PYTHONPATH=. .venv/Scripts/python.exe scripts/build_entity_results.py --full
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

import pandas as pd

from config.settings import SETTINGS
from config.universe import SECTOR_STOCKS
from core.calendar import TradingCalendar
from core.market_state import market_state_as_of
from data.sources.interfaces import DailyBar
from data.sources.local import load_corpus
from data.sources.local import LocalPriceSource
from labeling.windows import compute_outcome
from sector_mapping.entities import COMPETITORS, entity_matches, entity_spans
from sector_mapping.stance import entity_stance

BARS = "data/real/bars.csv"
OUT = "data/real/entity_results.csv"
HORIZONS = (1, 2, 3, 5, 10, 21)          # sessions: 1D,2D,3D,5D,10D(2wk),21D(1mo)
SUF = {1: "1D", 2: "2D", 3: "3D", 5: "5D", 10: "10D", 21: "1mo"}
T0, T1 = datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2026, 7, 7, tzinfo=timezone.utc)
LO = datetime(2008, 1, 1, tzinfo=timezone.utc)

# Semis names benchmark to SOXX (the sector's standard ETF, and the CHIPS-Act
# example's benchmark); every other stock to its SECTOR_STOCKS ETF; ETFs to SPY.
SEMIS = frozenset({"INTC", "TSM", "MU", "GFS", "TXN", "NVDA", "AMD", "AVGO", "SMH"})
_STOCK_SECTOR = {n: etf for etf, names in SECTOR_STOCKS.items() for n in names}


def benchmark_for(tk: str) -> str:
    if tk in SEMIS:
        return "SOXX"
    return _STOCK_SECTOR.get(tk, "SPY")


def _pre_vol(closes: list[float]) -> float | None:
    w = SETTINGS.vol_window_sessions
    if len(closes) < w + 1:
        return None
    dr = [closes[i] / closes[i - 1] - 1 for i in range(len(closes) - w, len(closes))]
    return (sum(d * d for d in dr) / len(dr)) ** 0.5


def _verdict(stance: str, abn: float | None, band: float | None) -> str:
    if stance in ("neutral", "no_clear_stance"):
        return "no_claim"
    if abn is None or band is None:
        return "no_call"
    if abs(abn) <= band:
        return "flat"
    want_up = stance == "positive"
    return "match" if (abn > 0) == want_up else "opposite"


def build_rows(post_id: str, text: str, t0: datetime,
               bars: dict[str, list[DailyBar]], cal: TradingCalendar) -> list[dict]:
    """One row per tweet x primary matched entity (competitor-tier skipped)."""
    rows: list[dict] = []
    for tk, m in entity_matches(text).items():
        if m.tier == "competitor":
            continue
        etf = benchmark_for(tk)
        if tk not in bars or etf not in bars:
            rows.append(_row(post_id, text, t0, tk, m, etf, None, None, None))
            continue
        st = market_state_as_of(t0, tk, bars[tk], bars[etf], cal)
        vol = _pre_vol([b.close for b in st.prior_bars])
        out = compute_outcome(t0, bars[tk], cal, HORIZONS)
        eo = compute_outcome(t0, bars[etf], cal, HORIZONS)
        rows.append(_row(post_id, text, t0, tk, m, etf, out, eo, vol))
    return rows


def _row(post_id, text, t0, tk, m, etf, out, eo, vol) -> dict:
    stance = entity_stance(text, entity_spans(text, tk))
    row = {
        "post_id": post_id, "tweet_text": text[:200],
        "tweet_date": t0.date().isoformat(),
        "primary_entity": tk, "match_tier": m.tier, "rel_reason": m.reason,
        "stance": stance.label, "stance_evidence": "|".join(stance.evidence),
        "competitor_tickers": " ".join(COMPETITORS.get(tk, ())),
        "sector_etf": etf, "missing_bars": int(out is None),
    }
    for h in HORIZONS:
        raw = None if out is None else out.ret[h]
        eraw = None if eo is None else eo.ret[h]
        abn = None if (raw is None or eraw is None) else raw - eraw
        band = None if vol is None else SETTINGS.k * vol * (h ** 0.5)
        row[f"raw_{SUF[h]}"] = None if raw is None else round(raw, 4)
        row[f"abn_{SUF[h]}"] = None if abn is None else round(abn, 4)
        row[f"verdict_{SUF[h]}"] = _verdict(stance.label, abn, band)
    return row


def _load(tickers: set[str]) -> tuple[dict, TradingCalendar]:
    price = LocalPriceSource(BARS)
    bars = {t: price.get_daily_bars(t, LO, T1 + timedelta(days=100)) for t in tickers}
    cal = TradingCalendar([b.session_date.date() for b in bars["SPY"]])
    return bars, cal


# The real March 4 2025 CHIPS-Act statement (Congressional address, NOT a Truth
# Social post -> absent from corpus_v3; injected here as an external event). t0 =
# address start ~21:00 ET Mar 4 = 02:00 UTC Mar 5, so s0 resolves to Mar 5 open.
CHIPS_TEXT = ("Your CHIPS Act is a horrible, horrible thing. We give hundreds of "
              "billions of dollars and it doesn't mean a thing. They take our money "
              "and they don't spend it.")
CHIPS_T0 = datetime(2025, 3, 5, 2, 0, tzinfo=timezone.utc)


def worked_example() -> None:
    tickers = {"INTC", "MU", "TSM", "GFS", "TXN", "SOXX", "SPY"}
    bars, cal = _load(tickers)
    rows = build_rows("EXTERNAL:chips_2025-03-04", CHIPS_TEXT, CHIPS_T0, bars, cal)
    print("== WORKED EXAMPLE: March 4 2025 CHIPS-Act statement (INJECTED external "
          "event; benchmark SOXX; t0=2025-03-05 02:00Z -> s0=Mar 5 open) ==")
    print(f'text: "{CHIPS_TEXT}"\n')
    hdr = f'{"ticker":6}{"tier":9}{"stance":16}' + "".join(f"{SUF[h]:>8}" for h in HORIZONS)
    for kind in ("raw_return", "abnormal_return (vs SOXX)", "verdict"):
        print(kind + ":")
        print("  " + hdr)
        for r in sorted(rows, key=lambda r: r["primary_entity"]):
            pfx = "raw_" if kind.startswith("raw") else "abn_" if kind.startswith("ab") else "verdict_"
            def cell(h: int) -> str:
                v = r[f"{pfx}{SUF[h]}"]
                if v is None:
                    return f'{"n/a":>8}'
                return f'{v*100:>7.2f}%' if pfx != "verdict_" else f'{v:>8}'
            flag = "  [MISSING BARS]" if r["missing_bars"] else ""
            print(f'  {r["primary_entity"]:6}{r["match_tier"]:9}{r["stance"]:16}'
                  + "".join(cell(h) for h in HORIZONS) + flag)
        print()


def stance_audit() -> None:
    tweets = load_corpus("data/real/corpus_v3.csv", T0, T1)
    dist, by_tier, nonneu = Counter(), Counter(), Counter()
    for tw in tweets:
        for tk, m in entity_matches(tw.text).items():
            if m.tier == "competitor":
                continue
            s = entity_stance(tw.text, entity_spans(tw.text, tk))
            dist[s.label] += 1
            by_tier[m.tier] += 1
            if s.label in ("positive", "negative"):
                nonneu[m.tier] += 1
    tot = sum(dist.values())
    directional = dist["positive"] + dist["negative"] + dist["neutral"]
    print("== STANCE AUDIT (all non-competitor entity-linked pairs, full 2025+ corpus) ==")
    print(f"total pairs: {tot}   (of these {directional} carry any sentiment word)")
    for lab in ("positive", "negative", "neutral", "no_clear_stance"):
        print(f"  {lab:16} {dist[lab]:5}  {100*dist[lab]/tot:5.1f}% of all"
              + (f"   {100*dist[lab]/directional:5.1f}% of sentiment-bearing"
                 if lab in ("positive", "negative", "neutral") and directional else ""))
    print("\nnon-neutral (pos|neg) stance rate by match_tier "
          "(sanity: is stance tracking the entity or random adjectives?):")
    for tier in ("direct", "indirect"):
        n = by_tier[tier]
        if n:
            print(f"  {tier:9} {nonneu[tier]:5}/{n:5} = {100*nonneu[tier]/n:5.1f}% non-neutral")


def _dir(abn: float | None, band: float | None) -> str:
    if abn is None or band is None:
        return "n/a"
    return "UP" if abn > band else "DOWN" if abn < -band else "FLAT"


def _ret(t0, tk, bars, cal, h) -> tuple[float | None, float | None]:
    """(raw_h, abnormal_h vs benchmark_for(tk)) for one ticker at horizon h."""
    etf = benchmark_for(tk)
    if tk not in bars or etf not in bars:
        return None, None
    o = compute_outcome(t0, bars[tk], cal, (h,))
    e = compute_outcome(t0, bars[etf], cal, (h,))
    raw = None if o is None else o.ret[h]
    er = None if e is None else e.ret[h]
    return raw, (None if (raw is None or er is None) else raw - er)


def _band(t0, tk, bars, cal, h) -> float | None:
    etf = benchmark_for(tk)
    if tk not in bars:
        return None
    st = market_state_as_of(t0, tk, bars[tk], bars.get(etf, bars["SPY"]), cal)
    vol = _pre_vol([b.close for b in st.prior_bars])
    return None if vol is None else SETTINGS.k * vol * (h ** 0.5)


def _event_block(text, t0, core, comps, bench, bars, cal, hs=(1, 5, 21)) -> None:
    print(f'  "{text[:120]}"')
    print(f"  core: {core}   benchmark(sector): {bench}")
    print(f'    {"horizon":8}{"core raw":>10}{"core abn":>10}{"dir vs band":>13}'
          f'{"  | sector spillover (raw / abn):":<20}')
    for h in hs:
        craw, cabn = _ret(t0, core, bars, cal, h)
        d = _dir(cabn, _band(t0, core, bars, cal, h))
        braw, _ = _ret(t0, bench, bars, cal, h)  # benchmark abn vs itself ~ 0
        spill = "  ".join(
            f"{c} {('n/a' if r is None else f'{r*100:+.1f}%')}/"
            f"{('n/a' if a is None else f'{a*100:+.1f}%')}"
            for c in comps for r, a in [_ret(t0, c, bars, cal, h)])
        bs = "n/a" if braw is None else f"{braw*100:+.1f}%"
        cr = "n/a" if craw is None else f"{craw*100:+.1f}%"
        ca = "n/a" if cabn is None else f"{cabn*100:+.1f}%"
        print(f"    {SUF[h]:8}{cr:>10}{ca:>10}{d:>13}   sector {bench} {bs}")
        if spill:
            print(f"             competitors: {spill}")
    print()


def event_report() -> None:
    universe = {"SPY", "SOXX", *SECTOR_STOCKS.keys(),
                *(n for ns in SECTOR_STOCKS.values() for n in ns),
                "MU", "GFS", "TXN"}
    bars, cal = _load(universe)

    print("=" * 78)
    print("A. FLAGSHIP SEMIS EVENT — CHIPS Act statement (injected, 2025-03-04)")
    print("=" * 78)
    _event_block(CHIPS_TEXT, CHIPS_T0, "INTC", ("MU", "TSM", "GFS", "TXN", "AMD", "NVDA"),
                 "SOXX", bars, cal)

    print("=" * 78)
    print("B. TOP CORPUS EVENTS — direct mention + explicit stance, by |abnormal 5D|")
    print("=" * 78)
    tweets = load_corpus("data/real/corpus_v3.csv", T0, T1)
    cand = []
    for tw in tweets:
        for tk, m in entity_matches(tw.text).items():
            if m.tier != "direct":
                continue
            s = entity_stance(tw.text, entity_spans(tw.text, tk))
            if s.label not in ("positive", "negative"):
                continue
            _, abn = _ret(tw.timestamp_utc, tk, bars, cal, 5)
            if abn is not None:
                cand.append((abs(abn), tw.text, tw.timestamp_utc, tk, s.label))
    for _, text, t0, core, stance in sorted(cand, key=lambda x: -x[0])[:8]:
        print(f"[stance={stance}]")
        _event_block(text, t0, core, COMPETITORS.get(core, ()), benchmark_for(core),
                     bars, cal)


def _event_dict(text, t0, core, comps, bench, bars, cal, hs=(1, 3, 5, 10, 21),
                platform="", source="") -> dict:
    stance = entity_stance(text, entity_spans(text, core))
    horizons = {}
    for h in hs:
        craw, cabn = _ret(t0, core, bars, cal, h)
        braw, _ = _ret(t0, bench, bars, cal, h)
        horizons[SUF[h]] = {
            "core_raw": craw, "core_abn": cabn,
            "dir": _dir(cabn, _band(t0, core, bars, cal, h)),
            "bench_raw": braw,
            "competitors": [
                {"tk": c, "raw": r, "abn": a}
                for c in comps for r, a in [_ret(t0, c, bars, cal, h)]
            ],
        }
    return {"text": text, "date": t0.date().isoformat(), "core": core,
            "sector": bench, "stance": stance.label, "platform": platform, "source": source,
            "stance_evidence": list(stance.evidence), "horizons": horizons}


def events_json(path: str) -> None:
    import json
    universe = {"SPY", "SOXX", *SECTOR_STOCKS.keys(),
                *(n for ns in SECTOR_STOCKS.values() for n in ns), "MU", "GFS", "TXN"}
    bars, cal = _load(universe)
    flagship = _event_dict(CHIPS_TEXT, CHIPS_T0, "INTC",
                           ("MU", "TSM", "GFS", "TXN", "AMD", "NVDA"), "SOXX", bars, cal,
                           platform="Congressional address", source="C-SPAN (injected)")
    tweets = load_corpus("data/real/corpus_v3.csv", T0, T1)
    cand = []
    for tw in tweets:
        for tk, m in entity_matches(tw.text).items():
            if m.tier != "direct":
                continue
            s = entity_stance(tw.text, entity_spans(tw.text, tk))
            if s.label not in ("positive", "negative"):
                continue
            _, abn = _ret(tw.timestamp_utc, tk, bars, cal, 5)
            if abn is not None:
                cand.append((abs(abn), tw.text, tw.timestamp_utc, tk, tw.platform, tw.source))
    events = [_event_dict(text, t0, core, COMPETITORS.get(core, ()),
                          benchmark_for(core), bars, cal, platform=plat, source=src)
              for _, text, t0, core, plat, src in sorted(cand, key=lambda x: -x[0])[:9]]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"flagship": flagship, "events": events}, f, indent=1)
    print(f"wrote {1 + len(events)} events -> {path}")


SEMIS_VIEW = ("INTC", "TSM", "MU", "GFS", "TXN", "AMD", "NVDA")


def sector_view(core: str, etf: str) -> tuple[list[str], str]:
    """(member tickers incl. core, sector-ETF) to plot for a core stock."""
    if etf in ("SOXX", "SMH") or core in SEMIS_VIEW:
        members = list(dict.fromkeys([core, *SEMIS_VIEW]))
        return members, "SOXX"
    members = list(dict.fromkeys([core, *SECTOR_STOCKS.get(etf, ())]))
    return members, etf


def _norm_series(t0: datetime, tk: str, bars: dict, cal: TradingCalendar,
                 span: int = 63) -> list[dict] | None:
    """[{d, v}] rebased to the last close BEFORE the tweet = day 0 = 0%.
    d = session offset from that reference bar; the tweet fires just after d=0,
    so d>0 is the reaction and d<0 is the run-up. v = pct vs the reference close."""
    if tk not in bars:
        return None
    st = market_state_as_of(t0, tk, bars[tk], bars[tk], cal)
    if not st.prior_bars:
        return None
    ref_date = st.prior_bars[-1].session_date.date()    # last session closed before t0
    ref = st.prior_bars[-1].close
    by_date = {b.session_date.date(): b.close for b in bars[tk]}
    out = []
    for d in range(-span, span + 1):
        sd = cal.session_at_offset(ref_date, d)
        if sd is not None and sd in by_date:
            out.append({"d": d, "v": round((by_date[sd] / ref - 1) * 100, 3)})
    return out or None


def series_json(path: str) -> None:
    import json
    universe = {"SPY", "SOXX", *SECTOR_STOCKS.keys(),
                *(n for ns in SECTOR_STOCKS.values() for n in ns), "MU", "GFS", "TXN"}
    bars, cal = _load(universe)

    def event_obj(text, t0, core, etf, platform, source) -> dict | None:
        members, sect_etf = sector_view(core, etf)
        series = {}
        for tk in [*members, sect_etf]:
            s = _norm_series(t0, tk, bars, cal)
            if s is not None:
                series[tk] = s
        if core not in series:
            return None
        stance = entity_stance(text, entity_spans(text, core))
        h5 = _ret(t0, core, bars, cal, 5)[1]
        return {"text": text, "date": t0.date().isoformat(),
                "platform": platform, "source": source,
                "core": core, "core_name": None, "sector_etf": sect_etf,
                "stance": stance.label, "members": members,
                "abn_5D": None if h5 is None else round(h5 * 100, 2),
                "series": series}

    flag = event_obj(CHIPS_TEXT, CHIPS_T0, "INTC", "SOXX",
                     "Congressional address", "C-SPAN (injected)")
    tweets = load_corpus("data/real/corpus_v3.csv", T0, T1)
    cand = []
    for tw in tweets:
        for tk, m in entity_matches(tw.text).items():
            if m.tier != "direct":
                continue
            s = entity_stance(tw.text, entity_spans(tw.text, tk))
            if s.label not in ("positive", "negative"):
                continue
            abn = _ret(tw.timestamp_utc, tk, bars, cal, 5)[1]
            if abn is not None:
                cand.append((abs(abn), tw.text, tw.timestamp_utc, tk,
                             benchmark_for(tk), tw.platform, tw.source))
    seen: set = set()
    events = [flag] if flag else []
    for _, text, t0, core, etf, plat, src in sorted(cand, key=lambda x: -x[0]):
        key = (core, t0.date())
        if key in seen:
            continue
        seen.add(key)
        obj = event_obj(text, t0, core, etf, plat, src)
        if obj:
            events.append(obj)
        if len(events) >= 16:
            break
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"events": events}, f, indent=1)
    print(f"wrote {len(events)} events -> {path}")
    for e in events[:3]:
        d0 = next((p["v"] for p in e["series"][e["core"]] if p["d"] == 0), None)
        print(f"  {e['core']:5} {e['date']}  lines={len(e['series'])}  core v@d0={d0}")


def cards_json(path: str) -> None:
    """One card per clear-stance tweet: prior-month trend + after moves (+30m/+1h
    intraday, EOD/+3d/+5d daily), stock and sector, joined from the two panels."""
    import json
    intra = pd.read_csv("data/real/intraday_reactions.csv")
    intra = intra[intra["stance"].isin(("positive", "negative"))]
    ts_by_post = {t.tweet_id: t for t in load_corpus("data/real/corpus_v3.csv", T0, T1)}

    universe = {"SPY", "SOXX", *SECTOR_STOCKS.keys(),
                *(n for ns in SECTOR_STOCKS.values() for n in ns), "MU", "GFS", "TXN"}
    bars, cal = _load(universe)

    def spark_at(series: list[dict], d: int) -> float | None:
        return next((p["v"] for p in series if p["d"] == d), None)

    cards = []
    for _, r in intra.iterrows():
        tw = ts_by_post.get(str(r["post_id"]))
        if tw is None:
            continue
        core, etf = str(r["entity"]), str(r["sector_etf"])
        t0 = tw.timestamp_utc
        s_stock = _norm_series(t0, core, bars, cal, span=21)
        s_etf = _norm_series(t0, etf, bars, cal, span=21)
        if not s_stock or not s_etf:
            continue
        # intraday sector-ETF move = stock_raw - abnormal (abn = stock - etf)
        def sec_intra(raw: str, abn: str) -> float | None:
            a, b = r[raw], r[abn]
            return None if pd.isna(a) or pd.isna(b) else round((float(a) - float(b)) * 100, 2)
        def stock_intra(raw: str) -> float | None:
            return None if pd.isna(r[raw]) else round(float(r[raw]) * 100, 2)
        after = {
            "30m": {"s": stock_intra("raw_30m"), "sec": sec_intra("raw_30m", "abn_30m")},
            "1h":  {"s": stock_intra("raw_1h"),  "sec": sec_intra("raw_1h", "abn_1h")},
            "EOD": {"s": spark_at(s_stock, 1),  "sec": spark_at(s_etf, 1)},
            "3d":  {"s": spark_at(s_stock, 3),  "sec": spark_at(s_etf, 3)},
            "5d":  {"s": spark_at(s_stock, 5),  "sec": spark_at(s_etf, 5)},
        }
        pre = spark_at(s_stock, -21)                 # % of ref vs a month earlier
        pre_trend = None if pre is None else round(-pre, 2)  # rise INTO the tweet
        five = after["5d"]["s"]
        stance = str(r["stance"])
        verdict = ("no_call" if five is None else
                   "flat" if abs(five) < 1.5 else
                   "match" if (five > 0) == (stance == "positive") else "opposite")
        cards.append({
            "text": tw.text, "date": t0.date().isoformat(),
            "platform": tw.platform, "source": tw.source,
            "core": core, "sector_etf": etf, "stance": stance,
            "expected": "up" if stance == "positive" else "down",
            "pre_trend": pre_trend, "verdict": verdict,
            "spark": {core: [p for p in s_stock if -21 <= p["d"] <= 5],
                      etf: [p for p in s_etf if -21 <= p["d"] <= 5]},
            "after": after,
        })

    cards.sort(key=lambda c: c["date"])
    tally = Counter(c["verdict"] for c in cards)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cards": cards, "tally": dict(tally)}, f, indent=1)
    print(f"wrote {len(cards)} cards -> {path}")
    print("5d verdict tally:", dict(tally))


def full() -> None:
    universe = {"SPY", "SOXX", *SECTOR_STOCKS.keys(),
                *(n for ns in SECTOR_STOCKS.values() for n in ns),
                "MU", "GFS", "TXN"}
    bars, cal = _load(universe)
    tweets = load_corpus("data/real/corpus_v3.csv", T0, T1)
    rows: list[dict] = []
    for tw in tweets:
        rows += build_rows(tw.tweet_id, tw.text, tw.timestamp_utc, bars, cal)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {len(df)} rows ({df['post_id'].nunique()} tweets) -> {OUT}")
    print("verdict balance (5D):", df["verdict_5D"].value_counts(dropna=False).to_dict())
    # ponytail: abnormal identity check on a sample
    chk = df.dropna(subset=["raw_5D", "abn_5D"]).head(500)
    assert (chk["missing_bars"] == 0).all()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--example"
    if mode == "--full":
        full()
    elif mode == "--report":
        event_report()
    elif mode == "--json":
        events_json(sys.argv[2] if len(sys.argv) > 2 else "events.json")
    elif mode == "--series":
        series_json(sys.argv[2] if len(sys.argv) > 2 else "events_series.json")
    elif mode == "--cards":
        cards_json(sys.argv[2] if len(sys.argv) > 2 else "tweet_cards.json")
    else:
        worked_example()
        print()
        stance_audit()
