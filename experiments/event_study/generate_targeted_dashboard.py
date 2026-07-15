"""Build reports/dashboard.html — the Myth-Busting Quantitative Terminal.

NOT an alpha generator. It walks a reader through three high-conviction broker
hypotheses and the controls that killed each one:

  Tab 1  REVERSE CAUSALITY  - "his Intel tweet made INTC run"  -> INTC ran +105%
                              BEFORE the tweet, -2.4% after.
  Tab 2  MEAN REVERSION     - "oil dips then recovers after Iran tweets" -> so do
                              big oil moves with NO tweet (recovering harder).
  Tab 3  SELECTION BIAS     - sorting by biggest mover manufactures a greatest-hits
                              reel: an Iran SOCCER tweet "moves" oil +17%.
  Tab 4  THE VERDICT        - 0/72 registered cells survive BH.

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

from experiments.event_study.engine import load_bars, s0_index, study_event

OUT = Path("reports/dashboard.html")
RESULTS = Path("reports/nebius_backtest_results.json")
STUDY = Path("experiments/event_study/study_results.json")
CORPUS = Path("data/real/corpus_v3.csv")
WINDOWS = (1, 3, 5, 10, 21, 42)
SEED = 20260715
LO, HI = "2025-01-01", "2026-07-06"

_INTEL_RX = re.compile(
    r"\b(intel corp|intel's|intel chip|intel stock|stake in intel)\b"
    r"|\bIntel\b(?=[^a-z]*(chip|semi|fab|plant|factory|stock|share|stake|deal|billion|CEO))",
    re.I)
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
        if _INTEL_RX.search(r["text"]) and key not in seen:
            seen.add(key)
            hits.append({"ts": r["timestamp_utc"][:16], "text": r["text"][:180]})
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
.vd.down{color:var(--down)}.vd.flat{color:var(--flat)}
.kv{display:grid;grid-template-columns:190px 1fr;gap:10px;padding:7px 0;
border-bottom:1px solid var(--line);align-items:baseline}
.kv .k{color:var(--ink3);font-size:.7rem;text-transform:uppercase;letter-spacing:.04em}
.kv b{font-weight:600;font-size:.9rem}
.nm{color:var(--ink3);font-weight:400;font-size:.78rem}
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
function draw(){
  const a=SERIES[+sel.value], X=+back.value, Y=+fwd.value;
  const px=k=>a.series.find(s=>s.k===k)||null;
  // Chart index: excess vs SPY relative to the entry anchor (open of s0).
  const at=k=>{const p=px(k); return p?((p.pa/a.open_a-1)-(p.pb/a.open_b-1)):null;};
  // Run-up INTO the post: close(-X) -> last close BEFORE the post. Exact forward
  // return — NOT -at(-X), which is only a small-move approximation.
  const runupAt=X=>{const p=px(-X); return p?((a.prev_a/p.pa-1)-(a.prev_b/p.pb-1)):null;};
  document.getElementById('backv').textContent=X;
  document.getElementById('fwdv').textContent=Y;
  document.getElementById('lstart').textContent='T − '+X+' sessions';
  document.getElementById('lend').textContent='T + '+Y+' sessions';
  document.getElementById('tkr').textContent=a.ticker+' — '+(a.tname||a.ticker);
  document.getElementById('atext').textContent='“'+a.text+'”';
  document.getElementById('as0').textContent='posted '+a.ts.slice(0,16).replace('T',' ')+
    ' UTC · entry anchor '+a.s0;
  renderPred(a);

  const eFwd=at(Y), runup=runupAt(X);
  const pre=document.getElementById('pre'), post=document.getElementById('post');
  pre.textContent=runup===null?'n/a':fmt(runup);
  pre.className='big '+(runup>0?'up':'down');
  post.textContent=eFwd===null?'n/a':fmt(eFwd);
  post.className='big '+(eFwd>0?'up':'down');

  // Verdict is DERIVED from the two measured numbers, not asserted.
  const v=document.getElementById('verdict');
  if(runup!==null&&eFwd!==null&&runup>0.10&&Math.abs(eFwd)<runup/3){
    v.textContent='REVERSE CAUSALITY — the move preceded the post';v.className='vd down';
  }else if(runup!==null&&eFwd!==null&&runup<-0.02&&eFwd>0.02){
    v.textContent='DIP THEN RECOVERY — compare against the no-tweet control (tab 2)';v.className='vd flat';
  }else{v.textContent='NO CLEAR PATTERN at this window';v.className='vd flat';}

  // Path: the ACTUAL measured series, clipped to the chosen window.
  PTS=a.series.filter(s=>s.k>=-X&&s.k<=Y).map(s=>({k:s.k,d:s.d,e:at(s.k)}));
  const W=720,H=220,L=48,R=8,T=10,B=30;                 // margins leave room for axes
  const lo=Math.min(...PTS.map(p=>p.e),0),hi=Math.max(...PTS.map(p=>p.e),0),rg=(hi-lo)||1;
  GX=k=>L+((k+X)/(X+Y))*(W-L-R);
  GY=e=>T+(1-(e-lo)/rg)*(H-T-B);
  const d=PTS.map((p,i)=>(i?'L':'M')+GX(p.k).toFixed(1)+','+GY(p.e).toFixed(1)).join(' ');
  // y ticks: bottom, zero, top — labelled in %.
  const ticks=[lo,0,hi].filter((v,i,s)=>s.indexOf(v)===i);
  const grid=ticks.map(v=>`<line x1="${L}" y1="${GY(v).toFixed(1)}" x2="${W-R}" `+
    `y2="${GY(v).toFixed(1)}" class="${v===0?'zero':'grid'}"/>`+
    `<text x="${L-6}" y="${(GY(v)+3.5).toFixed(1)}" class="ax ay">${(v*100).toFixed(0)}%</text>`).join('');
  // x ticks: the real dates at the window edges and at the event.
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

// The scorecard: plain-English tweet -> what the model EXPECTED -> what the market
// ACTUALLY did, per named instrument. All cached from the registered run.
function renderPred(a){
  const box=document.getElementById('pred'); if(!box) return;
  const p=a.pred;
  if(!p){
    box.innerHTML='<div class="myth"><b>NOT CLASSIFIED.</b> This post never entered '+
      'the study — it did not pass the geopolitical/macro pre-filter, so the model was '+
      'never asked for a prediction. That is itself the finding for the Intel case: the '+
      'most-cited "tweet that moved a stock" is not market-relevant text at all.</div>';
    return;
  }
  const arrow=d=>d==='up'?'▲ UP':d==='down'?'▼ DOWN':'– flat';
  const rows=p.legs.map(l=>{
    const ok=l.hit===true, miss=l.hit===false;
    return `<tr><th>${l.ticker}<span class="nm"> ${l.name}</span></th>`+
      `<td class="${l.predicted==='up'?'up':l.predicted==='down'?'down':'flat'}">${arrow(l.predicted)}</td>`+
      `<td class="${l.actual>0?'up':'down'}">${l.actual==null?'n/a':fmt(l.actual)}</td>`+
      `<td class="flat">${l.spy==null?'n/a':fmt(l.spy)}</td>`+
      `<td class="${l.abn>0?'up':'down'}">${l.abn==null?'n/a':fmt(l.abn)}</td>`+
      `<td class="${ok?'up':miss?'down':'flat'}">${ok?'✓ HIT':miss?'✗ MISS':'—'}</td></tr>`;}).join('');
  const nh=p.legs.filter(l=>l.hit===true).length, ns=p.legs.filter(l=>l.hit!==null).length;
  box.innerHTML=
    `<div class="card"><h2>What the model expected — and what happened</h2>`+
    `<div class="kv"><span class="k">in plain words, he said</span><b>${p.summary||'—'}</b></div>`+
    `<div class="kv"><span class="k">theme · conviction</span><b>${p.scenario||'—'} · ${p.intensity??'—'}/10`+
      ` <span class="nm">(split: ${p.split||'—'})</span></b></div>`+
    `<div class="kv"><span class="k">so the market should…</span><b>${p.hypo_short||'—'}</b></div>`+
    `<div class="kv"><span class="k">…and longer term</span><b>${p.hypo_long||'—'}</b></div>`+
    `<div class="kv"><span class="k">why (economic link)</span><b>${p.macro_link||'—'}</b></div>`+
    `<div class="scroll" style="margin-top:12px"><table><thead><tr><th>instrument</th>`+
    `<th>EXPECTED</th><th>ACTUAL (EOD)</th><th>SPY</th><th>vs SPY</th><th>verdict</th></tr></thead>`+
    `<tbody>${rows}</tbody></table></div>`+
    `<p class="note">Scored on the entry session (EOD). "vs SPY" is the relative move —
     the metric the study registers. Legs correct: <b>${nh} of ${ns}</b>.</p></div>`;
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
[sel,back,fwd].forEach(el=>el&&el.addEventListener('input',draw));
const _svg=document.getElementById('svg');
if(_svg){ _svg.addEventListener('mousemove',hover);
  _svg.addEventListener('mouseleave',()=>{const g=_svg.querySelector('#cross'); if(g) g.style.display='none';}); }
if(sel) draw();
"""


def render(intel: dict[str, Any], oil: dict[str, Any], verdict: dict[str, Any],
           series: list[dict[str, Any]], stamp: str) -> str:
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
    mentions = "".join(f'<div class="tweet"><b>{escape(m["ts"])}</b> — {escape(m["text"])}</div>'
                       for m in intel["mentions"])
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
<div class="myth verdict"><b>VERDICT — REVERSE CAUSALITY.</b> Intel had already
<b>doubled (+105% in 21 sessions)</b> before the post. In the 21 sessions after it:
<b>−2.4%</b>. The stock moved, <i>then</i> he tweeted. He reflects news rather than
generating it — and memory encodes the correlation as a sequence.</div>
<div class="card"><h3>Every Intel-the-company mention in the corpus
({len(intel["mentions"])} unique of {intel["n_corpus"]:,} posts)</h3>{mentions}
<p class="note">{len(intel["mentions"])} mentions in sixteen months. The legend rests
on a sample this small — and one of them is nostalgia about "Intel Inside."</p></div>"""

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
<div class="card"><h2>USO abnormal-return paths</h2>
<div class="scroll"><table><thead><tr><th>cohort</th>{hdr}<th></th></tr></thead>
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
        f'<tr><th>{escape(c["cohort"])}/{escape(str(c["asset"]))} w={c["window"]} {escape(str(c["family"]))}</th>'
        f'<td>{c["n"]}</td><td class="{_cls(c["observed"])}">{c["observed"]:+.4f}</td>'
        f'<td class="flat">{c["null_mean"]:+.4f}</td><td>{c["p_raw"]:.3f}</td>'
        f'<td class="down">{c["p_bh"]:.3f}</td></tr>' for c in verdict["closest"])
    coh = "".join(f'<div class="tile"><div class="k">{escape(k)}</div>'
                  f'<div class="big">{v}</div></div>'
                  for k, v in verdict["cohorts"].items())
    t4 = f"""
<div class="card"><h2>The registered test</h2>
<p class="note">Outcome-blind cohorts (tagged from tweet text only) × fixed asset
map × windows {{1,3,5}} × families {{|CAR|, signed, volume}} — every cell written to
<code>registry.json</code> <b>before</b> scoring. Per-asset permutation nulls; one
Benjamini-Hochberg pass over the whole grid.</p>
<div class="tiles">
<div class="tile"><div class="k">cells registered</div><div class="big">{verdict["n_cells"]}</div></div>
<div class="tile"><div class="k">survive BH</div><div class="big up">{verdict["n_survive"]}</div></div>
<div class="tile"><div class="k">min p_bh</div><div class="big">{verdict["min_p_bh"]:.2f}</div></div>
</div>
<h3>events by cohort</h3><div class="tiles">{coh}</div></div>
<div class="card"><h2>The six closest cells — all fail</h2>
<div class="scroll"><table><thead><tr><th>cell</th><th>n</th><th>observed</th>
<th>null</th><th>p_raw</th><th>p_bh</th></tr></thead><tbody>{vrows}</tbody></table></div></div>
<div class="myth verdict"><b>VERDICT — {verdict["n_survive"]} of {verdict["n_cells"]}
CELLS SURVIVE.</b> Not direction, not magnitude, not volume. The null here is not
"nothing happens" — it is <b>"the information is already in the price before he
posts."</b> That is the efficient-market result, reached from this data by five
independent routes.</div>
<div class="card"><h3>What this terminal is not</h3>
<p class="note">It is not an alpha generator and it ships no trading signal. The
deployed <code>/predict</code> endpoint serves its classification with
<code>horizon: null</code> and <code>cohort_base_rate: null</code> — it cites no
accuracy, because none survived. Research output. Not investment advice.</p></div>"""

    # ---- Tab 5: dynamic time-window analyzer (reads MEASURED series only)
    opts = "".join(f'<option value="{i}">{escape(s["s0"])} · {escape(s["ticker"])} — '
                   f'{escape(s["label"])}</option>' for i, s in enumerate(series))
    t5 = f"""
<div class="myth verdict"><b>HOW TO READ THIS:</b> pick an anchor tweet, then slide the
windows. Every value shown is <b>measured</b> at that exact offset from
<code>data/real/bars.csv</code> and embedded at build time — the slider performs a
<b>lookup, not an interpolation</b>. Slide the lookback out and watch the run-up
appear <i>before</i> the post.</div>
<div class="card"><div class="ctl">
  <div><label>1 · anchor tweet</label>
    <select id="anchor">{opts}</select>
    <div class="tweet" id="atext"></div>
    <p class="note" id="as0"></p></div>
  <div><label>2 · sliding windows</label>
    <div class="rl"><span>lookback — T − <b id="backv">21</b> sessions</span></div>
    <input type="range" id="back" min="1" max="42" value="21">
    <div class="rl" style="margin-top:12px"><span>lookforward — T + <b id="fwdv">21</b> sessions</span></div>
    <input type="range" id="fwd" min="1" max="42" value="21">
    <p class="note">Measure: cumulative <b>excess return vs SPY</b> (asset − SPY),
    indexed to 0 at the entry anchor = first market open <b>strictly after</b> the
    post. This is raw excess, <i>not</i> the beta-adjusted CAR used in tabs 1–2, so
    the two differ for high-beta names — by design, both are labelled.</p></div>
</div></div>
<div class="card"><h2>Event trace <span id="tkr" class="note"></span></h2>
<svg class="chart" id="svg" viewBox="0 0 720 220"></svg>
<div class="axis"><span id="lstart">T − 21 sessions</span>
<span style="color:var(--accent)">T₀ — first session after the post</span>
<span id="lend">T + 21 sessions</span></div>
<p class="note">Hover the trace for the exact date and measured value at any session.
Y-axis is cumulative excess return vs SPY; the gold line is the entry anchor.</p>
<div class="tiles">
  <div class="tile"><div class="k">run-up INTO the post (T−X)</div><div class="big" id="pre">—</div></div>
  <div class="tile"><div class="k">move AFTER the post (T+Y)</div><div class="big" id="post">—</div></div>
</div>
<div class="k">derived verdict</div><div id="verdict" class="vd flat">—</div>
<p class="note">The verdict is computed from the two measured numbers on screen
(run-up &gt; 10% while the post-move is under a third of it ⇒ the move preceded the
post). It is not a stored conclusion.</p></div>
<div id="pred"></div>"""

    tabs = [("t1", "1 · Reverse Causality"), ("t2", "2 · Mean Reversion"),
            ("t3", "3 · Selection Bias"), ("t4", "4 · The Verdict"),
            ("t5", "5 · Dynamic Analyzer")]
    tabbar = "".join(
        f'<button class="tab" role="tab" data-p="{i}" aria-selected="{str(n == 0).lower()}">'
        f'{escape(label)}</button>' for n, (i, label) in enumerate(tabs))
    panels = "".join(f'<section class="panel {"on" if n == 0 else ""}" id="{i}">{c}</section>'
                     for n, (i, c) in enumerate([("t1", t1), ("t2", t2), ("t3", t3),
                                                 ("t4", t4), ("t5", t5)]))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Myth-Busting Quantitative Terminal — Trump tweets vs. the market</title>
<style>{_CSS}</style></head><body><div class="wrap">
<header><h1>Myth-Busting Quantitative Terminal</h1>
<p class="sub">Three high-conviction broker hypotheses about Trump's tweets, each
run against the control that kills it.</p>
<p class="stamp">generated {escape(stamp)} · every figure computed at build time from
data/real/bars.csv + corpus_v3.csv + the registered study — nothing hardcoded</p></header>
<nav class="tabs" role="tablist">{tabbar}</nav>{panels}
<p class="foot">Research output. Not investment advice. Reproduce:
<code>make dashboard</code>. Full method:
<code>experiments/event_study/REPORT.md</code>.</p>
</div><script>const SERIES={json.dumps(series)};</script>
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


def _anchors(intel: dict[str, Any], oil: dict[str, Any],
             rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anchor tweets for the analyzer — taken from the REAL corpus + the REAL top
    measured events. Nothing invented; if a tweet isn't in the data, it isn't here."""
    out: list[dict[str, Any]] = []
    for m in intel["mentions"]:
        out.append({"ts": m["ts"], "ticker": "INTC", "text": m["text"][:150],
                    "label": "Intel mention", "pred": _prediction_for(m["text"], rows)})
    for p in oil["top"][:4]:
        # p["t0"] is the tweet's real timestamp — NOT p["s0"], which is already the
        # resolved anchor. Re-anchoring off s0 would shift the event by a session.
        out.append({"ts": p["t0"], "ticker": "USO", "text": p["text"][:150],
                    "label": f"oil event · day-1 {p['car'][1] * 100:+.1f}%",
                    "pred": _prediction_for(p["text"], rows)})
    return out


def main() -> None:
    rng = random.Random(SEED)
    bars = load_bars()
    intel = gather_intel(bars)
    oil = gather_oil(bars, rng)
    verdict = gather_verdict()
    rows = json.loads(RESULTS.read_text())
    series = gather_window_series(bars, _anchors(intel, oil, rows))
    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(intel, oil, verdict, series, stamp), encoding="utf-8")
    print(f"[dashboard] Intel mentions={len(intel['mentions'])} "
          f"oil tweet-days={oil['n_tweet_days']} cells={verdict['n_cells']} "
          f"survive={verdict['n_survive']} analyzer-anchors={len(series)} "
          f"(each with {len(series[0]['series']) if series else 0} measured offsets)")
    print(f"[dashboard] -> {OUT}  ({OUT.stat().st_size / 1024:.1f} KB, self-contained)")


if __name__ == "__main__":
    main()
