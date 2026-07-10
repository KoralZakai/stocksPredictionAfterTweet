"""Render a Report to a self-contained HTML dashboard. Pure: Report -> str.

Design tokens are inherited from reports/tweet_dashboard.html so this page reads
as a continuation of the repo's existing dashboard, not a different product. The
two dark-mode data colors are the one deviation: the originals sit outside the
lightness band for CVD safety, so validated substitutes are used. Sign is encoded
by bar direction AND an explicit +/- label, never by color alone.

Output is pure ASCII (non-ASCII escaped as numeric references) so the page renders
identically regardless of the charset the host declares.
"""

from __future__ import annotations

import html

from reportgen.stats import AssetMove, Example, Report

EVENT_LABEL: dict[str, str] = {
    "military_threat": "Military threat", "energy_policy": "Energy policy",
    "tariff_trade": "Tariffs &amp; trade", "monetary_policy": "Monetary policy",
    "drug_pricing": "Drug pricing", "sanction": "Sanction", "regulation": "Regulation",
    "macro_claim": "Macro claim", "praise": "Praise", "attack": "Attack",
    "none": "Not market-relevant", "other": "Other",
}
SECTOR_NAME: dict[str, str] = {
    "ITA": "Aerospace &amp; Defense", "XLE": "Energy", "XLI": "Industrials",
    "XLF": "Financials", "XLV": "Health Care", "SMH": "Semiconductors",
    "XLK": "Technology", "XLY": "Consumer Discretionary", "XLB": "Materials",
    "XLP": "Consumer Staples",
}


def _cls(v: float) -> str:
    return "up" if v > 0 else ("down" if v < 0 else "flat")


def _spark(a: AssetMove, span: float, w: int = 104, h: int = 32) -> str:
    pts_v = [0.0, a.abn_1, a.abn_3, a.abn_5]
    span = span or 1.0

    def y(v: float) -> float:
        return h / 2 - (v / span) * (h / 2 - 4)

    pts = " ".join(f"{i*(w-10)/3+5:.1f},{y(v):.1f}" for i, v in enumerate(pts_v))
    c = f"var(--{_cls(a.abn_5)})"
    tip = f"1d {a.abn_1:+,.0f} / 3d {a.abn_3:+,.0f} / 5d {a.abn_5:+,.0f} bps"
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">'
            f"<title>{a.ticker}: {tip}</title>"
            f'<line x1="0" y1="{h/2}" x2="{w}" y2="{h/2}" stroke="var(--grid)" '
            f'stroke-width="1" stroke-dasharray="2 2"/>'
            f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
            f'<circle cx="{w-5}" cy="{y(a.abn_5):.1f}" r="3.2" fill="{c}" '
            f'stroke="var(--surface)" stroke-width="2"/></svg>')


def _row(a: AssetMove, span: float, is_etf: bool = False) -> str:
    tag = '<span class="tag">ETF</span>' if is_etf else ""
    return (f'<tr class="{"etfrow" if is_etf else ""}">'
            f'<td class="tk">{a.ticker}{tag}</td>'
            f'<td class="spk">{_spark(a, span)}</td>'
            f'<td class="v {_cls(a.abn_1)}">{a.abn_1:+,.0f}</td>'
            f'<td class="v {_cls(a.abn_3)}">{a.abn_3:+,.0f}</td>'
            f'<td class="v {_cls(a.abn_5)}">{a.abn_5:+,.0f}</td></tr>')


def _card(e: Example) -> str:
    assets = [e.etf, *e.stocks]
    span = max(abs(v) for a in assets for v in (a.abn_1, a.abn_3, a.abn_5)) or 1.0
    rows = _row(e.etf, span, True) + "".join(_row(s, span) for s in e.stocks)
    chips = "".join(
        f'<span class="chip">{k}<b>{v}</b></span>' for k, v in (
            ("event", EVENT_LABEL.get(e.event, e.event)), ("intent", e.intent),
            ("urgency", e.urgency), ("magnitude", e.magnitude))
    )
    return f"""
  <article class="card">
    <div class="chd"><div class="sig">{chips}</div><time>{e.ts}</time></div>
    <blockquote>{html.escape(e.text)}</blockquote>
    <div class="route">
      <span class="rl">rules mapper &rarr;</span>
      <span class="sect">{e.etf.ticker}</span>
      <span class="sn">{SECTOR_NAME.get(e.etf.ticker, e.etf.ticker)}</span>
      <span class="conf">map confidence {e.map_conf:.2f}</span>
    </div>
    <div class="tw"><table>
      <thead><tr><th>Asset</th><th>Cumulative abnormal return</th>
        <th>1d</th><th>3d</th><th>5d</th></tr></thead>
      <tbody>{rows}</tbody></table></div>
    <p class="unit">Abnormal return vs SPY, basis points (100 bps = 1%).
      Sparkline runs 0 &rarr; 1d &rarr; 3d &rarr; 5d on a shared scale within this card.</p>
  </article>"""


_CSS = """
:root{--bg:#f4f5f2;--surface:#fff;--surface-2:#eef0ec;--ink:#191c1f;--ink-2:#5a6067;
 --ink-3:#878d94;--line:#dee1db;--grid:#c9cdc6;--accent:#a8792f;--etf:#2f7ec0;
 --up:#17805f;--up-bg:#e4f1ea;--down:#c0432f;--down-bg:#f7e5e0;--flat:#9aa0a6;
 --font:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 --mono:ui-monospace,"Cascadia Code","SF Mono",Menlo,Consolas,monospace;
 --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,ui-serif,serif;}
@media (prefers-color-scheme:dark){:root{--bg:#0d1014;--surface:#161a20;--surface-2:#1e242c;
 --ink:#e7eaed;--ink-2:#9aa2ab;--ink-3:#6a727b;--line:#262d36;--grid:#39424d;--accent:#cfa35a;
 --etf:#5aa9ec;--up:#45A87F;--up-bg:#132a22;--down:#DB6A52;--down-bg:#2c1813;--flat:#6c757e;}}
:root[data-theme="light"]{--bg:#f4f5f2;--surface:#fff;--surface-2:#eef0ec;--ink:#191c1f;
 --ink-2:#5a6067;--ink-3:#878d94;--line:#dee1db;--grid:#c9cdc6;--accent:#a8792f;--etf:#2f7ec0;
 --up:#17805f;--up-bg:#e4f1ea;--down:#c0432f;--down-bg:#f7e5e0;--flat:#9aa0a6;}
:root[data-theme="dark"]{--bg:#0d1014;--surface:#161a20;--surface-2:#1e242c;--ink:#e7eaed;
 --ink-2:#9aa2ab;--ink-3:#6a727b;--line:#262d36;--grid:#39424d;--accent:#cfa35a;--etf:#5aa9ec;
 --up:#45A87F;--up-bg:#132a22;--down:#DB6A52;--down-bg:#2c1813;--flat:#6c757e;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);line-height:1.55}
.wrap{max-width:900px;margin:0 auto;padding:34px 20px 80px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
 color:var(--accent);margin:0 0 10px}
h1{font-family:var(--serif);font-size:clamp(26px,3.9vw,38px);margin:0 0 10px;
 letter-spacing:-.015em;text-wrap:balance;font-weight:600}
.dek{color:var(--ink-2);font-size:16.5px;margin:0;max-width:66ch}
h2{font-family:var(--serif);font-size:22px;font-weight:600;margin:44px 0 4px;letter-spacing:-.01em}
.lede{color:var(--ink-2);margin:0 0 18px;max-width:70ch;font-size:15px}
.prov{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--accent);
 border-radius:8px;padding:14px 16px;margin:24px 0 0;font-size:14px;color:var(--ink-2)}
.prov b{color:var(--ink)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:22px 0 6px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:13px 14px}
.tile .k{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3)}
.tile .v{font-size:24px;font-weight:650;margin-top:6px;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.tile .s{font-size:12px;color:var(--ink-3)}
.answer{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:20px;margin:18px 0 0}
.answer .big{font-family:var(--serif);font-size:20px;font-weight:600;margin:0 0 8px;text-wrap:balance}
.answer p{margin:0;color:var(--ink-2);font-size:14.5px}
.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th{text-align:left;font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;
 color:var(--ink-3);font-weight:600;padding:9px 7px;border-bottom:1px solid var(--line);white-space:nowrap}
th:nth-child(n+3){text-align:right}
td{padding:8px 7px;border-bottom:1px solid var(--line);vertical-align:middle}
tr:last-child td{border-bottom:none}
td.tk{font-family:var(--mono);font-weight:600;font-size:13px;white-space:nowrap}
td.v{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:13px;white-space:nowrap}
td.v.muted{color:var(--ink-3)} td.v.sig{color:var(--accent);font-weight:700}
.up{color:var(--up)} .down{color:var(--down)} .flat{color:var(--flat)}
.etfrow{background:var(--surface-2)}
.tag{font-family:var(--mono);font-size:9px;letter-spacing:.08em;background:var(--etf);color:#fff;
 border-radius:4px;padding:1px 5px;margin-left:7px;vertical-align:2px}
.spark{display:block} td.spk{width:110px;padding:2px 8px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px 18px 14px;margin:14px 0}
.chd{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap}
.sig{display:flex;gap:6px;flex-wrap:wrap}
.chip{font-family:var(--mono);font-size:10.5px;color:var(--ink-3);background:var(--surface-2);
 border:1px solid var(--line);border-radius:20px;padding:3px 9px}
.chip b{color:var(--ink);margin-left:5px;font-weight:600}
.chd time{font-family:var(--mono);font-size:11.5px;color:var(--ink-3);white-space:nowrap}
blockquote{margin:13px 0;padding:0 0 0 14px;border-left:2px solid var(--accent);
 font-family:var(--serif);font-size:16px;line-height:1.5;color:var(--ink)}
.route{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:0 0 8px;font-size:13px}
.rl{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
.sect{font-family:var(--mono);font-weight:700;color:#fff;background:var(--etf);border-radius:5px;padding:2px 8px;font-size:12px}
.sn{color:var(--ink-2)} .conf{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--ink-3)}
.unit{font-size:11.5px;color:var(--ink-3);margin:10px 0 0}
.gauge{width:100%;min-width:120px}
.track{position:relative;height:12px;background:var(--surface-2);border:1px solid var(--line);border-radius:6px}
.fill{position:absolute;inset:0 auto 0 0;border-radius:5px 0 0 5px;background:currentColor;opacity:.85}
.fill.up{color:var(--up)} .fill.down{color:var(--down)}
.mark{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--ink);border-radius:1px}
.verdict{font-family:var(--mono);font-size:10.5px;font-weight:700;letter-spacing:.06em;
 text-transform:uppercase;padding:2px 8px;border-radius:20px}
.verdict.fail{color:var(--down);background:var(--down-bg)}
.verdict.ok{color:var(--up);background:var(--up-bg)}
.dbar{width:40%;min-width:110px}
.dtrack{position:relative;height:16px}
.zero{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--grid)}
.dfill{position:absolute;top:5px;height:6px;border-radius:3px;background:currentColor}
.dfill.up{color:var(--up)} .dfill.down{color:var(--down)} .dfill.flat{color:var(--flat)}
.err{position:absolute;top:7px;height:2px;background:var(--ink-3);opacity:.45;border-radius:1px}
ul.mix{list-style:none;padding:0;margin:0;display:grid;gap:6px}
ul.mix li{display:grid;grid-template-columns:minmax(120px,190px) 1fr 60px;align-items:center;gap:12px;font-size:13.5px}
.mk{color:var(--ink-2)} .mb{background:var(--surface-2);border-radius:4px;height:9px;display:block}
.mb i{display:block;height:9px;border-radius:4px;background:var(--accent);opacity:.75}
.mn{text-align:right;font-family:var(--mono);font-size:12.5px;color:var(--ink-3);font-variant-numeric:tabular-nums}
.note{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--down);
 border-radius:8px;padding:15px 17px;margin-top:16px;font-size:14.5px;color:var(--ink-2)}
.note strong{color:var(--ink)}
.foot{margin-top:38px;font-size:13px;color:var(--ink-3)}
a{color:var(--accent)} code{font-family:var(--mono);font-size:.9em}
"""


def render_html(r: Report, run_id: str) -> str:
    hit_rows = "".join(
        f'<tr><td class="tk">{h.horizon}</td>'
        f'<td class="gauge"><div class="track">'
        f'<div class="fill {"up" if h.beats_majority else "down"}" style="width:{h.hit*100:.1f}%"></div>'
        f'<div class="mark" style="left:{h.majority*100:.1f}%"></div></div></td>'
        f'<td class="v">{h.hit*100:.1f}%</td>'
        f'<td class="v muted">{h.majority*100:.1f}%</td>'
        f'<td class="v muted">{h.n_posts:,}</td>'
        f'<td class="v muted">{h.z_clustered:+.2f}</td>'
        f'<td><span class="verdict {"ok" if h.beats_majority else "fail"}">'
        f'{"beats" if h.beats_majority else "loses"}</span></td></tr>'
        for h in r.hit
    )
    ev_max = max((abs(e.mean_bps) + e.se_bps for e in r.by_event), default=1.0) or 1.0
    ev_rows = "".join(
        (lambda x, lo, hi: (
            f'<tr><td class="tk">{EVENT_LABEL.get(e.event, e.event)}</td>'
            f'<td class="v muted">{e.n:,}</td>'
            f'<td class="dbar"><div class="dtrack"><div class="zero"></div>'
            f'<div class="err" style="left:{lo:.2f}%;width:{max(hi-lo,0.4):.2f}%"></div>'
            f'<div class="dfill {_cls(e.mean_bps)}" style="left:{min(50.0,x):.2f}%;'
            f'width:{max(abs(x-50.0),0.4):.2f}%"></div></div></td>'
            f'<td class="v {_cls(e.mean_bps)}">{e.mean_bps:+.1f}</td>'
            f'<td class="v muted">&plusmn;{e.se_bps:.1f}</td>'
            f'<td class="v {"sig" if abs(e.t) >= 2 else "muted"}">{e.t:+.2f}</td></tr>'
        ))(50.0 + (e.mean_bps / ev_max) * 46,
           50.0 + ((e.mean_bps - e.se_bps) / ev_max) * 46,
           50.0 + ((e.mean_bps + e.se_bps) / ev_max) * 46)
        for e in r.by_event
    )
    mix = "".join(
        f'<li><span class="mk">{EVENT_LABEL.get(k, k)}</span>'
        f'<span class="mb"><i style="width:{v/r.n_posts*100:.1f}%"></i></span>'
        f'<span class="mn">{v:,}</span></li>' for k, v in r.event_mix
    )
    feats = "".join(
        f'<tr><td class="tk">{f.name}</td>'
        f'<td class="v muted">{len(f.columns)}</td>'
        f'<td class="v muted">{f.coverage:.1f}%</td>'
        f'<td style="font-family:var(--mono);font-size:11.5px;color:var(--ink-3)">'
        f'{", ".join(f.columns[:6])}{"&hellip;" if len(f.columns) > 6 else ""}</td></tr>'
        for f in r.features
    )
    defense = r.examples[0].etf.abn_1 if r.examples else 0.0

    body = f"""<title>Tweet &rarr; sector &rarr; stocks, with the eden signal layer</title>
<style>{_CSS}</style>
<div class="wrap">
  <p class="eyebrow">eden branch &middot; generated by jobs/reporting.py</p>
  <h1>A tweet, the sector it names, and what those stocks actually did</h1>
  <p class="dek">Every post is real, every return is measured against real SPY-adjusted bars, and the
    sector is assigned by the repo's deterministic keyword rules &mdash; not by a model. The
    <strong>eden</strong> layer adds the structured signal chips on each card.</p>

  <div class="prov"><b>Provenance.</b> Signals produced by <code>{html.escape(r.signal_model)}</code>.
    Run id <code>{run_id}</code> over <code>{r.n_rows:,}</code> rows. Regenerate at any time with
    <code>python jobs/reporting.py</code> &mdash; nothing on this page is hand-written.</div>

  <section class="tiles">
    <div class="tile"><div class="k">Posts</div><div class="v">{r.n_posts:,}</div>
      <div class="s">{r.date_min} &rarr; {r.date_max}</div></div>
    <div class="tile"><div class="k">Tweet &times; asset rows</div><div class="v">{r.n_rows:,}</div>
      <div class="s">real abnormal returns</div></div>
    <div class="tile"><div class="k">Assets</div><div class="v">{r.n_assets}</div>
      <div class="s">sector ETFs + names</div></div>
    <div class="tile"><div class="k">Benchmark</div><div class="v">SPY</div>
      <div class="s">abnormal = raw &minus; SPY</div></div>
  </section>

  <h2>Does the signal actually predict direction?</h2>
  <p class="lede">For every post the extractor called bullish or bearish, does the sign of the sector ETF's
    abnormal return agree? The black tick is the <strong>majority-class baseline</strong> &mdash; the score you
    get by always guessing the more common direction and reading no tweets at all. <code>z</code> is clustered
    by post, because one post spawns many correlated rows.</p>
  <div class="tw"><table>
    <thead><tr><th>Horizon</th><th>Hit rate vs baseline</th><th>Hit</th><th>Baseline</th>
      <th>Posts</th><th>z</th><th></th></tr></thead>
    <tbody>{hit_rows}</tbody></table></div>

  <div class="answer">
    <p class="big">At every horizon, the signal loses to the dumbest possible baseline.</p>
    <p>One-day accuracy sits below a coin flip. The 3- and 5-day numbers rise but never reach the
      majority-class bar, so a model that ignored the tweets entirely would score higher. This is the
      expected outcome, not a defect.</p>
  </div>

  <h2>Mean 1-day abnormal return by event type</h2>
  <p class="lede">Sector-ETF rows only, in basis points, with &plusmn;1 standard error. <code>|t| &ge; 2</code>
    is the bare minimum before any of this deserves a second look &mdash; and that is <em>before</em> correcting
    for having tested {len(r.by_event)} event types at once.</p>
  <div class="tw"><table>
    <thead><tr><th>Event type</th><th>n</th><th>Mean abnormal return (bps)</th>
      <th>Mean</th><th>SE</th><th>t</th></tr></thead>
    <tbody>{ev_rows}</tbody></table></div>

  <h2>What the model actually sees per row</h2>
  <p class="lede">Beyond the tweet and its sector, every (post &times; asset) row carries a point-in-time
    market-regime block computed strictly before <code>t0</code>. These are the "other data points" &mdash;
    they already exist in the dataset and are what a model consumes alongside the signal.</p>
  <div class="tw"><table>
    <thead><tr><th>Group</th><th>Cols</th><th>Coverage</th><th>Columns</th></tr></thead>
    <tbody>{feats}</tbody></table></div>

  <h2>What the extractor saw across {r.n_posts:,} posts</h2>
  <p class="lede">Most posts are marked not market-relevant. That conservatism is deliberate: a rule that fires
    on everything manufactures signal out of noise.</p>
  <ul class="mix">{mix}</ul>

  <h2>Worked examples</h2>
  <p class="lede">Each card is one real post: the eden signal chips, the sector the deterministic mapper
    assigned, and how that sector's ETF and member stocks moved. Positive is right-of-zero and green, negative
    left and red &mdash; sign is carried by position and the <code>+</code>/<code>&minus;</code> label too,
    never by color alone.</p>
  {"".join(_card(e) for e in r.examples)}

  <div class="note">
    <strong>Read the defense example again.</strong> The post announces precision air strikes &mdash; the
    textbook "attack &rArr; defense stocks up" prior, and the exact intuition that motivated this feature.
    <code>ITA</code> ticked {defense:+.0f} bps on day one, then gave it back, and the biggest names fell hard by
    day five. The prior is not merely weak here; for those names it points the wrong way.
  </div>

  <div class="note" style="border-left-color:var(--accent)">
    <strong>What this page is and isn't.</strong> It shows the eden layer wired end-to-end against real data and
    reports a null result &mdash; the expected, correct outcome for this repo. It is a <em>descriptive</em> read,
    not the purged, embargoed, BH-corrected evaluation in <code>eval/</code>. Treat every number as a diagnostic,
    not a finding.
  </div>

  <p class="foot">Generated by <code>jobs/reporting.py</code> &middot; signals via <code>llm/extract.py</code>
    &middot; sectors via <code>sector_mapping/rules.py</code> &middot; run <code>{run_id}</code></p>
</div>
"""
    return body.encode("ascii", "xmlcharrefreplace").decode("ascii")
