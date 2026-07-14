"""Per-tweet multi-benchmark chart (v-multibench viz). Pure: row dict -> HTML/SVG.

One card per (tweet x stock) from data/real/labeled_multibench.csv. Each card shows,
over the horizon axis (EOD -> 1mo), four cumulative-return lines:

  * the STOCK (gold, bold)
  * the averaged INDICES it belongs to     (blue)   -- e.g. mean(SPY, QQQ)
  * the averaged SECTOR ETF(s)             (purple) -- e.g. SMH
  * the PEER median                        (gray)   -- its sector siblings

plus a header naming the stock in plain English and listing EXACTLY which
benchmarks it was compared against (the user requirement: "show which benchmarks
were used for each stock"), the LLM stance, and a per-horizon verdict strip
(UP / DOWN / NEUTRAL from the folded label). The benchmark lines are reconstructed
from the stored abnormals (benchmark_ret = stock_raw - abn), so the chart and the
label can never disagree.

Colours here are CATEGORY colours (which line), not sign colours; the verdict
badges use the sign palette. Self-contained ASCII HTML, same design tokens as
reportgen/render.py so it reads as one dashboard.
"""

from __future__ import annotations

import html

from config.membership import name_of

# horizon suffix -> x-axis label, in order.
AXIS: list[tuple[str, str]] = [
    ("eod", "EOD"), ("2d", "2d"), ("3d", "3d"), ("1w", "1w"),
    ("2w", "2w"), ("3w", "3w"), ("1mo", "1mo"),
]
# The stock is the HERO line — its colour is set by sign (green/red/gray) at draw
# time, not fixed here. The three benchmark lines are muted grey references so the
# green/red carries the "did it beat the market?" message, not a 4-colour rainbow.
BENCH_LINES = [
    ("index", "indices (avg)"),
    ("sector", "sector (avg)"),
    ("peer", "peers (median)"),
]
BENCH_COLOR = "var(--ink-3)"


def _f(row: dict, col: str) -> float | None:
    v = row.get(col)
    if v is None or v == "" or (isinstance(v, float) and v != v):  # None/""/NaN
        return None
    return float(v)


def _series(row: dict) -> dict[str, list[float | None]]:
    """Per line: cumulative return (%) at each horizon. benchmark = raw - abn."""
    out: dict[str, list[float | None]] = {k: [] for k in ("stock", "index", "sector", "peer")}
    for suf, _ in AXIS:
        raw = _f(row, f"raw_{suf}")
        out["stock"].append(None if raw is None else raw * 100)
        for name, abncol in (("index", "abn_idx"), ("sector", "abn_sec"), ("peer", "abn_peer")):
            abn = _f(row, f"{abncol}_{suf}")
            out[name].append(None if raw is None or abn is None else (raw - abn) * 100)
    return out


def _net_verdict(row: dict) -> str:
    """UP / DOWN / NEUTRAL summarising the card, from the folded per-horizon labels.

    Majority of directional (UP/DOWN) labels wins; ties or all-flat -> NEUTRAL.
    This drives BOTH the headline and the hero-line colour so they always agree.
    """
    ups = sum(str(row.get(f"label_{s}")) == "UP" for s, _ in AXIS)
    downs = sum(str(row.get(f"label_{s}")) == "DOWN" for s, _ in AXIS)
    if ups > downs:
        return "UP"
    if downs > ups:
        return "DOWN"
    return "NEUTRAL"


# net verdict -> (hero line colour, sign css class)
_SIGN = {
    "UP": ("var(--up)", "up"),
    "DOWN": ("var(--down)", "down"),
    "NEUTRAL": ("var(--flat)", "flat"),
}


def _headline(row: dict, override: str | None = None) -> tuple[str, str]:
    """(text, sign-class) bold banner. `override` lets the macro layer pass a
    scenario headline (e.g. 'Peace Rally Triggered') without touching this logic."""
    verdict = _net_verdict(row)
    if override:
        return override, _SIGN[verdict][1]
    name = str(row.get("entity_name") or row.get("entity") or "The stock")
    if verdict == "UP":
        return f"Result: {name} Outperformed its Market & Sector — Positive Drift", "up"
    if verdict == "DOWN":
        return f"Result: {name} Underperformed the Market — Negative Drift", "down"
    return "Result: Market Ignored This Tweet — Flat Across All Horizons", "flat"


def _svg(row: dict, w: int = 560, h: int = 220) -> str:
    series = _series(row)
    hero_color, _ = _SIGN[_net_verdict(row)]
    vals = [v for line in series.values() for v in line if v is not None]
    span = max((abs(v) for v in vals), default=1.0) or 1.0
    pad_l, pad_b, pad_t = 46, 26, 20
    plot_w, plot_h = w - pad_l - 90, h - pad_b - pad_t  # right gutter for zone labels
    n = len(AXIS)

    def x(i: int) -> float:
        return pad_l + (plot_w * i / (n - 1) if n > 1 else 0)

    def y(v: float) -> float:
        return pad_t + plot_h / 2 - (v / span) * (plot_h / 2 - 6)

    zero = y(0)
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" '
             f'class="mbchart"><title>{html.escape(str(row.get("entity", "")))} vs the market</title>']
    # up / down zone tints so "above the line = went up" is unmissable
    parts.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{zero - pad_t:.1f}" '
                 f'fill="var(--up-bg)" opacity="0.5"/>')
    parts.append(f'<rect x="{pad_l}" y="{zero:.1f}" width="{plot_w}" height="{pad_t + plot_h - zero:.1f}" '
                 f'fill="var(--down-bg)" opacity="0.5"/>')
    # zone captions (right gutter)
    parts.append(f'<text x="{pad_l + plot_w + 8}" y="{pad_t + 12}" class="zone up">'
                 f'&#9650; Price Went UP</text>')
    parts.append(f'<text x="{pad_l + plot_w + 8}" y="{pad_t + plot_h - 4}" class="zone down">'
                 f'&#9660; Price Went DOWN</text>')
    # 0% baseline = the tweet minute
    parts.append(f'<line x1="{pad_l}" y1="{zero:.1f}" x2="{pad_l + plot_w}" y2="{zero:.1f}" '
                 f'stroke="var(--ink-2)" stroke-width="1.2"/>')
    parts.append(f'<text x="{pad_l - 4}" y="{zero - 3:.1f}" text-anchor="end" class="ax base">'
                 f'0% &#183; tweet</text>')
    parts.append(f'<text x="6" y="{y(span):.1f}" class="ax">+{span:.0f}%</text>')
    parts.append(f'<text x="6" y="{y(-span):.1f}" class="ax">-{span:.0f}%</text>')
    for i, (_, lab) in enumerate(AXIS):
        parts.append(f'<text x="{x(i):.1f}" y="{h-6}" text-anchor="middle" class="ax">{lab}</text>')
    # benchmark reference lines: thin, muted grey (the "market")
    for key, _lab in BENCH_LINES:
        pts = [(x(i), y(v)) for i, v in enumerate(series[key]) if v is not None]
        if len(pts) < 2:
            continue
        d = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        parts.append(f'<polyline points="{d}" fill="none" stroke="{BENCH_COLOR}" '
                     f'stroke-width="1.3" stroke-dasharray="3 2" opacity="0.75"/>')
    # hero: the stock, thick, coloured by its net verdict (green/red/grey)
    pts = [(x(i), y(v)) for i, v in enumerate(series["stock"]) if v is not None]
    if len(pts) >= 2:
        d = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        parts.append(f'<polyline points="{d}" fill="none" stroke="{hero_color}" '
                     f'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        lpx, lpy = pts[-1]
        parts.append(f'<circle cx="{lpx:.1f}" cy="{lpy:.1f}" r="3.5" fill="{hero_color}"/>')
    parts.append("</svg>")
    return "".join(parts)


def _verdict_gauge(row: dict) -> str:
    """Colour-dot gauge per horizon: green=beat market, red=lagged, grey=flat."""
    cells = []
    for suf, lab in AXIS:
        v = str(row.get(f"label_{suf}", "NA"))
        cls = "up" if v == "UP" else "down" if v == "DOWN" else "flat"
        glyph = "&#9650;" if v == "UP" else "&#9660;" if v == "DOWN" else "&#8212;"
        cells.append(f'<div class="gcell {cls}"><span class="gh">{lab}</span>'
                     f'<span class="gg">{glyph}</span></div>')
    intra = ""
    for win in ("30m", "1h"):
        v = str(row.get(f"label_{win}", "NA"))
        if v not in ("NA", ""):
            cls = "up" if v == "UP" else "down" if v == "DOWN" else "flat"
            intra += f'<span class="chip sec">{win} (sector) <b class="{cls}">{v}</b></span>'
    return (f'<div class="gauge">{"".join(cells)}</div>'
            + (f'<div class="intra">{intra}</div>' if intra else ""))


def _bench_label(row: dict) -> str:
    """Human-readable 'compared vs' line: Name (TICKER), from membership.name_of."""
    def names(field: str) -> str:
        toks = str(row.get(field, "")).split()
        return ", ".join(f"{html.escape(name_of(t))} ({t})" for t in toks) or "(none)"
    fb = " &middot; broad-market fallback" if str(row.get("used_fallback", "0")) == "1" else ""
    return (f'<b>Indices:</b> {names("indices_used")} &nbsp; '
            f'<b>Sector:</b> {names("sectors_used")} &nbsp; '
            f'<b>Peers:</b> {names("peers_used")}{fb}')


def _legend() -> str:
    items = [("var(--up)", "stock beat the market"), ("var(--down)", "stock lagged"),
             (BENCH_COLOR, "market / sector / peers (reference)")]
    return "".join(f'<span class="lg"><i style="background:{c}"></i>{lab}</span>' for c, lab in items)


def card(row: dict, headline: str | None = None) -> str:
    ent = str(row.get("entity", ""))
    name = name_of(ent) if ent else str(row.get("entity_name", ""))
    stance = str(row.get("stance", ""))
    sc = "up" if stance == "positive" else "down" if stance == "negative" else "flat"
    head, hsign = _headline(row, headline)
    return f"""
  <article class="card">
    <div class="hl {hsign}">{html.escape(head)}</div>
    <div class="chd">
      <div class="ttl"><span class="tk">{ent}</span>
        <span class="nm">{html.escape(name)}</span>
        <span class="chip {sc}">tweet tone <b>{stance}</b></span></div>
      <time>{row.get("tweet_date", "")}</time>
    </div>
    <blockquote>{html.escape(str(row.get("text", "")))}</blockquote>
    <div class="bench">Compared against &nbsp; {_bench_label(row)}</div>
    <div class="legend">{_legend()}</div>
    {_svg(row)}
    <p class="unit">Green line above the 0% baseline = the stock rose faster than the market
      after the tweet. It only counts as a "beat" when it tops indices AND its sector AND its
      peers beyond normal daily noise. Dots below show that verdict at each point in time.</p>
    {_verdict_gauge(row)}
  </article>"""


_CSS = """
:root{--bg:#f4f5f2;--surface:#fff;--surface-2:#eef0ec;--ink:#191c1f;--ink-2:#5a6067;
 --ink-3:#878d94;--line:#dee1db;--grid:#c9cdc6;--accent:#a8792f;--etf:#2f7ec0;
 --up:#17805f;--up-bg:#e4f1ea;--down:#c0432f;--down-bg:#f7e5e0;--flat:#9aa0a6;
 --font:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);
 font-size:14px;line-height:1.5}
.wrap{max-width:920px;margin:0 auto;padding:28px 18px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--ink-2);margin:0 0 22px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;
 padding:16px 18px;margin:0 0 18px}
.chd{display:flex;justify-content:space-between;align-items:baseline;gap:12px}
.ttl{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.tk{font-weight:700;font-size:17px}
.nm{color:var(--ink-2)}
time{color:var(--ink-3);font-variant-numeric:tabular-nums}
.chip{background:var(--surface-2);border-radius:20px;padding:2px 10px;font-size:12px}
.chip.up b{color:var(--up)}.chip.down b{color:var(--down)}.chip.flat b{color:var(--ink-2)}
blockquote{margin:10px 0;padding:8px 12px;border-left:3px solid var(--accent);
 background:var(--surface-2);border-radius:0 6px 6px 0;color:var(--ink)}
.bench{font-size:12px;color:var(--ink-2);margin:6px 0}
.bench b{color:var(--ink)}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 2px}
.lg{font-size:12px;color:var(--ink-2);display:flex;align-items:center;gap:5px}
.lg i{width:14px;height:3px;border-radius:2px;display:inline-block}
.mbchart{display:block;margin:2px 0}
.ax{fill:var(--ink-3);font-size:10px}
.ax.base{fill:var(--ink-2);font-weight:600}
.zone{font-size:10px;font-weight:700}
.zone.up{fill:var(--up)}.zone.down{fill:var(--down)}
.unit{color:var(--ink-3);font-size:11px;margin:4px 0 8px}
.hl{font-size:15px;font-weight:700;padding:9px 12px;border-radius:8px;margin:0 0 12px;
 border-left:5px solid var(--flat);background:var(--surface-2)}
.hl.up{border-left-color:var(--up);background:var(--up-bg);color:var(--up)}
.hl.down{border-left-color:var(--down);background:var(--down-bg);color:var(--down)}
.hl.flat{color:var(--ink-2)}
.gauge{display:flex;gap:6px;margin:8px 0 2px}
.gcell{flex:1;text-align:center;padding:6px 2px;border-radius:7px;background:var(--surface-2)}
.gcell .gh{display:block;font-size:10px;color:var(--ink-2);font-weight:600}
.gcell .gg{font-size:15px;font-weight:700;color:var(--ink-3)}
.gcell.up{background:var(--up-bg)}.gcell.up .gg{color:var(--up)}
.gcell.down{background:var(--down-bg)}.gcell.down .gg{color:var(--down)}
.intra{margin-top:6px;display:flex;gap:8px;flex-wrap:wrap}
.chip.sec{font-size:11px}.chip.sec b.up{color:var(--up)}.chip.sec b.down{color:var(--down)}
.chip.sec b.flat{color:var(--ink-3)}
footer{color:var(--ink-3);font-size:11px;margin-top:24px}
"""


def render_page(rows: list[dict], run_id: str = "", title: str = "Tweet -> stock: multi-benchmark reactions") -> str:
    cards = "".join(card(r) for r in rows)
    stamp = f' &middot; run {run_id}' if run_id else ""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{title}</title><style>{_CSS}</style></head><body><div class=\"wrap\">"
            f"<h1>{title}</h1><p class=\"sub\">Each card: one tweet, one stock, judged "
            f"against the average of every index it belongs to, its sector ETF(s), and "
            f"its sector peers.{stamp}</p>{cards}"
            f"<footer>Abnormal return = stock return minus the averaged benchmark over the "
            f"same window. Association at the drift horizon, not a causal claim. Intraday "
            f"badges are sector-only. </footer></div></body></html>").encode("ascii", "xmlcharrefreplace").decode("ascii")


def _demo() -> None:
    """Self-check: a hand row renders a card with a stock line and a verdict strip."""
    row = {
        "entity": "INTC", "entity_name": "Intel", "stance": "positive",
        "tweet_date": "2025-08-11", "text": "positive Intel example",
        "indices_used": "SPY QQQ", "sectors_used": "SMH",
        "peers_used": "NVDA AMD TSM AVGO", "used_fallback": "0",
        "raw_3d": 0.1255, "abn_idx_3d": 0.085, "abn_sec_3d": 0.0849,
        "abn_peer_3d": 0.04, "band_3d": 0.02, "label_3d": "UP",
        "raw_1mo": 0.1684, "abn_idx_1mo": 0.1307, "abn_sec_1mo": 0.1307,
        "abn_peer_1mo": 0.05, "band_1mo": 0.03, "label_1mo": "UP",
        "label_30m": "NEUTRAL",
    }
    page = render_page([row], run_id="demo")
    assert "Outperformed" in page and "polyline" in page          # dynamic headline fired
    assert "S&amp;P 500 ETF (SPY)" in page and "Price Went UP" in page  # human labels + zone marker
    assert 'class="gauge"' in page and page.isascii()              # verdict gauge present
    print(f"tweet_chart _demo OK: {len(page)} bytes, {page.count('polyline')} lines drawn")


if __name__ == "__main__":
    _demo()
