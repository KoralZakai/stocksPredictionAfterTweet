"""Non-expert visual card for a Nebius macro-validation run (pure: result -> HTML).

One card summarises: what scenario the LLM detected, what it predicted (per
instrument, with an up/down arrow + the macro reasoning), what ACTUALLY happened
in the market (returns per horizon), and whether each call was right (green) or
wrong (red). A big headline states the overall hit-rate in one scannable line.

Self-contained ASCII HTML/SVG, same design tokens as reportgen/tweet_chart.py so
the two pages read as one product. Consumes the dict that
scripts/nebius_macro_validate.py assembles — see `render_page` for the shape.
"""

from __future__ import annotations

import html

_ARROW = {"up": "&#9650;", "down": "&#9660;", "neutral": "&#8212;"}  # ▲ ▼ —


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:+.1f}%"


def _rate(hits: list[int]) -> float | None:
    hit, tot = hits
    return None if tot == 0 else hit / tot


def _headline(result: dict) -> tuple[str, str]:
    """(text, sign-class) — overall verdict at the longest horizon with data."""
    horizons = result["horizons"]
    rep = next((h for h in reversed(horizons) if _rate(result["hits"][h]) is not None), None)
    if rep is None:
        return "No directional calls to score", "flat"
    hit, tot = result["hits"][rep]
    pct = round(100 * hit / tot)
    scen = result.get("scenario", "Scenario")
    if pct >= 60:
        return f"{scen}: CONFIRMED — {pct}% of the LLM's calls were right at {rep} ({hit}/{tot})", "up"
    if pct <= 40:
        return f"{scen}: NOT CONFIRMED — only {pct}% right at {rep} ({hit}/{tot})", "down"
    return f"{scen}: MIXED — {pct}% right at {rep} ({hit}/{tot})", "flat"


def _callouts(result: dict) -> str:
    """Auto-highlight the strongest confirmation and the biggest miss."""
    horizons = result["horizons"]
    last = horizons[-1]

    def move(ins: dict) -> float:
        return abs((ins.get("returns") or {}).get(last, 0.0) or 0.0)

    scored = [i for i in result["instruments"] if isinstance(i.get("hit", {}).get(last), bool)]
    if not scored:
        return ""
    best = max((i for i in scored if i["hit"][last]), key=move, default=None)
    miss = next((i for i in scored if not i["hit"][last]), None)
    chips = []
    if best is not None:
        r = best["returns"][last]
        chips.append(f'<span class="call up">&#10003; Biggest hit: {html.escape(best["name"])} '
                     f'({best["ticker"]}) {_pct(r)} at {last}, exactly as predicted '
                     f'({best["predicted"]})</span>')
    if miss is not None:
        r = miss["returns"][last]
        chips.append(f'<span class="call down">&#10007; Anomaly: {html.escape(miss["name"])} '
                     f'({miss["ticker"]}) predicted {miss["predicted"]}, actually {_pct(r)} at {last}</span>')
    return f'<div class="calls">{"".join(chips)}</div>' if chips else ""


def _rows(result: dict) -> str:
    horizons = result["horizons"]
    out = []
    for ins in result["instruments"]:
        pred = str(ins.get("predicted", "neutral"))
        pcls = "up" if pred == "up" else "down" if pred == "down" else "flat"
        cells = [f'<td class="inst"><b>{html.escape(str(ins.get("name", "")))}</b>'
                 f'<span class="tkr">{ins.get("ticker", "")}</span>'
                 f'<span class="role">{html.escape(str(ins.get("role", "")))}</span></td>',
                 f'<td class="pred {pcls}">{_ARROW.get(pred, "&#8212;")} {pred}</td>']
        for h in horizons:
            r = (ins.get("returns") or {}).get(h)
            hit = ins.get("hit", {}).get(h)
            cls = "up" if hit is True else "down" if hit is False else "flat"
            cells.append(f'<td class="ret {cls}">{_pct(r)}</td>')
        out.append(f"<tr>{''.join(cells)}</tr>")
    return "".join(out)


def _gauge(result: dict) -> str:
    cells = []
    for h in result["horizons"]:
        r = _rate(result["hits"][h])
        if r is None:
            cells.append(f'<div class="gcell flat"><span class="gh">{h}</span>'
                         f'<span class="gg">n/a</span></div>')
            continue
        cls = "up" if r >= 0.6 else "down" if r <= 0.4 else "flat"
        hit, tot = result["hits"][h]
        cells.append(f'<div class="gcell {cls}"><span class="gh">{h}</span>'
                     f'<span class="gg">{round(100*r)}%</span>'
                     f'<span class="gs">{hit}/{tot}</span></div>')
    return f'<div class="gauge">{"".join(cells)}</div>'


def _card(result: dict) -> str:
    """One <article> card for a single validated tweet.

    result = {text, date, model, scenario, rationale, horizons:[...],
    instruments:[{ticker,name,role,predicted, returns{h:ret}, hit{h:bool}}],
    hits:{h:[nhit,ntot]}}."""
    head, hsign = _headline(result)
    hz = "".join(f"<th>{h}</th>" for h in result["horizons"])
    tag = str(result.get("tag", ""))
    tcls = {"Flash Shock": "flash", "Structural Drift": "struct"}.get(tag, "mix")
    tag_html = f'<span class="tag {tcls}">{html.escape(tag)}</span>' if tag else ""
    body = (
        f'<div class="hl {hsign}">{html.escape(head)}{tag_html}</div>'
        f'<div class="meta"><b>Scenario:</b> {html.escape(str(result.get("scenario","")))}'
        f' &nbsp;&#183;&nbsp; <b>Model:</b> {html.escape(str(result.get("model","")))}'
        f' &nbsp;&#183;&nbsp; <b>Tweet date:</b> {result.get("date","")}</div>'
        f'<blockquote>{html.escape(str(result.get("text","")))}</blockquote>'
        f'<div class="why"><b>Why the LLM expects this:</b> '
        f'{html.escape(str(result.get("rationale","")))}</div>'
        f'{_callouts(result)}'
        f'<table class="grid"><thead><tr><th>Instrument</th><th>LLM predicted</th>'
        f'{hz}</tr></thead><tbody>{_rows(result)}</tbody></table>'
        f'<p class="unit">Each % is the real move from the first market open after the tweet. '
        f'<span class="k up">Green</span> = the market moved the way the LLM predicted; '
        f'<span class="k down">red</span> = it moved the opposite way. Hit-rate per horizon:</p>'
        f'{_gauge(result)}'
    )
    return f'<article class="card">{body}</article>'


def _doc(title: str, inner: str) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{_CSS}</style></head>'
            f'<body><div class="wrap">{inner}</div></body></html>'
            ).encode("ascii", "xmlcharrefreplace").decode("ascii")


def render_page(result: dict) -> str:
    """Single-tweet page (option B)."""
    foot = ('<footer>One tweet is descriptive, not proof. A real edge needs this hit-rate to '
            'beat a coin-flip across many tweets after correction.</footer>')
    return _doc("Nebius macro validation", _card(result) + foot)


def render_gallery(results: list[dict], title: str, subtitle: str = "") -> str:
    """Multi-tweet page (option A: the top-N most significant historical examples)."""
    head = f'<h1 class="ph">{html.escape(title)}</h1>'
    sub = f'<p class="psub">{html.escape(subtitle)}</p>' if subtitle else ""
    return _doc(title, head + sub + "".join(_card(r) for r in results))


_NARR_H = ("30m", "1h", "1mo")   # short-term vs long-term, the two clocks


def _mini_grid(result: dict) -> str:
    """Compact instrument x {30m,1h,1mo} grid, each cell green/red by hit."""
    hz = "".join(f"<th>{h}</th>" for h in _NARR_H)
    rows = []
    for ins in result["instruments"]:
        pred = str(ins.get("predicted", "neutral"))
        pcls = "up" if pred == "up" else "down" if pred == "down" else "flat"
        cells = [f'<td class="inst"><b>{html.escape(str(ins.get("name","")))}</b>'
                 f'<span class="tkr">{ins.get("ticker","")}</span></td>',
                 f'<td class="pred {pcls}">{_ARROW.get(pred,"&#8212;")}</td>']
        for h in _NARR_H:
            r = (ins.get("returns") or {}).get(h)
            hit = ins.get("hit", {}).get(h)
            cls = "up" if hit is True else "down" if hit is False else "flat"
            cells.append(f'<td class="ret {cls}">{_pct(r)}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (f'<table class="grid mini"><thead><tr><th>Instrument</th><th>call</th>'
            f'{hz}</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def _narr_card(ex: dict) -> str:
    """ex = {result, role, conclusion}. role in {Flash Shock, Structural Drift, Failed Anomaly}."""
    r = ex["result"]
    role = ex["role"]
    rcls = {"Flash Shock": "flash", "Structural Drift": "struct"}.get(role, "down")
    def sec(label: str, val: str) -> str:
        return f'<div class="nsec"><b>{label}</b> {html.escape(str(val or "&#8212;"))}</div>' \
            if val else ""
    return (
        f'<article class="card">'
        f'<div class="hl {rcls}">{html.escape(role)}<span class="tag {rcls}">{r.get("date","")} '
        f'&#183; {html.escape(str(r.get("scenario","")))} &#183; intensity {r.get("intensity","?")}'
        f'</span></div>'
        f'<blockquote>{html.escape(str(r.get("text",""))[:400])}</blockquote>'
        f'{sec("What he actually said:", r.get("summary",""))}'
        f'{sec("Why it moves markets:", r.get("macro_link",""))}'
        f'{sec("Short-term call (30m-1h):", r.get("hypothesis_short",""))}'
        f'{sec("Long-term call (1mo):", r.get("hypothesis_long",""))}'
        f'<div class="nsec"><b>Reality check &#8212; short vs long (green=right, red=wrong):</b></div>'
        f'{_mini_grid(r)}'
        f'<div class="concl {rcls}"><b>Data-science conclusion:</b> {html.escape(ex["conclusion"])}</div>'
        f'</article>')


def render_narrative(examples: list[dict], title: str, subtitle: str = "") -> str:
    """Validation-set inspection page: narrative cards, one per example."""
    head = f'<h1 class="ph">{html.escape(title)}</h1>'
    sub = f'<p class="psub">{html.escape(subtitle)}</p>' if subtitle else ""
    return _doc(title, head + sub + "".join(_narr_card(e) for e in examples))


_CSS = """
:root{--bg:#f4f5f2;--surface:#fff;--surface-2:#eef0ec;--ink:#191c1f;--ink-2:#5a6067;
 --ink-3:#878d94;--line:#dee1db;--accent:#a8792f;
 --up:#17805f;--up-bg:#e4f1ea;--down:#c0432f;--down-bg:#f7e5e0;--flat:#9aa0a6;
 --font:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);font-size:14px;line-height:1.5}
.wrap{max-width:860px;margin:0 auto;padding:28px 18px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:0 0 18px}
.ph{font-size:22px;margin:0 0 4px}.psub{color:var(--ink-2);margin:0 0 20px}
.hl{font-size:18px;font-weight:800;padding:12px 14px;border-radius:9px;margin:0 0 12px;
 border-left:6px solid var(--flat);background:var(--surface-2)}
.hl.up{border-left-color:var(--up);background:var(--up-bg);color:var(--up)}
.hl.down{border-left-color:var(--down);background:var(--down-bg);color:var(--down)}
.tag{float:right;font-size:11px;font-weight:700;padding:3px 9px;border-radius:20px;
 background:var(--surface);border:1px solid var(--line);color:var(--ink-2)}
.tag.flash{color:#b06a00;border-color:#e6c68a}
.tag.struct{color:var(--up);border-color:#9cc9b6}
.meta{font-size:12px;color:var(--ink-2);margin:0 0 10px}.meta b{color:var(--ink)}
blockquote{margin:10px 0;padding:9px 13px;border-left:3px solid var(--accent);
 background:var(--surface-2);border-radius:0 6px 6px 0;font-style:italic}
.why{font-size:13px;color:var(--ink-2);margin:8px 0 12px}.why b{color:var(--ink)}
.calls{display:flex;flex-direction:column;gap:6px;margin:0 0 14px}
.call{font-size:13px;font-weight:600;padding:7px 11px;border-radius:7px}
.call.up{background:var(--up-bg);color:var(--up)}
.call.down{background:var(--down-bg);color:var(--down)}
.grid{width:100%;border-collapse:collapse;margin:4px 0;font-size:13px}
.grid th{text-align:right;font-size:11px;color:var(--ink-3);padding:6px 8px;border-bottom:1px solid var(--line)}
.grid th:first-child,.grid th:nth-child(2){text-align:left}
.grid td{padding:8px;border-bottom:1px solid var(--line);text-align:right;
 font-variant-numeric:tabular-nums}
.inst{text-align:left!important}.inst b{display:inline}
.inst .tkr{color:var(--ink-3);margin-left:6px;font-size:12px}
.inst .role{display:block;color:var(--ink-3);font-size:11px}
.pred{text-align:left!important;font-weight:700}
.pred.up{color:var(--up)}.pred.down{color:var(--down)}.pred.flat{color:var(--ink-3)}
.ret{font-weight:700;color:var(--ink-3)}
.ret.up{color:var(--up);background:var(--up-bg)}
.ret.down{color:var(--down);background:var(--down-bg)}
.unit{color:var(--ink-3);font-size:12px;margin:12px 0 6px}
.k{font-weight:700}.k.up{color:var(--up)}.k.down{color:var(--down)}
.gauge{display:flex;gap:8px;margin:4px 0}
.gcell{flex:1;text-align:center;padding:10px 4px;border-radius:9px;background:var(--surface-2)}
.gcell .gh{display:block;font-size:11px;color:var(--ink-2);font-weight:600}
.gcell .gg{font-size:20px;font-weight:800;color:var(--ink-3)}
.gcell .gs{display:block;font-size:11px;color:var(--ink-3)}
.gcell.up{background:var(--up-bg)}.gcell.up .gg{color:var(--up)}
.gcell.down{background:var(--down-bg)}.gcell.down .gg{color:var(--down)}
footer{color:var(--ink-3);font-size:11px;margin-top:16px}
.nsec{font-size:13px;margin:7px 0;color:var(--ink)}
.nsec b{color:var(--ink-2);font-weight:600;display:block;font-size:11px;text-transform:uppercase;
 letter-spacing:.03em;margin-bottom:1px}
.grid.mini th:nth-child(2){text-align:center}
.grid.mini .pred{text-align:center!important;font-size:15px}
.concl{margin-top:12px;padding:11px 13px;border-radius:8px;font-size:13px;
 background:var(--surface-2);border-left:5px solid var(--flat)}
.concl.flash{border-left-color:#b06a00;background:#fbf3e4}
.concl.struct{border-left-color:var(--up);background:var(--up-bg)}
.concl.down{border-left-color:var(--down);background:var(--down-bg)}
.concl b{color:var(--ink)}
"""


def _demo() -> None:
    """Self-check on the exact Russia/Ukraine run shape (real numbers from the CLI)."""
    result = {
        "text": "HOPEFULLY RUSSIA AND UKRAINE WILL MAKE A DEAL THIS WEEK...",
        "date": "2025-04-20", "model": "meta-llama/Llama-3.3-70B-Instruct",
        "scenario": "Geopolitics / Peace",
        "rationale": "Optimism about a Russia-Ukraine deal eases geopolitical risk.",
        "horizons": ["EOD", "3d", "1w", "1mo"],
        "instruments": [
            {"ticker": "VIXY", "name": "VIX Fear Gauge", "role": "fear_gauge", "predicted": "down",
             "returns": {"EOD": 0.0457, "3d": -0.0529, "1w": -0.1263, "1mo": -0.2762},
             "hit": {"EOD": False, "3d": True, "1w": True, "1mo": True}},
            {"ticker": "LMT", "name": "Lockheed Martin", "role": "bearish_sector", "predicted": "down",
             "returns": {"EOD": -0.002, "3d": 0.0093, "1w": 0.04, "1mo": 0.03},
             "hit": {"EOD": True, "3d": False, "1w": False, "1mo": False}},
        ],
        "hits": {"EOD": [1, 5], "3d": [4, 5], "1w": [4, 5], "1mo": [4, 5]},
    }
    page = render_page(result)
    assert "CONFIRMED" in page and "Biggest hit" in page and "Anomaly" in page
    assert page.isascii() and "-27.6%" in page
    print(f"macro_card _demo OK: {len(page)} bytes")


if __name__ == "__main__":
    _demo()
