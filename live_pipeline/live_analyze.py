"""Rich live LLM analysis. EXPERIMENTAL — isolated from the frozen decision engine.

A new, self-contained prompt that produces the structured literal/veiled/macro +
multi-asset analysis. It does NOT reuse alpha/classify's frozen prompt (that one is
pinned by the manifest) and it makes NO validated claim — see live_schema notes.
"""

from __future__ import annotations

import json
from typing import Any

import requests

from live_pipeline.live_schema import AssetPrediction, TrumpStatementAnalysis

_SYSTEM = (
    "You are a macro markets analyst reading a political social-media post. From the "
    "CONTENT ALONE (no live market data), produce a structured analysis: what was "
    "literally said, the veiled/implied agenda, the macro-economic impact, and per-asset "
    "directional predictions at two horizons. Return ONLY a JSON object, no prose, no fences. "
    "Be honest about uncertainty; if the post is not market-relevant, return empty predictions."
)
_INSTRUCT = (
    'Return ONLY this JSON shape:\n'
    '{"literal_translation":"...","veiled_meaning":"...","macro_economic_impact":"...",'
    '"predictions":[{"asset_name":"XLF","asset_type":"sector","horizon":"short_term",'
    '"direction":"UP","llm_conviction":0.6,"catalyst_reasoning":"..."}]}\n'
    "asset_type in {stock,sector,index}. horizon in {short_term,long_term}. "
    "direction in {UP,DOWN,NEUTRAL}. llm_conviction is your uncalibrated 0-1 self-rating "
    "(it is NOT a probability of being correct). Use real yfinance symbols. 0-8 predictions. "
    "Keep text fields under 240 chars."
)


def _parse(content: str) -> dict[str, Any]:
    s = content.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1].lstrip("json").strip() if "```" in s[3:] else s.strip("`")
    parsed: dict[str, Any] = json.loads(s[s.find("{"):s.rfind("}") + 1])
    return parsed


def analyze(text: str, *, base_url: str, api_key: str, model: str,
            timeout: int = 90) -> TrumpStatementAnalysis:
    """Call Nebius and coerce into the typed analysis. long_term calls are auto-flagged."""
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": 0,
              "messages": [{"role": "system", "content": _SYSTEM},
                           {"role": "user", "content": f"{_INSTRUCT}\n\nPOST:\n{text}"}]},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Nebius {resp.status_code}: {resp.text[:300]}")
    raw = _parse(resp.json()["choices"][0]["message"]["content"])
    return build_analysis(raw)


def build_analysis(raw: dict[str, Any]) -> TrumpStatementAnalysis:
    """Coerce a raw dict into the typed model, flagging long_term as speculative.

    Pure + LLM-free so it is unit-testable.
    """
    preds: list[AssetPrediction] = []
    for p in raw.get("predictions", []) or []:
        horizon = str(p.get("horizon", "short_term")).lower()
        conv = p.get("llm_conviction", p.get("confidence", 0.0))  # tolerate the old key name
        try:
            conv_f = max(0.0, min(1.0, float(conv)))
        except (TypeError, ValueError):
            conv_f = 0.0
        preds.append(AssetPrediction(
            asset_name=str(p.get("asset_name", "")),
            asset_type=str(p.get("asset_type", "index")),
            horizon=horizon,
            direction=str(p.get("direction", "NEUTRAL")).upper(),
            llm_conviction=conv_f,
            catalyst_reasoning=str(p.get("catalyst_reasoning", "")),
            speculative=(horizon == "long_term"),   # beta-dominated per our study
        ))
    return TrumpStatementAnalysis(
        literal_translation=str(raw.get("literal_translation", "")),
        veiled_meaning=str(raw.get("veiled_meaning", "")),
        macro_economic_impact=str(raw.get("macro_economic_impact", "")),
        predictions=preds,
    )
