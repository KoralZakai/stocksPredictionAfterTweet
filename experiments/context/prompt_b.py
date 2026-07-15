"""System-B dual-horizon prompt with point-in-time macro cross-referencing.
EXPERIMENTAL — the shipped System-A prompt in alpha/classify.py is untouched.

The macro_context injected here is ONLY the leak-safe public-calendar string from
macro_calendar.context_asof(t0) (events strictly before t0). The model is told to
treat 1h as a sentiment shock and EOD as weighing the structural macro trend.
"""

from __future__ import annotations

import json
from typing import Any

import requests

_SYSTEM_B = (
    "You are a macro markets analyst. From the CONTENT of a political social-media "
    "post plus a list of RECENT SCHEDULED MACRO EVENTS (a public calendar, all dated "
    "strictly BEFORE the post), decide which liquid US-listed instruments the post "
    "should move, and in which direction, at TWO horizons. Reason only from the post "
    "and the given events; invent no other market data. The macro events are context "
    "known before the post — never assume any outcome that came after it. "
    "Return ONLY a JSON object, no prose, no markdown fences."
)
_INSTRUCT_B = (
    'Return ONLY a JSON object with this exact shape:\n'
    '{"scenario":"Trade War","intensity":7,'
    '"summary":"one plain-English sentence a layperson gets",'
    '"macro_link":"the economic logic tying the post + macro context to markets",'
    '"rationale":"one-sentence thesis",'
    '"instruments":[{"ticker":"XLK","name":"Tech Sector ETF",'
    '"direction_1h":"down","direction_eod":"down"}]}\n'
    "direction_1h = the SUDDEN SENTIMENT SHOCK move in the first hour. "
    "direction_eod = the end-of-day move, which must WEIGH THE MACRO CONTEXT's "
    "structural trend, not just the initial reaction. Each direction in {up,down,neutral}. "
    "Use real yfinance symbols (SPY/QQQ/DIA; XLK/XLE/XLF/XLI/XLV/XLY/ITA/SMH; VIXY; "
    "USO/WEAT/CORN; CAT/LMT/AAPL). 3-8 instruments. Keep text fields under 240 chars. "
    "If not market-relevant, return an empty instruments list."
)


def classify_b(text: str, macro_context: str, *, base_url: str, api_key: str, model: str,
               timeout: int = 90) -> dict[str, Any]:
    """System-B dual-horizon classification. macro_context is the leak-safe calendar string."""
    ctx = macro_context.strip() or "(no scheduled macro events in the lookback window)"
    user = f"{_INSTRUCT_B}\n\nMACRO CONTEXT (public calendar, before the post):\n{ctx}\n\nPOST:\n{text}"
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": 0,
              "messages": [{"role": "system", "content": _SYSTEM_B},
                           {"role": "user", "content": user}]},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Nebius {resp.status_code}: {resp.text[:300]}")
    content = resp.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("```", 2)[1].lstrip("json").strip() if "```" in content[3:] else content.strip("`")
    a, b = content.find("{"), content.rfind("}")
    parsed: dict[str, Any] = json.loads(content[a:b + 1])
    return parsed
