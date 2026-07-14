"""Nebius zero-shot classification — the DECISION-PLANE input is tweet text ONLY.

Moved verbatim from scripts/nebius_macro_validate.py (the proven engine). This is
the call whose relative alpha was validated (N=443, EOD 58.2%, p<0.001). The
prompt template is frozen; `prompt_template_hash()` fingerprints it so the batch
Job records the hash in the manifest and the Endpoint refuses to boot if its live
prompt drifts from what was validated (the leakage/skew firewall).

CORRECTNESS: no market data, price, quote, or session state may EVER enter this
prompt. The 64.4%/58.2% edge holds precisely because the LLM saw only the text.
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

import requests

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # valid Nebius id; `--model` to override
DEFAULT_BASE = "https://api.studio.nebius.ai/v1"

_SYSTEM = (
    "You are a macro markets analyst. Read a political social-media post and decide, from "
    "the CONTENT ALONE (not any market data), which scenario it fits and which liquid, "
    "US-listed instruments its content should push, and in which direction. Reason from the "
    "event itself. Watch for typos: 'RUSSIA AMD UKRAINE' means 'AND', not the chip company AMD. "
    "Return ONLY a JSON object, no prose, no markdown fences."
)
_INSTRUCT = (
    'Return ONLY a JSON object with this exact shape:\n'
    '{"scenario":"Geopolitics / Peace","intensity":7,'
    '"summary":"plain-English: what the person actually said, one sentence a layperson gets",'
    '"macro_link":"the economic logic: why this scenario moves markets",'
    '"hypothesis_short":"what should happen in the first 30m-1h and WHY (immediate reaction)",'
    '"hypothesis_long":"what should happen by 1 month and WHY (as the policy plays out)",'
    '"rationale":"one-sentence thesis",'
    '"instruments":[{"ticker":"ITA","name":"Defense & Aerospace ETF",'
    '"role":"bearish_sector","predicted_direction":"down"}]}\n'
    "intensity = integer 1-10: how forceful, certain and market-moving the post is "
    "(10 = decisive policy action stated as fact / a done deal; 1 = vague musing or opinion). "
    "predicted_direction in {up,down,neutral}. Use real yfinance-valid symbols "
    "(indices SPY/QQQ/DIA; sectors XLI/ITA/XLE/XLK/XLF/XLV; infra PAVE; fear VIXY; "
    "commodities USO/WEAT/CORN/DBC; barometers CAT/LMT/AAPL). 3-8 instruments. "
    "Keep every text field under 240 characters. "
    "If the post is not market-relevant, return an empty instruments list."
)


def prompt_template_hash() -> str:
    """sha256 of the frozen classification prompt (system + instruction + params).

    The batch Job records this in validation_manifest.json; the Endpoint pins it
    at boot. A drift here means the served model is not the one that was validated.
    """
    payload = json.dumps(
        {"system": _SYSTEM, "instruct": _INSTRUCT, "temperature": 0},
        sort_keys=True, ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def classify_tweet(text: str, *, base_url: str, api_key: str, model: str) -> dict[str, Any]:
    """Call Nebius (OpenAI-compatible) and parse the JSON scenario/instrument prediction."""
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"{_INSTRUCT}\n\nPOST:\n{text}"},
            ],
        },
        timeout=90,
    )
    if resp.status_code != 200:
        sys.exit(f"Nebius API error {resp.status_code}: {resp.text[:400]}")
    content = resp.json()["choices"][0]["message"]["content"]
    return _parse_json(content)


def _parse_json(content: str) -> dict[str, Any]:
    """Tolerant JSON extraction: strip ``` fences, grab the outermost {...}."""
    s = content.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1].lstrip("json").strip() if "```" in s[3:] else s.strip("`")
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1:
        sys.exit(f"model did not return JSON:\n{content[:400]}")
    parsed: dict[str, Any] = json.loads(s[a : b + 1])
    return parsed
