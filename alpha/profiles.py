"""Two operational profiles, selected by env `SIGNAL_PROFILE`. Default = stable.

  stable  (Mode A, DEFAULT, SHIPPED) - the frozen indices/sector prompt + whitelist,
          pinned by reports/validation_manifest.json (EOD 61.8%). Its prompt string
          and hash are IMPORTED, never redefined here, so they cannot drift.
  macro   (Mode B, EXPERIMENTAL)     - expanded macro universe (+TLT/UUP/FXI/GLD),
          a stricter whitelist-only prompt, and a SEPARATE manifest
          reports/validation_manifest_macro_v1.json.

WHY a registry instead of editing classify.py: the manifest pins the exact prompt
that produced its numbers, and serving/app.py refuses to boot on drift. Mode B has
a different prompt, so it MUST carry its own manifest. Bundling (prompt, whitelist,
manifest) per profile is what keeps that pairing honest — you cannot serve Mode B's
prompt against Mode A's numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from alpha import classify as _stable
from alpha.route import WHITELIST as STABLE_WHITELIST

# ---------------------------------------------------------------- Mode B: universe
# XLE was already whitelisted; TLT/UUP/FXI/GLD are the genuinely new macro assets.
MACRO_EXTRA = frozenset({"TLT", "UUP", "FXI", "GLD", "XLE"})
MACRO_WHITELIST: frozenset[str] = STABLE_WHITELIST | MACRO_EXTRA

# The exact menu the Mode-B model is allowed to pick from (no off-menu tickers).
_MACRO_MENU = ("SPY, QQQ, DIA, XLK, XLE, XLF, XLI, XLV, XLY, XLB, ITA, SMH, "
               "VIXY, USO, WEAT, CORN, DBC, TLT, UUP, FXI, GLD")

# A fixed protocol fact — NOT model-generated (the LLM would only drift on it).
# Attached deterministically to every Mode-B signal.
MEASUREMENT_WINDOW = ("Measured from the first market open strictly after t0 to that "
                      "session's close, scored relative to SPY (beat-SPY). This is a "
                      "measured research signal, not a trade instruction: a beat-SPY hit "
                      "can still be a money-losing trade if the market falls.")

_SYSTEM_MACRO = (
    "You are a macro markets analyst. Read a political social-media post and decide, from "
    "the CONTENT ALONE (not any market data), which liquid US-listed instruments its content "
    "should push relative to the S&P 500, and in which direction. Reason from the event itself. "
    "Watch for typos: 'RUSSIA AMD UKRAINE' means 'AND', not the chip company AMD.\n"
    f"You may ONLY use tickers from this exact menu: {_MACRO_MENU}. Never invent a ticker "
    "outside that menu.\n"
    "ABSTAIN RULES — these override everything else. Return an EMPTY instruments list when:\n"
    "1. STRICT MACRO RELEVANCE: the post is a purely political endorsement, a get-out-the-vote "
    "or campaign message, a media/personal attack, a greeting, or otherwise has no direct, "
    "explicit economic or financial substance. DO NOT REACH.\n"
    "2. EXPLICIT ASSET CONNECTION: assign a direction only if the post itself contains a real "
    "economic concept (tariffs, taxes, inflation, the Federal Reserve or rates, oil/gas "
    "production, debt/deficit, currency or trade policy, sanctions, a named sector policy).\n"
    "3. NO HALOS: if the post merely praises or endorses a politician, treat it as a NON-EVENT "
    "and abstain — unless that post itself discusses a specific economic bill or sector policy. "
    "A politician's name is not an economic signal.\n"
    "Abstaining is a CORRECT, valued answer. An empty list is better than a reached-for guess. "
    "Return ONLY a JSON object, no prose, no markdown fences."
)
_INSTRUCT_MACRO = (
    'Return ONLY a JSON object with this exact shape:\n'
    '{"category":"Trade War","intensity":7,'
    '"summary":"plain-English: what the person actually said, one sentence a layperson gets",'
    '"trump_interpretation":"his economic intent/rhetoric in plain English",'
    '"macro_link":"the economic logic: why this scenario moves markets",'
    '"direction_rationale":"why these assets should OVER- or UNDER-perform SPY",'
    '"rationale":"one-sentence thesis",'
    '"instruments":[{"ticker":"FXI","name":"China Large-Cap ETF",'
    '"role":"bearish_country","predicted_direction":"down"}]}\n'
    'category is one of: "Trade War", "Fiscal Policy", "Energy", "Fed", "Geopolitics", '
    '"Non-Macro / Politician Endorsement", "Non-Macro / Other". '
    'If category starts with "Non-Macro", instruments MUST be an empty list.\n'
    "intensity = integer 1-10: how forceful, certain and market-moving the post is. "
    "predicted_direction in {up,down,neutral} — the move RELATIVE to SPY, not absolute. "
    f"ONLY these tickers are permitted: {_MACRO_MENU}. Guidance: rates/debt/Fed -> TLT; "
    "dollar/currency/trade-war -> UUP; China/tariffs -> FXI; oil/gas/drilling -> XLE/USO; "
    "geopolitical or inflation hedge -> GLD. 0-8 instruments (0 = abstain). "
    "Do NOT output position sizes, weights, or buy/sell timing — this is a research "
    "classification, not a trade plan. "
    "Keep every text field under 240 characters."
)

# ---------------------------------------------------------------- Mode B: pre-filter
# The stable geo filter ALREADY matches china/tariff/sanction/war/peace/opec/deal.
# These are the genuinely missing macro tokens (rates, energy, currency, gold, debt).
MACRO_EXTRA_RX = re.compile(
    r"\b(interest rate|interest rates|interest|rate cut|rate hike|rates|borrowed|debt|deficit|"
    r"oil|drilling|drill|gas|energy|gold|dollar|currency|devalue|fed|federal reserve|"
    r"powell|inflation|treasury|bonds?)\b", re.I)


def _hash_prompt(system: str, instruct: str) -> str:
    payload = json.dumps({"system": system, "instruct": instruct, "temperature": 0},
                         sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def classify_macro(text: str, *, base_url: str, api_key: str, model: str) -> dict[str, Any]:
    """Mode-B classification. Same transport as the stable call, different frozen prompt."""
    import requests
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": 0,
              "messages": [{"role": "system", "content": _SYSTEM_MACRO},
                           {"role": "user", "content": f"{_INSTRUCT_MACRO}\n\nPOST:\n{text}"}]},
        timeout=90,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Nebius {resp.status_code}: {resp.text[:300]}")
    return _stable._parse_json(resp.json()["choices"][0]["message"]["content"])


@dataclass(frozen=True)
class Profile:
    name: str
    manifest_path: str
    whitelist: frozenset[str]
    classify: Callable[..., dict[str, Any]]
    prompt_hash: Callable[[], str]
    experimental: bool


PROFILES: dict[str, Profile] = {
    # Mode A — frozen. Prompt + hash imported from alpha.classify: cannot drift.
    "stable": Profile(
        name="stable", manifest_path="reports/validation_manifest.json",
        whitelist=STABLE_WHITELIST, classify=_stable.classify_tweet,
        prompt_hash=_stable.prompt_template_hash, experimental=False),
    # Mode B — experimental, separate manifest.
    "macro": Profile(
        name="macro", manifest_path="reports/validation_manifest_macro_v1.json",
        whitelist=MACRO_WHITELIST, classify=classify_macro,
        prompt_hash=lambda: _hash_prompt(_SYSTEM_MACRO, _INSTRUCT_MACRO), experimental=True),
}


def active_profile() -> Profile:
    """Resolve SIGNAL_PROFILE (default 'stable'). Unknown name -> hard fail, never a
    silent fallback to the wrong prompt/manifest pairing."""
    name = os.environ.get("SIGNAL_PROFILE", "stable").strip().lower() or "stable"
    if name not in PROFILES:
        raise RuntimeError(f"Unknown SIGNAL_PROFILE={name!r}. Choose one of {sorted(PROFILES)}.")
    return PROFILES[name]
