"""Extract a TweetSignal from tweet text (product fork).

Two backends behind one interface:
  * AnthropicExtractor — Claude via the official SDK, structured output validated
    against TweetSignal. Runs OFFLINE in a batch Job; never on the /predict path.
  * HeuristicExtractor — deterministic keyword fallback so tests + CI are green
    with no API key and no network (also the honest default when unconfigured).

The extractor sees ONLY the tweet text — never any price/market data — so it
cannot break the point-in-time invariant (§3.1). It emits no ticker/sector (§6).
"""

from __future__ import annotations

import os
import re
from typing import Protocol

from llm.schema import TweetSignal

# Default to the most capable model (claude-api skill rule). For a few-thousand
# -tweet one-time batch, claude-haiku-4-5 is the cost option — set LLM_MODEL.
DEFAULT_MODEL = "claude-opus-4-8"

_SYSTEM = (
    "You label a political figure's social-media post for a markets-research "
    "pipeline. You see ONLY the post text — never any price or market data. "
    "Describe what the post does and the directional pressure its CONTENT implies "
    "on the affected sector, reasoning from the event itself, not from any "
    "market reaction (you have none). Do NOT name tickers or sectors. Be "
    "conservative: if the post is not market-relevant, use event_type 'none' and "
    "neutral/low values."
)


class Extractor(Protocol):
    def extract(self, text: str) -> TweetSignal: ...


class AnthropicExtractor:
    """Claude-backed structured extraction. Lazy SDK import keeps anthropic off
    the cold path (same pattern as serving/endpoint.py's xgboost import)."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        import anthropic  # local import: optional dep, only needed for live extraction

        self.model = model
        self._client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / ant profile

    def extract(self, text: str) -> TweetSignal:
        resp = self._client.messages.parse(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": text}],
            output_format=TweetSignal,
        )
        signal: TweetSignal | None = resp.parsed_output
        if signal is None:  # refusal / parse failure -> honest neutral, never a guess
            return HeuristicExtractor().extract(text)
        return signal


_KW = {
    "military_threat": r"\b(attack|strike|invade|troops|missiles?|war\b|bomb)\b",
    "sanction": r"\b(sanctions?|export ban|blockade|embargo)\b",
    "tariff_trade": r"\b(tariffs?|trade war|trade deal|import tax)\b",
    "monetary_policy": r"\b(federal reserve|the fed|interest rates?|jerome powell|dollar)\b",
    "energy_policy": r"\b(oil|opec|drill|pipelines?|crude|natural gas)\b",
    "drug_pricing": r"\b(drug prices?|pharma|big pharma|medicare|prescriptions?)\b",
    "regulation": r"\b(antitrust|dodd.frank|regulat\w+|section 230)\b",
    "macro_claim": r"\b(stock market|economy|jobs|gdp|nasdaq|dow)\b",
}
_NEG = re.compile(r"\b(attack|strike|war|sanctions?|tariffs?|disaster|threat|ban)\b", re.I)
_POS = re.compile(r"\b(great|deal|boom|record|strong|win|historic)\b", re.I)
_COUNTRY = re.compile(r"\b(iran|china|russia|mexico|canada|ukraine|israel|north korea|saudi)\b", re.I)
_COMPANY = re.compile(r"\b(apple|boeing|amazon|tesla|exxon|pfizer|nvidia|intel|goldman)\b", re.I)


class HeuristicExtractor:
    """Deterministic, no-network fallback. Coarse on purpose — it exists so the
    pipeline runs and tests pass without an API key, not to rival the LLM."""

    def extract(self, text: str) -> TweetSignal:
        event = next(
            (e for e, pat in _KW.items() if re.search(pat, text, re.I)), "none"
        )
        intent = "bearish" if _NEG.search(text) else "bullish" if _POS.search(text) else "neutral"
        # energy under threat is bullish for oil; keep the simple rule explicit.
        if event == "energy_policy" and _NEG.search(text):
            intent = "bullish"
        return TweetSignal(
            event_type=event,  # type: ignore[arg-type]
            direction_of_intent=intent,  # type: ignore[arg-type]
            urgency="high" if event == "military_threat" else "low",
            magnitude="medium" if event != "none" else "low",
            certainty="medium" if event != "none" else "low",
            names_country=bool(_COUNTRY.search(text)),
            names_company=bool(_COMPANY.search(text)),
        )


def default_extractor(model: str | None = None) -> Extractor:
    """AnthropicExtractor when an API key is available, else the heuristic. The
    batch Job picks this so a missing key degrades to a deterministic run, not a
    crash — and the /predict path never constructs a live extractor at all."""
    model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        try:
            return AnthropicExtractor(model)
        except Exception:  # SDK missing / auth misconfigured -> stay deterministic
            return HeuristicExtractor()
    return HeuristicExtractor()
