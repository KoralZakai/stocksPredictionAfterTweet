"""market/ — the MARKET PLANE. Live/near-live quotes for response enrichment ONLY.

CRITICAL: nothing in this package may ever be imported by the decision plane
(alpha.classify / alpha.route). Prices resolved here are attached to the response
AFTER the decision and never feed back into it (the leakage firewall, serving/app).

Providers degrade alpaca -> yfinance -> null. The endpoint runs with zero market
keys; any provider failure yields a null market_context, never a 5xx, and never
changes the decision.
"""

from __future__ import annotations

import os

from market.null import NullProvider
from market.provider import Provider


def select_provider(name: str | None = None) -> Provider:
    """Pick a provider. Explicit `name` (or $MARKET_PROVIDER) wins; otherwise use
    Alpaca when its keys are present, else yfinance (no key), else null."""
    choice = (name or os.environ.get("MARKET_PROVIDER") or "").lower().strip()

    if choice == "null":
        return NullProvider()
    if choice == "alpaca" or (not choice and _has_alpaca_keys()):
        try:
            from market.alpaca import AlpacaProvider
            return AlpacaProvider()
        except Exception:
            pass  # fall through to the next rung
    if choice in ("", "yfinance"):
        try:
            from market.yfinance import YFinanceProvider
            return YFinanceProvider()
        except Exception:
            pass
    return NullProvider()


def _has_alpaca_keys() -> bool:
    return bool(
        (os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID"))
        and (os.environ.get("ALPACA_API_SECRET") or os.environ.get("APCA_API_SECRET_KEY"))
    )
