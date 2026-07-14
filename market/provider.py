"""Provider contract for the market plane: ticker -> last price (or None)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Provider(Protocol):
    """A price source. `quote` returns the last price, or None if unavailable.

    Implementations MUST NOT raise for a missing/bad ticker — return None. The
    endpoint additionally wraps every call in a timeout + try/except, so a slow or
    throwing provider degrades to a null market_context, never a 5xx.
    """

    name: str

    def quote(self, ticker: str) -> float | None: ...
