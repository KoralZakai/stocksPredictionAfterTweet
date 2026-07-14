"""No-op provider: always returns None. The explicit, hermetic default when no
market source is wanted (offline runs, reproducible tests)."""

from __future__ import annotations


class NullProvider:
    name = "null"

    def quote(self, ticker: str) -> float | None:
        return None
