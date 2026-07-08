"""Asset universe: sector-ETF -> top liquid stocks (the fine-grained signal layer).

Each sector maps to its ETF (the coarse signal, already in SETTINGS.etfs) plus a
small set of top-liquidity single names (the fine-grained layer). Stocks get the
SAME abnormal-return labeling as ETFs. Some names sit in two sectors on purpose
(NVDA/AVGO in Tech+Semis, BA/RTX in Industrials+Defense) — that's real overlap,
not a bug.

XLB/XLP are ETF-only in v1 (no stock list requested).
"""

from __future__ import annotations

# ETF (sector proxy) -> top-5 liquid single names.
SECTOR_STOCKS: dict[str, tuple[str, ...]] = {
    "XLE": ("XOM", "CVX", "COP", "SLB", "EOG"),       # Energy
    "XLF": ("JPM", "BAC", "WFC", "GS", "MS"),         # Financials
    "XLK": ("AAPL", "MSFT", "NVDA", "AMD", "AVGO"),   # Tech
    "XLI": ("BA", "CAT", "GE", "RTX", "UNP"),         # Industrials
    "XLV": ("UNH", "JNJ", "PFE", "MRK", "ABBV"),      # Healthcare
    "XLY": ("AMZN", "TSLA", "HD", "MCD", "NKE"),      # Consumer
    "SMH": ("NVDA", "AMD", "INTC", "TSM", "AVGO"),    # Semiconductors
    "ITA": ("LMT", "NOC", "RTX", "GD", "BA"),         # Defense
    "XLB": (),                                        # Materials — ETF only
    "XLP": (),                                        # Staples   — ETF only
}

# Assets that carry Trump personal/business exposure (§4 feature, NOT a label).
# In the 2017-2021 train era this is effectively empty (DJT/Trump Media is 2024+),
# so the flag is near-constant there — kept as a hook, honest about being ~null.
TRUMP_EXPOSED: frozenset[str] = frozenset({"DJT"})


def all_stocks() -> list[str]:
    """Deduplicated flat list of every single name across sectors."""
    seen: dict[str, None] = {}
    for names in SECTOR_STOCKS.values():
        for n in names:
            seen.setdefault(n, None)
    return list(seen)


def assets_for_sector(etf: str) -> list[str]:
    """The ETF itself + its single names (the assets scored for that sector)."""
    return [etf, *SECTOR_STOCKS.get(etf, ())]
