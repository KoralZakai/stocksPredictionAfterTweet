"""Stock -> benchmark membership: the set of benchmarks each stock is judged against.

The core methodology change (v-multibench): a stock is NOT compared to one fixed
benchmark. For each stock we resolve EVERY relevant benchmark it belongs to:

  * indices  -- the broad/index ETFs whose basket contains the stock
                (SPY = S&P 500, QQQ = Nasdaq-100, DIA = Dow 30). Membership is
                real: TSM (foreign ADR) and GFS are in none of the three; DJT
                (Trump Media) is in none, so both fall back to SPY as the honest
                broad-market proxy.
  * sectors  -- the sector ETF(s) the stock sits in, inverted from
                config.universe.SECTOR_STOCKS. A name in two sectors (NVDA in
                XLK+SMH) carries both; they get averaged downstream.
  * peers    -- the OTHER single names sharing any of the stock's sectors. Used
                for the peer-relative check (did the stock beat its siblings, or
                just ride a sector-wide move). Never includes the stock itself.

`benchmarks_for()` is the single resolver; labeling/benchmarks.py and the
per-tweet report both call it so the benchmark set a row is judged against and
the one the chart draws can never disagree. Membership here is a pre-registered
config choice (like config/settings.py) -- editing it is a git-tracked decision,
not a runtime tweak.

Index membership is hand-curated for the ~45-name universe and approximate at
the edges (index reconstitutions move a few names/year); it is honest-to-2025
and good enough for a benchmark set, not a compliance-grade constituents feed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.universe import SECTOR_STOCKS

# --- index ETF proxies (must have bars in data/real/bars.csv) ------------------
INDEX_ETFS: dict[str, str] = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq-100",
    "DIA": "Dow Jones Industrial Average (30)",
}

# Which universe tickers each index actually contains (2025 membership).
# SPY: all our large-caps EXCEPT foreign ADR TSM, non-S&P GFS, and DJT.
# QQQ: Nasdaq-listed only. DIA: the price-weighted Dow 30 members we cover.
_NASDAQ_100: frozenset[str] = frozenset(
    {"AAPL", "MSFT", "NVDA", "AMD", "AVGO", "AMZN", "TSLA", "INTC", "MU", "TXN"}
)
_DOW_30: frozenset[str] = frozenset(
    {"AAPL", "MSFT", "NVDA", "AMZN", "JPM", "GS", "CAT", "BA", "UNH",
     "JNJ", "MRK", "HD", "NKE", "CVX", "MCD"}
)
_NOT_SP500: frozenset[str] = frozenset({"TSM", "GFS", "DJT"})

_ALL_STOCKS: frozenset[str] = frozenset(
    {n for names in SECTOR_STOCKS.values() for n in names} | {"MU", "GFS", "TXN"}
)
INDEX_MEMBERS: dict[str, frozenset[str]] = {
    "SPY": _ALL_STOCKS - _NOT_SP500,
    "QQQ": _NASDAQ_100,
    "DIA": _DOW_30,
}

# Semis sector ETF: SECTOR_STOCKS registers SMH (with its constituents), so SMH
# is the sector benchmark AND the peer source. (SOXX, used by the older single-
# benchmark script, is a near-identical alternative — we standardise on SMH so
# peers and benchmark come from one table.)
_STOCK_SECTORS: dict[str, list[str]] = {}
for _etf, _names in SECTOR_STOCKS.items():
    for _n in _names:
        _STOCK_SECTORS.setdefault(_n, []).append(_etf)

# Plain-English names for the report header (ticker -> what it is).
TICKER_NAME: dict[str, str] = {
    # indices / benchmark ETFs
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq-100 ETF", "DIA": "Dow Jones 30 ETF",
    "XLE": "Energy sector ETF", "XLF": "Financials sector ETF",
    "XLK": "Technology sector ETF", "XLI": "Industrials sector ETF",
    "XLV": "Health Care sector ETF", "XLY": "Consumer Discretionary sector ETF",
    "XLP": "Consumer Staples sector ETF", "XLB": "Materials sector ETF",
    "SMH": "Semiconductor sector ETF", "SOXX": "Semiconductor sector ETF (iShares)",
    "ITA": "Aerospace & Defense ETF",
    # single names
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMD": "AMD",
    "AVGO": "Broadcom", "INTC": "Intel", "TSM": "TSMC (Taiwan Semiconductor)",
    "MU": "Micron", "TXN": "Texas Instruments", "GFS": "GlobalFoundries",
    "XOM": "Exxon Mobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "SLB": "SLB (Schlumberger)", "EOG": "EOG Resources",
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "WFC": "Wells Fargo",
    "GS": "Goldman Sachs", "MS": "Morgan Stanley",
    "BA": "Boeing", "CAT": "Caterpillar", "GE": "GE Aerospace",
    "RTX": "RTX (Raytheon)", "UNP": "Union Pacific",
    "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson", "PFE": "Pfizer",
    "MRK": "Merck", "ABBV": "AbbVie",
    "AMZN": "Amazon", "TSLA": "Tesla", "HD": "Home Depot",
    "MCD": "McDonald's", "NKE": "Nike",
    "LMT": "Lockheed Martin", "NOC": "Northrop Grumman", "GD": "General Dynamics",
    "DJT": "Trump Media & Technology Group",
}


def name_of(ticker: str) -> str:
    """Plain-English name for a ticker, or the ticker itself if unknown."""
    return TICKER_NAME.get(ticker, ticker)


@dataclass(frozen=True)
class BenchmarkSet:
    """Every benchmark a stock is judged against + its sector peers.

    `used_fallback` is True when the stock is in none of the tracked indices and
    no sector, so we fell back to SPY alone — surfaced in the report so a reader
    knows the comparison was the broad market only, not a real membership set.
    """

    ticker: str
    indices: list[str] = field(default_factory=list)   # e.g. ["SPY", "QQQ"]
    sectors: list[str] = field(default_factory=list)   # e.g. ["SMH"]
    peers: list[str] = field(default_factory=list)     # sibling single names
    used_fallback: bool = False

    @property
    def all_benchmarks(self) -> list[str]:
        """Indices + sectors, deduplicated, stable order — every ETF to fetch/plot."""
        return list(dict.fromkeys([*self.indices, *self.sectors]))


def benchmarks_for(ticker: str) -> BenchmarkSet:
    """Resolve the full benchmark set for a stock (or ETF).

    An ETF passed in (e.g. "SPY", "SMH") returns itself as its own single index
    with no sectors/peers — so ETF rows still get a well-defined, self-referential
    benchmark (abnormal ~ 0) rather than crashing the resolver.
    """
    if ticker in INDEX_ETFS or ticker in SECTOR_STOCKS or ticker == "SOXX":
        return BenchmarkSet(ticker, indices=[ticker])

    indices = [etf for etf in INDEX_ETFS if ticker in INDEX_MEMBERS[etf]]
    sectors = list(_STOCK_SECTORS.get(ticker, []))
    peers = [
        p for etf in sectors for p in SECTOR_STOCKS[etf] if p != ticker
    ]
    peers = list(dict.fromkeys(peers))  # dedup across shared sectors, keep order

    if not indices and not sectors:
        return BenchmarkSet(ticker, indices=["SPY"], used_fallback=True)
    return BenchmarkSet(ticker, indices=indices, sectors=sectors, peers=peers)


def _demo() -> None:
    """Self-check: the membership logic the whole pipeline hangs off (run me)."""
    intc = benchmarks_for("INTC")
    assert intc.indices == ["SPY", "QQQ"], intc.indices          # S&P + Nasdaq
    assert intc.sectors == ["SMH"], intc.sectors                 # semis
    assert set(intc.peers) == {"NVDA", "AMD", "TSM", "AVGO"}, intc.peers
    assert not intc.used_fallback

    nvda = benchmarks_for("NVDA")                                # two sectors
    assert nvda.indices == ["SPY", "QQQ", "DIA"], nvda.indices
    assert nvda.sectors == ["XLK", "SMH"], nvda.sectors
    assert "NVDA" not in nvda.peers

    tsm = benchmarks_for("TSM")                                  # foreign ADR
    assert tsm.indices == [], tsm.indices                        # in no index
    assert tsm.sectors == ["SMH"]

    djt = benchmarks_for("DJT")                                  # in nothing
    assert djt.used_fallback and djt.indices == ["SPY"], djt

    xom = benchmarks_for("XOM")                                  # S&P only, energy
    assert xom.indices == ["SPY"] and xom.sectors == ["XLE"], xom
    assert name_of("INTC") == "Intel"
    print("membership _demo OK:",
          {"INTC": intc.all_benchmarks, "NVDA": nvda.all_benchmarks})


if __name__ == "__main__":
    _demo()
