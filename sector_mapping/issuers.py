"""Issuer taxonomy + signal-decay profiles (multi-issuer scaffold, V2 pre-reg).

SCOPE (pre-registered amendment to CLAUDE.md's single-speaker V1): the pipeline
was one speaker (Trump). This module is the SCHEMA for scaling to a roster of
market influencers. It is a pure taxonomy — **nothing consumes it yet**. Wiring
it into labeling/eval requires two guarded changes first (see the DEBT notes
below), so this file only DECLARES the schema and keeps it human-verifiable.

Two axes, kept separate on purpose:
  * ISSUER (who spoke) — this module. author-handle -> Issuer -> InfluencerGroup.
  * ENTITY (who the message is about) — sector_mapping/entities.py, unchanged and
    issuer-agnostic. A CEO links to their own ticker via `Issuer.self_entities`;
    that's the only bridge between the two axes.

Signal-decay profile = the horizon set + benchmark a group is measured on. A
CEO's move is fast/idiosyncratic (short horizons, sector-ETF benchmark); a Fed/
legislative signal plays out slowly and market-wide (long horizons, and abnormal-
vs-SPY would erase it, so it's benchmarked on cross-sector rotation instead).

Seed roster is small and HUMAN-VERIFIED (§6) — extend it deliberately; every new
name is a surname-collision risk (cf. the Goldman guard in entities.py).

DEBT — before any job/eval reads this (do NOT skip; these break §3 invariants):
  1. Cross-issuer purge: eval/splits.py purges by tweet_id. Two issuers on the
     same ticker inside the embargo window overlap -> re-key purge on
     (ticker, session-window) or issuer B leaks into issuer A. [[eval-splits]]
  2. Per-group horizons re-explode the BH registry (§4) and split N per group;
     re-run the power gate per group and expect several underpowered-by-design.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InfluencerGroup(str, Enum):
    MACRO_REGULATORY = "macro_regulatory"    # Fed officials, SEC, congressional chairs
    CORPORATE_TITAN = "corporate_titan"      # mega-cap CEOs, activist investors


class BenchmarkMode(str, Enum):
    SECTOR_ETF = "sector_etf"                # abnormal = raw - sector ETF (V1 default)
    # macro signals move SPY itself, so SPY-abnormal nets them out; measure them
    # as sector dispersion/rotation vs the market instead (design marker; the
    # rotation metric itself lands in labeling/ when this is wired).
    CROSS_SECTOR_ROTATION = "cross_sector_rotation"


@dataclass(frozen=True)
class DecayProfile:
    """Horizons (trading sessions) + benchmark a group's signal is graded on."""
    horizons: tuple[int, ...]                # sessions computed
    label_horizons: tuple[int, ...]          # subset that gets a BH-tested verdict
    benchmark: BenchmarkMode


# Pre-registered decay profiles per group. Changing these is a pre-registration
# change recorded in git, not a runtime tweak (same rule as config/settings.py).
DECAY: dict[InfluencerGroup, DecayProfile] = {
    InfluencerGroup.CORPORATE_TITAN: DecayProfile(
        horizons=(1, 2, 3, 5), label_horizons=(1, 3),
        benchmark=BenchmarkMode.SECTOR_ETF),                 # fast decay
    InfluencerGroup.MACRO_REGULATORY: DecayProfile(
        horizons=(1, 5, 10, 21, 42), label_horizons=(5, 21),
        benchmark=BenchmarkMode.CROSS_SECTOR_ROTATION),      # slow, market-wide
}


@dataclass(frozen=True)
class Issuer:
    handle: str                              # corpus author handle (lowercased)
    name: str
    group: InfluencerGroup
    self_entities: tuple[str, ...] = ()      # tickers the issuer IS (CEO -> own co.)
    default_scope: tuple[str, ...] = ()      # tickers/ETFs they move by default


# Seed roster — small, human-verified (§6). handle keys are lowercased, '@' stripped.
ISSUERS: dict[str, Issuer] = {
    "realdonaldtrump": Issuer("realdonaldtrump", "Donald Trump",
                              InfluencerGroup.MACRO_REGULATORY, default_scope=("SPY",)),
    "elonmusk": Issuer("elonmusk", "Elon Musk",
                       InfluencerGroup.CORPORATE_TITAN, self_entities=("TSLA",)),
    "jeromepowell": Issuer("jeromepowell", "Jerome Powell (Fed)",
                           InfluencerGroup.MACRO_REGULATORY, default_scope=("SPY", "XLF")),
    "secgov": Issuer("secgov", "U.S. SEC",
                     InfluencerGroup.MACRO_REGULATORY, default_scope=("SPY", "XLF")),
    "carlicahn": Issuer("carlicahn", "Carl Icahn (activist)",
                        InfluencerGroup.CORPORATE_TITAN),
}


def _norm(author: str) -> str:
    return author.strip().lstrip("@").lower().replace(" ", "")


def classify_issuer(author: str) -> Issuer | None:
    """author handle -> Issuer, or None if not in the (human-verified) roster."""
    return ISSUERS.get(_norm(author))


def decay_for(author: str) -> DecayProfile | None:
    iss = classify_issuer(author)
    return None if iss is None else DECAY[iss.group]


def union_horizons() -> tuple[int, ...]:
    """All horizons across profiles, sorted — the rectangular table's columns."""
    return tuple(sorted({h for p in DECAY.values() for h in p.horizons}))


def in_profile(group: InfluencerGroup, h: int) -> bool:
    """Is horizon h part of `group`'s decay profile? (mask for the union table)."""
    return h in DECAY[group].horizons


def issuer_columns(author: str) -> dict[str, str | None]:
    """The row-schema fields a builder would stamp per post. Non-invasive: wiring
    later is one call. Unknown author -> nulls (an honest 'unclassified issuer')."""
    iss = classify_issuer(author)
    return {"issuer_handle": None if iss is None else iss.handle,
            "issuer_group": None if iss is None else iss.group.value}


if __name__ == "__main__":  # ponytail: runnable schema self-check
    assert set(DECAY) == set(InfluencerGroup), "every group needs a decay profile"
    for g, p in DECAY.items():
        assert set(p.label_horizons) <= set(p.horizons), f"{g}: labels not in horizons"
        assert list(p.horizons) == sorted(p.horizons), f"{g}: horizons unsorted"
    trump, musk = classify_issuer("@realDonaldTrump"), classify_issuer("elonmusk")
    ceo_p, fed_p = decay_for("elonmusk"), decay_for("jeromepowell")
    assert trump is not None and trump.group is InfluencerGroup.MACRO_REGULATORY
    assert musk is not None and musk.self_entities == ("TSLA",)
    assert ceo_p is not None and ceo_p.benchmark is BenchmarkMode.SECTOR_ETF
    assert fed_p is not None and fed_p.benchmark is BenchmarkMode.CROSS_SECTOR_ROTATION
    assert classify_issuer("nobody") is None
    assert issuer_columns("unknown") == {"issuer_handle": None, "issuer_group": None}
    assert union_horizons() == (1, 2, 3, 5, 10, 21, 42) and in_profile(
        InfluencerGroup.CORPORATE_TITAN, 5) and not in_profile(
        InfluencerGroup.CORPORATE_TITAN, 42)
    print("issuers schema self-check OK")
