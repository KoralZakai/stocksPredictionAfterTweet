"""Outcome-blind cohort tagging + pre-registered asset mapping. EXPERIMENTAL.

Tags come from tweet TEXT ONLY: the cached LLM `scenario` (computed from text) and
the deterministic entity matcher. No tag may depend on returns — that is the
tie-bug lesson encoded as an import-time property (nothing here reads prices).

Precedence: CORPORATE (a direct company mention beats everything) -> GEO_SHOCK ->
NOISE. The asset map per cohort is FIXED here, before any scoring.
"""

from __future__ import annotations

import re
from typing import Any

from config.membership import benchmarks_for
from sector_mapping.entities import entity_matches

# Scenario substrings that mark a geopolitical/macro shock (case-insensitive).
_GEO_RX = re.compile(
    r"geopolit|trade|tariff|war|conflict|iran|china|energy|oil|sanction|military|"
    r"fed|fiscal|monetary|inflation|peace|ceasefire", re.I)

# Pre-registered asset universes (fixed BEFORE scoring; do not extend after).
GEO_SHOCK_ASSETS: tuple[str, ...] = ("USO", "VIXY", "GLD", "ITA", "FXI", "SPY")
NOISE_ASSETS: tuple[str, ...] = ("SPY",)          # control cohort
EVENT_WINDOWS: tuple[int, ...] = (1, 3, 5)        # sessions from entry
FAMILIES: tuple[str, ...] = ("abs", "signed")     # |CAR| reaction-size, signed drift


def tag_cohort(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """(cohort, assets) for one cached result row — text-only inputs."""
    text = row.get("text", "")
    direct = [t for t, m in entity_matches(text).items() if m.tier == "direct"]
    if direct:
        tk = direct[0].upper()
        try:
            sectors = tuple(s for s in benchmarks_for(tk).sectors if s != tk)
        except Exception:
            sectors = ()
        return "CORPORATE", (tk, *sectors[:1])
    if _GEO_RX.search(str(row.get("scenario", ""))):
        return "GEO_SHOCK", GEO_SHOCK_ASSETS
    return "NOISE", NOISE_ASSETS
