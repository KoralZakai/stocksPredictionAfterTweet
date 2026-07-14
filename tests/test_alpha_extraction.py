"""Stage 1 guards: the signal engine moved to alpha/ with NO behavior change, and
the decision-plane router is pure and tweet-only.

Two things are proven here:
  1. `scripts/*` still import the SAME engine objects now living in alpha.* (the
     re-export is identity, so the backtest path is untouched).
  2. `alpha.route.route_decision` collapses an instrument basket to LONG/SHORT/
     ABSTAIN exactly like the validated `signed_eod` majority vote.

The byte-identical --from-results regression (sha256 of macro_dataset.csv) is an
optional slow guard: it self-skips when the cached results artifact is absent.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from alpha.classify import classify_tweet as alpha_classify
from alpha.classify import prompt_template_hash
from alpha.route import WHITELIST, route_decision
from alpha.schema import RoutedDecision

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- no train/serve skew
def test_scripts_reexport_is_identity() -> None:
    """The backtest imports from scripts.nebius_macro_validate; those names must be
    the very objects defined in alpha.* — otherwise there are two code paths."""
    from scripts import nebius_macro_validate as w

    from alpha import benchmark, classify
    assert w.classify_tweet is alpha_classify is classify.classify_tweet
    assert w.validate is benchmark.validate
    assert w.forward_returns is benchmark.forward_returns
    assert w.relative_hit is benchmark.relative_hit
    assert w.HORIZONS is benchmark.HORIZONS


def test_prompt_hash_is_deterministic() -> None:
    assert prompt_template_hash() == prompt_template_hash()
    assert len(prompt_template_hash()) == 64  # sha256 hex


# ---------------------------------------------------------------- decision-plane router
def _classified(instruments: list[dict], scenario: str = "Trade Policy") -> dict:
    return {"scenario": scenario, "rationale": "why", "instruments": instruments}


def test_route_abstains_on_no_instruments() -> None:
    d = route_decision(_classified([]))
    assert d.decision == "ABSTAIN"
    assert "not market-relevant" in d.abstain_reason


def test_route_abstains_on_unknown_ticker() -> None:
    d = route_decision(_classified([{"ticker": "ZZZZ", "predicted_direction": "up"}]))
    assert d.decision == "ABSTAIN"


def test_route_long_on_up_majority() -> None:
    d = route_decision(_classified([
        {"ticker": "XLK", "predicted_direction": "up"},
        {"ticker": "QQQ", "predicted_direction": "up"},
        {"ticker": "VIXY", "predicted_direction": "down"},
    ]))
    assert d.decision == "LONG"
    assert {i.ticker for i in d.instruments} == {"XLK", "QQQ", "VIXY"}
    assert all(i.benchmark == "SPY" for i in d.instruments)


def test_route_short_on_down_majority() -> None:
    d = route_decision(_classified([
        {"ticker": "XLK", "predicted_direction": "down"},
        {"ticker": "XLF", "predicted_direction": "down"},
        {"ticker": "ITA", "predicted_direction": "up"},
    ]))
    assert d.decision == "SHORT"


def test_route_tie_breaks_long() -> None:
    """ups >= downs -> LONG, mirroring signed_eod's dominant-direction tie-break."""
    d = route_decision(_classified([
        {"ticker": "XLK", "predicted_direction": "up"},
        {"ticker": "XLF", "predicted_direction": "down"},
    ]))
    assert d.decision == "LONG"


def test_route_drops_spy_and_neutral() -> None:
    """SPY (the benchmark) and neutral calls are not scoreable instruments."""
    d = route_decision(_classified([
        {"ticker": "SPY", "predicted_direction": "up"},
        {"ticker": "XLE", "predicted_direction": "neutral"},
        {"ticker": "USO", "predicted_direction": "up"},
    ]))
    assert d.decision == "LONG"
    assert [i.ticker for i in d.instruments] == ["USO"]


def test_route_returns_frozen_decision() -> None:
    d = route_decision(_classified([{"ticker": "XLK", "predicted_direction": "up"}]))
    assert isinstance(d, RoutedDecision)
    with pytest.raises(Exception):
        d.decision = "SHORT"  # type: ignore[misc]  # frozen dataclass


def test_whitelist_covers_prompt_universe() -> None:
    for tk in ("XLK", "ITA", "VIXY", "USO", "QQQ"):
        assert tk in WHITELIST


# ---------------------------------------------------------------- byte-identical regression
def test_from_results_byte_identical() -> None:
    """Re-scoring from the cached results JSON must reproduce the committed
    macro_dataset.csv byte-for-byte — the proof the alpha/ extraction changed no
    behavior. Self-skips if the cached artifacts are absent (clean checkout)."""
    results = ROOT / "reports" / "nebius_backtest_results.json"
    dataset = ROOT / "reports" / "macro_dataset.csv"
    if not results.exists() or not dataset.exists():
        pytest.skip("cached results/dataset artifact not present")
    before = hashlib.sha256(dataset.read_bytes()).hexdigest()
    subprocess.run(
        [sys.executable, "scripts/nebius_macro_backtest.py", "--from-results"],
        cwd=ROOT, check=True, capture_output=True,
        env={"PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8",
             **_os_environ_without_pythonpath()},
    )
    after = hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert before == after, "alpha/ extraction changed macro_dataset.csv output"


def _os_environ_without_pythonpath() -> dict[str, str]:
    import os
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
