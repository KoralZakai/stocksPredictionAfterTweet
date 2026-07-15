"""Stage 3 guards for the shipped /predict endpoint (serving/app.py) — the leakage
firewall above all.

Hermetic: classification is stubbed (no Nebius) and the market provider is injected
(no network). The real validation_manifest.json is reused for the happy path; boot
tampering is done on temp copies.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

os.environ["PREDICT_SKIP_BOOT"] = "1"   # don't build the module-level app on import

from fastapi.testclient import TestClient  # noqa: E402

from alpha.benchmark import _session_anchor  # noqa: E402
from market.null import NullProvider  # noqa: E402
from serving.app import create_app, verify_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports" / "validation_manifest.json"
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="manifest not generated")


# ---- stub classifiers (decision plane input = tweet text only) ----
def _classify_macro(_text: str) -> dict[str, Any]:
    return {"scenario": "Trade War", "rationale": "tariffs bite tech",
            "instruments": [{"ticker": "XLK", "predicted_direction": "down"},
                            {"ticker": "XLF", "predicted_direction": "down"}]}


def _classify_empty(_text: str) -> dict[str, Any]:
    return {"scenario": "", "rationale": "", "instruments": []}


def _classify_unknown(_text: str) -> dict[str, Any]:
    return {"scenario": "x", "rationale": "y",
            "instruments": [{"ticker": "ZZZZ", "predicted_direction": "up"}]}


# ---- adversarial market providers (must NEVER change the decision) ----
class RaisingProvider:
    name = "raiser"

    def quote(self, ticker: str) -> float | None:
        raise RuntimeError("boom")


class TimeoutProvider:
    name = "slow"

    def quote(self, ticker: str) -> float | None:
        time.sleep(3.0)          # > MARKET_TIMEOUT_S
        return 1.0


class AbsurdProvider:
    name = "absurd"

    def quote(self, ticker: str) -> float | None:
        return 9.99e18


def _client(classify: Any, provider: Any) -> TestClient:
    return TestClient(create_app(manifest_path=str(MANIFEST), classify_fn=classify, provider=provider))


def _decision_core(resp: dict[str, Any]) -> dict[str, Any]:
    return {"decision": resp["decision"], "instruments": resp["instruments"],
            "reasoning": resp["reasoning"], "scenario": resp["scenario"]}


# ---------------------------------------------------------------- THE FIREWALL
def test_decision_invariant_to_market_data() -> None:
    """decision/instruments/reasoning must be byte-identical whether the market
    provider is null, raises, times out, or returns absurd values."""
    body = {"tweet_text": "China tariffs incoming", "t0_utc": "2025-03-03T14:00:00+00:00"}
    cores = []
    for provider in (NullProvider(), RaisingProvider(), TimeoutProvider(), AbsurdProvider()):
        r = _client(_classify_macro, provider).post("/predict", json=body)
        assert r.status_code == 200
        cores.append(_decision_core(r.json()))
    assert all(c == cores[0] for c in cores), "market data leaked into the decision!"
    # raise/timeout must null the market plane, never 5xx.
    for provider in (RaisingProvider(), TimeoutProvider()):
        r = _client(_classify_macro, provider).post("/predict", json=body)
        assert r.status_code == 200 and r.json()["market_context"] is None


# ---------------------------------------------------------------- abstention
def test_abstain_on_thanksgiving() -> None:
    r = _client(_classify_empty, NullProvider()).post(
        "/predict", json={"tweet_text": "Happy Thanksgiving!", "t0_utc": "2025-11-27T14:00:00+00:00"})
    body = r.json()
    assert body["decision"] == "ABSTAIN"
    assert body["market_context"] is None and body["cohort_base_rate"] is None


def test_abstain_on_unknown_ticker() -> None:
    r = _client(_classify_unknown, NullProvider()).post(
        "/predict", json={"tweet_text": "buy ZZZZ", "t0_utc": "2025-03-03T14:00:00+00:00"})
    assert r.json()["decision"] == "ABSTAIN"


def test_abstain_on_unanchorable_t0() -> None:
    r = _client(_classify_macro, NullProvider()).post(
        "/predict", json={"tweet_text": "tariffs", "t0_utc": "not-a-date"})
    body = r.json()
    assert body["decision"] == "ABSTAIN" and "t0 unanchorable" in body["abstain_reason"]


# ---------------------------------------------------------------- boot pins
def _tampered(field_path: list[str], value: Any, tmp: Path) -> str:
    m = json.loads(MANIFEST.read_text())
    node = m
    for k in field_path[:-1]:
        node = node[k]
    node[field_path[-1]] = value
    out = tmp / "manifest.json"
    out.write_text(json.dumps(m))
    return str(out)


def test_boot_fails_on_prompt_hash_mismatch(tmp_path: Path) -> None:
    bad = _tampered(["prompt_template_hash"], "deadbeef" * 8, tmp_path)
    with pytest.raises(RuntimeError, match="PROMPT HASH MISMATCH"):
        create_app(manifest_path=bad, classify_fn=_classify_macro, provider=NullProvider())


def test_boot_fails_on_corpus_hash_mismatch(tmp_path: Path) -> None:
    bad = _tampered(["corpus", "sha256"], "0" * 64, tmp_path)
    with pytest.raises(RuntimeError, match="CORPUS HASH MISMATCH"):
        create_app(manifest_path=bad, classify_fn=_classify_macro, provider=NullProvider())


def test_boot_fails_on_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="MANIFEST MISSING"):
        create_app(manifest_path=str(tmp_path / "nope.json"),
                   classify_fn=_classify_macro, provider=NullProvider())


def test_verify_manifest_accepts_real() -> None:
    verify_manifest(json.loads(MANIFEST.read_text()))   # must not raise


# ---------------------------------------------------------------- resilience / schema
def test_serves_with_no_market_keys() -> None:
    """Null provider (zero market keys) still serves a 200 with an intact decision."""
    r = _client(_classify_macro, NullProvider()).post(
        "/predict", json={"tweet_text": "tariffs", "t0_utc": "2025-03-03T14:00:00+00:00"})
    assert r.status_code == 200 and r.json()["decision"] in ("LONG", "SHORT")


def test_saturday_t0_anchors_to_monday() -> None:
    from datetime import datetime, timezone
    sat = datetime(2025, 3, 1, 18, 0, tzinfo=timezone.utc)   # Saturday
    anchor = _session_anchor(sat)
    assert anchor.weekday() == 0                              # Monday, not Friday
    assert anchor > sat


def test_no_per_tweet_probability_field() -> None:
    r = _client(_classify_macro, NullProvider()).post(
        "/predict", json={"tweet_text": "tariffs", "t0_utc": "2025-03-03T14:00:00+00:00"})

    def scan(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert "probab" not in k.lower() and k.lower() != "confidence", f"leaked: {k}"
                scan(v)
        elif isinstance(obj, list):
            for v in obj:
                scan(v)
    scan(r.json())


def test_health_reports_pins() -> None:
    r = _client(_classify_macro, NullProvider()).get("/health")
    h = r.json()
    assert h["status"] == "ok"
    assert h["prompt_template_hash"] and h["corpus_sha256"]
    # Assert against the manifest, NOT a hardcoded expectation. This previously
    # asserted ["EOD"] and so baked in a result that turned out to be a tie-break
    # artifact; a test must not assert what we wish were true.
    assert h["shipped_horizons"] == (json.loads(MANIFEST.read_text()).get("shipped_horizons") or [])


def test_no_validated_horizon_is_disclosed() -> None:
    """With no horizon surviving BH, /predict must not imply a validated horizon:
    horizon is null and no cohort base rate is cited."""
    shipped = json.loads(MANIFEST.read_text()).get("shipped_horizons") or []
    r = _client(_classify_macro, NullProvider()).post(
        "/predict", json={"tweet_text": "tariffs", "t0_utc": "2025-03-03T14:00:00+00:00"})
    body = r.json()
    if not shipped:
        assert body["horizon"] is None
        assert body["cohort_base_rate"] is None    # never quote an unvalidated rate


# ---------------------------------------------------------------- golden
def test_golden_three_corpus_tweets_stable() -> None:
    """3 cached teacher classifications -> the endpoint's decision matches route_decision
    on the same input (the endpoint uses the shared router, deterministically)."""
    from alpha.route import route_decision
    results = json.loads((ROOT / "reports" / "nebius_backtest_results.json").read_text())
    picked = [r for r in results if r.get("instruments")][:3]
    assert len(picked) == 3
    for r in picked:
        classified = {"scenario": r.get("scenario", ""), "rationale": r.get("macro_link", ""),
                      "instruments": [{"ticker": i["ticker"],
                                       "predicted_direction": i.get("predicted", "neutral")}
                                      for i in r["instruments"]]}
        expected = route_decision(classified).decision
        resp = _client(lambda _t, c=classified: c, NullProvider()).post(
            "/predict", json={"tweet_text": r["text"][:80], "t0_utc": "2025-03-03T14:00:00+00:00"})
        assert resp.json()["decision"] == expected
