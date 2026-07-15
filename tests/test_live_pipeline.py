"""Guards for the EXPERIMENTAL live pipeline — isolation + honest labelling.

Hermetic: the analyzer is stubbed, no Nebius. Proves the route works, abstains
cleanly, flags long_term as speculative, and never presents a calibrated
per-tweet probability.
"""

from __future__ import annotations

import os
from typing import Any

os.environ["LIVE_SKIP_BOOT"] = "1"   # don't build the module-level app on import

from fastapi.testclient import TestClient  # noqa: E402

from live_pipeline.live_analyze import build_analysis  # noqa: E402
from live_pipeline.live_app import create_live_app  # noqa: E402
from live_pipeline.live_schema import TrumpStatementAnalysis  # noqa: E402


def _analysis(preds: list[dict[str, Any]]) -> TrumpStatementAnalysis:
    return build_analysis({"literal_translation": "L", "veiled_meaning": "V",
                           "macro_economic_impact": "M", "predictions": preds})


def _client(analysis: TrumpStatementAnalysis) -> TestClient:
    return TestClient(create_live_app(analyzer=lambda _t: analysis))


def test_long_term_flagged_speculative() -> None:
    a = _analysis([
        {"asset_name": "XLF", "asset_type": "sector", "horizon": "short_term",
         "direction": "UP", "llm_conviction": 0.6, "catalyst_reasoning": "x"},
        {"asset_name": "SPY", "asset_type": "index", "horizon": "long_term",
         "direction": "DOWN", "llm_conviction": 0.5, "catalyst_reasoning": "y"},
    ])
    by_h = {p.horizon: p.speculative for p in a.predictions}
    assert by_h["short_term"] is False
    assert by_h["long_term"] is True          # beta-dominated -> flagged


def test_conviction_clamped_and_old_key_tolerated() -> None:
    a = _analysis([{"asset_name": "QQQ", "asset_type": "index", "horizon": "short_term",
                    "direction": "UP", "confidence": 5.0, "catalyst_reasoning": "z"}])
    assert a.predictions[0].llm_conviction == 1.0    # clamped from the tolerated old "confidence" key


def test_live_predict_route_and_abstain() -> None:
    client = _client(_analysis([{"asset_name": "XLE", "asset_type": "sector",
                                 "horizon": "short_term", "direction": "UP",
                                 "llm_conviction": 0.7, "catalyst_reasoning": "oil"}]))
    r = client.post("/live-predict", json={"tweet_text": "drill baby drill", "t0_utc": "2025-03-03T14:00:00+00:00"})
    body = r.json()
    assert r.status_code == 200
    assert body["abstained"] is False
    assert body["validated"] is False and body["disclaimer"]

    empty = _client(_analysis([]))
    rb = empty.post("/live-predict", json={"tweet_text": "Happy Thanksgiving!"}).json()
    assert rb["abstained"] is True


def test_no_calibrated_probability_field() -> None:
    """The response may carry llm_conviction (uncalibrated) but NOT a 'probability'
    or 'confidence' field that would read as calibrated P(correct)."""
    client = _client(_analysis([{"asset_name": "SPY", "asset_type": "index",
                                 "horizon": "short_term", "direction": "UP",
                                 "llm_conviction": 0.6, "catalyst_reasoning": "x"}]))
    body = client.post("/live-predict", json={"tweet_text": "tariffs"}).json()

    def scan(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                assert "probab" not in k.lower() and k.lower() != "confidence", f"leaked: {k}"
                scan(v)
        elif isinstance(o, list):
            for v in o:
                scan(v)
    scan(body)


def test_frozen_predict_app_untouched() -> None:
    """The live app is a DIFFERENT FastAPI instance from the shipped one."""
    os.environ["PREDICT_SKIP_BOOT"] = "1"
    from serving.app import create_app
    assert create_live_app.__module__ == "live_pipeline.live_app"
    assert create_app.__module__ == "serving.app"   # distinct, unshared apps
