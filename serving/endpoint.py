"""Nebius Serverless AI Endpoint: POST /predict (§13).

Caller-fed and offline (§2): the request carries the tweet text + t0, features
come from the historical price store as-of t0 via the SAME decide() the batch
pipeline uses (§3.2 — no train/serve skew). If no bars exist strictly before t0
(a live/future tweet), it abstains rather than fabricating features.

Phase 0 has no trained model, so direction is always ABSTAIN — an honest,
well-defined output. Stdlib http.server on purpose (no web-framework dep);
FastAPI/uvicorn is the upgrade path if the endpoint grows.

Run:  BARS_CSV=data/fixtures/bars.csv python serving/endpoint.py   # :8080
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, cast

from config.membership import benchmarks_for, name_of
from config.settings import SETTINGS, Settings
from core.calendar import TradingCalendar
from core.decide import DirectionPredictor, decide
from core.market_state import market_state_as_of
from data.sources.interfaces import PriceSource, Tweet
from data.sources.local import LocalPriceSource
from sector_mapping.entities import entity_matches
from sector_mapping.rules import map_tweet

WIDE = (datetime(1990, 1, 1, tzinfo=timezone.utc), datetime(2100, 1, 1, tzinfo=timezone.utc))


def predict_multibench(
    tweet_text: str,
    timestamp: str,
    prices: PriceSource,
    model: object | None = None,
    cfg: Settings = SETTINGS,
) -> dict[str, Any]:
    """Per-stock multi-benchmark prediction (v-multibench).

    Resolves the tweet's primary STOCK (entity linking, not sector mapping), lists
    the exact benchmarks it is judged against (benchmarks_for), builds the SHARED
    point-in-time feature vector, and — if a multibench model is mounted —
    predicts direction. Stance uses the deterministic offline extractor (text-only,
    no live LLM on the serve path, §2); with no model it abstains honestly.
    """
    from llm.extract import HeuristicExtractor
    from models.multibench_features import FEATURE_ORDER, feature_vector, pre_context

    t0 = datetime.fromisoformat(timestamp)
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)

    matches = {tk: m for tk, m in entity_matches(tweet_text).items() if m.tier != "competitor"}
    if not matches:
        return {"entity": None, "direction": "ABSTAIN", "abstain": True,
                "reason": "no_entity_match"}
    tk = max(matches, key=lambda k: matches[k].conf)
    bset = benchmarks_for(tk)
    benchmarks = {"indices": bset.indices, "sectors": bset.sectors, "peers": bset.peers}

    bars = prices.get_daily_bars(tk, *WIDE)
    spy = prices.get_daily_bars(cfg.benchmark, *WIDE)
    if not bars:
        return {"entity": tk, "entity_name": name_of(tk), "benchmarks": benchmarks,
                "direction": "ABSTAIN", "abstain": True, "reason": "no_bars_for_entity"}
    cal = TradingCalendar([b.session_date.date() for b in bars])
    state = market_state_as_of(t0, tk, bars, spy, cal)
    if not state.prior_bars:
        return {"entity": tk, "entity_name": name_of(tk), "benchmarks": benchmarks,
                "direction": "ABSTAIN", "abstain": True, "reason": "no_history_before_t0"}

    stance_raw = HeuristicExtractor().extract(tweet_text).direction_of_intent
    stance = {"bullish": "positive", "bearish": "negative"}.get(stance_raw, "neutral")
    pv, pr1, pr3 = pre_context([b.close for b in state.prior_bars])
    hour = t0.hour + t0.minute / 60.0
    fv = feature_vector(
        stance=stance, match_tier=matches[tk].tier, sectors=bset.sectors,
        indices=bset.indices, pre_vol=pv, pre_ret_1=pr1, pre_ret_3=pr3,
        weekday=t0.weekday(), after_hours=int(t0.weekday() >= 5 or not (13.5 <= hour < 20.0)),
        used_fallback=int(bset.used_fallback),
    )
    base = {"entity": tk, "entity_name": name_of(tk), "stance": stance,
            "benchmarks": benchmarks}
    if model is None:
        return {**base, "direction": "ABSTAIN", "confidence": 0.0, "abstain": True,
                "reason": "no_model_mounted"}

    import numpy as np  # local: only when a model is actually mounted
    import xgboost as xgb

    booster, classes = cast("tuple[Any, list[str]]", model)
    row = np.array([[fv[k] for k in FEATURE_ORDER]], dtype=float)
    proba = booster.predict(xgb.DMatrix(row, feature_names=list(FEATURE_ORDER)))[0]
    i = int(proba.argmax())
    return {**base, "direction": classes[i], "confidence": round(float(proba[i]), 3),
            "abstain": False}


def predict(
    tweet_text: str,
    timestamp: str,
    prices: PriceSource,
    predictor: DirectionPredictor | None = None,
    cfg: Settings = SETTINGS,
) -> dict[str, Any]:
    t0 = datetime.fromisoformat(timestamp)
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)

    m = map_tweet(tweet_text)
    if m.ticker is None:
        return {"ticker": None, "direction": "ABSTAIN", "confidence": 0.0,
                "abstain": True, "reason": "no_sector_match"}

    bars = prices.get_daily_bars(m.ticker, *WIDE)
    spy = prices.get_daily_bars(cfg.benchmark, *WIDE)
    cal = TradingCalendar([b.session_date.date() for b in bars])
    state = market_state_as_of(t0, m.ticker, bars, spy, cal)
    if not state.prior_bars:
        return {"ticker": m.ticker, "direction": "ABSTAIN", "confidence": 0.0,
                "abstain": True, "reason": "no_history_before_t0"}

    d = decide(Tweet("live", "endpoint", tweet_text, t0), state, predictor)
    return {"ticker": d.ticker, "direction": d.direction, "confidence": d.confidence,
            "abstain": d.abstain, "map_confidence": m.confidence}


def _make_handler(
    prices: PriceSource, predictor: DirectionPredictor | None, mb_model: object | None = None
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            self._send(200, {"status": "ok"}) if self.path == "/health" else self._send(
                404, {"error": "not_found"}
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in ("/predict", "/predict_stock"):
                self._send(404, {"error": "not_found"})
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/predict_stock":
                    result = predict_multibench(
                        req["tweet_text"], req["timestamp"], prices, mb_model)
                else:
                    result = predict(req["tweet_text"], req["timestamp"], prices, predictor)
            except (KeyError, ValueError) as e:
                self._send(400, {"error": str(e)})
                return
            self._send(200, result)

        def log_message(self, *_: Any) -> None:  # silence default stderr logging
            return

    return Handler


def main() -> None:
    bars_csv = os.environ.get("BARS_CSV", "data/fixtures/bars.csv")
    port = int(os.environ.get("PORT", "8080"))
    prices = LocalPriceSource(Path(bars_csv))

    predictor: DirectionPredictor | None = None
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        from models.predictor import Predictor  # local import: keeps xgboost off the cold path

        predictor = Predictor.load(model_dir)
        print(f"loaded model from {model_dir}")
    else:
        print("no MODEL_DIR set — /predict will abstain (no model).")

    # Optional multi-benchmark model for /predict_stock (v-multibench).
    mb_model: object | None = None
    mb_dir = os.environ.get("MB_MODEL_DIR")
    if mb_dir:
        import json as _json

        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(str(Path(mb_dir) / "model.json"))
        classes = _json.loads((Path(mb_dir) / "meta.json").read_text())["classes"]
        mb_model = (b, classes)
        print(f"loaded multibench model from {mb_dir}")
    else:
        print("no MB_MODEL_DIR set — /predict_stock will abstain (no model).")

    server = HTTPServer(("0.0.0.0", port), _make_handler(prices, predictor, mb_model))
    print(f"serving /predict + /predict_stock on :{port} (bars={bars_csv})")
    server.serve_forever()


if __name__ == "__main__":
    main()
