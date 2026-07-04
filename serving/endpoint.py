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
from typing import Any

from config.settings import SETTINGS, Settings
from core.calendar import TradingCalendar
from core.decide import decide
from core.market_state import market_state_as_of
from data.sources.interfaces import PriceSource, Tweet
from data.sources.local import LocalPriceSource
from sector_mapping.rules import map_tweet

WIDE = (datetime(1990, 1, 1, tzinfo=timezone.utc), datetime(2100, 1, 1, tzinfo=timezone.utc))


def predict(
    tweet_text: str, timestamp: str, prices: PriceSource, cfg: Settings = SETTINGS
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

    d = decide(Tweet("live", "endpoint", tweet_text, t0), state)
    return {"ticker": d.ticker, "direction": d.direction, "confidence": d.confidence,
            "abstain": d.abstain, "map_confidence": m.confidence}


def _make_handler(prices: PriceSource) -> type[BaseHTTPRequestHandler]:
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
            if self.path != "/predict":
                self._send(404, {"error": "not_found"})
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                result = predict(req["tweet_text"], req["timestamp"], prices)
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
    server = HTTPServer(("0.0.0.0", port), _make_handler(prices))
    print(f"serving /predict on :{port} (bars={bars_csv})")
    server.serve_forever()


if __name__ == "__main__":
    main()
