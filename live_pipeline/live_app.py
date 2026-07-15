"""EXPERIMENTAL live-analysis FastAPI app — SEPARATE from the shipped serving/app.py.

Isolation is structural: this is its own FastAPI() instance in its own package. It
does not import, modify, or share routes with the frozen /predict app, so the
shipped endpoint, its manifest, and its tests are untouched.

This route makes NO validated claim. It has no manifest hash pin (nothing to pin —
it's not the validated model) and every response carries validated=false.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from fastapi import FastAPI
from pydantic import BaseModel

from live_pipeline.live_analyze import analyze
from live_pipeline.live_schema import LiveResponse, TrumpStatementAnalysis

DISCLAIMER = LiveResponse.model_fields["disclaimer"].default


class LivePredictIn(BaseModel):
    tweet_text: str
    t0_utc: str = ""
    author: str = ""


def _default_analyzer(text: str) -> TrumpStatementAnalysis:
    from alpha.env import env, load_dotenv
    load_dotenv()
    api_key = env("NEBIUS_API_KEY", "EXPO_PUBLIC_NEBIUS_API_KEY")
    if not api_key:
        raise RuntimeError("No NEBIUS_API_KEY for the live analyzer.")
    base = env("NEBIUS_BASE_URL", "EXPO_PUBLIC_NEBIUS_BASE_URL",
               default="https://api.studio.nebius.ai/v1")
    model = env("NEBIUS_MODEL", "EXPO_PUBLIC_NEBIUS_MODEL", default="meta-llama/Llama-3.3-70B-Instruct")
    return analyze(text, base_url=base, api_key=api_key, model=model)


def create_live_app(analyzer: Callable[[str], TrumpStatementAnalysis] | None = None) -> FastAPI:
    """Build the isolated live app. `analyzer` is injectable for hermetic tests."""
    do_analyze = analyzer or _default_analyzer
    app = FastAPI(title="live-predict (experimental, unvalidated)")

    @app.post("/live-predict")
    def live_predict(req: LivePredictIn) -> dict[str, Any]:
        analysis = do_analyze(req.tweet_text)
        abstained = not analysis.predictions
        return LiveResponse(analysis=analysis, abstained=abstained).model_dump()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "route": "/live-predict",
                "validated": False, "disclaimer": DISCLAIMER}

    return app


# Module-level app for `uvicorn live_pipeline.live_app:app`. Skipped in tests.
app = None if os.environ.get("LIVE_SKIP_BOOT") == "1" else create_live_app()
