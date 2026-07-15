# live_pipeline/ — event-driven live analysis (EXPERIMENTAL, NOT shipped)

> A separate, isolated pathway: a post arrives → a rich structured LLM analysis
> (literal vs. veiled meaning, macro impact, multi-asset/multi-horizon) is returned.
> **It shares nothing with the validated `/predict` engine** — the shipped endpoint,
> its manifest, and its tests stay 100% untouched and green.

## What it is / isn't

- **Its own FastAPI app** (`live_app.py`, `/live-predict`) — a different `FastAPI()`
  instance, not a route added to `serving/app.py`.
- **Unvalidated by design.** Every response carries `validated: false` + a disclaimer.
  It has no manifest-hash pin because it is not the validated model.
- **Not investment advice, not calibrated.**

## Three honest departures from the requested spec

1. **No calibrated `confidence`.** The field is `llm_conviction` — the model's
   *uncalibrated* self-report — because per-tweet confidence was tested and failed
   the sacred test (Val 0.593 → Test 0.431). It must never be read as P(correct).
2. **`long_term` predictions are auto-flagged `speculative: true`.** Our study shows
   the edge is front-loaded; the multi-week "drift" was market beta, not signal.
3. **The listener is a stub, not a scraper.** `listener.py` gives a pluggable
   `SourceAdapter` (RSS / webhook you control / official API with your keys). Direct
   scraping of X/Truth Social is intentionally `NotImplementedError` — ToS + bot
   detection. Real-time streaming is a parked phase (CLAUDE.md §2), not shipped here.

## Run (experimental)

```bash
# Serve the isolated live app (needs NEBIUS_API_KEY):
uvicorn live_pipeline.live_app:app --host 0.0.0.0 --port 8090

curl -X POST localhost:8090/live-predict -H 'content-type: application/json' \
  -d '{"tweet_text":"We will put 100% tariffs on all foreign cars.","t0_utc":"2025-03-03T14:00:00+00:00"}'
```

Listener: implement an authorised `SourceAdapter.poll()`, then
`run_loop(adapter, "http://<endpoint>", interval_s=30)`.

## Files
`live_schema.py` (Pydantic contract) · `live_analyze.py` (rich LLM call) ·
`live_app.py` (isolated FastAPI `/live-predict`) · `listener.py` (trigger stub).
