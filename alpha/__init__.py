"""alpha/ — the ONE shared engine.

The proven zero-shot Nebius relative-alpha pipeline, extracted from scripts/ so
the batch Job (jobs/backtest) and the serving Endpoint (serving/app.py) call the
SAME code path — no train/serve skew (CLAUDE.md 3.2).

Modules:
  env        - .env loading + env-var lookup (stdlib only)
  classify   - Nebius zero-shot tweet -> scenario/instruments (+ prompt hash)
  benchmark  - point-in-time forward returns + beat-SPY relative-hit scoring
  route      - macro/micro routing + benchmark resolution (decision plane)
  schema     - typed request/response dataclasses (the endpoint contract)

Nothing here touches the network except classify (Nebius) and benchmark's
optional market fetch. The DECISION path is classify -> route only; market data
never enters it (the leakage firewall, serving/app.py).
"""
