"""Offline LLM signal-extraction layer (product fork, not in the null-result main).

The LLM runs ONCE, OFFLINE, over each tweet's TEXT ONLY (which is knowable
pre-t0 by construction — §3.1), and writes a cache keyed by tweet_id + content
hash + schema version + model id. Serving never calls the LLM: it reads the
cache, so `decide()` stays deterministic and the no-skew test (§3.2) still holds.

The LLM PROPOSES structured signals (event type, target, direction-of-intent,
urgency, magnitude); the GBT still DECIDES. The LLM is deliberately kept OUT of
the sector-mapping causal chain (§6) — it emits no ticker and no sector, so a
null result stays interpretable.
"""
