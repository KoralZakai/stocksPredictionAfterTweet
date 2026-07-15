"""Append-only prediction log — the Endpoint's half of the feedback loop.

    /predict  ->  prediction_log.jsonl (bucket)  ->  jobs/feedback.py  ->  report

ZERO SCIENCE HERE (CLAUDE.md 13). This module marshals one JSON line per served
request onto a bucket-mounted file. It computes no feature, resolves no label, joins
no price, and makes no decision. Everything that could ever count as evidence is
derived offline by jobs/feedback.py, from these lines, after the fact.

WHY NO OUTCOME IS WRITTEN HERE. At t0 the outcome does not exist yet: a 1-day call
cannot be scored for another session, a 1-month call for another month. Writing an
outcome at serve time would mean reading a price at or after t0 into the record of a
decision made at t0 — the exact point-in-time leak of CLAUDE.md 3.1. So this file
records only what was knowable at t0: the text, the anchor, and what the model said.
Maturity is jobs/feedback.py's problem, and it is decided by whether the bar exists.

FAILURE POLICY: logging must never break serving. Every error is swallowed and
warned. A dropped log line costs one row of a replication set; a 500 on /predict
costs the caller their request.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

log = logging.getLogger("predict")

SCHEMA = 1
DEFAULT_PATH = "runs/real/prediction_log.jsonl"


def log_path() -> Path:
    """Bucket-mounted in deployment: deploy.sh mounts the bucket at /app/runs/real, so
    the default already lands in durable storage and survives the container."""
    return Path(os.environ.get("PREDICTION_LOG", DEFAULT_PATH))


def build_record(*, tweet_text: str, t0_utc: str, author: str,
                 response: dict[str, Any], manifest: dict[str, Any],
                 profile: str, served_decision: str = "",
                 now: datetime | None = None) -> dict[str, Any]:
    """The line we would append. Pure — split out from the write so a test can assert
    the contents (and the ABSENCE of any outcome field) without touching a disk.

    `tweet_sha256` is the dedupe key: the same post replayed twice is ONE observation,
    not two. Counting a replay twice is the same-bar duplicate bug that fabricated 9
    intraday cells at p=0.0018 (see README), arriving through a new door.
    """
    stamp = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    return {
        "schema": SCHEMA,
        "logged_at_utc": stamp.isoformat(),
        "tweet_sha256": sha256(tweet_text.encode("utf-8")).hexdigest(),
        "tweet_text": tweet_text,
        "t0_utc": t0_utc,
        "author": author,
        # ---- what the model said, at t0, from text alone ----
        "scenario": response.get("scenario"),
        "reasoning": response.get("reasoning"),
        # TWO decisions, and they routinely differ. `model_decision` is the raw call the
        # classifier made; `served_decision` is what the caller actually got, which is
        # ABSTAIN for every tweet while shipped_horizons is empty. Collapsing them into
        # one "decision" field would make the log unreadable later: you could not tell a
        # model that said nothing from a model we refused to repeat.
        "model_decision": response.get("decision"),
        "served_decision": served_decision,
        # `direction` is the wire field (alpha.schema.Instrument); `predicted` is the key
        # the scorer reads (scripts/nebius_macro_backtest._rescore_relative). Map here,
        # once, so the Job feeds the SAME scorer the same shape the backtest does.
        "instruments": [{"ticker": i.get("ticker"),
                         "predicted": i.get("direction") or i.get("predicted")}
                        for i in (response.get("instruments") or [])],
        # ---- provenance: which exact artifact served this ----
        "profile": profile,
        "manifest_version": manifest.get("code_rev"),
        "prompt_template_hash": manifest.get("prompt_template_hash"),
        "shipped_horizons": manifest.get("shipped_horizons") or [],
        # NOTE: deliberately NO returns / hit / label field. See module docstring.
    }


def log_prediction(**kw: Any) -> None:
    """Append one line. Never raises — see FAILURE POLICY."""
    try:
        rec = build_record(**kw)
        p = log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:                                  # noqa: BLE001 - see policy
        log.warning("prediction log write failed (%s) — serving unaffected", type(e).__name__)
