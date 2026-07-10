"""On-disk cache for LLM signals — the seam that keeps serving deterministic.

Keyed by tweet_id; an entry is only reused when its content hash, schema version,
and model id all still match (so re-wording a tweet, bumping the schema, or
switching models forces a clean re-extract). This is what lets the batch job and
the /predict endpoint read the SAME signal without ever calling the LLM live.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm.schema import SCHEMA_VERSION, TweetSignal


def content_key(text: str, model: str) -> str:
    """Stable fingerprint of (schema, model, text). Change any -> new key."""
    h = hashlib.sha256(f"{SCHEMA_VERSION}\x00{model}\x00{text}".encode()).hexdigest()
    return h[:16]


class SignalCache:
    """tweet_id -> {key, model, schema_version, signal}. Plain JSON, git-friendly."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._store: dict[str, dict[str, object]] = {}
        if self.path.exists():
            self._store = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, tweet_id: str, text: str, model: str) -> TweetSignal | None:
        row = self._store.get(tweet_id)
        if row is None or row.get("key") != content_key(text, model):
            return None
        return TweetSignal.model_validate(row["signal"])

    def put(self, tweet_id: str, text: str, model: str, signal: TweetSignal) -> None:
        self._store[tweet_id] = {
            "key": content_key(text, model),
            "model": model,
            "schema_version": SCHEMA_VERSION,
            "signal": signal.model_dump(),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._store, indent=2, sort_keys=True), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._store)
