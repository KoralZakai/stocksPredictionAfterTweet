"""Stage 2 guards: the validation manifest is built correctly and honestly.

Covers the mission's required checks:
  - BH correction + Wilson CI math (delegates to alpha.stats self-check).
  - manifest schema round-trips (all required keys, JSON-serialisable).
  - job succeeds with NO Alpaca keys: intraday horizons are skipped, not shipped,
    and never crash the build.
  - shipped_horizons = survives_bh AND reproducible (daily only).
"""

from __future__ import annotations

import json

from alpha import stats
from jobs.backtest.entrypoint import build_manifest


def test_stats_selfcheck() -> None:
    stats._demo()  # asserts inside; raises on regression


def _tweet(split: str, hits: dict[str, bool]) -> dict:
    """A minimal scored-tweet record: build_manifest only reads split + instrument hits."""
    return {"split": split, "date": "2025-01-01",
            "instruments": [{"ticker": "XLK", "predicted": "up", "hit": dict(hits)}]}


def test_manifest_schema_round_trips() -> None:
    # 60 EOD hits out of 89 test tweets -> a real, BH-surviving daily signal.
    results = [_tweet("test", {"EOD": i < 60}) for i in range(89)]
    results += [_tweet("train", {"EOD": True}) for _ in range(10)]  # non-test ignored
    m = build_manifest(results, alpha=0.05, corpus_file="data/real/corpus_v3.csv",
                       now_utc="2026-07-14T00:00:00+00:00")

    for key in ("schema_version", "generated_at_utc", "code_rev", "corpus", "metric",
                "split", "alpha", "prompt_template_hash", "horizons", "registry",
                "shipped_horizons", "disclaimer"):
        assert key in m, f"missing manifest key: {key}"
    assert json.loads(json.dumps(m)) == m            # fully JSON-serialisable
    assert m["corpus"]["n"] == 99
    eod = m["horizons"]["EOD"]
    assert eod["n_test"] == 89 and eod["hits_test"] == 60
    assert eod["survives_bh"] is True
    assert m["shipped_horizons"] == ["EOD"]


def test_no_alpaca_skips_intraday_not_crash() -> None:
    """With no intraday hits (no Alpaca), 30m/1h are skipped and excluded from
    shipped_horizons; the build still succeeds."""
    results = [_tweet("test", {"EOD": i < 60}) for i in range(89)]  # no 30m/1h keys
    m = build_manifest(results, alpha=0.05, corpus_file="data/real/corpus_v3.csv",
                       now_utc="2026-07-14T00:00:00+00:00")
    for h in ("30m", "1h"):
        assert m["horizons"][h]["skipped"] is True
        assert m["horizons"][h]["requires_alpaca"] is True
    assert "30m" not in m["shipped_horizons"] and "1h" not in m["shipped_horizons"]


def test_intraday_survives_but_not_shipped() -> None:
    """Even a hugely significant intraday horizon is NOT shipped (needs private keys)."""
    results = [_tweet("test", {"EOD": i < 45, "1h": i < 80}) for i in range(89)]
    m = build_manifest(results, alpha=0.05, corpus_file="data/real/corpus_v3.csv",
                       now_utc="2026-07-14T00:00:00+00:00")
    assert m["horizons"]["1h"]["survives_bh"] is True   # 80/89 is overwhelming
    assert m["horizons"]["1h"]["requires_alpaca"] is True
    assert "1h" not in m["shipped_horizons"]            # excluded for reproducibility


def test_weak_signal_does_not_survive() -> None:
    """A coin-flip EOD (46/89) must NOT survive BH -> empty shipped_horizons."""
    results = [_tweet("test", {"EOD": i < 46}) for i in range(89)]
    m = build_manifest(results, alpha=0.05, corpus_file="data/real/corpus_v3.csv",
                       now_utc="2026-07-14T00:00:00+00:00")
    assert m["horizons"]["EOD"]["survives_bh"] is False
    assert m["shipped_horizons"] == []
