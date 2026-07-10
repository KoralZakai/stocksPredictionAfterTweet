"""LLM signal layer: schema->features stability, cache invalidation, heuristic
fallback, and the point-in-time guarantee (extractor sees text only). No network."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm.cache import SignalCache
from llm.extract import HeuristicExtractor, default_extractor
from llm.schema import EVENT_TYPES, TweetSignal, signal_to_features, zero_features


def test_feature_layout_is_stable_and_zero_default_matches() -> None:
    sig = TweetSignal(
        event_type="military_threat", direction_of_intent="bullish", urgency="high",
        magnitude="high", certainty="high", names_country=True, names_company=False,
    )
    feats = signal_to_features(sig)
    # every event gets a one-hot column; keys identical to the zero fallback
    assert sum(v for k, v in feats.items() if k.startswith("llm_event_")) == 1.0
    assert set(feats) == set(zero_features())
    assert feats["llm_event_military_threat"] == 1.0
    assert feats["llm_intent"] == 1.0 and feats["llm_urgency"] == 1.0


def test_heuristic_pure_energy_tweet_is_bullish_under_threat() -> None:
    # A pure supply-disruption tweet -> energy_policy, and the explicit
    # threat->oil-up rule flips intent bullish.
    sig = HeuristicExtractor().extract("OPEC oil supply cut, ban all crude exports.")
    assert sig.event_type == "energy_policy"
    assert sig.direction_of_intent == "bullish"


def test_heuristic_is_coarse_military_beats_energy() -> None:
    # DOCUMENTS the fallback's limitation: "strike Iran ... oil" resolves to
    # military_threat, not the oil-up read a real LLM would give. This coarseness
    # is exactly why the offline LLM extractor exists.
    sig = HeuristicExtractor().extract("We will strike Iran and its oil facilities.")
    assert sig.event_type == "military_threat"
    assert sig.names_country is True


def test_default_extractor_offline_is_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert isinstance(default_extractor(), HeuristicExtractor)


def test_cache_invalidates_on_text_change(tmp_path: Path) -> None:
    c = SignalCache(tmp_path / "sig.json")
    sig = HeuristicExtractor().extract("tariffs on China")
    c.put("t1", "tariffs on China", "m", sig)
    assert c.get("t1", "tariffs on China", "m") is not None      # same text -> hit
    assert c.get("t1", "tariffs on Mexico", "m") is None         # changed text -> miss
    assert c.get("t1", "tariffs on China", "other-model") is None  # model swap -> miss


def test_cache_roundtrips_through_disk(tmp_path: Path) -> None:
    p = tmp_path / "sig.json"
    c = SignalCache(p)
    c.put("t1", "oil", "m", HeuristicExtractor().extract("drill baby drill oil"))
    c.save()
    reloaded = SignalCache(p)
    assert reloaded.get("t1", "oil", "m") is not None
    assert len(reloaded) == 1


def test_every_event_type_has_a_feature_column() -> None:
    feats = zero_features()
    for e in EVENT_TYPES:
        assert f"llm_event_{e}" in feats
