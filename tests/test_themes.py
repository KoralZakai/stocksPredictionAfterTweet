"""Theme taxonomy + market-relevance filter: deterministic rule checks (§6)."""

from __future__ import annotations

from sector_mapping.themes import (
    active_themes,
    combined_relevance,
    market_relevance,
    theme_asset_relevance,
)


def test_tariff_china_post_links_expected_assets() -> None:
    text = "We will impose big Tariffs on China until they stop cheating!"
    ok, reasons = market_relevance(text)
    assert ok
    assert any(r.startswith("theme:tariffs_trade") for r in reasons)
    rel = combined_relevance(text)
    assert rel["XLI"] >= 0.6 and rel["XLB"] >= 0.3 and rel["SMH"] >= 0.4
    assert "CAT" in rel and rel["CAT"] <= rel["XLI"]  # member stock never beats its ETF


def test_junk_post_is_filtered() -> None:
    text = "MAKE AMERICA GREAT AGAIN! Thank you Iowa, big crowds, tremendous love!"
    ok, reasons = market_relevance(text)
    assert not ok and reasons == []
    assert combined_relevance(text) == {}


def test_direct_entity_mention_wins_over_theme() -> None:
    rel = combined_relevance("Boeing is building a brand new 747 Air Force One!")
    assert rel["BA"] == 1.0  # direct mention outranks the 0.5 aviation-theme link


def test_macro_theme_is_relevant_but_unlinked() -> None:
    text = "The Stock Market just hit another ALL-TIME HIGH! Your 401(k)s are way up!"
    ok, reasons = market_relevance(text)
    assert ok and any("stock_market_macro" in r for r in reasons)
    assert theme_asset_relevance(text) == {}  # macro themes map to no asset


def test_lone_generic_token_does_not_fire() -> None:
    assert "jobs_economy" not in active_themes("Great meeting today about jobs")


def test_relevance_always_in_unit_interval() -> None:
    for text in (
        "Tariffs on Chinese semiconductors and electric vehicles now!",
        "OPEC must keep oil prices down! Iran sanctions are working.",
        "The Fed and Jerome Powell should cut interest rates!",
    ):
        rel = combined_relevance(text)
        assert rel and all(0.0 < v <= 1.0 for v in rel.values())
