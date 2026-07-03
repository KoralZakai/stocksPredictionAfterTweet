from sector_mapping.rules import map_tweet


def test_maps_energy() -> None:
    m = map_tweet("We will drill baby drill, American energy dominance")
    assert m.ticker == "XLE" and m.score >= 2 and m.confidence == 1.0


def test_no_match_maps_to_none() -> None:
    m = map_tweet("Happy birthday to my wonderful friend")
    assert m.ticker is None and m.confidence == 0.0


def test_confidence_below_one_when_mixed() -> None:
    # "factory" (XLI) + "energy" (XLE) -> winner shares < 1.0.
    m = map_tweet("energy factory jobs")
    assert 0.0 < m.confidence < 1.0
