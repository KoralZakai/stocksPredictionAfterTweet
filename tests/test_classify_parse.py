"""alpha.classify._parse_json failure-mode contract. No network.

A live backtest is a multi-hour, paid, SEQUENTIAL run over ~1000 tweets. It died at
890/1000 because a single malformed model response raised JSONDecodeError out of
_parse_json and nothing caught it. The run loop in scripts/nebius_macro_backtest.py
now distinguishes the two failure modes, and depends on them staying distinct:

  no {...} in the response  -> SystemExit      -> systemic (auth/model/quota); STOP
  malformed {...}           -> JSONDecodeError -> transient; SKIP that tweet, continue

If _parse_json is ever "helpfully" changed to sys.exit on malformed JSON too, the
loop's `except SystemExit: break` would silently truncate every run at the first bad
response. These tests exist to fail loudly if that happens.
"""

from __future__ import annotations

import json

import pytest

from alpha.classify import _parse_json


def test_valid_json_parses() -> None:
    assert _parse_json('{"scenario":"Trade War","instruments":[]}')["scenario"] == "Trade War"


def test_fenced_json_parses() -> None:
    """The model often wraps output in ```json fences; that is not a failure."""
    assert _parse_json('```json\n{"scenario":"Peace"}\n```')["scenario"] == "Peace"


def test_malformed_json_raises_decode_error_not_systemexit() -> None:
    """THE REGRESSION GUARD: malformed JSON must be a skippable JSONDecodeError.

    This is the exact shape that killed the 890/1000 run — a missing ',' delimiter.
    SystemExit here would make one bad response stop the whole backtest.
    """
    bad = '{"scenario":"Trade War" "instruments":[{"ticker":"SMH"}]}'
    with pytest.raises(json.JSONDecodeError):
        _parse_json(bad)


def test_truncated_json_is_systemexit_which_STOPS_the_run() -> None:
    """DOCUMENTS A SHARP EDGE, not an endorsement of it.

    A response cut off at the token cap has no closing brace, so rfind("}") == -1 and
    _parse_json sys.exit()s -- indistinguishable from "the model refused". The run loop
    treats SystemExit as systemic and BREAKS, then still scores whatever it collected.

    So a single truncated response mid-run silently yields a manifest built on a
    smaller N, with only the "Nebius error, stopping" log line as evidence. If that
    ever bites, the fix is to make truncation distinguishable from refusal here (e.g.
    detect an unbalanced brace) and skip rather than stop.
    """
    with pytest.raises(SystemExit):
        _parse_json('{"scenario":"Trade War","instruments":[{"ticker":"SM')


def test_no_json_at_all_is_systemexit() -> None:
    """No braces => the model is not answering the prompt. Systemic; stop the run."""
    with pytest.raises(SystemExit):
        _parse_json("I'm sorry, I cannot help with that request.")
