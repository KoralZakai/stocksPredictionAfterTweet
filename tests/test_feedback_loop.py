"""The Endpoint -> Job feedback loop, and the four ways it must refuse to fool itself.

    /predict -> prediction_log.jsonl -> jobs/feedback.py -> prospective_replication.json

Each test here pins one property that, if it broke, would silently turn a null result
into a "finding". They are cheap; the failure mode they guard is not.
"""

from __future__ import annotations

import json
from pathlib import Path

from jobs.feedback import load_log, score
from serving.observe import build_record

MANIFEST = {"code_rev": "abc1234", "prompt_template_hash": "d" * 64, "shipped_horizons": []}


def _resp(decision: str = "LONG") -> dict:
    # `direction` is the wire field name from alpha.schema.Instrument.
    return {"scenario": "Geopolitics / Conflict", "reasoning": "r", "decision": decision,
            "instruments": [{"ticker": "USO", "direction": "up", "benchmark": "SPY"}]}


def test_log_record_carries_no_outcome() -> None:
    """The log is written AT t0, when the outcome does not exist. If an outcome field
    ever appears here it means a price at/after t0 was read into the record of a
    decision made at t0 — the point-in-time leak of CLAUDE.md 3.1, in the one place
    nobody would look for it."""
    rec = build_record(tweet_text="Iran!", t0_utc="2026-03-09T14:00:00Z", author="",
                       response=_resp(), manifest=MANIFEST, profile="stable",
                       served_decision="ABSTAIN")
    flat = json.dumps(rec).lower()
    for banned in ('"returns"', '"hit"', '"label"', '"realized"', '"pnl"', '"outcome"'):
        assert banned not in flat, f"outcome field {banned} leaked into the serve-time log"


def test_log_maps_direction_to_the_scorer_key() -> None:
    """The wire says `direction`; the scorer (_relabel) reads `predicted`. If this
    mapping breaks, every instrument logs predicted=None, the Job scores nothing, and
    the report says a confident 'n=0' forever instead of failing loudly."""
    rec = build_record(tweet_text="t", t0_utc="2026-03-09T14:00:00Z", author="",
                       response=_resp(), manifest=MANIFEST, profile="stable",
                       served_decision="ABSTAIN")
    assert rec["instruments"] == [{"ticker": "USO", "predicted": "up"}]


def test_log_separates_what_the_model_said_from_what_we_served() -> None:
    """While shipped_horizons is empty every caller gets ABSTAIN, but the classifier
    still made a call. Collapsing the two into one field would make the replication set
    unreadable: you could not tell a model that said nothing from one we refused to
    repeat."""
    rec = build_record(tweet_text="t", t0_utc="2026-03-09T14:00:00Z", author="",
                       response=_resp("LONG"), manifest=MANIFEST, profile="stable",
                       served_decision="ABSTAIN")
    assert rec["model_decision"] == "LONG"
    assert rec["served_decision"] == "ABSTAIN"


def test_replayed_tweet_counts_once(tmp_path: Path) -> None:
    """A caller hammering /predict with the same post must not become N independent
    observations. Counting duplicates is exactly the same-bar artifact that fabricated
    9 intraday cells at p=0.0018 (README) — here it would arrive via HTTP."""
    p = tmp_path / "log.jsonl"
    rec = build_record(tweet_text="same post", t0_utc="2026-03-09T14:00:00Z", author="",
                       response=_resp(), manifest=MANIFEST, profile="stable",
                       served_decision="ABSTAIN")
    other = build_record(tweet_text="different post", t0_utc="2026-03-09T14:00:00Z",
                         author="", response=_resp(), manifest=MANIFEST, profile="stable",
                         served_decision="ABSTAIN")
    with p.open("w", encoding="utf-8") as f:
        for _ in range(50):
            f.write(json.dumps(rec) + "\n")
        f.write(json.dumps(other) + "\n")
    assert len(load_log(p)) == 2, "50 replays of one post must collapse to one observation"


def test_torn_line_does_not_kill_the_job(tmp_path: Path) -> None:
    """A killed container leaves a half-written last line. The Job must skip it, not
    crash — otherwise one bad byte permanently blocks every future replication."""
    p = tmp_path / "log.jsonl"
    rec = build_record(tweet_text="ok", t0_utc="2026-03-09T14:00:00Z", author="",
                       response=_resp(), manifest=MANIFEST, profile="stable",
                       served_decision="ABSTAIN")
    p.write_text(json.dumps(rec) + "\n" + '{"tweet_sha256": "trunc', encoding="utf-8")
    assert len(load_log(p)) == 1


def test_looking_again_cannot_manufacture_significance() -> None:
    """THE ONE THAT MATTERS. Re-scoring a growing log and shipping the first time
    p < alpha is optional stopping: on pure noise it reaches "significance" with
    probability 1 given enough looks. So the correction must get STRICTER with each
    look. Same data, more looks -> p_bh must rise, never fall.
    """
    rows = [{"instruments": [{"ticker": "USO", "predicted": "up",
                             "hit": {h: True for h in ("EOD", "3d", "1w", "1mo")}}]}
            for _ in range(8)]
    first = score(rows, n_looks=1)
    tenth = score(rows, n_looks=10)
    p1 = {c["horizon"]: c["p_bh"] for c in first["cells"]}
    p10 = {c["horizon"]: c["p_bh"] for c in tenth["cells"]}
    assert all(p10[h] >= p1[h] for h in p1), (
        f"looking again LOOSENED the bar ({p1} -> {p10}) — the job would eventually "
        "ship noise just by being re-run on a schedule")
    assert tenth["bh_denominator"] > first["bh_denominator"]


def test_job_never_ships_a_horizon() -> None:
    """The Job writes a report, never the manifest the Endpoint boots from. Shipping
    stays a human decision against the pre-registered schedule."""
    src = Path("jobs/feedback.py").read_text(encoding="utf-8")
    body = src.split('"""', 2)[2]                      # ignore the docstring's prose
    assert "validation_manifest" not in body, "the feedback Job must not touch the manifest"
    assert '"shipped_horizons"' not in body.replace('"shipped_horizons": []', "")
