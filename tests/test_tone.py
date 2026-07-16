"""The tone scorer — the properties that keep it honest.

The scorer is pre-registered (tone.py docstring). These tests pin the behaviours a
drift in the regexes would silently break: acronyms are not shouting, URLs are not
his voice, and the composite stays in [0, 1] no matter the input.
"""

from __future__ import annotations

from experiments.event_study.tone import tone


def test_calm_post_scores_near_zero() -> None:
    t = tone("I met with the Prime Minister today to discuss trade and security.")
    assert t.score < 0.05
    assert t.exclaim == 0.0 and t.anger == 0.0


def test_signature_rage_post_scores_high() -> None:
    t = tone("The Radical Left LUNATICS are running a WITCH HUNT — a total DISGRACE. "
             "FAKE NEWS! Crooked, corrupt, and RIGGED!!!")
    assert t.exclaim >= 0.8, "four exclamation marks must near-max the component"
    assert t.shouting > 0.15, "LUNATICS/DISGRACE/etc are shouting"
    assert t.anger >= 0.8, "witch hunt/hoax-class lexicon must register"
    assert t.score > 0.6


def test_acronyms_are_not_shouting() -> None:
    """USA/FBI/CEO are capitalized by convention. A scorer that counts them tags
    every routine post about institutions as rage — the DJT-signature artifact's
    shape, recurring in a new feature."""
    t = tone("The USA and the UK signed with NATO. The FBI and CIA briefed the CEO.")
    assert t.shouting == 0.0


def test_urls_do_not_leak_into_the_score() -> None:
    """Half his corpus is quoted-headline + link. The URL is not his voice."""
    plain = tone("Interesting story about the border.")
    linked = tone("Interesting story about the border. "
                  "https://justthenews.com/CORRUPT-FBI-WITCH-HUNT-DISGRACE!!!")
    assert linked.score == plain.score


def test_score_bounded() -> None:
    for text in ["", "!", "!!!!!!!!!!!!!!!!", "A" * 500,
                 "WITCH HUNT " * 50 + "!" * 50]:
        s = tone(text).score
        assert 0.0 <= s <= 1.0


def test_deterministic() -> None:
    x = "Tariffs will make America WEALTHY again! The corrupt media won't say it!"
    assert tone(x) == tone(x)
