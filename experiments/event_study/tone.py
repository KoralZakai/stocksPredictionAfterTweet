"""Tone scoring for posts — anger/emphasis intensity, mechanically measured.

PRE-REGISTERED before any outcome was examined (this docstring is the registration;
see test_tone.py and tone_study.py). The hypothesis to be TESTED — not assumed:

    H_tone: higher-tone posts are followed by larger abnormal moves in their
            mapped assets than lower-tone posts.

We deliberately do NOT bake "angry = more influential" into any weight until the
data says so; a weight that presumes influence is the conclusion smuggled into the
features. This module only measures the text.

Why mechanical instead of the LLM's own 1-10 `intensity`: that field is degenerate
(343 of 476 tweets scored 8) and it already fed the meta-model that failed the
held-out test. The three components below are deterministic, distributed, and
auditable line by line.

Components (each normalized to [0, 1], registered weights: equal thirds):

  exclaim    min(#'!', 5)/5              "AMERICA WILL BE GREAT AGAIN!!!"
  shouting   share of ALL-CAPS words     his signature emphasis: "WITCH HUNT",
             (len >= 3, alphabetic,       "I PAID ZERO FOR INTEL". Acronyms are
             excluding the whitelist)     excluded so USA/FBI/CEO don't count.
  anger      min(lexicon hits, 5)/5      a small fixed list of his recurring
                                          attack vocabulary (see ANGER below).

tone(text) = (exclaim + shouting + anger) / 3   in [0, 1].

The composite and the component list are FROZEN as of registration; changing them
after seeing results would be tuning toward an outcome (charter 2). Anything this
scores is exploratory until it survives the registered test with a permutation
null and BH correction alongside the rest of the registry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Common acronyms/tickers in this corpus that are capitalized by convention, not anger.
# Frozen at registration. Deliberately does NOT include words he shouts for emphasis.
ACRONYMS = frozenset("""
USA US U.S UK EU UN NATO FBI CIA DOJ NSA IRS SEC GOP DNC RNC CNN NBC ABC CBS FOX
MSNBC NYT WSJ CEO COO CFO GDP AI EV LNG OPEC NAFTA USMCA ISIS MAGA DJT TV AM PM
EST PST GMT UTC II III IV OK RT VP POTUS SCOTUS WHO WTO IMF NASA SUV CPU NANOMETER
""".split())

# His recurring attack/outrage vocabulary. Word-boundary, case-insensitive, frozen.
ANGER = re.compile(
    r"\b(witch hunt|hoax|fake news|corrupt(?:ion)?|crooked|disgrace(?:ful)?|"
    r"scam|rigged|treason(?:ous)?|traitor|loser|dumb(?:est)?|stupid|weak|"
    r"pathetic|terrible|horrible|worst|disaster|failing|failed|dishonest|"
    r"radical left|lunatic|crazed|deranged|thug|criminal|illegal|invasion|"
    r"enem(?:y|ies) of the people|angry|furious|shame(?:ful)?|liar|lying|"
    r"incompetent|nasty|evil|sleaz[ey]|con job|coup|betray(?:al|ed)?)\b",
    re.IGNORECASE)

_WORD = re.compile(r"[A-Za-z][A-Za-z'.\-]{2,}")


@dataclass(frozen=True)
class Tone:
    exclaim: float      # [0,1] exclamation-mark load
    shouting: float     # [0,1] ALL-CAPS word share
    anger: float        # [0,1] attack-lexicon load
    n_words: int

    @property
    def score(self) -> float:
        """The registered composite: equal thirds, in [0, 1]."""
        return (self.exclaim + self.shouting + self.anger) / 3.0


def tone(text: str) -> Tone:
    """Score one post. Pure, deterministic, no I/O."""
    # URLs are quoted headlines/links, not his voice — strip before scoring.
    body = re.sub(r"https?://\S+", " ", text)

    exclaim = min(body.count("!"), 5) / 5.0

    # strip clinging punctuation before the whitelist lookup: "NATO." must hit NATO
    words = [w.strip(".'-") for w in _WORD.findall(body)]
    caps = [w for w in words
            if len(w) >= 3 and w.isupper() and w not in ACRONYMS]
    shouting = (len(caps) / len(words)) if words else 0.0

    anger = min(len(ANGER.findall(body)), 5) / 5.0

    return Tone(exclaim=round(exclaim, 4), shouting=round(shouting, 4),
                anger=round(anger, 4), n_words=len(words))
