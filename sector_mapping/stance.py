"""Per-entity STANCE: deterministic, auditable sentiment around the mention.

Separate from entity resolution (entities.py) and from relevance/confidence — a
tweet can name a ticker with zero stance (a policy statement) or with strong
stance ("horrible"). Stance is reported as its own field so we can later TEST
"does explicit stance predict a bigger abnormal return than mention-only?"
rather than baking that assumption into the label.

Rules only, no ML (CLAUDE.md §6), same philosophy as the entity gazetteer:
  * hand-curated positive / negative lexicons (Trump-vocabulary-flavored),
  * scored on the SENTENCE(S) containing the entity mention — NOT the whole
    tweet, so "great news for Apple, but Boeing is a disaster" scores AAPL
    positive and BA negative independently,
  * simple negation flip ("not great" -> negative) within a short window.

Labels:
  positive / negative  — net sentiment score > 0 / < 0 in the entity's sentence
  neutral              — sentiment words present but net == 0 (mixed / cancelled)
  no_clear_stance      — no sentiment words near the entity at all

The lexicons below are the tuning knob: edit them, re-run the audit counts
(fraction pos/neg/neutral, direct-vs-indirect non-neutral rate) to recalibrate.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

# Single-word lexicons (lowercased, word-boundaried at match time).
POSITIVE: frozenset[str] = frozenset({
    "great", "greatest", "historic", "tremendous", "incredible", "amazing",
    "fantastic", "wonderful", "success", "successful", "win", "winning", "won",
    "strong", "strongest", "best", "love", "loved", "congratulations", "congrats",
    "proud", "honored", "thank", "thanks", "booming", "boom", "record", "beautiful",
    "brilliant", "genius", "respected", "powerful", "prosperity", "prosperous",
    "growth", "thriving", "excellent", "outstanding", "smart", "safe", "grateful",
})
NEGATIVE: frozenset[str] = frozenset({
    "horrible", "disaster", "disastrous", "terrible", "terribly", "failure",
    "failed", "fail", "failing", "weak", "weakest", "sad", "pathetic", "disgrace",
    "disgraceful", "corrupt", "crooked", "fake", "worst", "unfair", "dangerous",
    "crazy", "dishonest", "incompetent", "stupid", "ridiculous", "scam",
    "catastrophe", "collapse", "collapsing", "disgusting", "nasty", "phony",
    "loser", "dumb", "betrayed", "threat", "hoax", "broken", "angry", "shameful",
    "wrong", "bad", "worse", "declining", "plunge", "plunging", "crisis",
})
# Multiword phrases (regex, sentence-level). Sign only; each counts once.
PHRASES: tuple[tuple[str, int], ...] = (
    (r"\bwitch hunt\b", -1), (r"\bripp?(?:ing|ed)? (?:us |america )?off\b", -1),
    (r"\bout of control\b", -1), (r"\btotal disaster\b", -1),
    (r"\brip[- ]off\b", -1), (r"\bmake america great\b", 1),
    (r"\bnever been (?:better|stronger)\b", 1), (r"\ball[- ]time high\b", 1),
)
NEGATORS: frozenset[str] = frozenset({"not", "no", "never", "without", "hardly",
                                      "barely", "isn't", "wasn't", "aren't", "don't"})

_SENT_SPLIT = re.compile(r"[.!?\n]+")
_WORD = re.compile(r"[a-z']+")
_PHRASES = tuple((re.compile(p, re.I), s) for p, s in PHRASES)


@dataclass(frozen=True)
class Stance:
    label: str                    # positive | negative | neutral | no_clear_stance
    score: int
    evidence: tuple[str, ...]     # the sentiment tokens/phrases that fired


def _sentences(text: str) -> list[tuple[int, int]]:
    """Char spans (start, end) of each sentence."""
    out, i = [], 0
    for m in _SENT_SPLIT.finditer(text):
        out.append((i, m.start()))
        i = m.end()
    if i < len(text):
        out.append((i, len(text)))
    return out


def _score_window(window: str) -> tuple[int, list[str]]:
    toks = _WORD.findall(window.lower())
    score, evidence = 0, []
    for j, tok in enumerate(toks):
        base = 1 if tok in POSITIVE else -1 if tok in NEGATIVE else 0
        if base == 0:
            continue
        negated = any(toks[k] in NEGATORS for k in range(max(0, j - 2), j))
        s = -base if negated else base
        score += s
        evidence.append(f"not {tok}" if negated else tok)
    for rx, s in _PHRASES:
        if rx.search(window):
            score += s
            evidence.append(rx.pattern)
    return score, evidence


def entity_stance(text: str, spans: Sequence[tuple[int, int]]) -> Stance:
    """Stance toward the entity whose mention lives at `spans` (char offsets).
    Scores only the sentence(s) that contain a span — per-entity, not per-tweet."""
    if not spans:
        return Stance("no_clear_stance", 0, ())
    windows = [text[a:b] for a, b in _sentences(text)
               if any(a <= s < b or a < e <= b for s, e in spans)]
    score, evidence = 0, []
    for w in windows:
        sc, ev = _score_window(w)
        score += sc
        evidence += ev
    if not evidence:
        return Stance("no_clear_stance", 0, ())
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return Stance(label, score, tuple(evidence))


if __name__ == "__main__":  # ponytail: runnable self-check for the non-trivial paths
    def st(text: str, ent: str) -> Stance:
        from sector_mapping.entities import entity_spans
        return entity_stance(text, entity_spans(text, ent))

    # positive / negative / negation / mixed-entity / no-stance policy mention
    assert st("Apple is doing a GREAT job, historic success!", "AAPL").label == "positive"
    assert st("Boeing is a total disaster, terrible failure.", "BA").label == "negative"
    assert st("Apple is not great, very weak.", "AAPL").label == "negative"
    mix = "Great news for Apple. Boeing is a disaster."
    assert st(mix, "AAPL").label == "positive" and st(mix, "BA").label == "negative"
    assert st("The CHIPS Act will fund Intel fabs in Ohio.", "INTC").label == "no_clear_stance"
    print("stance self-check OK")
