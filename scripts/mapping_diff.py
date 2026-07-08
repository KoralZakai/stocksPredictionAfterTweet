"""Mapping v1 -> v2 audit: before/after comparison + mis-mapped-trigger report.

Quantifies how much the stricter mapper (sector_mapping/rules.py) reduces false
positives vs the old single-word-substring mapper, on the real 2017-2021 corpus.

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/mapping_diff.py
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone

from data.sources.local import load_corpus
from sector_mapping.rules import map_tweet_multi, matched_triggers

CORPUS = "data/real/corpus.csv"

# The OLD v1 mapper: single-word SUBSTRING matching (the thing we're replacing).
OLD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "XLE": ("oil", "energy", "drill", "opec", "gas", "petroleum"),
    "SMH": ("chip", "chips", "semiconductor", "semiconductors"),
    "XLK": ("tech", "technology", "software"),
    "XLF": ("bank", "banks", "fed", "interest rate", "rates", "financial"),
    "ITA": ("military", "defense", "army", "war", "weapons", "troops"),
    "XLI": ("tariff", "tariffs", "manufacturing", "trade war", "industrial", "factory"),
    "XLB": ("steel", "aluminum", "materials", "mining"),
    "XLV": ("drug", "drugs", "pharma", "health", "healthcare", "medicare"),
    "XLY": ("car", "cars", "auto", "autos", "retail", "consumer"),
    "XLP": ("grocery", "groceries", "food", "staples"),
}


def old_sectors(text: str) -> set[str]:
    tl = text.lower()
    return {tk for tk, ws in OLD_KEYWORDS.items() if any(w in tl for w in ws)}


def new_sectors(text: str) -> set[str]:
    return {m.ticker for m in map_tweet_multi(text) if m.ticker}


def main() -> None:
    tweets = load_corpus(
        CORPUS, datetime(2017, 1, 1, tzinfo=timezone.utc),
        datetime(2021, 1, 9, tzinfo=timezone.utc), platforms=("twitter",))
    n = len(tweets)

    old_n = sum(bool(old_sectors(t.text)) for t in tweets)
    new_n = sum(bool(new_sectors(t.text)) for t in tweets)
    old_rows = sum(len(old_sectors(t.text)) for t in tweets)
    new_rows = sum(len(new_sectors(t.text)) for t in tweets)
    print(f"corpus tweets (twitter 2017-2021 originals): {n}")
    print(f"  mapped tweets   OLD: {old_n:5}   NEW: {new_n:5}")
    print(f"  (tweet,sector)  OLD: {old_rows:5}   NEW: {new_rows:5}   "
          f"(-{old_rows - new_rows} candidate rows pruned)")

    # ---- Misclassification report: top substring-only false triggers (OLD) ----
    false_trig: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for t in tweets:
        tl = t.text.lower()
        for ws in OLD_KEYWORDS.values():
            for w in ws:
                if w in tl and not re.search(rf"\b{re.escape(w)}\b", tl):
                    false_trig[w] += 1
                    examples.setdefault(w, t.text)
    print("\n--- TOP 20 MIS-MAPPED TRIGGERS (OLD substring false positives) ---")
    print(f"{'trigger':10}{'false hits':>11}  example (substring matched inside another word)")
    for w, c in false_trig.most_common(20):
        inside = next((word for word in examples[w].lower().split()
                       if w in word and w != word), "")
        print(f"{w:10}{c:>11}  '{inside}'  <- {examples[w][:46]}")

    # ---- Before/after examples (>=20): tweets where OLD and NEW disagree ----
    print("\n--- BEFORE / AFTER (>=20 tweets where mapping changed) ---")
    print(f"{'OLD sectors':22}{'NEW sectors':16} text")
    shown = 0
    for t in tweets:
        o, ne = old_sectors(t.text), new_sectors(t.text)
        if o != ne and (o or ne):
            print(f"{','.join(sorted(o)) or '-':22}{','.join(sorted(ne)) or '-':16} {t.text[:50]}")
            shown += 1
        if shown >= 22:
            break

    # ---- A few PRESERVED true positives (both map, same sector) ----
    print("\n--- PRESERVED TRUE POSITIVES (both map the same) ---")
    kept = 0
    for t in tweets:
        o, ne = old_sectors(t.text), new_sectors(t.text)
        if o and o == ne:
            trig = matched_triggers(t.text)
            print(f"{','.join(sorted(ne)):16} triggers={trig}  {t.text[:44]}")
            kept += 1
        if kept >= 6:
            break


if __name__ == "__main__":
    main()
