"""Purged + embargoed walk-forward CV (§3.6).

Overlapping ret_2d/3d windows correlate nearby rows, so a random KFold leaks.
This yields time-ordered folds where the training set is (a) strictly before
the test fold, (b) embargoed by >= `embargo` sessions, and (c) purged of any
tweet id that appears in the test fold (matters for the multi-sector secondary
analysis, where one tweet -> several rows).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date


def _blocks(order: list[int], n_splits: int) -> list[list[int]]:
    k, m = divmod(len(order), n_splits + 1)
    out, start = [], 0
    for b in range(n_splits + 1):
        size = k + (1 if b < m else 0)
        out.append(order[start : start + size])
        start += size
    return out


def purged_walk_forward(
    s0_dates: Sequence[date],
    tweet_ids: Sequence[str],
    sessions: Sequence[date],
    n_splits: int = 3,
    embargo: int = 3,
) -> list[tuple[list[int], list[int]]]:
    n = len(s0_dates)
    if n < n_splits + 1:
        return []
    sess_idx = {d: i for i, d in enumerate(sorted(set(sessions)))}
    order = sorted(range(n), key=lambda i: s0_dates[i])
    blocks = _blocks(order, n_splits)

    folds: list[tuple[list[int], list[int]]] = []
    for b in range(1, len(blocks)):
        test = blocks[b]
        if not test:
            continue
        test_start = min(sess_idx[s0_dates[i]] for i in test)
        test_tweets = {tweet_ids[i] for i in test}
        train = [
            i
            for prev in blocks[:b]
            for i in prev
            if tweet_ids[i] not in test_tweets
            and test_start - sess_idx[s0_dates[i]] > embargo  # embargo gap in sessions
        ]
        if train and test:
            folds.append((train, test))
    return folds
