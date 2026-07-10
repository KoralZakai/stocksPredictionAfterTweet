"""Pure aggregates for the eden dashboard. No I/O, no HTML, no plotting.

Every statistic here is DESCRIPTIVE. This is not the purged/embargoed,
BH-corrected evaluation in eval/ — it is a diagnostic view. Two rules keep it
from lying:

  * the majority-class baseline is reported next to every hit rate, because a
    hit rate alone is meaningless (§4 baselines are first-class);
  * rows are NOT i.i.d. — one post spawns many asset rows and the 3d/5d windows
    overlap — so significance is computed on POST-CLUSTERED means, never on the
    naive row count. Both z-scores are surfaced so the gap is visible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.universe import SECTOR_STOCKS
from llm.schema import TweetSignal
from sector_mapping.rules import map_tweet

HORIZONS: tuple[str, ...] = ("abn_1", "abn_3", "abn_5")
EXAMPLE_EVENTS: tuple[str, ...] = (
    "military_threat", "energy_policy", "tariff_trade", "monetary_policy", "drug_pricing",
)
MIN_MEMBERS = 3
MIN_EVENT_N = 30


@dataclass(frozen=True)
class HitRate:
    horizon: str
    n_rows: int
    n_posts: int
    hit: float
    majority: float
    z_naive: float
    z_clustered: float

    @property
    def beats_majority(self) -> bool:
        return self.hit > self.majority


@dataclass(frozen=True)
class EventStat:
    event: str
    n: int
    mean_bps: float
    se_bps: float
    t: float


@dataclass(frozen=True)
class AssetMove:
    ticker: str
    abn_1: float
    abn_3: float
    abn_5: float
    relevance: float


@dataclass(frozen=True)
class Example:
    post_id: str
    ts: str
    text: str
    event: str
    intent: str
    urgency: str
    magnitude: str
    map_conf: float
    etf: AssetMove
    stocks: tuple[AssetMove, ...]


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    columns: tuple[str, ...]
    coverage: float


@dataclass(frozen=True)
class Report:
    n_posts: int
    n_rows: int
    n_assets: int
    date_min: str
    date_max: str
    signal_model: str
    event_mix: tuple[tuple[str, int], ...]
    hit: tuple[HitRate, ...]
    by_event: tuple[EventStat, ...]
    examples: tuple[Example, ...]
    features: tuple[FeatureGroup, ...] = field(default_factory=tuple)


def _bps(v: object) -> float:
    return round(float(v) * 1e4, 1)  # type: ignore[arg-type]


def _hit_rates(etf: pd.DataFrame) -> list[HitRate]:
    signed = etf[etf["intent"] != "neutral"].copy()
    signed["pred"] = np.where(signed["intent"] == "bullish", 1, -1)
    out: list[HitRate] = []
    for h in HORIZONS:
        s = signed[signed[h].notna()]
        if s.empty:
            continue
        hit = float((np.sign(s[h]) == s["pred"]).mean())
        pos = float((etf[h] > 0).mean())
        majority = max(pos, 1.0 - pos)
        n_rows = int(len(s))
        per_post = s.groupby("post_id").apply(
            lambda g: float((np.sign(g[h]) == g["pred"]).mean()), include_groups=False
        )
        n_posts = int(len(per_post))
        se_c = float(per_post.std(ddof=1) / np.sqrt(n_posts)) if n_posts > 1 else 0.0
        z_c = (float(per_post.mean()) - 0.5) / se_c if se_c else 0.0
        z_n = (hit - 0.5) / (0.25 / n_rows) ** 0.5
        out.append(HitRate(h.replace("abn_", "") + "d", n_rows, n_posts,
                           round(hit, 4), round(majority, 4), round(z_n, 2), round(z_c, 2)))
    return out


def _by_event(etf: pd.DataFrame) -> list[EventStat]:
    out: list[EventStat] = []
    for event, g in etf.groupby("event"):
        g1 = g[g["abn_1"].notna()]
        if len(g1) < MIN_EVENT_N:
            continue
        mean = float(g1["abn_1"].mean())
        se = float(g1["abn_1"].std(ddof=1) / np.sqrt(len(g1)))
        out.append(EventStat(str(event), int(len(g1)), round(mean * 1e4, 1),
                             round(se * 1e4, 1), round(mean / se, 2) if se else 0.0))
    out.sort(key=lambda r: r.mean_bps)
    return out


def _move(sub: pd.DataFrame, ticker: str) -> AssetMove:
    r = sub[sub["asset"] == ticker].iloc[0]
    return AssetMove(ticker, _bps(r["abn_1"]), _bps(r["abn_3"]), _bps(r["abn_5"]),
                     round(float(r["relevance"]), 2))


def _examples(d: pd.DataFrame, posts: pd.DataFrame,
              signals: Mapping[str, TweetSignal]) -> list[Example]:
    out: list[Example] = []
    for event in EXAMPLE_EVENTS:
        best: tuple[tuple[float, int], Example] | None = None
        for p in posts[posts["event"] == event].itertuples():
            # Sector from the DETERMINISTIC mapper (§6) — never the LLM, and never
            # "whichever ETF moved most" (that would be cherry-picking the outcome).
            m = map_tweet(str(p.text))
            if m.ticker is None:
                continue
            sub = d[d["post_id"] == p.post_id]
            if m.ticker not in set(sub[sub["is_etf"] == 1]["asset"]):
                continue
            members = [x for x in SECTOR_STOCKS.get(m.ticker, ()) if x in set(sub["asset"])]
            if len(members) < MIN_MEMBERS:
                continue
            score = (float(m.confidence), len(members))  # ranked on mapping, not on returns
            if best is not None and score <= best[0]:
                continue
            s = signals[str(p.post_id)]
            text = str(p.text)
            best = (score, Example(
                post_id=str(p.post_id), ts=str(p.ts)[:16].replace("T", " ") + "Z",
                text=(text[:260] + "…") if len(text) > 260 else text,
                event=s.event_type, intent=s.direction_of_intent, urgency=s.urgency,
                magnitude=s.magnitude, map_conf=round(float(m.confidence), 2),
                etf=_move(sub, m.ticker),
                stocks=tuple(_move(sub, x) for x in members),
            ))
        if best is not None:
            out.append(best[1])
    return out


FEATURE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Pre-event market regime (strictly < t0)",
     ("pre_vol", "pre_ret_1", "pre_abn_1", "pre_ret_3", "pre_abn_3", "pre_ret_5",
      "pre_abn_5", "pre_ret_10", "pre_abn_10", "pre_ret_20", "pre_abn_20")),
    ("Asset linkage", ("asset", "is_etf", "sectors", "relevance", "rel_source", "is_direct")),
    ("Text / policy themes", ("themes",)),
    ("Event-time context", ("weekday", "hour_utc", "after_hours", "trump_exposed")),
)


def _features(d: pd.DataFrame) -> list[FeatureGroup]:
    out: list[FeatureGroup] = []
    for name, cols in FEATURE_GROUPS:
        present = tuple(c for c in cols if c in d.columns)
        if not present:
            continue
        cov = float(np.mean([d[c].notna().mean() for c in present])) * 100.0
        out.append(FeatureGroup(name, present, round(cov, 1)))
    return out


def build_report(d: pd.DataFrame, signals: Mapping[str, TweetSignal],
                 signal_model: str) -> Report:
    """Join the event dataset to the eden signal layer and compute every figure."""
    d = d.copy()
    for h in HORIZONS:
        d[h] = pd.to_numeric(d[h], errors="coerce")
    d["post_id"] = d["post_id"].astype(str)

    posts = (d.groupby("post_id")
               .agg(ts=("timestamp_utc", "first"), text=("text", "first"))
               .reset_index())
    missing = [p for p in posts["post_id"] if p not in signals]
    if missing:
        raise KeyError(f"{len(missing)} posts have no signal (e.g. {missing[:3]})")
    posts["event"] = [signals[p].event_type for p in posts["post_id"]]
    posts["intent"] = [signals[p].direction_of_intent for p in posts["post_id"]]

    d = d.merge(posts[["post_id", "event", "intent"]], on="post_id", how="left")
    etf = d[d["is_etf"] == 1]

    counts = posts["event"].value_counts()
    return Report(
        n_posts=int(len(posts)), n_rows=int(len(d)), n_assets=int(d["asset"].nunique()),
        date_min=str(d["timestamp_utc"].min())[:10], date_max=str(d["timestamp_utc"].max())[:10],
        signal_model=signal_model,
        event_mix=tuple((str(k), int(v)) for k, v in counts.items()),
        hit=tuple(_hit_rates(etf)),
        by_event=tuple(_by_event(etf)),
        examples=tuple(_examples(d, posts, signals)),
        features=tuple(_features(d)),
    )


def horizon_labels() -> Sequence[str]:
    return [h.replace("abn_", "") + "d" for h in HORIZONS]
