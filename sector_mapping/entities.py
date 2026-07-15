"""Soft entity -> stock relevance scoring (0..1 per asset per tweet).

Upgrade from hard sector mapping: instead of "tweet -> sector -> its 6 assets all
equal", each asset gets a graded relevance in [0,1] combining two deterministic,
pre-t0 (text-only) signals — no ML in the causal chain (§6):

  * DIRECT entity mention (company / exec / brand named in the text)  -> 1.0
  * SECTOR membership: the asset's sector confidence (share of matched
      keyword weight); the ETF inherits the full share, member stocks half
      (a sector tweet is more clearly about the ETF than any single name).

relevance(asset) = max(direct, sector_component). A tweet that names "Boeing"
links BA directly even with no industrial keyword; a generic "tariffs" tweet
gives XLI=conf and its stocks 0.5*conf. This is the stock-level linking layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config.universe import SECTOR_STOCKS
from sector_mapping.rules import sector_scores

# "intel" the word ("US intel flagged...") is intelligence, not the chipmaker. In a
# political corpus that sense DOMINATES: of 55 bare-"intel" matches in corpus_v3, 51
# (93%) were intelligence. This is the DJT-signature artifact's twin — a ticker rule
# matching a common English word — so it gets the same treatment.
#
# The old guard only blocked the word AFTER intel ("intel leak") and a few words before
# ("US intel"). It therefore let through "leaking intel", "destroy intel", "Danish intel",
# "Obama Intel Chief", "Ex-intel official", and every URL slug ("us-intel-has-known"),
# while the direction it did block was the rarer one.
#
# Now precision-first — the same call as GS_GUARDED below: an "intel" only resolves to
# INTC when the post ALSO carries company context. Both orders are matched because the
# context can precede the word ("the CEO of INTEL") or follow it ("Intel Stock"); `re`
# has no variable-width lookbehind, so the second alternative spans from the context word
# to the mention. Only truthiness and the label are used, so a wide span is harmless.
_INTC_CTX = (r"(?:chips?\b|semiconductor|foundry|fabs?\b|wafer|nanometer|cpu|processor|"
             r"ceo|chief executive|stock|shares?\b|stake|shareholder|equity|owns?\b|"
             # `deals?` is whole-word on BOTH sides on purpose: a bare `\bdeal` prefix
             # matched "incapable of DEALing", which dragged a Snowden intelligence
             # tweet into INTC. Context words this generic must not match substrings.
             r"corp\b|inside\b|lip-?bu|gelsinger|billion|\bdeals?\b)")
# (?<!-) and [-\s] in the lookahead cover URL slugs, where there is no space to anchor on.
_INTC_BARE = (r"(?<!\bus )(?<!\bu\.s\. )(?<!american )(?<!\bthe )(?<!russia )(?<!russian )"
              r"(?<!house )(?<!senate )(?<!obama )(?<!cia )(?<!fbi )(?<!danish )(?<!spy )"
              r"(?<!ex-)(?<!leaking )(?<!destroy )(?<!\bmy )(?<!nat )(?<!-)"
              r"\bintel\b"
              r"(?![-\s]+(?:community|agenc|official|secret|report|source|chief|director|"
              r"briefing|assessment|committee|leak|panel|team|warn|driven|agent|comm\b))")
INTC_GUARDED = rf"{_INTC_BARE}(?=(?s:.)*{_INTC_CTX})|{_INTC_CTX}(?s:.)*?{_INTC_BARE}"

# "Goldman" is a common politician surname (Reps. Dan/Craig Goldman), always
# adjacent to a first name — no lookbehind saves a bare surname without a name
# DB. Precision-first (same call as the INTC guard): require the FULL bank name.
# ponytail: bare "Goldman" for the bank is now a miss; add a bank-context
# allowlist ("goldman" near sachs/stock/CEO/Solomon) only if such misses recur.
GS_GUARDED = r"\bgoldman sachs\b"

# Trump SIGNS his posts "DJT" — 754 of 762 DJT "mentions" in the 2025-26 corpus were
# that sign-off, not a reference to the stock. A trailing DJT (optionally followed by
# punctuation and/or a link) is a signature. Mid-text "DJT stock is up" still matches.
#
# This REPLACES an earlier deliberate choice to let the signature match, on the theory
# that "downstream views must treat signature-only hits as non-company content". No
# downstream filter was ever written, so every consumer inherited the false positives.
# The intended use case (DJT as a Trump-exposed ticker) needs no matcher at all —
# every post in this corpus is already by Trump. tests/test_entities.py pins the guard.
DJT_TICKER_GUARDED = (r"(?<!president )(?<!\bpres\. )"
                      r"\bdjt\b(?![\s\-–—!.,:;\"')\]]*(?:https?://\S+\s*)?$)")

# "Anti Trump Media" / "the Trump media" is the PRESS, not Trump Media & Technology.
# Require the company sense: not preceded by anti-/the, not followed by press words.
DJT_MEDIA_GUARDED = (r"(?<!anti[- ])(?<!\bthe )\btrump media\b"
                     r"(?!\s+(?:coverage|outlets?|bias|machine|complex|empire|hoax))")

# ticker -> name/alias/exec/brand patterns Trump actually tweeted (word-boundaried).
ENTITY_TRIGGERS: dict[str, tuple[str, ...]] = {
    "XOM": (r"\bexxon\b", r"\bexxonmobil\b"), "CVX": (r"\bchevron\b",),
    "COP": (r"\bconoco(?:phillips)?\b",), "SLB": (r"\bschlumberger\b",), "EOG": (r"\beog\b",),
    "JPM": (r"\bjp ?morgan\b", r"\bjamie dimon\b"), "BAC": (r"\bbank of america\b",),
    "WFC": (r"\bwells fargo\b",), "GS": (GS_GUARDED,),
    "MS": (r"\bmorgan stanley\b",),
    "AAPL": (r"\bapple\b", r"\btim cook\b"), "MSFT": (r"\bmicrosoft\b",),
    "NVDA": (r"\bnvidia\b",), "AMD": (r"\bamd\b",), "AVGO": (r"\bbroadcom\b",),
    "BA": (r"\bboeing\b",), "CAT": (r"\bcaterpillar\b",), "GE": (r"\bgeneral electric\b",),
    "RTX": (r"\braytheon\b",), "UNP": (r"\bunion pacific\b",),
    "UNH": (r"\bunitedhealth\b",), "JNJ": (r"\bjohnson & johnson\b", r"\bj&j\b"),
    "PFE": (r"\bpfizer\b",), "MRK": (r"\bmerck\b",), "ABBV": (r"\babbvie\b",),
    "AMZN": (r"\bamazon\b", r"\bbezos\b"), "TSLA": (r"\btesla\b", r"\belon musk\b", r"\bmusk\b"),
    "HD": (r"\bhome depot\b",), "MCD": (r"\bmcdonald'?s?\b",), "NKE": (r"\bnike\b",),
    # ponytail: bare "intel" ("US intel flagged...") is intelligence, not the
    # chipmaker — require it NOT preceded by us/american nor followed by spy-speak.
    # ponytail: blocklist disambiguation has a ceiling — swap to corp-context
    # allowlist (intel + corp/stock/chips/ceo/fab context) if misses keep coming.
    "INTC": (INTC_GUARDED,),
    "TSM": (r"\btaiwan semiconductor\b", r"\btsmc\b"),
    "LMT": (r"\block ?heed\b", r"\bf-?35\b"), "NOC": (r"\bnorthrop\b",),
    "GD": (r"\bgeneral dynamics\b",),
    # Trump-related public holding (Trump Media, lists 2024-03; DATASEARCH §3)
    "DJT": (DJT_MEDIA_GUARDED, r"\btruth social\b", DJT_TICKER_GUARDED),
}
_ENT = {tk: tuple(re.compile(p, re.I) for p in pats) for tk, pats in ENTITY_TRIGGERS.items()}

# reverse: stock -> the sectors it belongs to (a stock can sit in two, e.g. NVDA).
_STOCK_SECTORS: dict[str, list[str]] = {}
for _etf, _names in SECTOR_STOCKS.items():
    for _n in _names:
        _STOCK_SECTORS.setdefault(_n, []).append(_etf)

MIN_RELEVANCE = 0.1


def asset_relevance(text: str) -> dict[str, float]:
    """ticker -> relevance in [0,1]. Includes ETFs and stocks; drops < MIN_RELEVANCE."""
    scores = sector_scores(text)
    total = sum(scores.values())
    rel: dict[str, float] = {}
    if total:
        for etf, s in scores.items():
            conf = s / total
            rel[etf] = max(rel.get(etf, 0.0), conf)  # ETF: full share
            # NO fan-out to member stocks: a sector word ("cars") must not link
            # AMZN/HD/MCD; equities link only via their own entity triggers.
    for tk, pats in _ENT.items():                                    # direct mention wins
        if any(p.search(text) for p in pats):
            rel[tk] = 1.0
    return {tk: round(v, 4) for tk, v in rel.items() if v >= MIN_RELEVANCE}


def is_direct_mention(text: str, ticker: str) -> bool:
    return any(p.search(text) for p in _ENT.get(ticker, ()))


# ---------------------------------------------------------------------------
# Tiered gazetteer (v4 — rollout gated on review). direct 1.0 / indirect 0.7 /
# competitor 0.5. Only AMZN + INTC are fully tiered (the review sample); every
# other ticker auto-migrates its ENTITY_TRIGGERS as direct-only until approved.
# ---------------------------------------------------------------------------
TIER_CONF: dict[str, float] = {"direct": 1.0, "indirect": 0.7, "competitor": 0.5}

# Policy-vehicle triggers fold INTO the indirect tier at a distinct, LOWER conf
# than CEO/brand triggers: a tweet naming a policy that a company depends on
# ("CHIPS Act") links the company indirectly, but weaker than naming its CEO.
# rel_reason records the specific vehicle ("policy: 'CHIPS Act'"). These do NOT
# fire competitor ride-along — a policy already spans the whole sector, so
# adding each name's rivals would double-count the sector-wide move that the
# abnormal-return (vs sector-ETF) adjustment is there to net out. Phase 2 will
# instead capture the sector/competitor reaction as FEATURES, not label rows.
POLICY_CONF: float = 0.55  # between competitor (0.5) and CEO/brand indirect (0.7)

# (regex, human label, tickers the policy vehicle links). Small + auditable;
# this mapping is the tuning knob — extend it, re-run the audit counts.
POLICY_TRIGGERS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (r"\bchips (?:and science )?act\b", "CHIPS Act",
     ("INTC", "MU", "TSM", "GFS", "TXN")),
    (r"\b(?:tariffs?|export bans?|export controls?) on (?:semiconductors?|chips)\b",
     "chip trade curbs", ("INTC", "MU", "TSM", "GFS", "TXN")),
    (r"\bsemiconductor (?:subsid\w+|manufacturing incentives?)\b",
     "semi subsidies", ("INTC", "MU", "TSM", "GFS", "TXN")),
)

TIERED_TRIGGERS: dict[str, dict[str, tuple[str, ...]]] = {
    "AMZN": {
        "direct": (r"\bamazon(?:\.com)?\b", r"\$amzn\b"),
        # CEO/execs, owned brands/products, nicknames Trump has actually used.
        # Ambiguous common words ("prime") match ONLY inside unambiguous phrases
        # — same precision-first principle as the INTC guard.
        "indirect": (r"\bbezos\b", r"\bandy jassy\b", r"\bjassy\b", r"\baws\b",
                     r"\bwashington post\b", r"\bamazon prime\b", r"\bprime video\b",
                     r"\bwhole foods\b", r"\bthe everything store\b"),
    },
    "INTC": {
        "direct": (INTC_GUARDED, r"\$intc\b", r"\bintel corp(?:oration)?\b"),
        "indirect": (r"\bpat gelsinger\b", r"\bgelsinger\b", r"\blip[- ]bu tan\b",
                     r"\bintel foundry\b"),
    },
}

# Static, hand-curated competitor sets: rows RIDE ALONG with a primary hit,
# never fire standalone. WMT/TGT/COST are outside the current bar universe —
# their bars must be fetched before any rebuild emits their rows.
COMPETITORS: dict[str, tuple[str, ...]] = {
    "AMZN": ("WMT", "TGT", "COST"),
    "INTC": ("AMD", "NVDA", "TSM"),
}


@dataclass(frozen=True)
class EntityMatch:
    ticker: str
    conf: float
    tier: str                    # direct | indirect | competitor
    reason: str                  # trigger that fired, or "competitor of X"
    competitor_of: str | None = None
    is_policy: bool = False       # indirect-via-policy-vehicle; suppresses ride-along


def _lab(pattern: str) -> str:
    return re.sub(r"\\b|\(\?[:<!=][^)]*\)|[\\()?$]", "", pattern).replace("|", "/").strip()


def _compile_tiers() -> dict[str, dict[str, tuple[tuple[re.Pattern[str], str], ...]]]:
    out: dict[str, dict[str, tuple[tuple[re.Pattern[str], str], ...]]] = {}
    for tk, pats in ENTITY_TRIGGERS.items():          # legacy: direct-only
        if tk not in TIERED_TRIGGERS:
            out[tk] = {"direct": tuple((re.compile(p, re.I), _lab(p)) for p in pats)}
    for tk, tiers in TIERED_TRIGGERS.items():
        out[tk] = {tier: tuple((re.compile(p, re.I), _lab(p)) for p in pats)
                   for tier, pats in tiers.items()}
    return out


_TIERED = _compile_tiers()
_POLICY = tuple((re.compile(p, re.I), label, tickers) for p, label, tickers in POLICY_TRIGGERS)


def entity_spans(text: str, ticker: str) -> list[tuple[int, int]]:
    """Char spans where `ticker`'s triggers fire — lets the stance scorer scope
    sentiment to the sentence around the mention (not the whole tweet)."""
    out: list[tuple[int, int]] = []
    for tier in _TIERED.get(ticker, {}).values():
        for rx, _lab in tier:
            out += [m.span() for m in rx.finditer(text)]
    return sorted(set(out))


def entity_matches(text: str) -> dict[str, EntityMatch]:
    """Tiered entity scan: best (highest-conf) tier per ticker; competitor rows
    ride along only when their primary ticker matched. Text-only, pre-t0."""
    out: dict[str, EntityMatch] = {}
    for tk, tiers in _TIERED.items():
        for tier in ("direct", "indirect"):
            hit = next((lab for rx, lab in tiers.get(tier, ()) if rx.search(text)), None)
            if hit is not None:
                out[tk] = EntityMatch(tk, TIER_CONF[tier], tier, f"{tier}: '{hit}'")
                break
    # policy vehicles -> indirect tier, lower conf; never overrides a direct or
    # CEO/brand-indirect hit already found for the same ticker.
    for rx, label, tickers in _POLICY:
        if rx.search(text):
            for tk in tickers:
                if tk not in out:
                    out[tk] = EntityMatch(tk, POLICY_CONF, "indirect",
                                          f"policy: '{label}'", is_policy=True)
    for tk, comps in COMPETITORS.items():
        if tk in out and out[tk].tier != "competitor" and not out[tk].is_policy:
            for c in comps:
                if c not in out:
                    out[c] = EntityMatch(c, TIER_CONF["competitor"], "competitor",
                                         f"competitor of {tk}", tk)
    return out
