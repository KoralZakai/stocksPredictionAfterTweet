"""Nebius Serverless AI Endpoint: POST /predict — the SHIPPED live path.

Serves the VALIDATED raw Llama-3.3-70B zero-shot call. Two planes, one-way flow
(the leakage firewall):

  DECISION PLANE   input = tweet text ONLY -> classify -> route -> LONG/SHORT/ABSTAIN.
                   Never sees a price, quote, or session phase.
  MARKET PLANE     runs AFTER the decision, on the resolved tickers only, to enrich
                   the response. Never feeds back into the decision. Any failure ->
                   market_context: null, never a 5xx, decision unchanged.

Boot pins (REFUSE TO START on mismatch):
  - alpha.classify.prompt_template_hash() == manifest.prompt_template_hash
    (the live model is the one that was validated),
  - sha256(corpus file) == manifest.corpus.sha256 (the data is the one measured).
No hit-rate is hardcoded here: the Job produces the numbers, the Endpoint cites them.

Imports alpha/ and market/ only — NOTHING from scripts/.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI
from pydantic import BaseModel

from alpha.benchmark import daily_returns, _session_anchor, session_phase
from alpha.classify import prompt_template_hash
from alpha.profiles import PROFILES, Profile, active_profile
from alpha.route import route_decision
from alpha.schema import MarketContext, Quote, RealizedAlpha
from market import select_provider
from market.provider import Provider

log = logging.getLogger("predict")
DISCLAIMER = "Research output. Not investment advice."
# Emitted for every tweet when the manifest ships no horizon (nothing survived BH
# correction). The endpoint still runs and explains itself — it just refuses to call.
NO_VALIDATED_EDGE = (
    "no validated edge: no horizon survived Benjamini-Hochberg correction over the "
    "test registry, so this deployment ships no tradeable horizon (shipped_horizons "
    "is empty). The scenario and reasoning are research output, not a call."
)
MARKET_TIMEOUT_S = 1.5
_QUOTE_TTL_S = 15.0


# ---------------------------------------------------------------- boot verification
def verify_manifest(manifest: dict[str, Any],
                    prompt_hash_fn: Callable[[], str] | None = None) -> None:
    """Refuse to start unless the live code + data match what the manifest certifies.

    `prompt_hash_fn` defaults to the frozen stable prompt hash; alpha.profiles passes
    the ACTIVE profile's hash so Mode B is checked against its own manifest. The
    pairing is what stops a profile's prompt being served against another's numbers.
    """
    live_prompt = (prompt_hash_fn or prompt_template_hash)()
    if manifest.get("prompt_template_hash") != live_prompt:
        raise RuntimeError(
            "PROMPT HASH MISMATCH: live classification prompt "
            f"({live_prompt[:12]}...) != manifest ({str(manifest.get('prompt_template_hash'))[:12]}...). "
            "The served model is not the one that was validated — refusing to start.")
    corpus = manifest.get("corpus", {})
    cfile = Path(str(corpus.get("file", "")))
    if not cfile.exists():
        raise RuntimeError(f"CORPUS MISSING: cannot verify {cfile} against the manifest.")
    actual = sha256(cfile.read_bytes()).hexdigest()
    if actual != corpus.get("sha256"):
        raise RuntimeError(
            f"CORPUS HASH MISMATCH: {cfile} ({actual[:12]}...) != manifest "
            f"({str(corpus.get('sha256'))[:12]}...) — refusing to start.")


# ---------------------------------------------------------------- decision plane
def _default_classify(text: str) -> dict[str, Any]:
    """Real Nebius zero-shot classification (tweet text only) — stable profile."""
    return _profile_classify(PROFILES["stable"], text)


def _profile_classify(prof: Profile, text: str) -> dict[str, Any]:
    """Run the ACTIVE profile's classifier. Tweet text only — the decision plane."""
    from alpha.classify import DEFAULT_BASE, DEFAULT_MODEL
    from alpha.env import env, load_dotenv
    load_dotenv()
    api_key = env("NEBIUS_API_KEY", "EXPO_PUBLIC_NEBIUS_API_KEY")
    if not api_key:
        raise RuntimeError("No NEBIUS_API_KEY for classification.")
    model = env("NEBIUS_MODEL", "EXPO_PUBLIC_NEBIUS_MODEL", default=DEFAULT_MODEL)
    base = env("NEBIUS_BASE_URL", "EXPO_PUBLIC_NEBIUS_BASE_URL", default=DEFAULT_BASE)
    return prof.classify(text, base_url=base, api_key=api_key, model=model)


def _parse_t0(t0_utc: str) -> datetime | None:
    """Parse the request timestamp. None -> unanchorable -> ABSTAIN (decision-plane
    safe: t0 is a request field, not market-derived)."""
    try:
        dt = datetime.fromisoformat(t0_utc.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------- market plane
class _Breaker:
    """Minimal circuit breaker: after `trip` consecutive failures, skip the market
    plane for `cooldown` seconds. ponytail: a global counter, not per-provider — a
    single upstream is all we call; upgrade to per-ticker if that changes."""

    def __init__(self, trip: int = 3, cooldown: float = 30.0) -> None:
        self.trip, self.cooldown = trip, cooldown
        self.fails = 0
        self.open_until = 0.0

    def is_open(self) -> bool:
        return time.monotonic() < self.open_until

    def record(self, ok: bool) -> None:
        if ok:
            self.fails = 0
        else:
            self.fails += 1
            if self.fails >= self.trip:
                self.open_until = time.monotonic() + self.cooldown


class MarketPlane:
    """Post-decision enrichment. Owns the timeout, TTL cache and breaker. Guarantees
    it can only ever RETURN None or a MarketContext — it cannot alter a decision."""

    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        self._cache: dict[str, tuple[float, float | None]] = {}
        self._breaker = _Breaker()
        self._pool = ThreadPoolExecutor(max_workers=4)

    def quote(self, ticker: str) -> float | None:
        hit = self._cache.get(ticker)
        now = time.monotonic()
        if hit and now - hit[0] < _QUOTE_TTL_S:
            return hit[1]
        val = self.provider.quote(ticker)
        self._cache[ticker] = (now, val)
        return val

    def context(self, instruments: list[Any], t0: datetime) -> MarketContext | None:
        if self._breaker.is_open():
            return None
        try:
            fut = self._pool.submit(self._build, instruments, t0)
            mc = fut.result(timeout=MARKET_TIMEOUT_S)   # raises TimeoutError past the deadline
            self._breaker.record(True)
            return mc
        except Exception as e:  # noqa: BLE001 - ANY failure (incl. timeout) -> null context
            self._breaker.record(False)
            log.warning("market plane failed (%s) -> market_context=null", type(e).__name__)
            return None

    def _build(self, instruments: list[Any], t0: datetime) -> MarketContext:
        now = datetime.now(timezone.utc)
        spy_last = self.quote("SPY")
        quotes: list[Quote] = []
        for ins in instruments:
            last = self.quote(ins.ticker)
            if last is not None:
                quotes.append(Quote(ins.ticker, last, "SPY", spy_last or 0.0))
        realized = self._realized(instruments, t0) if (
            self.provider.name != "null" and t0 < now) else None
        return MarketContext(
            as_of_utc=now.replace(microsecond=0).isoformat(),
            provider=self.provider.name,
            session_phase=session_phase(t0),
            entry_anchor_utc=_session_anchor(t0).replace(tzinfo=timezone.utc).isoformat(),
            quotes=quotes,
            realized_alpha_since_t0=realized,
        )

    def _realized(self, instruments: list[Any], t0: datetime) -> list[RealizedAlpha] | None:
        """Post-hoc REALIZED alpha vs SPY (never PREDICTED). Daily horizons only."""
        spy = daily_returns("SPY", t0)
        if not spy:
            return None
        out: list[RealizedAlpha] = []
        for ins in instruments:
            ret = daily_returns(ins.ticker, t0)
            for h, sret in spy.items():
                if h in ret:
                    out.append(RealizedAlpha(ins.ticker, h, ret[h], sret, ret[h] - sret,
                                             beat=(ret[h] - sret > 0) == (ins.direction == "up")))
        return out or None


# ---------------------------------------------------------------- response assembly
def _cohort(manifest: dict[str, Any], decision: str) -> dict[str, Any] | None:
    shipped = manifest.get("shipped_horizons") or []
    if decision == "ABSTAIN" or not shipped:
        return None
    h = shipped[0]
    e = manifest["horizons"][h]
    return {
        "value": e["hit_rate_test"], "ci95": e["ci95"], "n": e["n_test"], "horizon": h,
        "note": ("Historical hit-rate of ALL calls of this type on a held-out chronological "
                 "test set. This is NOT a probability for THIS tweet. We tested per-tweet "
                 "confidence; it did not generalize."),
    }


# ---------------------------------------------------------------- app factory
class PredictIn(BaseModel):
    tweet_text: str
    t0_utc: str = ""
    author: str = ""


def create_app(*, manifest_path: str | None = None,
               classify_fn: Callable[[str], dict[str, Any]] | None = None,
               provider: Provider | None = None,
               profile: Profile | None = None) -> FastAPI:
    """Build the app for the ACTIVE profile (env SIGNAL_PROFILE, default 'stable').

    Path precedence: MANIFEST_PATH env > explicit manifest_path arg > profile default.
    Mode A (default) is byte-identical to before profiles existed.
    """
    prof = profile or active_profile()
    mpath = Path(os.environ.get("MANIFEST_PATH", manifest_path or prof.manifest_path))
    if not mpath.exists():
        raise RuntimeError(f"MANIFEST MISSING: {mpath} — the Endpoint refuses to start.")
    manifest: dict[str, Any] = json.loads(mpath.read_text())
    verify_manifest(manifest, prompt_hash_fn=prof.prompt_hash)   # drift -> refuse to start

    classify = classify_fn or (lambda text: _profile_classify(prof, text))
    market = MarketPlane(provider or select_provider())
    shipped = manifest.get("shipped_horizons") or []
    app = FastAPI(title=f"tweet-alpha /predict [{prof.name}]")

    @app.post("/predict")
    def predict(req: PredictIn) -> dict[str, Any]:
        t0 = _parse_t0(req.t0_utc)
        if t0 is None:
            return _abstain(manifest, "t0 unanchorable (bad or missing t0_utc)")

        # ---- DECISION PLANE: tweet text ONLY ----
        classified = classify(req.tweet_text)
        routed = route_decision(classified, whitelist=prof.whitelist)

        # No horizon survived BH correction over the registry => there is no validated
        # edge, so there is no call to make. Refuse EVERY tweet, not just the marginal
        # ones: a null result must not be dressed up as a signal (CLAUDE.md 4, 8). The
        # classification is still returned as research output — it explains what the
        # model saw, while `decision` refuses to act on it.
        if not shipped:
            out = _abstain(manifest, NO_VALIDATED_EDGE)
            out["scenario"] = routed.scenario
            out["reasoning"] = routed.reasoning
            return out

        horizon = shipped[0]
        resp: dict[str, Any] = {
            "decision": routed.decision,
            "instruments": [asdict(i) for i in routed.instruments],
            "scenario": routed.scenario,
            "reasoning": routed.reasoning,
            "horizon": horizon,
            "cohort_base_rate": _cohort(manifest, routed.decision),
            "manifest_version": manifest.get("code_rev"),
            "disclaimer": DISCLAIMER,
        }
        if routed.decision == "ABSTAIN":
            resp["abstain_reason"] = routed.abstain_reason

        # ---- MARKET PLANE: after the fact, nullable, never feeds back ----
        mc = market.context(routed.instruments, t0) if routed.instruments else None
        resp["market_context"] = asdict(mc) if mc else None
        return resp

    @app.get("/market/{ticker}")
    def market_quote(ticker: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        last = market.quote(ticker.upper())
        spy = market.quote("SPY")
        return {"ticker": ticker.upper(), "last": last, "benchmark_ticker": "SPY",
                "benchmark_last": spy, "session_phase": session_phase(now),
                "provider": market.provider.name, "as_of_utc": now.replace(microsecond=0).isoformat()}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "profile": prof.name,
            "experimental": prof.experimental,
            "manifest_path": str(mpath),
            "manifest_version": manifest.get("code_rev"),
            "corpus_sha256": manifest.get("corpus", {}).get("sha256"),
            "prompt_template_hash": manifest.get("prompt_template_hash"),
            "shipped_horizons": shipped,
            "market_provider": market.provider.name,
        }

    return app


def _abstain(manifest: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "decision": "ABSTAIN", "instruments": [], "scenario": "", "reasoning": "",
        "abstain_reason": reason, "horizon": (manifest.get("shipped_horizons") or [None])[0],
        "cohort_base_rate": None, "market_context": None,
        "manifest_version": manifest.get("code_rev"), "disclaimer": DISCLAIMER,
    }


# uvicorn entrypoint: `uvicorn serving.app:app`. Import-time boot verification means
# a drifted prompt / corpus / missing manifest fails fast, before serving traffic.
app = create_app() if os.environ.get("PREDICT_SKIP_BOOT") != "1" else None
