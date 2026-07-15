"""Live post listener — ARCHITECTURE + STUBS. EXPERIMENTAL, not a shipped scraper.

Scope + ethics (read before extending):
- CLAUDE.md 2 parks real-time ingestion/streaming to a later phase; this file is a
  design skeleton, not a running service.
- We do NOT ship a scraper that evades bot detection or violates a platform's ToS.
  Plug in a SOURCE YOU ARE AUTHORISED TO USE: the official X/Truth API with your own
  keys, a public RSS feed, or a webhook you control. The unauthorised-scrape path is
  intentionally left as NotImplementedError.

Flow: a SourceAdapter yields new posts -> dedupe -> POST each to the live endpoint.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Protocol

import requests


class Post(Protocol):
    id: str
    text: str
    author: str
    t0_utc: str


class SourceAdapter(Protocol):
    """Yield posts newer than the last seen id. Implement ONE you're authorised to use."""

    def poll(self) -> Iterable[dict[str, Any]]: ...


class RSSAdapter:
    """Poll a public RSS/Atom feed (a legitimate, ToS-friendly source when available).

    Stub: wire your feed parser here (e.g. stdlib xml.etree or feedparser). Return a
    list of {id, text, author, t0_utc} dicts.
    """

    def __init__(self, feed_url: str) -> None:
        self.feed_url = feed_url

    def poll(self) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "Wire a real RSS/Atom parser for a feed you are authorised to read.")


class WebhookAdapter:
    """Preferred: a platform/webhook you control pushes posts to you (no polling).

    Stub: in production this is a receiver (its own FastAPI route) that enqueues posts;
    the loop below then drains the queue. Left unimplemented on purpose.
    """

    def poll(self) -> list[dict[str, Any]]:
        raise NotImplementedError("Run a webhook receiver you control and drain its queue here.")


class UnauthorisedScrapeAdapter:
    """Intentionally NOT implemented. Scraping X/Truth Social directly can violate ToS
    and trip bot detection; we will not build that."""

    def poll(self) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "Direct scraping of X/Truth Social is out of scope (ToS / bot-detection). "
            "Use the official API with your keys, RSS, or a webhook instead.")


def post_to_endpoint(endpoint_url: str, post: dict[str, Any], *, token: str = "",
                     timeout: float = 5.0) -> dict[str, Any]:
    """POST one post to the live /live-predict endpoint. Real, usable."""
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    r = requests.post(
        f"{endpoint_url.rstrip('/')}/live-predict",
        json={"tweet_text": post["text"], "t0_utc": post.get("t0_utc", ""),
              "author": post.get("author", "")},
        headers=headers, timeout=timeout)
    r.raise_for_status()
    result: dict[str, Any] = r.json()
    return result


def run_loop(adapter: SourceAdapter, endpoint_url: str, *, token: str = "",
             interval_s: float = 30.0, max_iterations: int | None = None) -> None:
    """Poll -> dedupe -> forward. `max_iterations` bounds it for tests/demos.

    Note: `sleep` runs only in a real long-lived deployment. Respect the source's
    rate limits; do not tighten `interval_s` to hammer a platform.
    """
    seen: set[str] = set()
    it = 0
    while max_iterations is None or it < max_iterations:
        for post in adapter.poll():
            pid = str(post.get("id", ""))
            if pid and pid in seen:
                continue
            seen.add(pid)
            try:
                out = post_to_endpoint(endpoint_url, post, token=token)
                print(f"[listener] {pid}: {out.get('abstained') and 'ABSTAIN' or 'analysed'}")
            except Exception as exc:                     # a bad post must not kill the loop
                print(f"[listener] {pid}: forward failed: {exc}")
        it += 1
        if max_iterations is None or it < max_iterations:
            time.sleep(interval_s)


if __name__ == "__main__":
    raise SystemExit("Stub service. Implement an authorised SourceAdapter, then call run_loop().")
