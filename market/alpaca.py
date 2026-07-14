"""Alpaca IEX provider: latest trade price. Keys from env (paper key works).

Reads ALPACA_API_KEY / ALPACA_API_SECRET (or the APCA_* aliases). Best-effort:
any failure returns None. Constructing without keys raises so select_provider can
fall through to yfinance.
"""

from __future__ import annotations

import os

import requests

_DATA_URL = "https://data.alpaca.markets/v2/stocks/{sym}/trades/latest"


class AlpacaProvider:
    name = "alpaca"

    def __init__(self) -> None:
        self._key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
        self._secret = os.environ.get("ALPACA_API_SECRET") or os.environ.get("APCA_API_SECRET_KEY")
        if not (self._key and self._secret):
            raise RuntimeError("Alpaca keys absent")

    def quote(self, ticker: str) -> float | None:
        try:
            resp = requests.get(
                _DATA_URL.format(sym=ticker.upper()),
                headers={"APCA-API-KEY-ID": self._key or "",
                         "APCA-API-SECRET-KEY": self._secret or ""},
                params={"feed": "iex"},
                timeout=1.5,
            )
            if resp.status_code != 200:
                return None
            price = resp.json().get("trade", {}).get("p")
            return float(price) if price is not None else None
        except Exception:
            return None
