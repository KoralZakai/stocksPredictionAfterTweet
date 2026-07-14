"""yfinance provider: last daily close, no API key. The default when Alpaca keys
are absent. Best-effort — any failure returns None (never raises to the caller)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


class YFinanceProvider:
    name = "yfinance"

    def quote(self, ticker: str) -> float | None:
        try:
            import yfinance as yf
            end = datetime.now(timezone.utc).date() + timedelta(days=1)
            start = end - timedelta(days=8)
            df = yf.download(ticker, start=start, end=end, interval="1d",
                             auto_adjust=True, progress=False)
            if df is None or df.empty:
                return None
            close = df["Close"]
            val = close.iloc[-1]
            # single-ticker frames may hand back a 1-cell Series
            return float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
        except Exception:
            return None
