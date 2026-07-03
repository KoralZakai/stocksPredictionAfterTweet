"""DuckDB + parquet store (§5 data/storage).

Fail-fast invariants enforced here, atomically:
  * tz-naive timestamps are rejected before any write (§12).
  * duplicate (author, timestamp) tweets are rejected — the whole batch rolls
    back, so a bad row never lands (§data/storage).
Schema-versioned; ASOF joins (step 3) will build on the same tables.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timezone

import duckdb

from config.settings import SETTINGS
from data.sources.interfaces import DailyBar, Tweet


def _tz_aware(dt: datetime) -> bool:
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None


# ponytail: store instants as naive-UTC TIMESTAMP, not TIMESTAMPTZ — duckdb's
# tz type needs pytz at read time. tz-awareness is enforced at ingest and
# reattached on load, so everything in Python stays tz-aware UTC (§12).
def _to_naive_utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _from_naive_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc)


class Store:
    def __init__(self, path: str = ":memory:") -> None:
        self.con = duckdb.connect(path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.con.execute("CREATE TABLE IF NOT EXISTS meta(schema_version INTEGER)")
        if not self.con.execute("SELECT count(*) FROM meta").fetchone()[0]:
            self.con.execute("INSERT INTO meta VALUES (?)", [SETTINGS.schema_version])
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS tweets("
            "tweet_id VARCHAR, author VARCHAR, text VARCHAR, "
            "timestamp_utc TIMESTAMP, is_retweet BOOLEAN, is_deleted BOOLEAN)"
        )
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS bars("
            "ticker VARCHAR, session_date TIMESTAMP, open DOUBLE, high DOUBLE, "
            "low DOUBLE, close DOUBLE, volume BIGINT)"
        )

    # --- ingest (atomic; validation before commit) ---------------------------

    def ingest_tweets(self, tweets: Iterable[Tweet]) -> None:
        rows = []
        for t in tweets:
            if not _tz_aware(t.timestamp_utc):
                raise ValueError(f"tz-naive tweet timestamp: {t.tweet_id}")
            rows.append(
                [t.tweet_id, t.author, t.text, _to_naive_utc(t.timestamp_utc),
                 t.is_retweet, t.is_deleted]
            )
        self._atomic_insert("tweets", "(?,?,?,?,?,?)", rows, self._assert_no_dup_tweets)

    def ingest_bars(self, bars: Iterable[DailyBar]) -> None:
        rows = []
        for b in bars:
            if not _tz_aware(b.session_date):
                raise ValueError(f"tz-naive bar session_date: {b.ticker} {b.session_date}")
            rows.append(
                [b.ticker, _to_naive_utc(b.session_date), b.open, b.high, b.low,
                 b.close, b.volume]
            )
        self._atomic_insert("bars", "(?,?,?,?,?,?,?)", rows, self._assert_no_dup_bars)

    def _atomic_insert(
        self,
        table: str,
        placeholders: str,
        rows: list[list[object]],
        check: Callable[[], None],
    ) -> None:
        self.con.begin()
        try:
            self.con.executemany(f"INSERT INTO {table} VALUES {placeholders}", rows)
            check()
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    def _assert_no_dup_tweets(self) -> None:
        dup = self.con.execute(
            "SELECT author, timestamp_utc FROM tweets GROUP BY 1,2 HAVING count(*)>1"
        ).fetchall()
        if dup:
            raise ValueError(f"duplicate (author,timestamp): {dup}")

    def _assert_no_dup_bars(self) -> None:
        dup = self.con.execute(
            "SELECT ticker, session_date FROM bars GROUP BY 1,2 HAVING count(*)>1"
        ).fetchall()
        if dup:
            raise ValueError(f"duplicate (ticker,session_date): {dup}")

    # --- load ---------------------------------------------------------------

    def load_tweets(self) -> list[Tweet]:
        return [
            Tweet(tid, author, text, _from_naive_utc(ts), is_rt, is_del)
            for tid, author, text, ts, is_rt, is_del in self.con.execute(
                "SELECT * FROM tweets ORDER BY timestamp_utc"
            ).fetchall()
        ]

    def load_bars(self) -> list[DailyBar]:
        return [
            DailyBar(tick, _from_naive_utc(sd), o, h, low, c, vol)
            for tick, sd, o, h, low, c, vol in self.con.execute(
                "SELECT * FROM bars ORDER BY ticker, session_date"
            ).fetchall()
        ]
