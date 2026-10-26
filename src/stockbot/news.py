"""Finnhub company-news ingestion with throttling and session mapping (FR-11 to FR-15)."""

from __future__ import annotations

import bisect
import logging
import sqlite3
import time
from collections import deque
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime

from stockbot.calendar import NY
from stockbot.db import upsert
from stockbot.retry import with_retries
from stockbot.sentiment import score
from stockbot.universe import active_symbols

log = logging.getLogger(__name__)

MARKET_CLOSE = dtime(16, 0)


class RateLimiter:
    """Sliding-window limiter: at most max_calls in any period seconds."""

    def __init__(
        self,
        max_calls: int,
        period: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_calls = max_calls
        self.period = period
        self.clock = clock
        self.sleep = sleep
        self._calls: deque[float] = deque()

    def wait(self) -> None:
        now = self.clock()
        while self._calls and now - self._calls[0] >= self.period:
            self._calls.popleft()
        if len(self._calls) >= self.max_calls:
            self.sleep(self.period - (now - self._calls[0]))
            now = self.clock()
            while self._calls and now - self._calls[0] >= self.period:
                self._calls.popleft()
        self._calls.append(now)


def assign_session(published_utc: datetime, trading_days: list[date]) -> date | None:
    """Map a headline to the session whose close it can influence.

    At or before 16:00 ET on a trading day -> that day. Otherwise -> next trading day.
    """
    et = published_utc.astimezone(NY)
    day = et.date()
    i = bisect.bisect_left(trading_days, day)
    if i < len(trading_days) and trading_days[i] == day and et.time() <= MARKET_CLOSE:
        return day
    j = bisect.bisect_right(trading_days, day)
    return trading_days[j] if j < len(trading_days) else None


def date_chunks(start: date, end: date, days: int) -> Iterator[tuple[date, date]]:
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=days - 1), end)
        yield cur, stop
        cur = stop + timedelta(days=1)


def _rows(symbol: str, items: list[dict], trading_days: list[date]) -> list[dict]:
    rows = []
    for item in items:
        headline = (item.get("headline") or "").strip()
        if not headline or not item.get("datetime"):
            continue
        published = datetime.fromtimestamp(item["datetime"], UTC)
        session = assign_session(published, trading_days)
        if session is None:
            continue
        rows.append(
            {
                "symbol": symbol,
                "published_at_utc": published.isoformat(),
                "session_date": session.isoformat(),
                "headline": headline,
                "source": item.get("source"),
                "url": item.get("url"),
                "vader_compound": score(headline),
            }
        )
    return rows


def ingest_news(
    conn: sqlite3.Connection,
    client,
    *,
    trading_days: list[date],
    today: date,
    default_start: date,
    chunk_days: int,
    limiter: RateLimiter,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[int, list[str]]:
    """Fetch headlines since the last stored date per symbol. Rows are never deleted."""
    last = dict(
        conn.execute(
            "SELECT symbol, MAX(substr(published_at_utc, 1, 10)) FROM news GROUP BY symbol"
        ).fetchall()
    )
    written, failed = 0, []
    for symbol in active_symbols(conn):
        start = date.fromisoformat(last[symbol]) if symbol in last else default_start
        rows: list[dict] = []
        try:
            for lo, hi in date_chunks(start, today, chunk_days):

                def call(symbol=symbol, lo=lo, hi=hi):
                    limiter.wait()
                    return client.company_news(symbol, _from=lo.isoformat(), to=hi.isoformat())

                rows.extend(_rows(symbol, with_retries(call, sleep=sleep), trading_days))
        except Exception as exc:
            log.error("news failed for %s: %s", symbol, exc)
            failed.append(symbol)
            continue
        # First insert wins on duplicate (symbol, timestamp, headline).
        upsert(conn, "news", rows, ("symbol", "published_at_utc", "headline"), update=False)
        written += len(rows)
    log.info("headlines fetched: %d, failed symbols: %d", written, len(failed))
    return written, failed
