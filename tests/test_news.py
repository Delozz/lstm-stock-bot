"""AC-10 to AC-13: throttle, retries, VADER, session mapping."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from stockbot.db import upsert
from stockbot.news import RateLimiter, assign_session, ingest_news
from stockbot.retry import with_retries

# 2026-03-08 is the US switch to daylight time.
TRADING = [date(2026, 3, 5), date(2026, 3, 6), date(2026, 3, 9), date(2026, 3, 10)]


@pytest.mark.parametrize(
    "utc, expected",
    [
        (datetime(2026, 3, 6, 20, 59, tzinfo=UTC), date(2026, 3, 6)),  # Fri 15:59 EST
        (datetime(2026, 3, 6, 21, 1, tzinfo=UTC), date(2026, 3, 9)),  # Fri 16:01 EST
        (datetime(2026, 3, 7, 15, 0, tzinfo=UTC), date(2026, 3, 9)),  # Sat 10:00 EST
        (datetime(2026, 3, 9, 19, 30, tzinfo=UTC), date(2026, 3, 9)),  # Mon 15:30 EDT
        (datetime(2026, 3, 9, 20, 30, tzinfo=UTC), date(2026, 3, 10)),  # Mon 16:30 EDT
    ],
)
def test_assign_session(utc, expected):
    assert assign_session(utc, TRADING) == expected


def test_rate_limiter_never_exceeds_50_per_minute():
    now = [0.0]
    limiter = RateLimiter(
        50, 60, clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s)
    )
    stamps = []
    for _ in range(120):
        limiter.wait()
        stamps.append(now[0])
    assert now[0] >= 120
    assert max(sum(1 for t in stamps if s <= t < s + 60) for s in stamps) <= 50


class TooMany(Exception):
    status_code = 429


def test_retry_backoff_then_success():
    calls, sleeps = [0], []

    def flaky():
        calls[0] += 1
        if calls[0] <= 3:
            raise TooMany()
        return "ok"

    assert with_retries(flaky, sleep=sleeps.append) == "ok"
    assert sleeps == [1, 2, 4]


def test_retry_gives_up_after_five_delays():
    sleeps = []

    def always():
        raise TooMany()

    with pytest.raises(TooMany):
        with_retries(always, sleep=sleeps.append)
    assert sleeps == [1, 2, 4, 8, 16]


class FakeFinnhub:
    def __init__(self, items, failing=()):
        self.items, self.failing, self.calls = items, set(failing), []

    def company_news(self, symbol, _from, to):
        self.calls.append((symbol, _from, to))
        if symbol in self.failing:
            raise TooMany()
        return self.items


def test_ingest_continues_past_failed_symbol_and_scores_with_vader(conn):
    upsert(
        conn,
        "tickers",
        [{"symbol": s, "name": s, "sector": None, "is_benchmark": 0} for s in ("AAA", "BBB")],
        ("symbol",),
    )
    headline = "Company beats earnings estimates and raises guidance"
    stamp = int(datetime(2026, 3, 6, 15, 0, tzinfo=UTC).timestamp())
    client = FakeFinnhub(
        [{"datetime": stamp, "headline": headline, "source": "x", "url": "u"}], failing={"AAA"}
    )
    now = [0.0]
    written, failed = ingest_news(
        conn,
        client,
        trading_days=TRADING,
        today=date(2026, 3, 6),
        default_start=date(2026, 3, 1),
        chunk_days=30,
        limiter=RateLimiter(50, clock=lambda: now[0], sleep=lambda s: None),
        sleep=lambda s: None,
    )
    assert failed == ["AAA"]
    assert written == 1
    stored = conn.execute("SELECT session_date, vader_compound FROM news").fetchone()
    assert stored[0] == "2026-03-06"
    expected = SentimentIntensityAnalyzer().polarity_scores(headline)["compound"]
    assert stored[1] == pytest.approx(expected, abs=1e-9)
