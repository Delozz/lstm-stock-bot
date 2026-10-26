"""AC-28: refuse to signal when too many tickers are stale."""

from __future__ import annotations

import pytest

from stockbot.config import DataError
from stockbot.db import upsert
from stockbot.signals import check_staleness

LATEST, OLD = "2026-10-23", "2026-10-22"


def _universe(conn, n_stale: int, total: int = 100) -> None:
    upsert(
        conn,
        "tickers",
        [
            {"symbol": f"S{i:03d}", "name": "x", "sector": None, "is_benchmark": 0}
            for i in range(total)
        ],
        ("symbol",),
    )
    upsert(
        conn,
        "bars_daily",
        [
            {
                "symbol": f"S{i:03d}",
                "session_date": OLD if i < n_stale else LATEST,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1,
            }
            for i in range(total)
        ],
        ("symbol", "session_date"),
    )


def test_six_percent_stale_refuses(conn):
    _universe(conn, n_stale=6)
    with pytest.raises(DataError):
        check_staleness(conn, LATEST, 0.05)


def test_four_percent_stale_proceeds(conn):
    _universe(conn, n_stale=4)
    assert len(check_staleness(conn, LATEST, 0.05)) == 4
