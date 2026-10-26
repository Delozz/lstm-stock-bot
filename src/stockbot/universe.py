"""Ticker universe from data/sp500.csv plus the SPY benchmark (FR-7)."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from stockbot.db import upsert


def load_universe(conn: sqlite3.Connection, csv_path: str | Path, benchmark: str) -> int:
    with Path(csv_path).open(newline="") as f:
        rows = [
            {
                "symbol": r["symbol"].strip().upper(),
                "name": r["name"].strip(),
                "sector": r["sector"].strip() or None,
                "is_benchmark": 0,
            }
            for r in csv.DictReader(f)
        ]
    rows.append(
        {"symbol": benchmark, "name": f"{benchmark} (benchmark)", "sector": None, "is_benchmark": 1}
    )
    upsert(conn, "tickers", rows, ("symbol",))
    return len(rows)


def active_symbols(conn: sqlite3.Connection, *, include_benchmark: bool = False) -> list[str]:
    sql = "SELECT symbol FROM tickers WHERE active = 1"
    if not include_benchmark:
        sql += " AND is_benchmark = 0"
    return [r[0] for r in conn.execute(sql + " ORDER BY symbol")]


def deactivate(conn: sqlite3.Connection, symbol: str) -> None:
    conn.execute("UPDATE tickers SET active = 0 WHERE symbol = ?", (symbol,))
    conn.commit()
