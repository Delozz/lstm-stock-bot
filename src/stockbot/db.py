"""SQLite connection, schema (PRD Section 9), upserts, and run bookkeeping."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickers (
  symbol        TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  sector        TEXT,
  is_benchmark  INTEGER NOT NULL DEFAULT 0,
  active        INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS bars_daily (
  symbol        TEXT NOT NULL REFERENCES tickers(symbol),
  session_date  TEXT NOT NULL,
  open          REAL NOT NULL,
  high          REAL NOT NULL,
  low           REAL NOT NULL,
  close         REAL NOT NULL,
  volume        INTEGER NOT NULL,
  PRIMARY KEY (symbol, session_date)
);

CREATE TABLE IF NOT EXISTS news (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol           TEXT NOT NULL REFERENCES tickers(symbol),
  published_at_utc TEXT NOT NULL,
  session_date     TEXT NOT NULL,
  headline         TEXT NOT NULL,
  source           TEXT,
  url              TEXT,
  vader_compound   REAL NOT NULL,
  UNIQUE (symbol, published_at_utc, headline)
);
CREATE INDEX IF NOT EXISTS idx_news_symbol_session ON news (symbol, session_date);

CREATE TABLE IF NOT EXISTS features (
  symbol         TEXT NOT NULL REFERENCES tickers(symbol),
  session_date   TEXT NOT NULL,
  ret_1d REAL NOT NULL, ret_5d REAL NOT NULL, ret_20d REAL NOT NULL,
  vol_20d REAL NOT NULL, vol_z_20d REAL NOT NULL, hl_range REAL NOT NULL,
  gap REAL NOT NULL, oc_ret REAL NOT NULL,
  sent_mean REAL NOT NULL, sent_max REAL NOT NULL, sent_min REAL NOT NULL,
  news_count REAL NOT NULL, news_available INTEGER NOT NULL,
  next_oc_ret    REAL,
  label          INTEGER,
  PRIMARY KEY (symbol, session_date)
);

CREATE TABLE IF NOT EXISTS models (
  model_id       TEXT PRIMARY KEY,
  trained_at_utc TEXT NOT NULL,
  train_end_date TEXT NOT NULL,
  seed           INTEGER NOT NULL,
  git_sha        TEXT NOT NULL,
  val_loss       REAL NOT NULL,
  artifact_dir   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
  session_date   TEXT NOT NULL,
  symbol         TEXT NOT NULL REFERENCES tickers(symbol),
  model_id       TEXT NOT NULL REFERENCES models(model_id),
  p_up           REAL NOT NULL,
  rank           INTEGER NOT NULL,
  selected       INTEGER NOT NULL,
  PRIMARY KEY (session_date, symbol, model_id)
);

CREATE TABLE IF NOT EXISTS orders (
  client_order_id  TEXT PRIMARY KEY,
  alpaca_order_id  TEXT,
  session_date     TEXT NOT NULL,
  symbol           TEXT NOT NULL,
  side             TEXT NOT NULL CHECK (side IN ('buy','sell')),
  qty              INTEGER NOT NULL,
  time_in_force    TEXT NOT NULL CHECK (time_in_force IN ('opg','cls')),
  status           TEXT NOT NULL,
  submitted_at_utc TEXT NOT NULL,
  filled_qty       INTEGER,
  filled_avg_price REAL,
  filled_at_utc    TEXT
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
  session_date   TEXT PRIMARY KEY,
  equity         REAL NOT NULL,
  cash           REAL NOT NULL,
  realized_pnl   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  command         TEXT NOT NULL,
  started_at_utc  TEXT NOT NULL,
  finished_at_utc TEXT,
  status          TEXT,
  exit_code       INTEGER,
  error_summary   TEXT
);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def upsert(
    conn: sqlite3.Connection,
    table: str,
    rows: Iterable[Mapping[str, Any]],
    key: Sequence[str],
    *,
    update: bool = True,
) -> int:
    """Insert rows, updating non-key columns on conflict (or ignoring them if update=False).

    Table and column names come from this codebase, never from user input; values are
    always bound as parameters.
    """
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0])
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))}) "
        f"ON CONFLICT ({', '.join(key)}) DO "
    )
    non_key = [c for c in cols if c not in key]
    if update and non_key:
        sql += "UPDATE SET " + ", ".join(f"{c} = excluded.{c}" for c in non_key)
    else:
        sql += "NOTHING"
    cur = conn.executemany(sql, [tuple(r[c] for c in cols) for r in rows])
    conn.commit()
    return cur.rowcount


def start_run(conn: sqlite3.Connection, command: str) -> int:
    cur = conn.execute(
        "INSERT INTO runs (command, started_at_utc) VALUES (?, ?)", (command, utc_now())
    )
    conn.commit()
    return cur.lastrowid


def finish_run(
    conn: sqlite3.Connection, run_id: int, exit_code: int, error_summary: str | None = None
) -> None:
    conn.execute(
        "UPDATE runs SET finished_at_utc = ?, status = ?, exit_code = ?, error_summary = ? "
        "WHERE id = ?",
        (utc_now(), "ok" if exit_code == 0 else "error", exit_code, error_summary, run_id),
    )
    conn.commit()
