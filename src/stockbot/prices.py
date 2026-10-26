"""Daily bar ingestion from Alpaca market data, IEX feed, fully adjusted (FR-8 to FR-10)."""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, time

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from stockbot.calendar import NY
from stockbot.db import upsert
from stockbot.retry import with_retries
from stockbot.universe import active_symbols, deactivate

log = logging.getLogger(__name__)

BAR_COLUMNS = ["symbol", "session_date", "open", "high", "low", "close", "volume"]
SPLIT_TOLERANCE = 0.005


def fetch_bars(client, symbols: list[str], start: date, end: date) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=datetime.combine(start, time.min, NY),
        end=datetime.combine(end, time.max, NY),
        adjustment=Adjustment.ALL,
        feed=DataFeed.IEX,
    )
    df = with_retries(lambda: client.get_stock_bars(req).df)
    if df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    df = df.reset_index()
    # Daily bars are stamped at midnight exchange time; convert before taking the date.
    df["session_date"] = df["timestamp"].dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
    df["volume"] = df["volume"].round().astype("int64")
    return df[BAR_COLUMNS]


def _write(conn: sqlite3.Connection, bars: pd.DataFrame) -> int:
    if bars.empty:
        return 0
    return upsert(
        conn, "bars_daily", bars.astype(object).to_dict("records"), ("symbol", "session_date")
    )


def _stored_close(conn: sqlite3.Connection, symbol: str, session_date: str) -> float:
    row = conn.execute(
        "SELECT close FROM bars_daily WHERE symbol = ? AND session_date = ?",
        (symbol, session_date),
    ).fetchone()
    return row[0]


def ingest_prices(
    conn: sqlite3.Connection,
    client,
    *,
    end: date,
    default_start: date,
    batch_size: int = 100,
) -> tuple[int, list[str]]:
    """Incrementally upsert bars for every active ticker. Returns (bars written, failed symbols).

    Each batch re-requests the last stored bar per symbol. If its adjusted close moved by
    more than 0.5%, a split or dividend re-based history and the symbol is refetched in full.
    """
    symbols = active_symbols(conn, include_benchmark=True)
    last = dict(
        conn.execute("SELECT symbol, MAX(session_date) FROM bars_daily GROUP BY symbol").fetchall()
    )
    written, failed, refetch = 0, [], []

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        start = min(date.fromisoformat(last[s]) if s in last else default_start for s in batch)
        try:
            bars = fetch_bars(client, batch, start, end)
        except Exception as exc:
            log.error("bar batch failed (%s..%s): %s", batch[0], batch[-1], exc)
            failed.extend(batch)
            continue

        for symbol, sym_bars in bars.groupby("symbol"):
            if symbol in last:
                overlap = sym_bars[sym_bars["session_date"] == last[symbol]]
                if not overlap.empty:
                    stored = _stored_close(conn, symbol, last[symbol])
                    if abs(overlap["close"].iloc[0] / stored - 1) > SPLIT_TOLERANCE:
                        log.warning("adjusted close moved for %s, refetching history", symbol)
                        refetch.append(symbol)
                        continue
                sym_bars = sym_bars[sym_bars["session_date"] > last[symbol]]
            written += _write(conn, sym_bars)

        for symbol in set(batch) - set(bars["symbol"]) - set(last):
            log.warning("no bars returned for %s, marking inactive", symbol)
            deactivate(conn, symbol)

    for symbol in refetch:
        conn.execute("DELETE FROM bars_daily WHERE symbol = ?", (symbol,))
        conn.commit()
        try:
            written += _write(conn, fetch_bars(client, [symbol], default_start, end))
        except Exception as exc:
            log.error("full refetch failed for %s: %s", symbol, exc)
            failed.append(symbol)

    log.info("bars upserted: %d, failed symbols: %d", written, len(failed))
    return written, failed
