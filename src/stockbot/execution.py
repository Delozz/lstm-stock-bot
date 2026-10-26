"""trade (opg buys), exit (cls sells), and reconcile against Alpaca paper (FR-35 to FR-41)."""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from stockbot import broker
from stockbot.config import DataError
from stockbot.db import upsert, utc_now
from stockbot.retry import status_of
from stockbot.risk import RiskState, check_entry

log = logging.getLogger(__name__)

TERMINAL = {"filled", "canceled", "expired", "rejected", "replaced"}
EXIT_CUTOFF = timedelta(minutes=10)


def order_id(session_date: str, symbol: str, side: str) -> str:
    return f"{session_date}-{symbol}-{side}"


def _value(x: Any) -> Any:
    return getattr(x, "value", x)


def _known(conn: sqlite3.Connection, client, coid: str) -> bool:
    if conn.execute("SELECT 1 FROM orders WHERE client_order_id = ?", (coid,)).fetchone():
        return True
    return broker.find_order(client, coid) is not None


def _record(conn, coid, order, *, session_date, symbol, side, qty, tif, status=None) -> str:
    status = status or str(_value(order.status))
    upsert(
        conn,
        "orders",
        [
            {
                "client_order_id": coid,
                "alpaca_order_id": str(order.id) if order is not None else None,
                "session_date": session_date,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "time_in_force": tif,
                "status": status,
                "submitted_at_utc": utc_now(),
            }
        ],
        ("client_order_id",),
    )
    return status


def _submit(conn, client, submit, coid: str, **row) -> str:
    """Submit once. A broker rejection is stored and never retried (EC-7)."""
    try:
        order = submit()
    except Exception as exc:
        status = status_of(exc)
        if status is None or status >= 500:
            raise
        log.error("order %s rejected (%s): %s", coid, status, exc)
        return _record(conn, coid, None, status="rejected", **row)
    return _record(conn, coid, order, **row)


def run_trade(
    conn: sqlite3.Connection,
    client,
    cfg: dict[str, Any],
    *,
    today: str,
    signal_date: str,
    dry_run: bool = False,
) -> list[str]:
    scfg = cfg["strategy"]
    if not conn.execute("SELECT 1 FROM signals WHERE session_date = ?", (signal_date,)).fetchone():
        raise DataError(f"no signals for {signal_date}")
    picks = [
        r[0]
        for r in conn.execute(
            "SELECT symbol FROM signals WHERE session_date = ? AND selected = 1 "
            "AND model_id = (SELECT model_id FROM signals WHERE session_date = ? "
            "ORDER BY rowid DESC LIMIT 1) ORDER BY rank",
            (signal_date, signal_date),
        )
    ]

    equity = float(client.get_account().equity)
    prev = conn.execute(
        "SELECT equity FROM equity_snapshots WHERE session_date < ? "
        "ORDER BY session_date DESC LIMIT 1",
        (today,),
    ).fetchone()
    state = RiskState(
        equity=equity,
        prev_equity=prev[0] if prev else None,
        entries_today=conn.execute(
            "SELECT COUNT(*) FROM orders WHERE session_date = ? AND side = 'buy'", (today,)
        ).fetchone()[0],
    )

    lines = []
    for symbol in picks:
        coid = order_id(today, symbol, "buy")
        if _known(conn, client, coid):
            log.info("order %s already exists, skipping", coid)
            continue
        close = conn.execute(
            "SELECT close FROM bars_daily WHERE symbol = ? AND session_date = ?",
            (symbol, signal_date),
        ).fetchone()
        if close is None:
            log.warning("no close for %s on %s, skipping", symbol, signal_date)
            continue
        qty = math.floor(scfg["position_pct"] * equity / close[0])
        if qty < 1:
            log.warning("quantity below 1 for %s at %.2f, skipping", symbol, close[0])
            continue
        rule = check_entry(state, qty * close[0], scfg)
        if rule:
            log.warning("risk check %s blocked %s", rule, symbol)
            continue

        if dry_run:
            lines.append(f"DRY RUN buy {qty} {symbol} opg {coid}")
        else:
            status = _submit(
                conn,
                client,
                lambda s=symbol, q=qty, c=coid: broker.submit_buy(client, s, q, c),
                coid,
                session_date=today,
                symbol=symbol,
                side="buy",
                qty=qty,
                tif="opg",
            )
            lines.append(f"buy {qty} {symbol} opg {coid} {status}")
        state.entries_today += 1
    return lines or ["no orders"]


def run_exit(
    conn: sqlite3.Connection,
    client,
    *,
    today: str,
    now: datetime,
    session_close: datetime,
    dry_run: bool = False,
) -> list[str]:
    if now > session_close - EXIT_CUTOFF:
        raise DataError("exit window missed")
    entries = conn.execute(
        "SELECT symbol, qty FROM orders WHERE session_date = ? AND side = 'buy' "
        "AND status NOT IN ('canceled', 'expired', 'rejected')",
        (today,),
    ).fetchall()

    lines = []
    for symbol, bought in entries:
        coid = order_id(today, symbol, "sell")
        if _known(conn, client, coid):
            continue
        qty = min(bought, broker.held_qty(client, symbol))
        if qty < 1:
            log.warning("no position in %s to exit", symbol)
            continue
        if dry_run:
            lines.append(f"DRY RUN sell {qty} {symbol} cls {coid}")
            continue
        status = _submit(
            conn,
            client,
            lambda s=symbol, q=qty, c=coid: broker.submit_exit(client, s, q, c),
            coid,
            session_date=today,
            symbol=symbol,
            side="sell",
            qty=qty,
            tif="cls",
        )
        lines.append(f"sell {qty} {symbol} cls {coid} {status}")
    return lines or ["no orders"]


def run_reconcile(conn: sqlite3.Connection, client, *, today: str) -> list[str]:
    pending = conn.execute(
        f"SELECT client_order_id FROM orders WHERE status NOT IN "
        f"({', '.join('?' * len(TERMINAL))})",
        tuple(TERMINAL),
    ).fetchall()
    for (coid,) in pending:
        order = broker.find_order(client, coid)
        if order is None:
            continue
        conn.execute(
            "UPDATE orders SET alpaca_order_id = ?, status = ?, filled_qty = ?, "
            "filled_avg_price = ?, filled_at_utc = ? WHERE client_order_id = ?",
            (
                str(order.id),
                str(_value(order.status)),
                int(float(order.filled_qty or 0)),
                float(order.filled_avg_price) if order.filled_avg_price else None,
                order.filled_at.isoformat() if order.filled_at else None,
                coid,
            ),
        )
    conn.commit()

    fills = conn.execute(
        "SELECT b.filled_avg_price, s.filled_avg_price, s.filled_qty FROM orders b "
        "JOIN orders s ON s.symbol = b.symbol AND s.session_date = b.session_date "
        "WHERE b.session_date = ? AND b.side = 'buy' AND s.side = 'sell' "
        "AND b.filled_avg_price IS NOT NULL AND s.filled_avg_price IS NOT NULL",
        (today,),
    ).fetchall()
    realized = sum((sell - buy) * qty for buy, sell, qty in fills)

    account = client.get_account()
    equity = float(account.equity)
    upsert(
        conn,
        "equity_snapshots",
        [
            {
                "session_date": today,
                "equity": equity,
                "cash": float(account.cash),
                "realized_pnl": realized,
            }
        ],
        ("session_date",),
    )
    return [f"equity={equity:.2f} realized_pnl={realized:.2f}"]
