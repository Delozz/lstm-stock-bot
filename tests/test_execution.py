"""AC-29 to AC-35: sizing, idempotency, exit window, risk rules, no shorts, dry run."""

from __future__ import annotations

import ast
import inspect
from datetime import datetime

import pandas as pd
import pytest
from alpaca.trading.enums import OrderSide, TimeInForce

from stockbot import broker
from stockbot.calendar import NY
from stockbot.config import DataError, SafetyError
from stockbot.db import upsert
from stockbot.execution import run_exit, run_trade
from stockbot.risk import RiskState, check_entry

SIGNAL_DATE, TODAY = "2026-10-23", "2026-10-26"
CFG = {
    "strategy": {"top_n": 4, "threshold": 0.55, "position_pct": 0.05, "max_daily_drawdown": 0.03}
}
CLOSE = datetime(2026, 10, 26, 16, 0, tzinfo=NY)


@pytest.fixture
def seeded(conn, add_bars):
    for symbol, close in (("AAA", 250.0), ("BIG", 6000.0)):
        add_bars(
            conn,
            symbol,
            pd.DataFrame(
                [
                    {
                        "session_date": SIGNAL_DATE,
                        "open": close,
                        "high": close,
                        "low": close,
                        "close": close,
                        "volume": 1,
                    }
                ]
            ),
        )
    upsert(
        conn,
        "models",
        [
            {
                "model_id": "m1",
                "trained_at_utc": "2026-10-20T00:00:00+00:00",
                "train_end_date": "2026-10-20",
                "seed": 42,
                "git_sha": "abc",
                "val_loss": 0.69,
                "artifact_dir": "x",
            }
        ],
        ("model_id",),
    )
    upsert(
        conn,
        "signals",
        [
            {
                "session_date": SIGNAL_DATE,
                "symbol": s,
                "model_id": "m1",
                "p_up": p,
                "rank": r,
                "selected": 1,
            }
            for r, (s, p) in enumerate((("AAA", 0.6), ("BIG", 0.58)), start=1)
        ],
        ("session_date", "symbol", "model_id"),
    )
    return conn


def test_trade_sizes_and_skips_unaffordable(seeded, fake_client):
    lines = run_trade(seeded, fake_client, CFG, today=TODAY, signal_date=SIGNAL_DATE)
    assert len(fake_client.submitted) == 1
    order = fake_client.submitted[0]
    assert (order.symbol, order.qty, order.side) == ("AAA", 20, OrderSide.BUY)
    assert order.time_in_force == TimeInForce.OPG
    assert order.client_order_id == f"{TODAY}-AAA-buy"
    assert lines == [f"buy 20 AAA opg {TODAY}-AAA-buy accepted"]


def test_trade_twice_submits_once(seeded, fake_client):
    run_trade(seeded, fake_client, CFG, today=TODAY, signal_date=SIGNAL_DATE)
    run_trade(seeded, fake_client, CFG, today=TODAY, signal_date=SIGNAL_DATE)
    assert len(fake_client.submitted) == 1


def test_dry_run_submits_nothing(seeded, fake_client):
    lines = run_trade(seeded, fake_client, CFG, today=TODAY, signal_date=SIGNAL_DATE, dry_run=True)
    assert fake_client.submitted == []
    assert lines == [f"DRY RUN buy 20 AAA opg {TODAY}-AAA-buy"]


def test_no_signals_exits_4(conn, fake_client):
    with pytest.raises(DataError, match="no signals"):
        run_trade(conn, fake_client, CFG, today=TODAY, signal_date=SIGNAL_DATE)


def test_exit_sells_held_quantity_with_cls(seeded, fake_client):
    run_trade(seeded, fake_client, CFG, today=TODAY, signal_date=SIGNAL_DATE)
    fake_client.positions["AAA"] = 20
    run_exit(seeded, fake_client, today=TODAY, now=CLOSE.replace(hour=14), session_close=CLOSE)
    sell = fake_client.submitted[-1]
    assert (sell.symbol, sell.qty, sell.side) == ("AAA", 20, OrderSide.SELL)
    assert sell.time_in_force == TimeInForce.CLS


def test_exit_after_cutoff_refuses(seeded, fake_client):
    with pytest.raises(DataError, match="exit window missed"):
        run_exit(
            seeded,
            fake_client,
            today=TODAY,
            now=CLOSE.replace(hour=15, minute=51),
            session_close=CLOSE,
        )


def test_exit_without_position_submits_nothing(seeded, fake_client):
    run_trade(seeded, fake_client, CFG, today=TODAY, signal_date=SIGNAL_DATE)
    run_exit(seeded, fake_client, today=TODAY, now=CLOSE.replace(hour=14), session_close=CLOSE)
    assert [o.side for o in fake_client.submitted] == [OrderSide.BUY]


def test_broker_refuses_sell_above_position(fake_client):
    fake_client.positions["AAA"] = 5
    with pytest.raises(SafetyError):
        broker.submit_exit(fake_client, "AAA", 6, "x")
    with pytest.raises(SafetyError):
        broker.submit_exit(fake_client, "ZZZ", 1, "y")
    assert fake_client.submitted == []


def test_every_sell_path_checks_position():
    tree = ast.parse(inspect.getsource(broker))
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        src = ast.unparse(fn)
        if "OrderSide.SELL" in src:
            assert "held_qty(" in src, f"{fn.name} can sell without a position check"


@pytest.mark.parametrize(
    "state, notional, rule",
    [
        (RiskState(100_000, None, 4), 1_000, "max_entries_per_day"),
        (RiskState(100_000, None, 0), 5_001, "max_position_notional"),
        (RiskState(97_000, 100_000, 0), 1_000, "daily_drawdown_halt"),
        (RiskState(99_000, 100_000, 3), 4_900, None),
    ],
)
def test_risk_rules(state, notional, rule):
    assert check_entry(state, notional, CFG["strategy"]) == rule
