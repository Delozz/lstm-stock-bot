from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stockbot import config, db

ENV_VARS = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "FINNHUB_API_KEY",
    "ALPACA_PAPER",
    "ALPACA_BASE_URL",
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Tests never read the developer's .env or real credentials."""
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    db.init_db(c)
    yield c
    c.close()


def synthetic_bars(n: int = 60, seed: int = 0, start: str = "2025-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    return pd.DataFrame(
        {
            "session_date": pd.bdate_range(start, periods=n).strftime("%Y-%m-%d"),
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 2_000_000, n),
        }
    )


@pytest.fixture
def make_bars():
    return synthetic_bars


def insert_bars(conn, symbol: str, bars: pd.DataFrame, *, benchmark: bool = False) -> None:
    db.upsert(
        conn,
        "tickers",
        [{"symbol": symbol, "name": symbol, "sector": None, "is_benchmark": int(benchmark)}],
        ("symbol",),
    )
    db.upsert(
        conn,
        "bars_daily",
        bars.assign(symbol=symbol).astype(object).to_dict("records"),
        ("symbol", "session_date"),
    )


@pytest.fixture
def add_bars():
    return insert_bars


class NotFound(Exception):
    status_code = 404


class FakeTradingClient:
    """Records order submissions; never touches the network."""

    def __init__(self, equity: float = 100_000.0, positions: dict[str, int] | None = None):
        self.equity = equity
        self.positions = positions or {}
        self.submitted: list = []
        self.orders: dict[str, SimpleNamespace] = {}

    def get_account(self):
        return SimpleNamespace(equity=str(self.equity), cash=str(self.equity))

    def get_order_by_client_id(self, client_id):
        if client_id not in self.orders:
            raise NotFound(client_id)
        return self.orders[client_id]

    def get_open_position(self, symbol):
        if symbol not in self.positions:
            raise NotFound(symbol)
        return SimpleNamespace(qty=str(self.positions[symbol]))

    def submit_order(self, order_data):
        self.submitted.append(order_data)
        order = SimpleNamespace(
            id=f"alpaca-{len(self.submitted)}",
            status="accepted",
            client_order_id=order_data.client_order_id,
            filled_qty="0",
            filled_avg_price=None,
            filled_at=None,
        )
        self.orders[order_data.client_order_id] = order
        return order


@pytest.fixture
def fake_client():
    return FakeTradingClient()
