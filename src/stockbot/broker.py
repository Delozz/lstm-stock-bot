"""Alpaca client factories and the only functions that submit orders.

Every factory runs the paper-only guard before building a client (FR-2). The one sell
path, submit_exit, re-checks the live position and refuses anything that could open a
short (FR-40).
"""

from __future__ import annotations

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from stockbot.config import SafetyError, assert_paper, require_secret
from stockbot.retry import status_of


def make_trading_client() -> TradingClient:
    url = assert_paper()
    return TradingClient(
        require_secret("ALPACA_API_KEY"),
        require_secret("ALPACA_SECRET_KEY"),
        paper=True,
        url_override=url,
    )


def make_data_client() -> StockHistoricalDataClient:
    assert_paper()
    return StockHistoricalDataClient(
        require_secret("ALPACA_API_KEY"), require_secret("ALPACA_SECRET_KEY")
    )


def find_order(client: TradingClient, client_order_id: str):
    try:
        return client.get_order_by_client_id(client_order_id)
    except Exception as exc:
        if status_of(exc) == 404:
            return None
        raise


def held_qty(client: TradingClient, symbol: str) -> int:
    """Long shares currently held; 0 if there is no position."""
    try:
        position = client.get_open_position(symbol)
    except Exception as exc:
        if status_of(exc) == 404:
            return 0
        raise
    return max(0, int(float(position.qty)))


def submit_buy(client: TradingClient, symbol: str, qty: int, client_order_id: str):
    return client.submit_order(
        order_data=MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.OPG,
            client_order_id=client_order_id,
        )
    )


def submit_exit(client: TradingClient, symbol: str, qty: int, client_order_id: str):
    held = held_qty(client, symbol)
    if qty <= 0 or qty > held:
        raise SafetyError(f"refusing to sell {qty} {symbol}: only {held} held")
    return client.submit_order(
        order_data=MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.CLS,
            client_order_id=client_order_id,
        )
    )
