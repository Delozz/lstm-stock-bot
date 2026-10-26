"""Pre-trade risk checks (FR-38)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RiskState:
    equity: float
    prev_equity: float | None
    entries_today: int


def check_entry(state: RiskState, notional: float, scfg: dict[str, Any]) -> str | None:
    """Return the name of the first failed rule, or None if the entry is allowed."""
    if state.entries_today >= scfg["top_n"]:
        return "max_entries_per_day"
    if notional > scfg["position_pct"] * state.equity + 1e-6:
        return "max_position_notional"
    if state.prev_equity and state.equity <= state.prev_equity * (1 - scfg["max_daily_drawdown"]):
        return "daily_drawdown_halt"
    return None
