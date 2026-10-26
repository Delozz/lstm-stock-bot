"""AC-23, AC-24: selection rules, costs, and metric math."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stockbot.backtest import daily_returns, select_trades
from stockbot.metrics import bootstrap_mean_ci, max_drawdown, sharpe, wilson_interval


def fixture_preds() -> pd.DataFrame:
    p = {
        "2026-01-05": {"A": 0.90, "B": 0.80, "C": 0.70, "D": 0.60, "E": 0.56},
        "2026-01-06": {"A": 0.40, "B": 0.56, "C": 0.54, "D": 0.20, "E": 0.10},
        "2026-01-07": {"A": 0.70, "B": 0.70, "C": 0.70, "D": 0.70, "E": 0.70},
    }
    rows = [
        {"session_date": d, "symbol": s, "p_up": v, "next_oc_ret": 0.01}
        for d, picks in p.items()
        for s, v in picks.items()
    ]
    return pd.DataFrame(rows)


def test_selection_threshold_top_n_and_ties():
    trades = select_trades(fixture_preds(), top_n=4, threshold=0.55)
    by_day = trades.groupby("session_date")["symbol"].apply(list).to_dict()
    assert by_day == {
        "2026-01-05": ["A", "B", "C", "D"],
        "2026-01-06": ["B"],
        "2026-01-07": ["A", "B", "C", "D"],  # tie broken alphabetically, E dropped
    }


def test_daily_return_applies_round_trip_cost():
    preds = fixture_preds()
    trades = select_trades(preds, top_n=4, threshold=0.55)
    sessions = np.sort(preds["session_date"].unique())
    daily, trades = daily_returns(trades, sessions, position_pct=0.05, cost_bps=5)
    net = 0.01 - 0.001
    assert daily["2026-01-05"] == pytest.approx(0.05 * 4 * net)
    assert daily["2026-01-06"] == pytest.approx(0.05 * net)
    assert trades["net_ret"].tolist() == pytest.approx([net] * len(trades))


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(50, 100)
    assert (lo, hi) == (pytest.approx(0.40383, abs=1e-4), pytest.approx(0.59617, abs=1e-4))


def test_sharpe_and_drawdown():
    r = np.array([0.01, -0.01, 0.02, 0.0])
    assert sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))
    assert max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    assert max_drawdown(np.array([-0.1, 0.05])) == pytest.approx(-0.1)


def test_bootstrap_is_seeded_and_brackets_mean():
    x = np.random.default_rng(1).normal(0.001, 0.01, 500)
    ci_1 = bootstrap_mean_ci(x, seed=7)
    assert ci_1 == bootstrap_mean_ci(x, seed=7)
    assert ci_1[0] < x.mean() < ci_1[1]
