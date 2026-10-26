"""Look-ahead guards run before train and backtest (FR-31)."""

from __future__ import annotations

import random
import sqlite3
from collections.abc import Callable

import numpy as np
import pandas as pd

from stockbot.config import DataError
from stockbot.features import FEATURES, compute_features

MAX_TARGET_CORR = 0.5


class LeakError(DataError):
    pass


def assert_no_future_dependence(
    bars: pd.DataFrame,
    news: pd.DataFrame,
    t: str,
    compute: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame] = compute_features,
    tol: float = 1e-9,
) -> None:
    """Features at t must not change when every bar and headline after t is removed."""
    full = compute(bars, news).set_index("session_date").loc[t, FEATURES]
    truncated = (
        compute(bars[bars["session_date"] <= t], news[news["session_date"] <= t])
        .set_index("session_date")
        .loc[t, FEATURES]
    )
    a, b = full.astype(float).to_numpy(), truncated.astype(float).to_numpy()
    if not np.allclose(a, b, rtol=0, atol=tol, equal_nan=True):
        changed = [f for f, x, y in zip(FEATURES, a, b, strict=True) if not np.isclose(x, y)]
        raise LeakError(f"features at {t} depend on future data: {', '.join(changed)}")


def assert_no_target_proxy(features_df: pd.DataFrame) -> None:
    """No feature may equal the label or correlate with next_oc_ret above 0.5 in absolute value."""
    labeled = features_df[features_df["label"].notna()]
    label = labeled["label"].astype(float).to_numpy()
    for col in FEATURES:
        values = labeled[col].astype(float)
        if np.array_equal(values.to_numpy(), label):
            raise LeakError(f"feature {col} equals the label")
        corr = values.corr(labeled["next_oc_ret"].astype(float))
        if pd.notna(corr) and abs(corr) > MAX_TARGET_CORR:
            raise LeakError(f"feature {col} has correlation {corr:.3f} with next_oc_ret")


def run_leakchecks(conn: sqlite3.Connection, sample_size: int = 20, seed: int = 42) -> None:
    symbols = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT f.symbol FROM features f JOIN tickers t USING (symbol) "
            "WHERE t.is_benchmark = 0 ORDER BY f.symbol"
        )
    ]
    if not symbols:
        raise DataError("features table is empty; run `stockbot features` first")
    rng = random.Random(seed)
    for symbol in rng.sample(symbols, min(sample_size, len(symbols))):
        bars = pd.read_sql_query(
            "SELECT * FROM bars_daily WHERE symbol = ? ORDER BY session_date",
            conn,
            params=(symbol,),
        )
        news = pd.read_sql_query(
            "SELECT session_date, vader_compound FROM news WHERE symbol = ?",
            conn,
            params=(symbol,),
        )
        dates = [
            r[0]
            for r in conn.execute(
                "SELECT session_date FROM features WHERE symbol = ? ORDER BY session_date",
                (symbol,),
            )
        ]
        # Skip the last session so there is always future data to remove.
        assert_no_future_dependence(bars, news, rng.choice(dates[:-1] or dates))

    features_df = pd.read_sql_query("SELECT * FROM features WHERE label IS NOT NULL", conn)
    assert_no_target_proxy(features_df)
