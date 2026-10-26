"""Sliding windows, walk-forward folds, and train-only normalization (FR-22, FR-23)."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class Windows:
    X: np.ndarray  # (n, seq_len, n_features), raw (unnormalized) float32
    y: np.ndarray  # (n,) float32, NaN where the label is unknown
    symbols: np.ndarray
    dates: np.ndarray  # session_date of the last step in each window
    next_ret: np.ndarray


@dataclass(frozen=True)
class Fold:
    fit_dates: list[str]
    val_dates: list[str]
    test_dates: list[str]


def load_features(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT f.* FROM features f JOIN tickers t USING (symbol) "
        "WHERE t.is_benchmark = 0 AND t.active = 1 ORDER BY f.symbol, f.session_date",
        conn,
    )


def make_windows(df: pd.DataFrame, features: Sequence[str], seq_len: int) -> Windows:
    xs, ys, syms, dates, rets = [], [], [], [], []
    for symbol, g in df.groupby("symbol", sort=True):
        if len(g) < seq_len:
            continue
        values = g[list(features)].to_numpy(np.float32)
        xs.append(sliding_window_view(values, (seq_len, len(features)))[:, 0])
        tail = g.iloc[seq_len - 1 :]
        ys.append(tail["label"].to_numpy(np.float32, na_value=np.nan))
        rets.append(tail["next_oc_ret"].to_numpy(np.float64, na_value=np.nan))
        dates.append(tail["session_date"].to_numpy())
        syms.append(np.full(len(tail), symbol, dtype=object))
    if not xs:
        empty = np.empty(0)
        return Windows(
            np.empty((0, seq_len, len(features)), np.float32), empty, empty, empty, empty
        )
    return Windows(
        X=np.concatenate(xs),
        y=np.concatenate(ys),
        symbols=np.concatenate(syms),
        dates=np.concatenate(dates),
        next_ret=np.concatenate(rets),
    )


def fit_stats(rows: pd.DataFrame, features: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    values = rows[list(features)].to_numpy(np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def split_val(dates: Sequence[str], val_frac: float) -> tuple[list[str], list[str]]:
    """Hold out the last val_frac of dates (by time) for early stopping."""
    dates = sorted(dates)
    n_val = max(1, math.ceil(len(dates) * val_frac))
    return dates[:-n_val], dates[-n_val:]


def walk_forward_folds(
    dates: Sequence[str],
    *,
    oos_months: int = 24,
    n_folds: int = 8,
    embargo: int = 5,
    val_frac: float = 0.10,
) -> list[Fold]:
    """Split the last oos_months into n_folds consecutive test blocks.

    Each block trains on everything before it, minus `embargo` sessions so that labels
    (which look one session ahead) cannot overlap the test block.
    """
    idx = pd.DatetimeIndex(sorted(set(dates)))
    block = oos_months // n_folds
    oos_start = idx[-1] - pd.DateOffset(months=oos_months)
    folds = []
    for k in range(n_folds):
        lo = oos_start + pd.DateOffset(months=block * k)
        hi = oos_start + pd.DateOffset(months=block * (k + 1))
        test = idx[(idx > lo) & (idx <= hi)]
        before = idx[idx <= lo]
        train = before[:-embargo] if embargo else before
        if len(test) == 0 or len(train) == 0:
            continue
        fit, val = split_val([d.strftime("%Y-%m-%d") for d in train], val_frac)
        folds.append(Fold(fit, val, [d.strftime("%Y-%m-%d") for d in test]))
    return folds
