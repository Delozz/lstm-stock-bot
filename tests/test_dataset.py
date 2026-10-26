"""AC-20, AC-21: walk-forward folds and train-only normalization."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stockbot.dataset import fit_stats, make_windows, walk_forward_folds
from stockbot.features import FEATURES


def test_walk_forward_folds_respect_embargo_and_val_split():
    dates = list(pd.bdate_range("2021-01-01", "2025-12-31").strftime("%Y-%m-%d"))
    pos = {d: i for i, d in enumerate(dates)}
    folds = walk_forward_folds(dates, oos_months=24, n_folds=8, embargo=5, val_frac=0.10)

    assert len(folds) == 8
    for prev, nxt in zip(folds, folds[1:], strict=False):
        assert pos[prev.test_dates[-1]] + 1 == pos[nxt.test_dates[0]]  # consecutive blocks
    for fold in folds:
        assert 55 <= len(fold.test_dates) <= 70  # about 3 months of sessions
        train = fold.fit_dates + fold.val_dates
        assert pos[max(train)] + 5 < pos[min(fold.test_dates)]
        assert max(fold.fit_dates) < min(fold.val_dates)
        assert len(fold.val_dates) == math.ceil(0.10 * len(train))


def test_stats_come_from_training_rows_only():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(100, len(FEATURES))), columns=FEATURES)
    df["session_date"] = pd.bdate_range("2025-01-01", periods=100).strftime("%Y-%m-%d")
    df.loc[80:, FEATURES] += 1000  # later rows would wreck the stats if they leaked in
    train = df.iloc[:80]
    mean, std = fit_stats(train, FEATURES)
    np.testing.assert_allclose(mean, train[FEATURES].mean().to_numpy())
    np.testing.assert_allclose(std, train[FEATURES].std(ddof=0).to_numpy())


def test_windows_end_on_each_row_after_warmup():
    n = 30
    df = pd.DataFrame(np.arange(n * len(FEATURES), dtype=float).reshape(n, -1), columns=FEATURES)
    df["symbol"] = "AAA"
    df["session_date"] = pd.bdate_range("2025-01-01", periods=n).strftime("%Y-%m-%d")
    df["label"] = 1.0
    df["next_oc_ret"] = 0.0
    w = make_windows(df, FEATURES, 20)
    assert w.X.shape == (11, 20, len(FEATURES))
    assert w.dates[0] == df.session_date[19]
    np.testing.assert_array_equal(w.X[0, -1], df.loc[19, FEATURES].to_numpy())
