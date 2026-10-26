"""Wilson interval, bootstrap CI, Sharpe, and max drawdown (FR-28)."""

from __future__ import annotations

import math

import numpy as np

TRADING_DAYS = 252


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


def bootstrap_mean_ci(
    x: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
    chunk: int = 1_000,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - start)
        means.append(x[rng.integers(0, len(x), size=(size, len(x)))].mean(axis=1))
    means = np.concatenate(means)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def sharpe(daily_returns: np.ndarray) -> float:
    r = np.asarray(daily_returns, dtype=float)
    if len(r) < 2:
        return math.nan
    sd = r.std(ddof=1)
    return 0.0 if sd == 0 else float(r.mean() / sd * math.sqrt(TRADING_DAYS))


def max_drawdown(daily_returns: np.ndarray) -> float:
    """Most negative peak-to-trough move of the compounded equity curve, starting at 1.0."""
    equity = np.concatenate([[1.0], np.cumprod(1 + np.asarray(daily_returns, dtype=float))])
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1).min())
