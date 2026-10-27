"""Walk-forward out-of-sample predictions and the trading simulation (FR-27 to FR-29)."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

import numpy as np
import pandas as pd
import torch

from stockbot.dataset import (
    fit_stats,
    load_features,
    make_windows,
    normalize,
    walk_forward_folds,
)
from stockbot.features import FEATURES, PRICE_FEATURES
from stockbot.metrics import bootstrap_mean_ci, max_drawdown, sharpe, wilson_interval
from stockbot.models import make_baseline
from stockbot.train import fit_lstm, predict_proba

log = logging.getLogger(__name__)

RANDOM_RUNS = 100


def walk_forward_predictions(
    df: pd.DataFrame, cfg: dict[str, Any], device: torch.device, *, ablation: bool = True
) -> pd.DataFrame:
    """Out-of-sample p_up for every (model, symbol, session) in the test blocks."""
    mcfg, wcfg, seed = cfg["model"], cfg["walkforward"], cfg["seed"]
    labeled_dates = df.loc[df["label"].notna(), "session_date"].unique()
    folds = walk_forward_folds(
        labeled_dates,
        oos_months=wcfg["oos_months"],
        n_folds=wcfg["n_folds"],
        embargo=wcfg["embargo"],
        val_frac=wcfg["val_frac"],
    )
    w = make_windows(df, FEATURES, mcfg["seq_len"])
    has_label = ~np.isnan(w.y)
    variants = {"lstm": FEATURES}
    if ablation:
        variants["lstm_no_sentiment"] = PRICE_FEATURES

    out = []
    for k, fold in enumerate(folds):
        fit = has_label & np.isin(w.dates, fold.fit_dates)
        val = has_label & np.isin(w.dates, fold.val_dates)
        test = has_label & np.isin(w.dates, fold.test_dates)
        stats_rows = df[df["session_date"].isin(fold.fit_dates) & df["label"].notna()]
        log.info(
            "fold %d: fit %d, val %d, test %d windows (test %s..%s)",
            k + 1,
            fit.sum(),
            val.sum(),
            test.sum(),
            fold.test_dates[0],
            fold.test_dates[-1],
        )

        for name, feats in variants.items():
            cols = [FEATURES.index(f) for f in feats]
            X = normalize(w.X[:, :, cols], *fit_stats(stats_rows, feats))
            model, _ = fit_lstm(X[fit], w.y[fit], X[val], w.y[val], mcfg, device, seed)
            out.append(_frame(name, k, w, test, predict_proba(model, X[test], device)))

        X_last = normalize(w.X[:, -1, :], *fit_stats(stats_rows, FEATURES))
        clf = make_baseline(seed).fit(X_last[fit], w.y[fit])
        out.append(_frame("logreg", k, w, test, clf.predict_proba(X_last[test])[:, 1]))

    return pd.concat(out, ignore_index=True)


def _frame(name: str, fold: int, w, mask: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": name,
            "fold": fold,
            "symbol": w.symbols[mask],
            "session_date": w.dates[mask],
            "p_up": p,
            "label": w.y[mask],
            "next_oc_ret": w.next_ret[mask],
        }
    )


def select_trades(preds: pd.DataFrame, top_n: int, threshold: float) -> pd.DataFrame:
    """Top-N per session with p_up >= threshold; ties go to the alphabetically first symbol."""
    eligible = preds[preds["p_up"] >= threshold]
    ranked = eligible.sort_values(["session_date", "p_up", "symbol"], ascending=[True, False, True])
    return ranked.groupby("session_date", sort=True).head(top_n)


def daily_returns(
    trades: pd.DataFrame, sessions: np.ndarray, position_pct: float, cost_bps: float
) -> tuple[pd.Series, pd.DataFrame]:
    """Enter at the open of t+1, exit at its close, cost_bps per side, position_pct each."""
    trades = trades.assign(net_ret=trades["next_oc_ret"] - 2 * cost_bps / 1e4)
    daily = (trades.groupby("session_date")["net_ret"].sum() * position_pct).reindex(
        sessions, fill_value=0.0
    )
    return daily, trades


def summarize(
    daily: pd.Series, trades: pd.DataFrame, preds: pd.DataFrame | None, seed: int
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if preds is not None:
        correct = int(((preds["p_up"] >= 0.5) == (preds["label"] == 1)).sum())
        out["accuracy"] = correct / len(preds)
        out["accuracy_ci"] = wilson_interval(correct, len(preds))
    if trades is not None:
        out["hit_rate"] = float((trades["net_ret"] > 0).mean()) if len(trades) else float("nan")
        out["mean_trade_return"] = float(trades["net_ret"].mean()) if len(trades) else float("nan")
        out["mean_trade_return_ci"] = bootstrap_mean_ci(trades["net_ret"].to_numpy(), seed=seed)
        out["n_trades"] = len(trades)
        out["trades_per_day"] = len(trades) / len(daily) if len(daily) else 0.0
    out["sharpe"] = sharpe(daily.to_numpy())
    out["max_drawdown"] = max_drawdown(daily.to_numpy())
    out["total_return"] = float(np.prod(1 + daily.to_numpy()) - 1)
    return out


def random_baseline(
    preds: pd.DataFrame, sessions: np.ndarray, scfg: dict[str, Any], seed: int
) -> tuple[pd.Series, dict[str, Any]]:
    """Mean over RANDOM_RUNS seeded runs of picking top_n random tickers each session."""
    universe = preds[preds["model"] == "lstm"][["symbol", "session_date", "next_oc_ret"]]
    runs, curves = [], []
    for r in range(RANDOM_RUNS):
        rng = np.random.default_rng(seed + r)
        shuffled = universe.assign(key=rng.random(len(universe))).sort_values(
            ["session_date", "key"]
        )
        picks = shuffled.groupby("session_date").head(scfg["top_n"])
        daily, trades = daily_returns(picks, sessions, scfg["position_pct"], scfg["cost_bps"])
        curves.append(daily)
        runs.append(summarize(daily, trades, None, seed))
    summary = {
        k: float(np.mean([m[k] for m in runs]))
        for k in (
            "hit_rate",
            "mean_trade_return",
            "sharpe",
            "max_drawdown",
            "total_return",
            "n_trades",
            "trades_per_day",
        )
    }
    return pd.concat(curves, axis=1).mean(axis=1), summary


def spy_returns(conn: sqlite3.Connection, benchmark: str, sessions: np.ndarray) -> pd.Series:
    """SPY close-to-close return from t to t+1, keyed by t to line up with the strategy."""
    spy = pd.read_sql_query(
        "SELECT session_date, close FROM bars_daily WHERE symbol = ? ORDER BY session_date",
        conn,
        params=(benchmark,),
    ).set_index("session_date")["close"]
    return (spy.shift(-1) / spy - 1).reindex(sessions).fillna(0.0)


def run_backtest(
    conn: sqlite3.Connection, cfg: dict[str, Any], device: torch.device, *, ablation: bool = True
) -> tuple[dict[str, dict], dict[str, pd.Series], pd.DataFrame]:
    """Returns (metrics per model, daily return series per model, out-of-sample predictions)."""
    scfg, seed = cfg["strategy"], cfg["seed"]
    preds = walk_forward_predictions(load_features(conn), cfg, device, ablation=ablation)
    sessions = np.sort(preds["session_date"].unique())

    metrics, curves = {}, {}
    for name, p in preds.groupby("model"):
        trades = select_trades(p, scfg["top_n"], scfg["threshold"])
        daily, trades = daily_returns(trades, sessions, scfg["position_pct"], scfg["cost_bps"])
        metrics[name] = summarize(daily, trades, p, seed)
        curves[name] = daily

    curves["random"], metrics["random"] = random_baseline(preds, sessions, scfg, seed)
    curves["spy"] = spy_returns(conn, cfg["benchmark"], sessions)
    metrics["spy"] = summarize(curves["spy"], None, None, seed)
    return metrics, curves, preds
