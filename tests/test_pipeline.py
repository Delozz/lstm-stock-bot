"""End-to-end smoke test on synthetic data: features, backtest, report, train, signal.

Covers AC-22, AC-25, AC-26, and AC-27 with a tiny model so it runs in seconds.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pandas as pd
import pytest
import torch

from stockbot.backtest import run_backtest
from stockbot.features import build_features
from stockbot.leakcheck import run_leakchecks
from stockbot.reports import write_backtest_report
from stockbot.signals import generate_signals
from stockbot.train import train_final

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = [f"S{k}" for k in range(6)]


@pytest.fixture
def small_cfg():
    path = ROOT / "config" / "default.toml"
    cfg = tomllib.loads(path.read_text())
    cfg["_path"] = str(path)
    cfg["device"] = "cpu"
    cfg["model"].update(max_epochs=2, patience=1, hidden_size=8, batch_size=256)
    cfg["walkforward"].update(oos_months=4, n_folds=2)
    cfg["strategy"]["threshold"] = 0.0  # synthetic data: make sure some trades happen
    return cfg


@pytest.fixture
def populated(conn, make_bars, add_bars):
    for k, symbol in enumerate(SYMBOLS):
        add_bars(conn, symbol, make_bars(300, seed=k, start="2024-01-01"))
    add_bars(conn, "SPY", make_bars(300, seed=99, start="2024-01-01"), benchmark=True)
    build_features(conn)
    run_leakchecks(conn, sample_size=3)
    return conn


def test_backtest_and_report(populated, small_cfg, tmp_path):
    metrics, curves, preds = run_backtest(populated, small_cfg, torch.device("cpu"))
    assert set(metrics) == {"lstm", "lstm_no_sentiment", "logreg", "random", "spy"}
    assert metrics["lstm"]["n_trades"] > 0
    assert 0 <= metrics["lstm"]["accuracy"] <= 1

    report = write_backtest_report(
        metrics,
        curves,
        preds,
        run_id="lstm-test",
        cfg=small_cfg,
        git_sha="abc123",
        out_dir=tmp_path,
    )
    text = report.read_text()
    for needle in (
        "Accuracy (95% CI)",
        "Hit rate",
        "Sharpe",
        "Max DD",
        "Trades/day",
        "Seed 42",
        "abc123",
        "seq_len = 20",
        "Survivorship bias",
    ):
        assert needle in text
    assert report.with_suffix(".png").stat().st_size > 0
    stem = report.with_suffix("")
    assert Path(f"{stem}-deciles.png").stat().st_size > 0
    assert set(pd.read_csv(f"{stem}-daily.csv", index_col=0).columns) == set(curves)
    payload = json.loads(report.with_suffix(".json").read_text())
    assert payload["seed"] == 42 and payload["git_sha"] == "abc123"


def test_train_then_signal(populated, small_cfg, tmp_path):
    model_id, val_loss = train_final(populated, small_cfg, artifacts_root=tmp_path)
    assert re.fullmatch(r"lstm-\d{8}-\d{6}", model_id)
    for name in ("model.pt", "feature_stats.json", "config_snapshot.toml"):
        assert (tmp_path / model_id / name).is_file()
    assert populated.execute("SELECT git_sha FROM models").fetchone()[0]

    last = populated.execute("SELECT MAX(session_date) FROM bars_daily").fetchone()[0]
    selected = generate_signals(populated, small_cfg, last)
    assert len(selected) <= small_cfg["strategy"]["top_n"]
    rows = populated.execute("SELECT symbol, selected FROM signals").fetchall()
    assert sorted(r[0] for r in rows) == SYMBOLS  # one row per ticker, no SPY
    assert sum(r[1] for r in rows) == len(selected)
