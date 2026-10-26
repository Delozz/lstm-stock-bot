"""Nightly signal generation from the newest trained model (FR-32 to FR-34)."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from stockbot.config import DataError
from stockbot.dataset import load_features, make_windows, normalize
from stockbot.db import upsert
from stockbot.models import LSTMClassifier
from stockbot.train import predict_proba

log = logging.getLogger(__name__)


def latest_model(conn: sqlite3.Connection) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM models ORDER BY trained_at_utc DESC LIMIT 1").fetchone()
    if row is None:
        raise DataError("no trained model; run `stockbot train` first")
    return row


def already_processed(conn: sqlite3.Connection, session_date: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM signals s JOIN models m USING (model_id) WHERE s.session_date = ? "
        "AND m.model_id = (SELECT model_id FROM models ORDER BY trained_at_utc DESC LIMIT 1) "
        "LIMIT 1",
        (session_date,),
    ).fetchone()
    return row is not None


def check_staleness(conn: sqlite3.Connection, session_date: str, max_frac: float) -> list[str]:
    """Refuse to signal if too many active tickers are missing the latest session."""
    rows = conn.execute(
        "SELECT t.symbol, MAX(b.session_date) FROM tickers t "
        "LEFT JOIN bars_daily b USING (symbol) "
        "WHERE t.active = 1 AND t.is_benchmark = 0 GROUP BY t.symbol"
    ).fetchall()
    stale = [symbol for symbol, last in rows if last != session_date]
    if rows and len(stale) > max_frac * len(rows):
        raise DataError(f"{len(stale)} of {len(rows)} tickers have no bar for {session_date}")
    if stale:
        log.warning("stale tickers skipped: %s", ",".join(stale))
    return stale


def generate_signals(conn: sqlite3.Connection, cfg: dict[str, Any], session_date: str) -> list[str]:
    """Score every active ticker for session_date and mark the top picks. Returns selected."""
    scfg = cfg["strategy"]
    check_staleness(conn, session_date, cfg["data"]["stale_max_frac"])

    model_row = latest_model(conn)
    artifact_dir = Path(model_row["artifact_dir"])
    for name in ("model.pt", "feature_stats.json"):
        if not (artifact_dir / name).is_file():
            raise DataError(f"missing model artifact {artifact_dir / name}")
    stats = json.loads((artifact_dir / "feature_stats.json").read_text())
    features = stats["features"]

    w = make_windows(load_features(conn), features, cfg["model"]["seq_len"])
    mask = w.dates == session_date
    if not mask.any():
        raise DataError(f"no feature windows end on {session_date}; run `stockbot features`")
    X = normalize(w.X[mask], np.asarray(stats["mean"]), np.asarray(stats["std"]))

    mcfg = cfg["model"]
    model = LSTMClassifier(len(features), mcfg["hidden_size"], mcfg["num_layers"], mcfg["dropout"])
    model.load_state_dict(
        torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True)
    )
    p_up = predict_proba(model, X, torch.device("cpu"))

    ranked = pd.DataFrame({"symbol": w.symbols[mask], "p_up": p_up}).sort_values(
        ["p_up", "symbol"], ascending=[False, True]
    )
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["selected"] = (
        (ranked["rank"] <= scfg["top_n"]) & (ranked["p_up"] >= scfg["threshold"])
    ).astype(int)
    upsert(
        conn,
        "signals",
        [
            {
                "session_date": session_date,
                "symbol": r.symbol,
                "model_id": model_row["model_id"],
                "p_up": float(r.p_up),
                "rank": int(r.rank),
                "selected": int(r.selected),
            }
            for r in ranked.itertuples()
        ],
        ("session_date", "symbol", "model_id"),
    )

    selected = ranked.loc[ranked["selected"] == 1, "symbol"].tolist()
    log.info("signals for %s: %d scored, %d selected", session_date, len(ranked), len(selected))
    return selected
