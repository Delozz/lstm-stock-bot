"""LSTM training with early stopping, and the final `train` command (FR-21 to FR-26)."""

from __future__ import annotations

import copy
import json
import logging
import random
import shutil
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from stockbot.dataset import fit_stats, load_features, make_windows, normalize, split_val
from stockbot.features import FEATURES
from stockbot.models import LSTMClassifier

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


@torch.no_grad()
def predict_proba(
    model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 4096
) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        out.append(torch.sigmoid(model(xb)).squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, np.float32)


@torch.no_grad()
def _loss(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    p = np.clip(predict_proba(model, X, device), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_lstm(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mcfg: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[LSTMClassifier, float]:
    """Train with AdamW + BCE, early-stop on validation loss, return the best weights."""
    set_seed(seed)
    model = LSTMClassifier(
        X_fit.shape[2], mcfg["hidden_size"], mcfg["num_layers"], mcfg["dropout"]
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    X_t, y_t = torch.from_numpy(X_fit), torch.from_numpy(y_fit).unsqueeze(1)
    gen = torch.Generator().manual_seed(seed)

    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(mcfg["max_epochs"]):
        model.train()
        perm = torch.randperm(len(X_t), generator=gen)
        for i in range(0, len(perm), mcfg["batch_size"]):
            idx = perm[i : i + mcfg["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(X_t[idx].to(device)), y_t[idx].to(device))
            loss.backward()
            opt.step()

        val_loss = _loss(model, X_val, y_val, device)
        log.info("epoch %d val_loss %.5f", epoch + 1, val_loss)
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= mcfg["patience"]:
                break

    model.load_state_dict(best_state)
    return model, best_loss


def train_final(
    conn: sqlite3.Connection, cfg: dict[str, Any], artifacts_root: str | Path = "artifacts/models"
) -> tuple[str, float]:
    """Train one model on all labeled data and register it in `models`."""
    mcfg, seed = cfg["model"], cfg["seed"]
    device = pick_device(cfg.get("device", "auto"))
    log.info("seed %d, device %s", seed, device)

    df = load_features(conn)
    labeled = df[df["label"].notna()]
    fit_dates, val_dates = split_val(
        labeled["session_date"].unique(), cfg["walkforward"]["val_frac"]
    )
    mean, std = fit_stats(labeled[labeled["session_date"].isin(fit_dates)], FEATURES)

    w = make_windows(df, FEATURES, mcfg["seq_len"])
    X = normalize(w.X, mean, std)
    has_label = ~np.isnan(w.y)
    fit = has_label & np.isin(w.dates, fit_dates)
    val = has_label & np.isin(w.dates, val_dates)
    model, val_loss = fit_lstm(X[fit], w.y[fit], X[val], w.y[val], mcfg, device, seed)

    now = datetime.now(UTC)
    model_id = f"lstm-{now:%Y%m%d-%H%M%S}"
    out = Path(artifacts_root) / model_id
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), out / "model.pt")
    (out / "feature_stats.json").write_text(
        json.dumps({"features": FEATURES, "mean": mean.tolist(), "std": std.tolist()}, indent=2)
    )
    shutil.copyfile(cfg["_path"], out / "config_snapshot.toml")

    conn.execute(
        "INSERT INTO models (model_id, trained_at_utc, train_end_date, seed, git_sha, val_loss, "
        "artifact_dir) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            model_id,
            now.isoformat(timespec="seconds"),
            max(val_dates),
            seed,
            git_sha(),
            val_loss,
            str(out),
        ),
    )
    conn.commit()
    return model_id, val_loss
