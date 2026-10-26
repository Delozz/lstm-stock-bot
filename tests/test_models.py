"""AC-18, AC-19: model shapes, overfit sanity check, baseline output range."""

from __future__ import annotations

import numpy as np
import torch

from stockbot.models import LSTMClassifier, make_baseline
from stockbot.train import set_seed


def test_forward_shape():
    model = LSTMClassifier(13)
    assert model(torch.randn(8, 20, 13)).shape == (8, 1)


def test_lstm_can_overfit_small_batch():
    set_seed(0)
    X = torch.randn(64, 20, 13)
    y = torch.randint(0, 2, (64, 1)).float()
    model = LSTMClassifier(13)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(200):
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = ((model(X) > 0).float() == y).float().mean().item()
    assert acc >= 0.95


def test_baseline_probabilities_in_range():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 13))
    y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    p = make_baseline(42).fit(X, y).predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
