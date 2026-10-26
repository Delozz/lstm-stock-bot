"""The global LSTM classifier and the logistic-regression baseline (FR-20, FR-21)."""

from __future__ import annotations

import torch
from sklearn.linear_model import LogisticRegression
from torch import nn


class LSTMClassifier(nn.Module):
    """(batch, seq_len, n_features) -> one logit per sequence; p_up = sigmoid(logit)."""

    def __init__(
        self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :]))


def make_baseline(seed: int) -> LogisticRegression:
    """Logistic regression on the last day's features only."""
    return LogisticRegression(max_iter=1000, random_state=seed)
