# src/models/classifier.py
from __future__ import annotations

import torch
import torch.nn as nn


class FraudClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(embedding))