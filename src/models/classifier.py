# ============================================================
# src/models/classifier.py — Fraud classifier head (Fig. 1: "Classification").
# Linear(hidden->hidden) -> ReLU -> Dropout -> Linear(hidden->1)  [INFERRED: paper không nêu cấu trúc]
# ============================================================
from __future__ import annotations

import torch
import torch.nn as nn


class FraudClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb).squeeze(-1)


__all__ = ["FraudClassifier"]
