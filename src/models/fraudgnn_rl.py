# src/models/fraudgnn_rl.py
from __future__ import annotations

import torch
import torch.nn as nn
from .tssgc import TSSGCEncoder
from .classifier import FraudClassifier


class FraudGNNRL(nn.Module):
    """
    TSSGC encoder + classifier head.
    
    DQN/NAF threshold adjustment is trained separately in src/train/train_rl.py because it
    operates on validation/inference score streams.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_node_types: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        self.encoder = TSSGCEncoder(in_dim, hidden_dim, num_layers, num_node_types, dropout)
        self.classifier = FraudClassifier(hidden_dim, hidden_dim, dropout)

    def forward(self, data) -> torch.Tensor:
        emb = self.encoder(
            data.x,
            data.edge_index,
            getattr(data, "edge_time_delta", None),
            getattr(data, "node_type", None),
            getattr(data, "edge_weight", None),
        )
        return self.classifier(emb)

    @torch.no_grad()
    def embeddings(self, data) -> torch.Tensor:
        return self.encoder(
            data.x,
            data.edge_index,
            getattr(data, "edge_time_delta", None),
            getattr(data, "node_type", None),
            getattr(data, "edge_weight", None),
        )