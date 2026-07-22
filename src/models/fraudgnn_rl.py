# src/models/fraudgnn_rl.py
from __future__ import annotations

import torch
import torch.nn as nn
from .tssgc import TSSGCEncoder
from .classifier import FraudClassifier


class FraudGNNRL(nn.Module):
    """TSSGC encoder + classifier head + feature importance weighting."""
    
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
        
        # ✅ Feature importance weights (khởi tạo đều)
        self._feature_weights = nn.Parameter(
            torch.ones(hidden_dim) / hidden_dim,
            requires_grad=False
        )
    
    def set_feature_weights(self, weights: torch.Tensor):
        """Set feature importance weights from RL agent."""
        if weights.numel() == self._feature_weights.numel():
            self._feature_weights.data.copy_(weights.to(self._feature_weights.device))
    
    def forward(self, data) -> torch.Tensor:
        emb = self.encoder(
            data.x,
            data.edge_index,
            getattr(data, "edge_time_delta", None),
            getattr(data, "node_type", None),
            getattr(data, "edge_weight", None),
        )
        # ✅ Apply feature importance weights
        weighted_emb = emb * self._feature_weights.unsqueeze(0)
        return self.classifier(weighted_emb)
    
    @torch.no_grad()
    def embeddings(self, data) -> torch.Tensor:
        emb = self.encoder(
            data.x,
            data.edge_index,
            getattr(data, "edge_time_delta", None),
            getattr(data, "node_type", None),
            getattr(data, "edge_weight", None),
        )
        return emb * self._feature_weights.unsqueeze(0)