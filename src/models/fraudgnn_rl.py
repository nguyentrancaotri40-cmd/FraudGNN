# ============================================================
# src/models/fraudgnn_rl.py - FULL FIX
# ============================================================

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
        dropout: float = 0.2,
        use_projection: bool = False,
        projection_dim: int = 64,
    ):
        super().__init__()
        
        # ✅ FIX: Projection layer (trainable) cho shared_encoder
        self.use_projection = use_projection
        if use_projection:
            self.projection = nn.Linear(in_dim, projection_dim)
            actual_in_dim = projection_dim
            print(f"[FraudGNNRL] Using trainable projection: {in_dim} → {projection_dim}")
        else:
            self.projection = nn.Identity()
            actual_in_dim = in_dim
        
        self.encoder = TSSGCEncoder(actual_in_dim, hidden_dim, num_layers, num_node_types, dropout)
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
        # ✅ FIX: Đưa tất cả về cùng device
        device = self._feature_weights.device
        
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_time_delta = getattr(data, "edge_time_delta", None)
        node_type = getattr(data, "node_type", None)
        edge_weight = getattr(data, "edge_weight", None)
        
        if edge_time_delta is not None:
            edge_time_delta = edge_time_delta.to(device)
        if node_type is not None:
            node_type = node_type.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
        
        # ✅ FIX: Projection (trainable)
        x = self.projection(x)
        
        emb = self.encoder(x, edge_index, edge_time_delta, node_type, edge_weight)
        weighted_emb = emb * self._feature_weights.to(emb.device).unsqueeze(0)
        
        return self.classifier(weighted_emb)
    
    @torch.no_grad()
    def embeddings(self, data) -> torch.Tensor:
        """✅ FIX: Lấy embeddings với projection (dùng cho RL State)."""
        device = self._feature_weights.device
        
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_time_delta = getattr(data, "edge_time_delta", None)
        node_type = getattr(data, "node_type", None)
        edge_weight = getattr(data, "edge_weight", None)
        
        if edge_time_delta is not None:
            edge_time_delta = edge_time_delta.to(device)
        if node_type is not None:
            node_type = node_type.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
        
        # ✅ FIX: Áp dụng projection
        x = self.projection(x)
        
        emb = self.encoder(x, edge_index, edge_time_delta, node_type, edge_weight)
        return emb * self._feature_weights.to(emb.device).unsqueeze(0)