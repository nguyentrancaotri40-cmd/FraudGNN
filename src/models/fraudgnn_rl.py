# ============================================================
# src/models/fraudgnn_rl.py — encoder (TSSGC | MLP) + classifier + feature importance weights.
#
# Feature importance weights (Action của RL, IV-B) nhân vào embedding TRƯỚC classifier.
# Dạng chuẩn hoá: w = D * softmax-like(action)  -> trung bình = 1 (trung tính khi đều),
# nên classifier huấn luyện với w = 1 vẫn dùng được khi RL đổi trọng số.
#
# shared_projection (Graph alignment cho FL, IV-C) [INFERRED, đã xác nhận phương án B]:
#   Linear(in_dim -> hidden) dùng chung, được FedAvg. Áp dụng ở MỌI chế độ (FL & non-FL)
#   để ablation 'w/o FL' chỉ khác ở cách train.
# ============================================================
from __future__ import annotations

import torch
import torch.nn as nn

from .classifier import FraudClassifier
from .tssgc import MLPEncoder, TSSGCEncoder


class FraudGNNRL(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, num_node_types: int = 1,
                 dropout: float = 0.2, use_gnn: bool = True, shared_projection: bool = True,
                 temporal_max_seq_len: int | None = None):
        super().__init__()
        self.hidden_dim, self.use_gnn = hidden_dim, use_gnn
        self.projection = nn.Linear(in_dim, hidden_dim) if shared_projection else nn.Identity()
        enc_in = hidden_dim if shared_projection else in_dim
        self.encoder = (TSSGCEncoder(enc_in, hidden_dim, num_layers, num_node_types, dropout, temporal_max_seq_len)
                        if use_gnn else MLPEncoder(enc_in, hidden_dim, num_layers, dropout))
        self.classifier = FraudClassifier(hidden_dim, hidden_dim, dropout)
        self.register_buffer("feature_weights", torch.ones(hidden_dim))  # cập nhật bởi RL, không qua gradient

    # -- RL hooks -----------------------------------------------------
    def set_feature_weights(self, w: torch.Tensor) -> None:
        w = torch.as_tensor(w, dtype=torch.float32, device=self.feature_weights.device).flatten()
        if w.numel() != self.hidden_dim:
            raise ValueError(f"feature weights cần {self.hidden_dim} phần tử, nhận {w.numel()}")
        self.feature_weights.copy_(w)

    def reset_feature_weights(self) -> None:
        self.feature_weights.fill_(1.0)

    # -- forward ------------------------------------------------------
    def encode(self, data) -> torch.Tensor:
        dev = self.feature_weights.device
        x = self.projection(data.x.to(dev))
        ei = data.edge_index.to(dev)
        dt = getattr(data, "edge_time_delta", None)
        nt = getattr(data, "node_type", None)
        return self.encoder(x, ei, None if dt is None else dt.to(dev), None if nt is None else nt.to(dev))

    def logits_from_embedding(self, emb: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        w = self.feature_weights if weights is None else weights
        return self.classifier(emb * w)

    def forward(self, data) -> torch.Tensor:
        return self.logits_from_embedding(self.encode(data))

    @torch.no_grad()
    def embeddings(self, data) -> torch.Tensor:
        """Embedding TRƯỚC khi nhân feature weights (State của RL)."""
        return self.encode(data)


__all__ = ["FraudGNNRL"]
