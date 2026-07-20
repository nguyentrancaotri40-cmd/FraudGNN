# src/models/tssgc.py
from __future__ import annotations

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter


class TemporalAggregator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_msg = nn.Linear(in_dim, out_dim)
        self.gru = nn.GRUCell(out_dim, out_dim)
        self.beta_raw = nn.Parameter(torch.tensor(0.0))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time_delta: torch.Tensor | None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        src, dst = edge_index
        msg = self.lin_msg(x[src])
        
        if edge_time_delta is None or edge_time_delta.numel() != msg.size(0):
            time_weights = torch.ones(msg.size(0), device=x.device, dtype=msg.dtype)
        else:
            beta = F.softplus(self.beta_raw)
            delta = edge_time_delta.to(x.device).to(msg.dtype)
            delta = torch.clamp(delta, min=0.0, max=1e6)
            time_weights = torch.exp(-beta * delta)
        
        if edge_weight is None or edge_weight.numel() != msg.size(0):
            graph_weights = torch.ones(msg.size(0), device=x.device, dtype=msg.dtype)
        else:
            graph_weights = edge_weight.to(x.device).to(msg.dtype).clamp_min(0.0)
        
        weights = time_weights * graph_weights
        
        weighted_msg = msg * weights.unsqueeze(-1)
        denom = scatter(weights, dst, dim=0, dim_size=num_nodes, reduce="sum").clamp_min(1e-8).unsqueeze(-1)
        agg = scatter(weighted_msg, dst, dim=0, dim_size=num_nodes, reduce="sum") / denom
        
        h0 = torch.zeros(num_nodes, self.gru.hidden_size, device=x.device, dtype=x.dtype)
        h = self.gru(agg, h0)
        
        return h


class SemanticEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_node_types: int = 1):
        super().__init__()
        self.num_node_types = max(1, num_node_types)
        self.type_emb = nn.Embedding(self.num_node_types, out_dim)
        self.lin = nn.Linear(in_dim + out_dim, out_dim)
    
    def forward(self, x: torch.Tensor, node_type: torch.Tensor | None) -> torch.Tensor:
        if node_type is None:
            node_type = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        if node_type.min() < 0 or node_type.max() >= self.num_node_types:
            invalid_count = ((node_type < 0) | (node_type >= self.num_node_types)).sum().item()
            warnings.warn(
                f"SemanticEncoder: Found {invalid_count} invalid node_type indices "
                f"(min={node_type.min().item()}, max={node_type.max().item()}, "
                f"num_types={self.num_node_types}). Clamping to valid range.",
                UserWarning
            )
            node_type = node_type.clamp(min=0, max=self.num_node_types - 1)
        
        type_vec = self.type_emb(node_type)
        return self.lin(torch.cat([x, type_vec], dim=-1))


class TSSGCLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_node_types: int = 1, dropout: float = 0.2):
        super().__init__()
        self.temporal = TemporalAggregator(in_dim, out_dim)
        self.spatial = GATConv(in_dim, out_dim, heads=1, concat=False, dropout=dropout, add_self_loops=False)
        self.semantic = SemanticEncoder(in_dim, out_dim, num_node_types=num_node_types)
        self.combine = nn.Linear(out_dim * 3, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time_delta: torch.Tensor | None = None,
        node_type: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        temp = self.temporal(x, edge_index, edge_time_delta, edge_weight=edge_weight)
        spat = self.spatial(x, edge_index)
        sem = self.semantic(x, node_type)
        out = self.combine(torch.cat([temp, spat, sem], dim=-1))
        out = self.norm(out)
        out = F.relu(out)
        return self.dropout(out)


class TSSGCEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_node_types: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        
        layers = []
        for i in range(num_layers):
            layers.append(TSSGCLayer(
                in_dim=in_dim if i == 0 else hidden_dim,
                out_dim=hidden_dim,
                num_node_types=num_node_types,
                dropout=dropout,
            ))
        self.layers = nn.ModuleList(layers)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time_delta: torch.Tensor | None = None,
        node_type: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_time_delta, node_type, edge_weight)
        return x