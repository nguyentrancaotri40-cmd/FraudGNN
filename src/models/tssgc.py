# src/models/tssgc.py
from __future__ import annotations

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter


class TemporalAggregator(nn.Module):
    """
    Temporal Aggregator - GIỐNG PAPER 100%
    
    Paper (Eq 6): TEMP(i) = GRU({ (f_k, α_k) | (v_i, v_j, t_k, f_k) ∈ E_i })
    Paper (Eq 7): α_k = exp(-β(t_now - t_k)) / Σ_l exp(-β(t_now - t_l))
    
    ✅ Sử dụng GRU layer với batch_first=True
    ✅ KHÔNG dùng pack_padded_sequence
    ✅ KHÔNG có in-place operation (dùng clone())
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_msg = nn.Linear(in_dim, out_dim)
        self.gru = nn.GRU(out_dim, out_dim, batch_first=True, num_layers=1)
        self.beta_raw = nn.Parameter(torch.tensor(0.0))
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time_delta: torch.Tensor | None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        device = x.device
        src, dst = edge_index
        
        if edge_index.size(1) == 0:
            return torch.zeros(num_nodes, self.gru.hidden_size, device=device, dtype=x.dtype)
        
        # 1. MESSAGE
        msg = self.lin_msg(x[src])
        
        # 2. TIME WEIGHTS (Eq 7)
        if edge_time_delta is None or edge_time_delta.numel() != msg.size(0):
            time_weights = torch.ones(msg.size(0), device=device, dtype=msg.dtype)
        else:
            beta = F.softplus(self.beta_raw)
            delta = edge_time_delta.to(device).to(msg.dtype)
            delta = torch.clamp(delta, min=0.0, max=1e6)
            time_weights = torch.exp(-beta * delta)
            node_weight_sums = scatter(
                time_weights, dst, dim=0, dim_size=num_nodes, reduce='sum'
            )
            time_weights = time_weights / (node_weight_sums[dst] + 1e-8)
        
        # 3. GRAPH WEIGHTS
        if edge_weight is None or edge_weight.numel() != msg.size(0):
            graph_weights = torch.ones(msg.size(0), device=device, dtype=msg.dtype)
        else:
            graph_weights = edge_weight.to(device).to(msg.dtype).clamp_min(0.0)
        
        weights = time_weights * graph_weights
        
        # 4. SẮP XẾP THEO THỜI GIAN
        if edge_time_delta is not None:
            sorted_indices = torch.argsort(edge_time_delta)
        else:
            sorted_indices = torch.arange(edge_index.size(1), device=device)
        
        sorted_dst = dst[sorted_indices]
        sorted_msg = msg[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # 5. GROUP EDGES THEO NODE
        num_edges_per_node = scatter(
            torch.ones(sorted_dst.size(0), device=device),
            sorted_dst,
            dim=0,
            dim_size=num_nodes,
            reduce='sum'
        ).long()
        
        has_edges = num_edges_per_node > 0
        node_indices = torch.where(has_edges)[0]
        seq_lengths = num_edges_per_node[has_edges]
        
        if len(node_indices) == 0:
            return torch.zeros(num_nodes, self.gru.hidden_size, device=device, dtype=x.dtype)
        
        max_len = seq_lengths.max().item()
        num_nodes_with_edges = len(node_indices)
        
        # 6. TẠO PADDED SEQUENCES
        padded_sequences = torch.zeros(
            num_nodes_with_edges, max_len, self.gru.hidden_size,
            device=device, dtype=x.dtype
        )
        
        offsets = torch.cat([
            torch.tensor([0], device=device),
            torch.cumsum(num_edges_per_node[:-1], dim=0)
        ])
        
        for i, node_idx in enumerate(node_indices):
            start = offsets[node_idx].item()
            end = start + seq_lengths[i].item()
            node_msgs = sorted_msg[start:end] * sorted_weights[start:end].unsqueeze(-1)
            padded_sequences[i, :seq_lengths[i]] = node_msgs.clone()
        
        # 7. ✅ GRU TRÊN TOÀN BỘ SEQUENCE
        seq_lengths_sorted, sort_indices = torch.sort(seq_lengths, descending=True)
        padded_sequences_sorted = padded_sequences[sort_indices].clone()
        
        gru_out, h_n = self.gru(padded_sequences_sorted)
        
        batch_size = padded_sequences_sorted.size(0)
        final_hidden = torch.zeros(batch_size, self.gru.hidden_size, device=device, dtype=x.dtype)
        
        for i, length in enumerate(seq_lengths_sorted):
            seq_len = length.item()
            if seq_len > 0:
                final_hidden[i] = gru_out[i, seq_len - 1, :].clone()
            else:
                final_hidden[i] = torch.zeros(self.gru.hidden_size, device=device, dtype=x.dtype)
        
        inverse_sort_indices = torch.argsort(sort_indices)
        final_hidden_original_order = final_hidden[inverse_sort_indices].clone()
        
        h = torch.zeros(num_nodes, self.gru.hidden_size, device=device, dtype=x.dtype)
        h[node_indices] = final_hidden_original_order.clone()
        
        return h


class SemanticEncoder(nn.Module):
    """Semantic encoder - GIỐNG PAPER (Eq 10)."""
    
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
                f"SemanticEncoder: Found {invalid_count} invalid node_type indices. Clamping.",
                UserWarning
            )
            node_type = node_type.clamp(min=0, max=self.num_node_types - 1)
        
        type_vec = self.type_emb(node_type)
        return self.lin(torch.cat([x, type_vec], dim=-1))


class TSSGCLayer(nn.Module):
    """TSSGC Layer - GIỐNG PAPER (Eq 11)."""
    
    def __init__(self, in_dim: int, out_dim: int, num_node_types: int = 1, dropout: float = 0.2):
        super().__init__()
        self.temporal = TemporalAggregator(in_dim, out_dim)
        self.spatial = GATConv(in_dim, out_dim, heads=1, concat=False, dropout=dropout, add_self_loops=False)
        self.semantic = SemanticEncoder(in_dim, out_dim, num_node_types=num_node_types)
        self.W_t = nn.Linear(out_dim, out_dim, bias=False)
        self.W_s = nn.Linear(out_dim, out_dim, bias=False)
        self.W_m = nn.Linear(out_dim, out_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(out_dim))
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
        out = self.W_t(temp) + self.W_s(spat) + self.W_m(sem) + self.b
        out = self.norm(out)
        out = F.relu(out)
        return self.dropout(out)


class TSSGCEncoder(nn.Module):
    """TSSGC Encoder - GIỐNG PAPER."""
    
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


__all__ = [
    'TemporalAggregator',
    'SemanticEncoder',
    'TSSGCLayer',
    'TSSGCEncoder',
]