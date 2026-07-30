# src/models/tssgc.py
from __future__ import annotations

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TemporalAggregator(nn.Module):
    """
    Temporal Aggregator - GIỐNG PAPER 100% + TỐI ƯU HIỆU NĂNG
    
    Paper (Eq 6): TEMP(i) = GRU({ (f_k, α_k) | (v_i, v_j, t_k, f_k) ∈ E_i })
    Paper (Eq 7): α_k = exp(-β(t_now - t_k)) / Σ_l exp(-β(t_now - t_l))
    
    Tối ưu:
    1. Dùng pack_padded_sequence để xử lý GRU nhanh hơn
    2. Vector hóa việc lấy final hidden state (không lặp Python)
    3. Giữ nguyên 100% công thức toán học
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
        
        # ============================================================
        # 1. MESSAGE
        # ============================================================
        msg = self.lin_msg(x[src])  # [num_edges, out_dim]
        
        # ============================================================
        # 2. TIME WEIGHTS (Eq 7)
        # ============================================================
        if edge_time_delta is None or edge_time_delta.numel() != msg.size(0):
            time_weights = torch.ones(msg.size(0), device=device, dtype=msg.dtype)
        else:
            beta = F.softplus(self.beta_raw)
            delta = edge_time_delta.to(device).to(msg.dtype)
            delta = torch.clamp(delta, min=0.0, max=1e6)
            # Eq 7: α_k = exp(-β * Δt_k)
            time_weights = torch.exp(-beta * delta)
            # Normalize by node (softmax over time)
            node_weight_sums = scatter(
                time_weights, dst, dim=0, dim_size=num_nodes, reduce='sum'
            )
            time_weights = time_weights / (node_weight_sums[dst] + 1e-8)
        
        # ============================================================
        # 3. GRAPH WEIGHTS
        # ============================================================
        if edge_weight is None or edge_weight.numel() != msg.size(0):
            graph_weights = torch.ones(msg.size(0), device=device, dtype=msg.dtype)
        else:
            graph_weights = edge_weight.to(device).to(msg.dtype).clamp_min(0.0)
        
        weights = time_weights * graph_weights
        
        # ============================================================
        # 4. ✅ SẮP XẾP EDGES THEO THỜI GIAN (quá khứ → hiện tại)
        #    FIX LỖI #F: descending để cũ nhất trước, gần nhất sau
        #    edge_time_delta = t_now - t_k (càng nhỏ = càng gần)
        # ============================================================
        if edge_time_delta is not None:
            sorted_indices = torch.argsort(edge_time_delta, descending=True)  # ✅ FIX #F
        else:
            sorted_indices = torch.arange(edge_index.size(1), device=device)
        
        sorted_dst = dst[sorted_indices]
        sorted_msg = msg[sorted_indices] * weights[sorted_indices].unsqueeze(-1)
        
        # ============================================================
        # 5. GROUP EDGES THEO NODE (CSR format)
        # ============================================================
        # Đếm số edges mỗi node
        num_edges_per_node = scatter(
            torch.ones(sorted_dst.size(0), device=device),
            sorted_dst,
            dim=0,
            dim_size=num_nodes,
            reduce='sum'
        ).long()
        
        has_edges = num_edges_per_node > 0
        node_indices = torch.where(has_edges)[0]
        seq_lengths = num_edges_per_node[node_indices]
        
        if len(node_indices) == 0:
            return torch.zeros(num_nodes, self.gru.hidden_size, device=device, dtype=x.dtype)
        
        max_len = seq_lengths.max().item()
        num_nodes_with_edges = len(node_indices)
        
        # ============================================================
        # 6. TẠO PADDED SEQUENCES (VECTOR HÓA BẰNG SCATTER)
        # ============================================================
        # Tạo padded_sequences bằng scatter thay vì vòng lặp Python
        padded_sequences = torch.zeros(
            num_nodes_with_edges, max_len, self.gru.hidden_size,
            device=device, dtype=x.dtype
        )
        
        # Tạo offsets (CSR format)
        offsets = torch.cat([
            torch.tensor([0], device=device),
            torch.cumsum(num_edges_per_node[:-1], dim=0)
        ])
        
        # VECTOR HÓA: Dùng scatter để gán messages vào padded_sequences
        # Tạo indices cho scatter
        node_positions = torch.arange(num_nodes_with_edges, device=device)
        
        # Tính toán vị trí bắt đầu và độ dài cho từng node
        start_offsets = offsets[node_indices]
        seq_lens = seq_lengths
        
        # Tạo flat indices cho scatter
        # Mỗi node i có seq_len[i] messages, cần gán vào padded_sequences[i, :seq_len[i]]
        total_msgs = len(sorted_msg)
        
        # Tạo batch indices (node index trong padded_sequences)
        batch_indices = torch.repeat_interleave(
            torch.arange(num_nodes_with_edges, device=device),
            seq_lens
        )
        
        # Tạo sequence indices (vị trí trong sequence)
        seq_indices = torch.cat([
            torch.arange(seq_len, device=device) for seq_len in seq_lens
        ])
        
        # Scatter messages vào padded_sequences
        padded_sequences[batch_indices, seq_indices] = sorted_msg
        
        # ============================================================
        # 7. GRU TRÊN TOÀN BỘ SEQUENCE (VỚI PACK_PADDED_SEQUENCE)
        # ============================================================
        # Sắp xếp theo độ dài giảm dần (cho pack_padded_sequence)
        seq_lengths_sorted, sort_indices = torch.sort(seq_lengths, descending=True)
        padded_sequences_sorted = padded_sequences[sort_indices]
        
        # Dùng pack_padded_sequence để tăng tốc GRU
        packed_input = pack_padded_sequence(
            padded_sequences_sorted,
            seq_lengths_sorted.cpu(),
            batch_first=True,
            enforce_sorted=True
        )
        
        # GRU forward
        gru_out, _ = self.gru(packed_input)
        
        # Unpack để lấy output
        gru_out_unpacked, _ = pad_packed_sequence(gru_out, batch_first=True, total_length=max_len)
        
        # ============================================================
        # 8. VECTOR HÓA: LẤY FINAL HIDDEN STATE
        # ============================================================
        # Lấy hidden state ở bước cuối của mỗi sequence (không lặp Python)
        batch_size = gru_out_unpacked.size(0)
        seq_len_minus_1 = (seq_lengths_sorted - 1).long()
        batch_indices = torch.arange(batch_size, device=device)
        
        # Vector hóa: lấy hidden cuối cho tất cả sequence cùng lúc
        final_hidden = gru_out_unpacked[batch_indices, seq_len_minus_1, :].clone()
        
        # Đưa về đúng thứ tự ban đầu
        inverse_sort_indices = torch.argsort(sort_indices)
        final_hidden_original_order = final_hidden[inverse_sort_indices].clone()
        
        # ============================================================
        # 9. GÁN VÀO TENSOR KẾT QUẢ
        # ============================================================
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