# ============================================================
# src/models/tssgc.py
#
# TSSGC Encoder — Eq. 5–11 (Section IV-A).
#
# Paper (nguyên văn):
#   Eq. 5  : h_i^(l+1) = σ( W_t·TEMP(i) + W_s·SPAT(i) + W_m·SEM(i) + b )
#   Eq. 6  : TEMP(i) = GRU({(f_k, α_k) | (v_i, v_j, t_k, f_k) ∈ E_i})
#   Eq. 7  : α_k = exp(-β(t_now - t_k)) / Σ_l exp(-β(t_now - t_l))
#   Eq. 8  : SPAT(i) = Σ_{j∈N(i)} α_ij · W_s · h_j
#   Eq. 9  : α_ij = exp(LeakyReLU(a^T[W_s h_i || W_s h_j])) / Σ_k exp(...)
#   Eq. 10 : SEM(i) = W_m [h_i || e_type(i)]
#   Eq. 11 : h_i^(l+1) = σ(...) — cùng Eq. 5
#
# Implementation Details (V-A-4):
#   - 3 graph convolution layers, 64 hidden units
#   - ReLU activation, BatchNorm sau mỗi conv layer
#
# 32 dòng đã xác nhận:
#   #5 β = softplus(β_raw) để đảm bảo dương [INFERRED — paper không nói]
#   #6 GRU sequential (sort theo tăng dần, hidden truyền qua từng bước)
#   #7 Dùng GATConv 1-head (bám Eq. 8-9, không bám văn xuôi "GCN")
#   #8 e_type: PaySim 'type', IEEE-CIS 'ProductCD', CC2023 single-type
#   #32 BatchNorm: SAU combine 3 nhánh + bias, TRƯỚC ReLU [INFERRED — thứ tự
#       paper không rõ; paper chỉ nói "after each conv layer"]
# ============================================================

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ======================================================================
# TEMP(i) — Eq. 6, 7
# ======================================================================
class TemporalAggregator(nn.Module):
    """
    Eq. 6: TEMP(i) = GRU({(f_k, α_k) | (v_i, v_j, t_k, f_k) ∈ E_i})
    Eq. 7: α_k = exp(-β Δt_k) / Σ_l exp(-β Δt_l)

    β = softplus(β_raw) đảm bảo β > 0 [INFERRED — paper không nói cách đảm bảo].

    GRU sequential (32 dòng #6):
      - Sort cạnh theo tăng dần Δt (cũ → mới).
      - Feed lần lượt f_k · α_k qua GRUCell, hidden truyền từ bước trước.
      - Lấy hidden cuối cùng làm TEMP(i).
      - Dùng pack_padded_sequence để vector hóa trên batch.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_msg = nn.Linear(in_dim, out_dim)
        self.gru = nn.GRU(out_dim, out_dim, batch_first=True, num_layers=1)
        # β_raw → β = softplus(β_raw) > 0
        self.beta_raw = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time_delta: torch.Tensor | None,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        device = x.device
        src, dst = edge_index

        if edge_index.size(1) == 0:
            return torch.zeros(num_nodes, self.gru.hidden_size, device=device, dtype=x.dtype)

        # 1) message = W_msg · x[src]
        msg = self.lin_msg(x[src])  # [E, out_dim]

        # 2) α_k — Eq. 7
        beta = F.softplus(self.beta_raw)
        if edge_time_delta is None or edge_time_delta.numel() != msg.size(0):
            # Fallback khi không có thời gian — α_k = 1/deg (chuẩn hóa)
            weights = torch.ones(msg.size(0), device=device, dtype=msg.dtype)
        else:
            delta = edge_time_delta.to(device).to(msg.dtype)
            delta = torch.clamp(delta, min=0.0, max=1e6)
            weights = torch.exp(-beta * delta)

        # Chuẩn hóa theo dst (mẫu số Eq. 7)
        denom = scatter(weights, dst, dim=0, dim_size=num_nodes, reduce="sum")
        weights = weights / (denom[dst] + 1e-12)

        # 3) Sort edges: nhóm theo dst, trong mỗi dst sort Δt giảm dần (cũ→mới)
        if edge_time_delta is not None and edge_time_delta.numel() == msg.size(0):
            # Bước A: sort ổn định Δt giảm dần
            o1 = torch.argsort(edge_time_delta.to(device), descending=True, stable=True)
            # Bước B: sort ổn định dst
            sorted_idx = o1[torch.argsort(dst[o1], stable=True)]
        else:
            sorted_idx = torch.argsort(dst, stable=True)

        sorted_dst = dst[sorted_idx]
        sorted_msg = msg[sorted_idx] * weights[sorted_idx].unsqueeze(-1)

        # 4) Đếm số cạnh / node
        deg = scatter(
            torch.ones(sorted_dst.size(0), device=device),
            sorted_dst, dim=0, dim_size=num_nodes, reduce="sum",
        ).long()

        has_edge = deg > 0
        node_idx = torch.where(has_edge)[0]
        seq_len = deg[node_idx]
        if len(node_idx) == 0:
            return torch.zeros(num_nodes, self.gru.hidden_size, device=device, dtype=x.dtype)

        # 5) Padded sequences [M, max_len, D]
        max_len = int(seq_len.max().item())
        M = int(node_idx.numel())
        padded = torch.zeros(M, max_len, self.gru.hidden_size, device=device, dtype=x.dtype)

        # CSR offsets cho node có cạnh
        offsets = torch.cat([
            torch.tensor([0], device=device),
            torch.cumsum(deg[:-1], dim=0),
        ])
        offsets_valid = offsets[node_idx]

        # Gán message vào padded theo đúng thứ tự đã sort
        batch_indices = torch.repeat_interleave(
            torch.arange(M, device=device), seq_len,
        )
        seq_indices = torch.cat(
            [torch.arange(int(l), device=device) for l in seq_len.tolist()]
        )
        msg_idx = torch.cat([
            torch.arange(int(offsets_valid[i]),
                        int(offsets_valid[i]) + int(seq_len[i]),
                        device=device)
            for i in range(M)
        ])
        padded[batch_indices, seq_indices] = sorted_msg[msg_idx]

        # 6) GRU sequential qua packed sequence
        seq_len_sorted, sort_idx = torch.sort(seq_len, descending=True)
        padded_sorted = padded[sort_idx]

        packed = pack_padded_sequence(
            padded_sorted, seq_len_sorted.cpu(), batch_first=True, enforce_sorted=True,
        )
        gru_out, _ = self.gru(packed)
        out_unpacked, _ = pad_packed_sequence(gru_out, batch_first=True, total_length=max_len)

        # 7) Lấy hidden CUỐI CÙNG (tại vị trí seq_len-1)
        B = out_unpacked.size(0)
        last_idx = (seq_len_sorted - 1).long()
        batch_arange = torch.arange(B, device=device)
        final_hidden = out_unpacked[batch_arange, last_idx, :]  # [B, D]

        # Đảo sort
        inv_sort = torch.argsort(sort_idx)
        final_hidden = final_hidden[inv_sort]

        # 8) Ghi lại vào tensor [N, D]
        h = torch.zeros(num_nodes, self.gru.hidden_size, device=device, dtype=x.dtype)
        h[node_idx] = final_hidden
        return h


# ======================================================================
# SEM(i) — Eq. 10
# ======================================================================
class SemanticEncoder(nn.Module):
    """
    Eq. 10: SEM(i) = W_m [h_i || e_type(i)]

    - e_type: nn.Embedding(num_node_types, out_dim)
    - W_m: nn.Linear(in_dim + out_dim, out_dim) — KHÔNG activation trung gian
      (activation chỉ nằm ngoài Eq. 11)
    """

    def __init__(self, in_dim: int, out_dim: int, num_node_types: int = 1):
        super().__init__()
        self.num_node_types = max(1, num_node_types)
        self.type_emb = nn.Embedding(self.num_node_types, out_dim)
        self.lin = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, x: torch.Tensor, node_type: torch.Tensor | None) -> torch.Tensor:
        if node_type is None:
            node_type = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            node_type = node_type.to(x.device).long()

        if node_type.numel() > 0:
            bad = ((node_type < 0) | (node_type >= self.num_node_types)).sum().item()
            if bad > 0:
                warnings.warn(
                    f"SemanticEncoder: {bad} node_type ngoài [{0},{self.num_node_types}) → clamp.",
                    UserWarning,
                )
                node_type = node_type.clamp(0, self.num_node_types - 1)

        type_vec = self.type_emb(node_type)
        return self.lin(torch.cat([x, type_vec], dim=-1))


# ======================================================================
# TSSGC Layer — Eq. 11
# ======================================================================
class TSSGCLayer(nn.Module):
    """
    Eq. 11: h_i^(l+1) = σ( W_t·TEMP(i) + W_s·SPAT(i) + W_m·SEM(i) + b )

    - SPAT dùng GATConv 1-head (Eq. 8-9), add_self_loops=False
    - BatchNorm SAU combine 3 nhánh + bias, TRƯỚC ReLU [INFERRED — paper
      chỉ nói "after each conv layer", không rõ trước/sau ReLU]
    """

    def __init__(self, in_dim: int, out_dim: int,
                 num_node_types: int = 1, dropout: float = 0.2):
        super().__init__()
        self.temporal = TemporalAggregator(in_dim, out_dim)
        # GATConv 1-head, KHÔNG self-loop
        self.spatial = GATConv(
            in_dim, out_dim,
            heads=1, concat=False,
            dropout=dropout,
            add_self_loops=False,
        )
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
        edge_weight: torch.Tensor | None = None,  # giữ tham số để tương thích, không dùng
    ) -> torch.Tensor:
        temp = self.temporal(x, edge_index, edge_time_delta)
        spat = self.spatial(x, edge_index)
        sem = self.semantic(x, node_type)
        out = self.W_t(temp) + self.W_s(spat) + self.W_m(sem) + self.b
        out = self.norm(out)
        out = F.relu(out)
        return self.dropout(out)


# ======================================================================
# TSSGC Encoder
# ======================================================================
class TSSGCEncoder(nn.Module):
    """
    3 layers, hidden_dim=64 (paper V-A-4).
    """

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
            raise ValueError("num_layers >= 1")
        self.layers = nn.ModuleList([
            TSSGCLayer(
                in_dim=in_dim if i == 0 else hidden_dim,
                out_dim=hidden_dim,
                num_node_types=num_node_types,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

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


__all__ = ["TemporalAggregator", "SemanticEncoder", "TSSGCLayer", "TSSGCEncoder"]