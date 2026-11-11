# ============================================================
# src/models/tssgc.py — TSSGC (Paper IV-A, Eq. 5-11)
#
#   Eq. 5/11 : h_i^(l+1) = sigma( W_t.TEMP(i) + W_s.SPAT(i) + W_m.SEM(i) + b )
#   Eq. 6    : TEMP(i)  = GRU({(f_k, alpha_k) | (v_i,v_j,t_k,f_k) in E_i})
#   Eq. 7    : alpha_k  = exp(-beta (t_now - t_k)) / sum_l exp(-beta (t_now - t_l))
#   Eq. 8-9  : SPAT(i)  = sum_j alpha_ij W_s h_j,  alpha_ij = softmax(LeakyReLU(a^T[W_s h_i || W_s h_j]))
#   Eq. 10   : SEM(i)   = W_m [h_i || e_type(i)]
#   V-A-4    : 3 lớp, 64 hidden, ReLU, BatchNorm sau mỗi lớp conv.
#
# Lựa chọn [INFERRED] (paper không nói):
#   - beta = softplus(beta_raw) > 0 (learnable, Eq. 7 nói "learnable parameter").
#   - Dãy cạnh đưa vào GRU: sort theo |dt| giảm dần (cũ -> mới), đầu vào = alpha_k * W_msg(x_j);
#     lấy hidden bước cuối.
#   - |dt| = |t_i - t_j| của cạnh (graph vô hướng).
#   - SPAT dùng GATConv 1 head (bám Eq. 8-9 thay vì chữ "GCN" ở văn xuôi), không self-loop.
#   - BatchNorm đặt sau khi cộng 3 nhánh + bias, trước ReLU.
#   - temporal_max_seq_len (None = paper): cap kỹ thuật, giữ N cạnh gần nhất cho GRU.
# ============================================================
from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax as pyg_softmax


class TemporalAggregator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, max_seq_len: Optional[int] = None):
        super().__init__()
        self.out_dim = out_dim
        self.max_seq_len = max_seq_len
        self.lin_msg = nn.Linear(in_dim, out_dim)
        self.gru = nn.GRU(out_dim, out_dim, batch_first=True)
        self.beta_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, edge_index, edge_time_delta):
        N, D, dev = x.size(0), self.out_dim, x.device
        E = edge_index.size(1)
        if E == 0:
            return x.new_zeros(N, D)
        src, dst = edge_index
        msg = self.lin_msg(x[src])
        delta = (edge_time_delta.to(dev).to(msg.dtype).clamp(min=0)
                 if edge_time_delta is not None and edge_time_delta.numel() == E
                 else torch.zeros(E, device=dev, dtype=msg.dtype))

        alpha = pyg_softmax(-F.softplus(self.beta_raw) * delta, dst, num_nodes=N)  # Eq. 7
        msg = msg * alpha.unsqueeze(-1)

        # sắp xếp: theo dst, trong mỗi dst từ CŨ (delta lớn) -> MỚI (delta nhỏ)
        o1 = torch.argsort(delta, descending=True, stable=True)
        order = o1[torch.argsort(dst[o1], stable=True)]
        dst_s, msg_s = dst[order], msg[order]

        deg = torch.bincount(dst_s, minlength=N)
        ptr = torch.cumsum(deg, 0) - deg
        pos = torch.arange(E, device=dev) - ptr[dst_s]
        if self.max_seq_len is not None:  # giữ các cạnh MỚI nhất
            drop = (deg - self.max_seq_len).clamp(min=0)
            keep = pos >= drop[dst_s]
            dst_s, msg_s, pos = dst_s[keep], msg_s[keep], pos[keep] - drop[dst_s][keep]
            deg = deg.clamp(max=self.max_seq_len)

        nodes = (deg > 0).nonzero(as_tuple=True)[0]
        row = torch.full((N,), -1, dtype=torch.long, device=dev)
        row[nodes] = torch.arange(nodes.numel(), device=dev)
        L = int(deg.max().item())
        padded = msg_s.new_zeros(nodes.numel(), L, D)
        padded[row[dst_s], pos] = msg_s
        out, _ = self.gru(padded)                                   # GRU nhân quả: đệm 0 phía sau không ảnh hưởng
        last = out[torch.arange(nodes.numel(), device=dev), deg[nodes] - 1]
        h = x.new_zeros(N, D)
        h[nodes] = last
        return h


class SemanticEncoder(nn.Module):
    """Eq. 10: SEM(i) = W_m [h_i || e_type(i)]"""

    def __init__(self, in_dim: int, out_dim: int, num_node_types: int = 1):
        super().__init__()
        self.num_node_types = max(1, num_node_types)
        self.type_emb = nn.Embedding(self.num_node_types, out_dim)
        self.lin = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, x, node_type):
        if node_type is None:
            node_type = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        node_type = node_type.to(x.device).long()
        bad = int(((node_type < 0) | (node_type >= self.num_node_types)).sum())
        if bad:
            warnings.warn(f"SemanticEncoder: {bad} node_type ngoài phạm vi -> clamp.")
            node_type = node_type.clamp(0, self.num_node_types - 1)
        return self.lin(torch.cat([x, self.type_emb(node_type)], dim=-1))


class TSSGCLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_node_types=1, dropout=0.2, max_seq_len=None):
        super().__init__()
        self.temporal = TemporalAggregator(in_dim, out_dim, max_seq_len)
        self.spatial = GATConv(in_dim, out_dim, heads=1, concat=False, dropout=dropout, add_self_loops=False)
        self.semantic = SemanticEncoder(in_dim, out_dim, num_node_types)
        self.W_t = nn.Linear(out_dim, out_dim, bias=False)
        self.W_s = nn.Linear(out_dim, out_dim, bias=False)
        self.W_m = nn.Linear(out_dim, out_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(out_dim))
        self.norm = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_time_delta=None, node_type=None):
        out = (self.W_t(self.temporal(x, edge_index, edge_time_delta))
               + self.W_s(self.spatial(x, edge_index))
               + self.W_m(self.semantic(x, node_type)) + self.b)
        return self.dropout(F.relu(self.norm(out)))


class TSSGCEncoder(nn.Module):
    """3 lớp TSSGC, hidden 64 (paper V-A-4)."""

    def __init__(self, in_dim, hidden_dim=64, num_layers=3, num_node_types=1, dropout=0.2, max_seq_len=None):
        super().__init__()
        self.layers = nn.ModuleList([
            TSSGCLayer(in_dim if i == 0 else hidden_dim, hidden_dim, num_node_types, dropout, max_seq_len)
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_time_delta=None, node_type=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_time_delta, node_type)
        return x


class MLPEncoder(nn.Module):
    """Encoder "standard deep neural network" cho ablation 'w/o GNN' (cùng độ sâu/độ rộng, không dùng cạnh)."""

    def __init__(self, in_dim, hidden_dim=64, num_layers=3, dropout=0.2):
        super().__init__()
        blocks = []
        for i in range(num_layers):
            blocks += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                       nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x, edge_index=None, edge_time_delta=None, node_type=None):
        return self.net(x)


__all__ = ["TemporalAggregator", "SemanticEncoder", "TSSGCLayer", "TSSGCEncoder", "MLPEncoder"]
