# ============================================================
# src/extensions/soft_behavior_graph.py — EXTENSION (không có trong paper)
# Soft behavioral edges: cosine >= soft threshold, top-k láng giềng, tuỳ chọn causal.
# Dùng lại tìm kiếm chính xác của graph_utils.cosine_neighbors.
# ============================================================
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch

from src.graph.graph_utils import cosine_neighbors, dedupe_edges


def _select_features(x: np.ndarray, idx: Optional[Iterable[int]]) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    if not idx:
        return x
    valid = [int(i) for i in idx if 0 <= int(i) < x.shape[1]]
    return x[:, valid] if valid else x


def build_soft_behavior_edges(
    x: np.ndarray,
    times_hours: Optional[np.ndarray],
    cfg: Dict[str, Any],
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (edge_index [2,E], delta_hours [E], weight [E]). Rỗng nếu tắt."""
    sc = (cfg or {}).get("soft_graph", {})
    empty = (torch.empty(2, 0, dtype=torch.long), torch.empty(0), torch.empty(0))
    if not sc.get("enabled", True) or int(sc.get("max_neighbors_per_node", 5)) <= 0:
        return empty
    bx = _select_features(x, sc.get("feature_indices"))
    ei, dl, w = cosine_neighbors(
        bx, times_hours,
        threshold=float(sc.get("similarity_threshold", 0.85)),
        window_hours=sc.get("time_window_hours", 1.0),
        max_neighbors=int(sc.get("max_neighbors_per_node", 5)),
        causal=bool(sc.get("causal", True)) and times_hours is not None,
        device=device, verbose=False,
    )
    return ei, dl, w


def summarize_soft_edges(edge_index: torch.Tensor, weight: torch.Tensor, num_nodes: int) -> dict:
    e = int(edge_index.size(1))
    return {
        "num_nodes": int(num_nodes), "num_soft_edges": e,
        "avg_soft_degree": e / max(1, num_nodes),
        "weight_min": float(weight.min()) if e else 0.0,
        "weight_mean": float(weight.mean()) if e else 0.0,
        "weight_max": float(weight.max()) if e else 0.0,
    }


def merge_hard_soft_edges(hard_ei: torch.Tensor, hard_delta: torch.Tensor,
                          soft_ei: torch.Tensor, soft_delta: torch.Tensor, num_nodes: int):
    """Hợp nhất 2 tập cạnh, loại trùng (giữ delta nhỏ hơn)."""
    if soft_ei.size(1) == 0:
        return hard_ei, hard_delta
    ei = torch.cat([hard_ei, soft_ei], dim=1)
    d = torch.cat([hard_delta, soft_delta])
    order = torch.argsort(d)                       # delta nhỏ trước -> được giữ khi trùng
    ei, d = ei[:, order], d[order]
    ei, d = dedupe_edges(ei, d, num_nodes=num_nodes)
    return ei, d
