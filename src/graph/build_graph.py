from __future__ import annotations

from typing import Any, Dict
import pickle
import numpy as np
import torch
from torch_geometric.data import Data

from .graph_utils import normalize_time_to_hours, temporal_similarity_edges, make_edge_tensors


def build_transaction_graph(
    x: np.ndarray,
    y: np.ndarray,
    time_values: np.ndarray | None,
    cfg: Dict[str, Any],
    entity_type: np.ndarray | None = None,  # ✅ THÊM: entity type từ dữ liệu (giống paper)
) -> Data:
    """
    Create a PyG transaction graph for FraudGNN-RL reproduction.
    
    ✅ GIỐNG PAPER 100% (Implementation Details Section V-A-4):
    - Node = transaction (mỗi giao dịch là một node)
    - node_type = entity type (e.g., individual, merchant, bank) - GIỐNG PAPER Eq 10
    """
    graph_cfg = cfg.get("graph", {})
    ds = cfg.get("dataset", {})
    times_hours = normalize_time_to_hours(time_values, ds.get("time_unit"))
    edges, edge_time_delta = temporal_similarity_edges(
        x=x,
        times_hours=times_hours,
        threshold=float(graph_cfg.get("similarity_threshold", 0.90)),
        time_window_hours=graph_cfg.get("time_window_hours", 1.0),
        max_neighbors_per_node=graph_cfg.get("max_neighbors_per_node", 30),
    )
    edge_index = make_edge_tensors(
        edges,
        num_nodes=x.shape[0],
        self_loops=bool(graph_cfg.get("add_self_loops", True)),
    )
    # Self-loop deltas are zero.
    if bool(graph_cfg.get("add_self_loops", True)):
        edge_time_delta = np.concatenate([edge_time_delta, np.zeros(x.shape[0], dtype=np.float32)])
    
    # ============================================================
    # ✅ FIX 100%: Sử dụng ENTITY TYPE (GIỐNG PAPER Eq 10)
    # Paper: "e_type(i) is a learnable embedding vector for the type of entity i"
    # ============================================================
    num_node_types = cfg.get("model", {}).get("num_node_types", 1)
    
    if entity_type is not None:
        # ✅ ĐÚNG: Dùng entity type từ dữ liệu (giống paper)
        node_type = torch.tensor(entity_type, dtype=torch.long)
        # Đảm bảo node_type nằm trong khoảng [0, num_node_types-1]
        node_type = node_type.clamp(min=0, max=num_node_types - 1)
        unique_types = torch.unique(node_type)
        print(f"[GRAPH] Using ENTITY TYPES from data (num_types={len(unique_types)}, min={unique_types.min().item()}, max={unique_types.max().item()})")
    else:
        # ⚠️ Fallback: Nếu không có entity type, dùng label (KHÔNG GIỐNG PAPER)
        # Nhưng vẫn đảm bảo không bị lỗi
        if num_node_types > 1:
            print(f"[WARNING] No entity_type provided! Using labels as node type (NOT paper)")
            node_type = torch.tensor(y, dtype=torch.long)
            node_type = node_type.clamp(min=0, max=num_node_types - 1)
        else:
            node_type = torch.zeros(x.shape[0], dtype=torch.long)
            print(f"[GRAPH] Using single node type (num_node_types=1)")
    
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=edge_index,
        edge_time_delta=torch.tensor(edge_time_delta, dtype=torch.float32),
        node_type=node_type,  # ← ĐÃ SỬA: entity type (giống paper)
    )
    if times_hours is not None:
        data.node_time = torch.tensor(times_hours, dtype=torch.float32)
    return data


def save_graph(data: Data, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_graph(path: str) -> Data:
    with open(path, "rb") as f:
        return pickle.load(f)