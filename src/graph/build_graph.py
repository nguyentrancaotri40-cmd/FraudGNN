# src/graph/build_graph.py
from __future__ import annotations

from typing import Any, Dict
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
import warnings

from .graph_utils import normalize_time_to_hours, temporal_similarity_edges, make_edge_tensors


def build_transaction_graph(
    x: np.ndarray,
    y: np.ndarray,
    time_values: np.ndarray | None,
    cfg: Dict[str, Any],
    entity_types: np.ndarray | None = None,  # ✅ Thêm tham số entity_types
) -> Data:
    """Create a PyG transaction graph for FraudGNN-RL reproduction.

    Node = transaction.
    Node feature = preprocessed transaction vector.
    Edge = temporal+feature-similarity relation between transactions.
    Label = fraud/legitimate label per transaction node.
    
    Args:
        x: Node features
        y: Node labels
        time_values: Time values for each node
        cfg: Configuration dictionary
        entity_types: Optional array of entity types (User/Merchant/Bank/...)
            If provided, used for Semantic branch (Eq 10).
            If None, creates 3 artificial types to enable Semantic branch.
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
    # ✅ FIX LỖI #D: Kích hoạt Semantic branch với node types
    # ============================================================
    if entity_types is not None:
        # ✅ Dùng entity types thực tế
        node_type = torch.tensor(entity_types, dtype=torch.long)
        num_types = len(np.unique(entity_types))
        print(f"[GRAPH] Using {num_types} node types from entity_types")
    else:
        # ✅ Fallback: Tạo 3 artificial types để kích hoạt Semantic branch
        # Paper Eq 10: SEM(i) = W_m[h_i || e_type(i)] - cần ít nhất 2 types
        # để e_type(i) không phải hằng số
        warnings.warn(
            "[GRAPH] entity_types not provided. Using 3 artificial types (User/Merchant/Bank) "
            "to enable Semantic branch (Eq 10). For real data, provide actual entity types.",
            UserWarning
        )
        # Tạo 3 types luân phiên: 0, 1, 2, 0, 1, 2, ...
        node_type = torch.tensor([i % 3 for i in range(x.shape[0])], dtype=torch.long)
        print(f"[GRAPH] Using artificial 3 node types to enable Semantic branch")
    
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=edge_index,
        edge_time_delta=torch.tensor(edge_time_delta, dtype=torch.float32),
        node_type=node_type,  # ✅ SỬA: Dùng node_type từ tham số
    )
    if times_hours is not None:
        data.node_time = torch.tensor(times_hours, dtype=torch.float32)
    return data


def save_graph(data: Data, path: str) -> None:
    """Save graph to disk."""
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_graph(path: str) -> Data:
    """Load graph from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)