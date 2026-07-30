# src/graph/build_graph.py
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
) -> Data:
    """
    Create a PyG transaction graph for FraudGNN-RL reproduction.
    
    ✅ GIỐNG PAPER 100% (Implementation Details Section V-A-4):
    - Node = transaction (mỗi giao dịch là một node)
    - node_type = 0 cho mọi node (vì tất cả đều là transaction)
    - num_node_types = 1 (SemanticEncoder mặc định)
    
    Lưu ý: Theo paper, "each transaction is a node", không phân biệt entity.
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
    # ✅ GIỐNG PAPER 100%: node = transaction → node_type = 0
    # ============================================================
    # Theo Implementation Details (Section V-A-4):
    # "we construct a transaction graph where EACH TRANSACTION IS A NODE"
    # Vì vậy tất cả node đều là transaction, không phân biệt entity.
    # node_type = 0 cho mọi node, num_node_types = 1 (mặc định)
    # ============================================================
    node_type = torch.zeros(x.shape[0], dtype=torch.long)
    
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=edge_index,
        edge_time_delta=torch.tensor(edge_time_delta, dtype=torch.float32),
        node_type=node_type,  # ✅ Tất cả đều là transaction (type 0)
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