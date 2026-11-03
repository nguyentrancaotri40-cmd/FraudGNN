# ============================================================
# src/graph/build_graph.py
#
# Xây dựng PyG Data cho 1 split (train/val/test).
#
# Theo Phần 2 + Eq. 10 (SEM):
#   - Node = transaction (Section V-A-4)
#   - Edge = undirected, |Δt| <= 1h AND cosine >= 0.9 (Section V-A-4)
#   - node_type = loại giao dịch (PaySim: type; IEEE-CIS: ProductCD;
#                 CC2023: không có → num_node_types=1)
#
# [WARN] [INFERRED] e_type cho SEM Eq. 10:
#   Paper chỉ ví dụ PaySim có 5 loại (CASH-IN, CASH-OUT, DEBIT, PAYMENT,
#   TRANSFER). Với IEEE-CIS, nhóm chọn cột 'ProductCD' (5 giá trị: W, H, C,
#   S, R) — đây là cột "type" tự nhiên nhất của giao dịch. CC2023 không có
#   cột tương đương → num_node_types=1 (embedding hằng).
# ============================================================

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch_geometric.data import Data

from .graph_utils import (
    normalize_time_to_hours,
    temporal_similarity_edges,
    make_edge_tensors,
)


# ----------------------------------------------------------------------
# Trích entity_type (loại giao dịch) cho SEM Eq. 10
# ----------------------------------------------------------------------
def extract_entity_type(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Optional[np.ndarray]:
    """
    Trả về mảng int [N] hoặc None nếu không có cột loại.

    [INFERRED] Paper chỉ nói ví dụ PaySim; mapping cho dataset khác do nhóm chọn.
    """
    ds_name = cfg.get("experiment", {}).get("dataset", "").lower()

    # PaySim: cột 'type' (paper nói rõ 5 loại)
    if ds_name in ("paysim",):
        if "type" in df.columns:
            mapping = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2,
                       "PAYMENT": 3, "TRANSFER": 4}
            et = df["type"].map(mapping).fillna(0).astype(int).values
            print(f"[ENTITY] PaySim 'type' → {len(np.unique(et))} loại")
            return et

    # IEEE-CIS: ProductCD [INFERRED]
    if ds_name in ("ieee", "ieee-cis", "ieee_cis"):
        if "ProductCD" in df.columns:
            et = df["ProductCD"].astype("category").cat.codes.values.astype(int)
            print(f"[ENTITY] IEEE-CIS 'ProductCD' → {len(np.unique(et))} loại")
            return et

    # CC2023: không có cột loại → None → num_node_types=1
    if ds_name in ("creditcard", "creditcard2023", "credit_card"):
        print("[ENTITY] CC2023 không có cột loại → single type")
        return None

    # Fallback
    print(f"[ENTITY] Dataset '{ds_name}': không xác định cột loại → single type")
    return None


# ----------------------------------------------------------------------
# Build graph từ (x, y, t)
# ----------------------------------------------------------------------
def build_transaction_graph(
    x: np.ndarray,
    y: np.ndarray,
    time_values: np.ndarray | None,
    cfg: Dict[str, Any],
    entity_type: np.ndarray | None = None,
) -> Data:
    """
    Tạo PyG Data với:
      - x [N, d]
      - y [N]
      - edge_index [2, 2E] undirected, không self-loop
      - edge_time_delta [2E]
      - node_type [N] (cho SEM Eq. 10)
      - node_time [N] (giờ) nếu có
    """
    # Densify nếu sparse
    if sparse.issparse(x):
        print(f"[GRAPH] Densify sparse {x.shape} → dense")
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)

    graph_cfg = cfg.get("graph", {})
    ds_cfg = cfg.get("dataset", {})
    ds_name = cfg.get("experiment", {}).get("dataset", "").lower()

    threshold = float(graph_cfg.get("similarity_threshold", 0.90))
    time_window_hours = float(graph_cfg.get("time_window_hours", 1.0))
    max_neighbors = graph_cfg.get("max_neighbors_per_node", None)
    if max_neighbors is not None:
        max_neighbors = int(max_neighbors)

    # Thời gian → giờ
    time_unit = ds_cfg.get("time_unit")
    times_hours = normalize_time_to_hours(time_values, time_unit, ds_name)

    # Cạnh
    edges, deltas = temporal_similarity_edges(
        x=x,
        times_hours=times_hours,
        threshold=threshold,
        time_window_hours=time_window_hours,
        max_neighbors_per_node=max_neighbors,
    )

    edge_index, edge_delta = make_edge_tensors(
        edges, deltas, num_nodes=x.shape[0], add_self_loops=False,
    )

    # Node type (SEM Eq. 10)
    num_node_types = int(cfg.get("model", {}).get("num_node_types", 1))
    if entity_type is not None:
        node_type = torch.tensor(entity_type, dtype=torch.long).clamp_(
            min=0, max=num_node_types - 1
        )
        print(f"[GRAPH] node_type từ data (num_types={len(torch.unique(node_type))})")
    else:
        node_type = torch.zeros(x.shape[0], dtype=torch.long)
        print(f"[GRAPH] node_type = single type (num_node_types={num_node_types})")

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=edge_index,
        edge_time_delta=edge_delta,
        node_type=node_type,
    )

    if times_hours is not None:
        data.node_time = torch.tensor(times_hours, dtype=torch.float32)

    print(f"[GRAPH] x={tuple(data.x.shape)}, "
          f"edges={data.edge_index.size(1)}, "
          f"node_type={tuple(data.node_type.shape)}")
    return data


# ----------------------------------------------------------------------
# Save / Load (pickle)
# ----------------------------------------------------------------------
def save_graph(data: Data, path: str) -> None:
    import pickle
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_graph(path: str) -> Data:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


__all__ = [
    "extract_entity_type",
    "build_transaction_graph",
    "save_graph",
    "load_graph",
]