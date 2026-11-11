# ============================================================
# src/graph/build_graph.py — PyG Data cho 1 split.
#   Node = transaction (V-A-4); edge = |dt|<=1h AND cosine>=0.9 (hard edges).
#   node_type phục vụ SEM (Eq. 10): e_type(i) = loại giao dịch.
#   - PaySim : cột 'type' (5 loại — paper nêu rõ)
#   - IEEE   : 'ProductCD' (5 giá trị W,C,H,R,S) [INFERRED]
#   - CC2023 : không có -> single type
#   Mapping đọc từ config (dataset.entity_type_col / entity_type_values) và CỐ ĐỊNH
#   giữa train/val/test (bản cũ dùng cat.codes theo từng split nên lệch mã).
#
# flags: gnn=False -> Data không cạnh (dùng cho "w/o GNN");
#        soft_edges=True -> thêm soft edges (extension, NGOÀI paper).
# ============================================================
from __future__ import annotations

import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.utils.config import get_flags
from .graph_utils import cosine_neighbors, make_undirected, normalize_time_to_hours


def extract_entity_type(df: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    ds = cfg.get("dataset", {})
    col, values = ds.get("entity_type_col"), ds.get("entity_type_values")
    if not col or not values or col not in df.columns:
        print("[ENTITY] Không có entity_type_col -> single type")
        return None
    mapping = {str(v): i for i, v in enumerate(values)}
    mapped = df[col].astype(str).map(mapping)
    unk = int(mapped.isna().sum())
    if unk:
        print(f"[ENTITY] [WARN] {unk} giá trị '{col}' ngoài danh sách -> gán 0")
    return mapped.fillna(0).astype(int).to_numpy()


def build_transaction_graph(
    x: np.ndarray,
    y: np.ndarray,
    time_values: Optional[np.ndarray],
    cfg: Dict[str, Any],
    entity_type: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> Data:
    flags = get_flags(cfg)
    g, ds = cfg.get("graph", {}), cfg.get("dataset", {})
    n = x.shape[0]
    thr = float(g.get("similarity_threshold", 0.9))
    win = g.get("time_window_hours", 1.0)
    cap = g.get("max_neighbors_per_node")
    cap = int(cap) if cap is not None else None
    dev = device or g.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    hours = normalize_time_to_hours(time_values, ds.get("time_unit"))

    edge_index = torch.empty(2, 0, dtype=torch.long)
    delta = torch.empty(0)

    if flags["gnn"]:
        if flags["hard_edges"]:
            print(f"[GRAPH] Hard edges: N={n}, d={x.shape[1]}, cos>={thr}, window={win if hours is not None else 'n/a'}h, cap={cap}")
            ei, dl, _ = cosine_neighbors(x, hours, thr, win, cap, causal=False, device=dev,
                                         chunk_elems=float(g.get("chunk_elems", 2e7)))
            edge_index, delta = make_undirected(ei, dl, n)
        if flags["soft_edges"]:  # EXTENSION — ngoài paper
            from src.extensions.soft_behavior_graph import build_soft_behavior_edges, merge_hard_soft_edges
            s_ei, s_dl, s_w = build_soft_behavior_edges(x, hours, cfg, device=dev)
            edge_index, delta = merge_hard_soft_edges(edge_index, delta, s_ei, s_dl, n)

    node_type = (torch.tensor(entity_type, dtype=torch.long) if entity_type is not None
                 else torch.zeros(n, dtype=torch.long))
    node_type = node_type.clamp_(0, max(0, int(cfg.get("model", {}).get("num_node_types", 1)) - 1))

    data = Data(x=torch.tensor(x, dtype=torch.float32), y=torch.tensor(y, dtype=torch.long),
                edge_index=edge_index, edge_time_delta=delta, node_type=node_type)
    data.node_time = torch.tensor(hours if hours is not None else np.arange(n), dtype=torch.float64)
    data.has_time = bool(hours is not None)
    print(f"[GRAPH] x={tuple(data.x.shape)} edges={data.edge_index.size(1):,} "
          f"avg_deg={data.edge_index.size(1)/max(1,n):.2f}")
    return data


def save_graph(data: Data, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_graph(path: str) -> Data:
    with open(path, "rb") as f:
        return pickle.load(f)


__all__ = ["extract_entity_type", "build_transaction_graph", "save_graph", "load_graph"]
