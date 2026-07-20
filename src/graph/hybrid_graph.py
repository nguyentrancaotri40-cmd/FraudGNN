# src/graph/hybrid_graph.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np
import torch
import time
from torch_geometric.data import Data

from src.graph.graph_utils import normalize_time_to_hours
from src.graph.build_graph import build_transaction_graph
from src.graph.soft_behavior_graph import (
    build_soft_behavior_edges,
    summarize_soft_edges,
)


def _edge_index_to_list(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    """Convert PyG edge_index tensor [2, E] to list of (src, dst)."""
    if edge_index is None or edge_index.numel() == 0:
        return []

    edge_index = edge_index.detach().cpu().long()

    return [
        (int(edge_index[0, i]), int(edge_index[1, i]))
        for i in range(edge_index.size(1))
    ]


def _tensor_to_numpy_1d(
    x: torch.Tensor | None,
    expected_len: int | None = None,
) -> np.ndarray:
    """Convert 1D tensor to numpy float32."""
    if x is None:
        n = int(expected_len or 0)
        return np.zeros(n, dtype=np.float32)
    
    arr = x.detach().cpu().numpy().astype(np.float32).reshape(-1)
    
    if expected_len is not None and len(arr) != expected_len:
        raise ValueError(
            f"_tensor_to_numpy_1d: Length mismatch! "
            f"Actual len={len(arr)}, expected_len={expected_len}."
        )
    
    return arr


def _merge_edges(
    base_edges: list[tuple[int, int]],
    base_delta: np.ndarray,
    soft_edges: list[tuple[int, int]],
    soft_delta: np.ndarray,
    soft_weight: np.ndarray,
    prefer: str = "min_delta",
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    """Merge base and soft edges."""
    merged_edges = []
    merged_delta = []
    merged_soft_weight = []
    edge_source = []
    
    edge_to_indices: dict[tuple[int, int], list[int]] = {}
    
    # Add base edges
    for i, e in enumerate(base_edges):
        idx = len(merged_edges)
        merged_edges.append(e)
        merged_delta.append(float(base_delta[i]) if i < len(base_delta) else 0.0)
        merged_soft_weight.append(0.0)
        edge_source.append(0)
        if e not in edge_to_indices:
            edge_to_indices[e] = []
        edge_to_indices[e].append(idx)
    
    # Add soft edges with overlap detection
    for i, e in enumerate(soft_edges):
        delta = float(soft_delta[i]) if i < len(soft_delta) else 0.0
        sw = float(soft_weight[i]) if i < len(soft_weight) else 0.0
        
        if e in edge_to_indices:
            for idx in edge_to_indices[e]:
                edge_source[idx] = 2
                merged_soft_weight[idx] = sw
                if prefer == "soft":
                    merged_delta[idx] = delta
                elif prefer == "base":
                    pass
                else:
                    merged_delta[idx] = min(merged_delta[idx], delta)
        else:
            idx = len(merged_edges)
            merged_edges.append(e)
            merged_delta.append(delta)
            merged_soft_weight.append(sw)
            edge_source.append(1)
            edge_to_indices[e] = [idx]
    
    return (
        merged_edges,
        np.asarray(merged_delta, dtype=np.float32),
        np.asarray(edge_source, dtype=np.int64),
        np.asarray(merged_soft_weight, dtype=np.float32),
    )


def _make_edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _add_self_loops_if_missing(
    edges: list[tuple[int, int]],
    delta: np.ndarray,
    edge_source: np.ndarray,
    soft_weight: np.ndarray,
    num_nodes: int,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    """Ensure every node has one self-loop."""
    existing = set(edges)
    new_edges = list(edges)
    new_delta = list(delta.astype(np.float32))
    new_source = list(edge_source.astype(np.int64))
    new_soft_weight = list(soft_weight.astype(np.float32))

    for i in range(num_nodes):
        e = (i, i)
        if e not in existing:
            new_edges.append(e)
            new_delta.append(0.0)
            new_source.append(3)
            new_soft_weight.append(0.0)

    return (
        new_edges,
        np.asarray(new_delta, dtype=np.float32),
        np.asarray(new_source, dtype=np.int64),
        np.asarray(new_soft_weight, dtype=np.float32),
    )


def _make_edge_weight(
    edge_source: torch.Tensor,
    soft_weight: torch.Tensor,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """Create edge weights using actual soft_weight values."""
    fusion_cfg = cfg.get("hybrid_graph", {})
    
    # Check if weighted fusion is enabled
    weight_keys = ["base_edge_weight", "soft_edge_weight", "overlap_edge_weight", "self_loop_edge_weight"]
    has_explicit_weights = any(key in fusion_cfg for key in weight_keys)
    
    if not has_explicit_weights:
        # Unweighted: all edges = 1.0
        return torch.ones(edge_source.numel(), dtype=torch.float32)
    
    base_weight = float(fusion_cfg.get("base_edge_weight", 1.0))
    soft_weight_const = float(fusion_cfg.get("soft_edge_weight", 0.4))
    overlap_weight_const = float(fusion_cfg.get("overlap_edge_weight", 1.2))
    self_loop_weight = float(fusion_cfg.get("self_loop_edge_weight", 1.0))
    
    edge_weight = torch.ones(edge_source.numel(), dtype=torch.float32)
    
    edge_weight[edge_source == 0] = base_weight
    edge_weight[edge_source == 1] = soft_weight_const
    edge_weight[edge_source == 2] = overlap_weight_const
    edge_weight[edge_source == 3] = self_loop_weight
    
    return edge_weight


def build_hybrid_transaction_graph(
    x: np.ndarray,
    y: np.ndarray,
    time_values: np.ndarray | None,
    cfg: Dict[str, Any],
) -> Data:
    """Build Hybrid Graph for the proposed model."""
    graph_cfg = cfg.get("graph", {})
    hybrid_cfg = cfg.get("hybrid_graph", {})
    dataset_cfg = cfg.get("dataset", {})

    add_self_loops = bool(graph_cfg.get("add_self_loops", True))
    merge_prefer = str(hybrid_cfg.get("merge_prefer", "min_delta"))

    n_nodes = x.shape[0]
    threshold = graph_cfg.get("similarity_threshold", 0.98)
    time_window = graph_cfg.get("time_window_hours", 1000.0)
    max_neighbors = graph_cfg.get("max_neighbors_per_node", 3)
    
    print(f"[GRAPH] Building hybrid graph with {n_nodes} nodes...")
    print(f"[GRAPH] threshold={threshold}, time_window_hours={time_window}, max_neighbors={max_neighbors}")
    
    # 1. Build original/baseline graph
    base_start = time.perf_counter()
    base_graph = build_transaction_graph(x, y, time_values, cfg)
    base_time = time.perf_counter() - base_start

    base_edges = _edge_index_to_list(base_graph.edge_index)
    base_delta = _tensor_to_numpy_1d(
        getattr(base_graph, "edge_time_delta", None),
        expected_len=len(base_edges),
    )
    
    print(f"[GRAPH] Base graph done: {len(base_edges)} edges in {base_time:.2f}s")

    # 2. Build soft behavioral graph
    ds = cfg.get("dataset", {})
    times_hours = normalize_time_to_hours(time_values, ds.get("time_unit"))

    print(f"[GRAPH] Building soft behavior graph...")
    soft_start = time.perf_counter()
    
    soft_edges, soft_delta, soft_weight = build_soft_behavior_edges(
        x=x,
        times_hours=times_hours,
        cfg=cfg,
    )
    
    soft_time = time.perf_counter() - soft_start
    print(f"[GRAPH] Soft graph done: {len(soft_edges)} edges in {soft_time:.2f}s")

    # 3. Merge base graph and soft graph
    print(f"[GRAPH] Merging graphs...")
    merge_start = time.perf_counter()
    
    merged_edges, merged_delta, edge_source_np, merged_soft_weight = _merge_edges(
        base_edges=base_edges,
        base_delta=base_delta,
        soft_edges=soft_edges,
        soft_delta=soft_delta,
        soft_weight=soft_weight,
        prefer=merge_prefer,
    )
    
    merge_time = time.perf_counter() - merge_start
    print(f"[GRAPH] Merge done: {len(merged_edges)} edges in {merge_time:.2f}s")

    # 4. Ensure self-loops if requested
    if add_self_loops:
        print(f"[GRAPH] Adding self-loops...")
        merged_edges, merged_delta, edge_source_np, merged_soft_weight = _add_self_loops_if_missing(
            edges=merged_edges,
            delta=merged_delta,
            edge_source=edge_source_np,
            soft_weight=merged_soft_weight,
            num_nodes=x.shape[0],
        )

    # 5. Convert to tensors
    edge_index = _make_edge_index(merged_edges)
    edge_source = torch.tensor(edge_source_np, dtype=torch.long)
    edge_weight = _make_edge_weight(
        edge_source=edge_source,
        soft_weight=torch.tensor(merged_soft_weight, dtype=torch.float32),
        cfg=cfg,
    )

    # 6. Build PyG Data object
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=edge_index,
        edge_time_delta=torch.tensor(merged_delta, dtype=torch.float32),
        node_type=torch.zeros(x.shape[0], dtype=torch.long),
        edge_source=edge_source,
        edge_weight=edge_weight,
    )

    if times_hours is not None:
        data.node_time = torch.tensor(times_hours, dtype=torch.float32)

    # 7. Useful debug metadata
    summary = {
        "num_nodes": int(x.shape[0]),
        "num_base_edges": int(len(base_edges)),
        "num_soft_edges": int(len(soft_edges)),
        "num_hybrid_edges": int(len(merged_edges)),
        "num_base_only_edges": int(np.sum(edge_source_np == 0)),
        "num_soft_only_edges": int(np.sum(edge_source_np == 1)),
        "num_overlap_edges": int(np.sum(edge_source_np == 2)),
        "num_added_self_loops": int(np.sum(edge_source_np == 3)),
        "edge_weight_min": float(edge_weight.min().item()) if edge_weight.numel() else 0.0,
        "edge_weight_mean": float(edge_weight.mean().item()) if edge_weight.numel() else 0.0,
        "edge_weight_max": float(edge_weight.max().item()) if edge_weight.numel() else 0.0,
        "soft_summary": summarize_soft_edges(
            edges=soft_edges,
            edge_weight=soft_weight,
            num_nodes=x.shape[0],
        ),
    }

    data.hybrid_summary = summary
    
    print(f"[GRAPH] Hybrid graph complete! Total edges: {len(merged_edges)}")
    print(f"[GRAPH]   Base only: {summary['num_base_only_edges']}")
    print(f"[GRAPH]   Soft only: {summary['num_soft_only_edges']}")
    print(f"[GRAPH]   Overlap: {summary['num_overlap_edges']}")
    print(f"[GRAPH]   Self-loops: {summary['num_added_self_loops']}")

    return data