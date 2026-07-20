# src/graph/graph_utils.py
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def normalize_time_to_hours(
    time_values: np.ndarray | None,
    time_unit: str | None = None
) -> np.ndarray | None:
    """Convert time values to hours."""
    if time_values is None:
        return None
    t = np.asarray(time_values, dtype=np.float32)
    if time_unit == "second":
        return t / 3600.0
    return t.astype(np.float32)


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Add self-loops to edge_index."""
    loops = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    if edge_index.numel() == 0:
        return loops
    return torch.cat([edge_index, loops], dim=1)


def make_edge_tensors(
    edges: list[tuple[int, int]],
    num_nodes: int,
    self_loops: bool = True
) -> torch.Tensor:
    """Convert edge list to PyG edge_index tensor."""
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if self_loops:
        edge_index = add_self_loops(edge_index, num_nodes)
    return edge_index


def temporal_similarity_edges(
    x: np.ndarray,
    times_hours: np.ndarray | None,
    threshold: float = 0.90,
    time_window_hours: float | None = 1.0,
    max_neighbors_per_node: int | None = 30,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    Build transaction-transaction edges với CAUSAL direction.
    
    Edge direction: quá khứ → hiện tại (causal)
    
    Paper yêu cầu: "connect transactions if they occur within a time window"
    nhưng với temporal split, chỉ được dùng thông tin từ quá khứ.
    """
    import time
    start_time = time.time()
    
    n = int(x.shape[0])
    print(f"[GRAPH] Building graph with {n} nodes...")
    print(f"[GRAPH] threshold={threshold}, time_window_hours={time_window_hours}, max_neighbors={max_neighbors_per_node}")
    
    if n == 0:
        return [], np.empty((0,), dtype=np.float32)
    
    if times_hours is None or time_window_hours is None:
        order = np.arange(n)
        times_sorted = None
    else:
        order = np.argsort(times_hours)
        times_sorted = times_hours[order]
    
    x_sorted = x[order]
    edges: list[tuple[int, int]] = []
    deltas: list[float] = []
    
    last_log = 0
    
    for pos, node_i in enumerate(order):
        progress = (pos + 1) / n * 100
        if progress - last_log >= 10:
            print(f"[GRAPH] Progress: {progress:.0f}% ({pos+1}/{n} nodes), edges found: {len(edges)}")
            last_log = progress
        
        if times_sorted is None:
            start, end = 0, n
        else:
            # Chỉ nhìn quá khứ (causal)
            left_t = times_sorted[pos] - time_window_hours
            right_t = times_sorted[pos]  # Chỉ đến thời điểm hiện tại
            start = int(np.searchsorted(times_sorted, left_t, side="left"))
            end = int(np.searchsorted(times_sorted, right_t, side="right"))
        
        cand_positions = np.arange(start, end)
        cand_positions = cand_positions[cand_positions != pos]
        
        if cand_positions.size == 0:
            continue
        
        sims = cosine_similarity(x_sorted[pos:pos+1], x_sorted[cand_positions]).ravel()
        valid = np.where(sims >= threshold)[0]
        
        if valid.size == 0:
            continue
        
        if max_neighbors_per_node is not None and valid.size > max_neighbors_per_node:
            best = np.argsort(sims[valid])[-max_neighbors_per_node:]
            valid = valid[best]
        
        for idx in valid:
            jpos = int(cand_positions[idx])
            node_j = int(order[jpos])
            
            # Edge direction: node_j (quá khứ) → node_i (hiện tại)
            edges.append((node_j, node_i))
            
            if times_sorted is None:
                deltas.append(0.0)
            else:
                deltas.append(float(times_sorted[pos] - times_sorted[jpos]))
    
    elapsed = time.time() - start_time
    print(f"[GRAPH] Done! Found {len(edges)} edges in {elapsed:.2f}s")
    
    return edges, np.asarray(deltas, dtype=np.float32)