# src/graph/graph_utils.py
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


def normalize_time_to_hours(
    time_values: np.ndarray | None,
    time_unit: str | None = None
) -> np.ndarray | None:
    """Convert time values to hours.
    
    Args:
        time_values: Array of time values
        time_unit: 'second', 'hour', 'index', or None
    
    Returns:
        time_values in hours (float32)
    
    Note:
        - 'second': convert seconds to hours (divide by 3600)
        - 'hour': keep as hours
        - 'index': treat as sequential indices (0,1,2,...) - no conversion
        - None/other: keep as-is with warning
    """
    if time_values is None:
        return None
    
    t = np.asarray(time_values, dtype=np.float32)
    
    if time_unit == "second":
        return t / 3600.0
    elif time_unit == "hour":
        return t
    elif time_unit == "index":
        # index là số thứ tự, không có đơn vị thời gian thực
        # Giữ nguyên, chỉ chuyển sang float32
        print(f"[TIME] time_unit='index': treating values as sequential indices (not real hours)")
        return t.astype(np.float32)
    else:
        # Default: giữ nguyên (cảnh báo nếu giá trị lớn)
        if np.max(t) > 1000:
            print(f"[TIME] ⚠️ time_unit='{time_unit}' with large values ({np.max(t):.0f}) - may cause issues")
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


def temporal_similarity_edges_fast(
    x: np.ndarray,
    times_hours: np.ndarray | None,
    threshold: float = 0.90,
    time_window_hours: float | None = 1.0,
    max_neighbors_per_node: int | None = 30,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    ✅ VECTORIZED VERSION: Build transaction-transaction edges với CAUSAL direction.
    
    Sử dụng NearestNeighbors để vector hóa, tương tự soft_behavior_graph.py.
    Nhanh hơn nhiều so với vòng lặp for + cosine_similarity từng node.
    
    Edge direction: quá khứ → hiện tại (causal)
    """
    import time
    start_time = time.perf_counter()
    
    n = int(x.shape[0])
    print(f"[GRAPH FAST] Building graph with {n} nodes, {x.shape[1]} features...")
    print(f"[GRAPH FAST] threshold={threshold}, time_window_hours={time_window_hours}, max_neighbors={max_neighbors_per_node}")
    
    if n == 0:
        return [], np.empty((0,), dtype=np.float32)
    
    # L2-normalize features
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
    
    edges = []
    deltas = []
    
    if times_hours is not None and time_window_hours is not None:
        times = np.asarray(times_hours, dtype=np.float32)
        
        # Bucket theo thời gian
        bucket_size = max(0.5, time_window_hours / 2)
        bucket_ids = np.floor(times / bucket_size).astype(np.int64)
        unique_buckets = np.sort(np.unique(bucket_ids))
        
        print(f"[GRAPH FAST] {len(unique_buckets)} time buckets")
        last_log = 0
        
        for bucket_idx, b in enumerate(unique_buckets):
            progress = (bucket_idx + 1) / len(unique_buckets) * 100
            if progress - last_log >= 10:
                print(f"[GRAPH FAST] Progress: {progress:.0f}%, edges: {len(edges)}")
                last_log = progress
            
            # Lấy indices trong bucket và lân cận
            mask = (bucket_ids >= b - 1) & (bucket_ids <= b + 1)
            idx = np.where(mask)[0]
            
            if len(idx) <= 1:
                continue
            
            # Nearest neighbors trong bucket
            k = min(max_neighbors_per_node + 1, len(idx))
            nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute', n_jobs=-1)
            nn.fit(x_norm[idx])
            distances, neighbors = nn.kneighbors(x_norm[idx])
            
            for local_i, src in enumerate(idx):
                kept = 0
                for pos in range(1, k):
                    dst = idx[neighbors[local_i, pos]]
                    sim = 1.0 - distances[local_i, pos]
                    
                    if sim < threshold:
                        continue
                    
                    # Causal: quá khứ → hiện tại
                    if times[dst] <= times[src]:
                        edges.append((dst, src))
                        deltas.append(float(times[src] - times[dst]))
                        kept += 1
                        if kept >= max_neighbors_per_node:
                            break
    
    else:
        # Không có time → global search
        k = min(max_neighbors_per_node + 1, n)
        nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute', n_jobs=-1)
        nn.fit(x_norm)
        distances, neighbors = nn.kneighbors(x_norm)
        
        for src in range(n):
            for pos in range(1, k):
                dst = neighbors[src, pos]
                sim = 1.0 - distances[src, pos]
                if sim >= threshold:
                    edges.append((src, dst))
                    deltas.append(0.0)
    
    elapsed = time.perf_counter() - start_time
    print(f"[GRAPH FAST] Done! Found {len(edges)} edges in {elapsed:.2f}s")
    
    return edges, np.asarray(deltas, dtype=np.float32)


# src/graph/graph_utils.py

def temporal_similarity_edges(
    x: np.ndarray,
    times_hours: np.ndarray | None,
    threshold: float = 0.90,
    time_window_hours: float | None = 1.0,
    max_neighbors_per_node: int | None = 30,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    ✅ VECTORIZED VERSION: Build transaction-transaction edges với CAUSAL direction.
    
    Sử dụng NearestNeighbors để vector hóa, tương tự soft_behavior_graph.py.
    Nhanh hơn nhiều so với vòng lặp for + cosine_similarity từng node.
    
    Edge direction: quá khứ → hiện tại (causal)
    """
    import time
    from sklearn.neighbors import NearestNeighbors
    
    start_time = time.perf_counter()
    
    n = int(x.shape[0])
    print(f"[GRAPH] Building graph with {n} nodes, {x.shape[1]} features...")
    print(f"[GRAPH] threshold={threshold}, time_window_hours={time_window_hours}, max_neighbors={max_neighbors_per_node}")
    
    if n == 0:
        return [], np.empty((0,), dtype=np.float32)
    
    # L2-normalize features
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
    
    edges = []
    deltas = []
    
    if times_hours is not None and time_window_hours is not None:
        times = np.asarray(times_hours, dtype=np.float32)
        
        # ✅ Bucket theo thời gian (giống soft_behavior_graph.py)
        bucket_size = max(0.5, time_window_hours / 2)
        bucket_ids = np.floor(times / bucket_size).astype(np.int64)
        unique_buckets = np.sort(np.unique(bucket_ids))
        
        print(f"[GRAPH] {len(unique_buckets)} time buckets")
        last_log = 0
        
        for bucket_idx, b in enumerate(unique_buckets):
            progress = (bucket_idx + 1) / len(unique_buckets) * 100
            if progress - last_log >= 10:
                print(f"[GRAPH] Progress: {progress:.0f}%, edges: {len(edges)}")
                last_log = progress
            
            # Lấy indices trong bucket và lân cận
            mask = (bucket_ids >= b - 1) & (bucket_ids <= b + 1)
            idx = np.where(mask)[0]
            
            if len(idx) <= 1:
                continue
            
            # ✅ Nearest neighbors trong bucket (vector hóa!)
            k = min(max_neighbors_per_node + 1, len(idx))
            nn = NearestNeighbors(
                n_neighbors=k, 
                metric='cosine', 
                algorithm='brute', 
                n_jobs=-1
            )
            nn.fit(x_norm[idx])
            distances, neighbors = nn.kneighbors(x_norm[idx])
            
            for local_i, src in enumerate(idx):
                kept = 0
                for pos in range(1, k):
                    dst = idx[neighbors[local_i, pos]]
                    sim = 1.0 - distances[local_i, pos]
                    
                    if sim < threshold:
                        continue
                    
                    # ✅ Causal: quá khứ → hiện tại
                    if times[dst] <= times[src]:
                        edges.append((int(dst), int(src)))
                        deltas.append(float(times[src] - times[dst]))
                        kept += 1
                        if kept >= max_neighbors_per_node:
                            break
    
    else:
        # Không có time → global search
        k = min(max_neighbors_per_node + 1, n)
        nn = NearestNeighbors(
            n_neighbors=k, 
            metric='cosine', 
            algorithm='brute', 
            n_jobs=-1
        )
        nn.fit(x_norm)
        distances, neighbors = nn.kneighbors(x_norm)
        
        for src in range(n):
            for pos in range(1, k):
                dst = neighbors[src, pos]
                sim = 1.0 - distances[src, pos]
                if sim >= threshold:
                    edges.append((int(src), int(dst)))
                    deltas.append(0.0)
    
    elapsed = time.perf_counter() - start_time
    print(f"[GRAPH] Done! Found {len(edges)} edges in {elapsed:.2f}s")
    
    return edges, np.asarray(deltas, dtype=np.float32)