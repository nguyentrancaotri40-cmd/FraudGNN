from __future__ import annotations
from typing import Any, Dict, Iterable
import numpy as np
from sklearn.neighbors import NearestNeighbors
import warnings

def _as_float_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = _as_float_array(x)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return x / norm

def _select_behavior_features(x: np.ndarray, feature_indices: Iterable[int] | None = None) -> np.ndarray:
    x = _as_float_array(x)
    if feature_indices is None:
        return x
    indices = [int(i) for i in feature_indices]
    if not indices:
        return x
    max_idx = x.shape[1] - 1
    valid = [i for i in indices if 0 <= i <= max_idx]
    if not valid:
        return x
    return x[:, valid]

def build_soft_behavior_edges(x: np.ndarray, times_hours: np.ndarray | None = None, cfg: Dict[str, Any] | None = None) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    import time
    cfg = cfg or {}
    soft_cfg = cfg.get("soft_graph", cfg)
    enabled = bool(soft_cfg.get("enabled", True))
    if not enabled:
        return [], np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    x = _as_float_array(x)
    n = int(x.shape[0])
    if n <= 1:
        return [], np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    threshold = float(soft_cfg.get("similarity_threshold", 0.85))
    max_neighbors = int(soft_cfg.get("max_neighbors_per_node", 5))
    time_window_hours = soft_cfg.get("time_window_hours", None)
    feature_indices = soft_cfg.get("feature_indices", None)
    if threshold < 0.5:
        warnings.warn(f"Soft behavior threshold is very low ({threshold})", UserWarning)
    bucket_size_hours = float(soft_cfg.get("bucket_size_hours", time_window_hours or 1.0))
    candidate_bucket_radius = int(soft_cfg.get("candidate_bucket_radius", 1))
    max_candidates_per_bucket = int(soft_cfg.get("max_candidates_per_bucket", min(20000, n // 2)))
    max_candidates_per_bucket = max(1, max_candidates_per_bucket)
    random_seed = int(cfg.get("dataset", {}).get("random_state", 42))
    if max_neighbors <= 0:
        return [], np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    behavior_x = _select_behavior_features(x, feature_indices)
    behavior_x = _l2_normalize(behavior_x)
    edge_dict: dict[tuple[int, int], tuple[float, float]] = {}
    if times_hours is not None and time_window_hours is not None:
        times = np.asarray(times_hours, dtype=np.float32)
        if bucket_size_hours <= 0: bucket_size_hours = float(time_window_hours)
        bucket_ids = np.floor(times / bucket_size_hours).astype(np.int64)
        unique_buckets = np.sort(np.unique(bucket_ids))
        bucket_to_indices = {int(b): np.where(bucket_ids == b)[0] for b in unique_buckets}
        for bucket_idx, b in enumerate(unique_buckets):
            b = int(b)
            source_idx = bucket_to_indices[b]
            candidate_parts = []
            for nb in range(b - candidate_bucket_radius, b + candidate_bucket_radius + 1):
                if nb in bucket_to_indices: candidate_parts.append(bucket_to_indices[nb])
            if not candidate_parts: continue
            candidate_idx = np.concatenate(candidate_parts)
            if len(candidate_idx) > max_candidates_per_bucket:
                rng = np.random.default_rng(random_seed + b)
                candidate_idx = rng.choice(candidate_idx, size=max_candidates_per_bucket, replace=False)
            if len(candidate_idx) <= 1: continue
            candidate_x = behavior_x[candidate_idx]
            source_x = behavior_x[source_idx]
            k = min(len(candidate_idx), max(max_neighbors + 1, max_neighbors * 4 + 1))
            nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
            nn.fit(candidate_x)
            distances, neighbors = nn.kneighbors(source_x)
            for local_src_pos, src in enumerate(source_idx):
                kept = 0
                for pos in range(k):
                    dst = int(candidate_idx[neighbors[local_src_pos, pos]])
                    if src == dst: continue
                    delta = abs(float(times[src] - times[dst]))
                    if delta > float(time_window_hours): continue
                    sim = float(1.0 - distances[local_src_pos, pos])
                    if sim < threshold: continue
                    if times[dst] <= times[src]:
                        edge_dict[(int(dst), int(src))] = (delta, sim)
                        kept += 1
                    if kept >= max_neighbors: break
    else:
        k = min(n, max(max_neighbors + 1, max_neighbors * 4 + 1))
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(behavior_x)
        distances, neighbors = nn.kneighbors(behavior_x)
        for src in range(n):
            kept = 0
            for pos in range(1, k):
                dst = int(neighbors[src, pos])
                if src == dst: continue
                sim = float(1.0 - distances[src, pos])
                if sim < threshold: continue
                edge_dict[(src, dst)] = (0.0, sim)
                if soft_cfg.get("bidirectional", True): edge_dict[(dst, src)] = (0.0, sim)
                kept += 1
                if kept >= max_neighbors: break
    if not edge_dict:
        return [], np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    edges = list(edge_dict.keys())
    edge_time_delta = np.asarray([edge_dict[e][0] for e in edges], dtype=np.float32)
    edge_weight = np.asarray([edge_dict[e][1] for e in edges], dtype=np.float32)
    return edges, edge_time_delta, edge_weight

def summarize_soft_edges(edges: list[tuple[int, int]], edge_weight: np.ndarray, num_nodes: int) -> dict:
    num_edges = len(edges)
    avg_degree = float(num_edges / num_nodes) if num_nodes > 0 else 0.0
    if edge_weight.size == 0:
        weight_min = weight_mean = weight_max = 0.0
    else:
        weight_min = float(np.min(edge_weight))
        weight_mean = float(np.mean(edge_weight))
        weight_max = float(np.max(edge_weight))
    return {"num_nodes": int(num_nodes), "num_soft_edges": int(num_edges), "avg_soft_out_degree": avg_degree, "edge_weight_min": weight_min, "edge_weight_mean": weight_mean, "edge_weight_max": weight_max}

def merge_hard_soft_edges(hard_edges: list[tuple[int, int]], hard_delta: np.ndarray, soft_edges: list[tuple[int, int]], soft_delta: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    edge_to_delta: dict[tuple[int, int], float] = {}
    for e, d in zip(hard_edges, hard_delta): edge_to_delta[e] = float(d)
    for e, d in zip(soft_edges, soft_delta):
        d = float(d)
        edge_to_delta[e] = min(edge_to_delta[e], d) if e in edge_to_delta else d
    edges = list(edge_to_delta.keys())
    delta = np.asarray([edge_to_delta[e] for e in edges], dtype=np.float32)
    return edges, delta
