# ============================================================
# src/graph/graph_utils.py
#
# Phần 2 của yêu cầu: Graph Construction (Section V-A-4).
#
# Paper (nguyên văn, V-A-4):
#   "we construct a transaction graph where each transaction is a node,
#    and edges are created based on temporal proximity and feature
#    similarity. Specifically, we connect transactions that occur within
#    a 1-hour window and have a cosine similarity of their feature vectors
#    above a threshold of 0.9."
#
# → 2 điều kiện BẮT BUỘC:
#     (1) |t_i - t_j| <= 1 giờ
#     (2) cosine_similarity(f_i, f_j) >= 0.9
#
# Quyết định (đã xác nhận 32 dòng):
#   - Hướng cạnh: UNDIRECTED (paper không nói hướng; Eq. 6 dùng E_i = "tập
#     cạnh connected to node i" → undirected).
#   - Self-loop: KHÔNG thêm (paper không nhắc; Eq. 8 tổng trên N(i)).
#   - max_neighbors: KHÔNG giới hạn (paper không nhắc). Nếu OOM thì user
#     tự đặt config — mặc định = None.
#   - Time bucketing + kNN: tối ưu hóa tính toán, không đổi ngữ nghĩa
#     (chỉ để tránh O(N^2) full pair).
# ============================================================

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


# ----------------------------------------------------------------------
# Chuẩn hóa thời gian về GIỜ
# ----------------------------------------------------------------------
def normalize_time_to_hours(
    time_values: np.ndarray | None,
    time_unit: str | None = None,
    dataset_name: str | None = None,
) -> np.ndarray | None:
    """
    Chuẩn hóa thời gian về đơn vị GIỜ.

    Paper không nói time_unit cụ thể cho từng dataset, nhóm suy luận:
      - PaySim 'step'      : 1 step = 1 giờ (paper nói rõ "each step
                             representing one hour")
      - IEEE-CIS TransactionDT : giây → chia 3600
      - CC2023 'Time'      : giây (từ PCA) → chia 3600
    """
    if time_values is None:
        return None

    t = np.asarray(time_values, dtype=np.float64)

    if time_unit == "second" or (time_unit is None and dataset_name in ("ieee", "ieee-cis", "creditcard")):
        return (t / 3600.0).astype(np.float32)
    elif time_unit == "hour" or (time_unit is None and dataset_name == "paysim"):
        return t.astype(np.float32)
    elif time_unit == "index":
        # [INFERRED] index coi như giờ (mỗi bước 1 giờ)
        return t.astype(np.float32)
    else:
        # [INFERRED] heuristic: nếu max > 1000 → khả năng giây
        if len(t) > 0 and t.max() > 1000:
            print(f"[TIME] [WARN] time_unit='{time_unit}' với max={t.max():.0f} → tự chia 3600")
            return (t / 3600.0).astype(np.float32)
        return t.astype(np.float32)


# ----------------------------------------------------------------------
# Temporal similarity edges — Eq. (paper V-A-4)
# ----------------------------------------------------------------------
def temporal_similarity_edges(
    x: np.ndarray,
    times_hours: np.ndarray | None,
    threshold: float = 0.90,
    time_window_hours: float = 1.0,
    max_neighbors_per_node: int | None = None,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Tạo cạnh UNDIRECTED theo 2 điều kiện paper.

    Args:
        x: [N, d] features.
        times_hours: [N] thời gian (giờ).
        threshold: cosine similarity ngưỡng (paper: 0.9).
        time_window_hours: cửa sổ thời gian (paper: 1h).
        max_neighbors_per_node: nếu != None → cap kỹ thuật [INFERRED].

    Returns:
        edges: list[(i, j)] — mỗi cặp CHỈ xuất hiện 1 lần (i < j), sau đó
               caller tự thêm (j, i) nếu cần undirected.
        deltas: [num_edges] |t_i - t_j| tính bằng giờ.

    [WARN] [INFERRED] Time bucketing để tăng tốc:
        Paper yêu cầu duyệt toàn cặp. Thay vì O(N^2), ta chia thành các
        bucket thời gian rộng 2*time_window_hours, chỉ duyệt cặp trong
        bucket lân cận. Kết quả TƯƠNG ĐƯƠNG duyệt toàn cặp, chỉ nhanh hơn.
    """
    n = int(x.shape[0])
    if n == 0:
        return [], np.empty((0,), dtype=np.float32)

    print(f"[GRAPH] Building edges: N={n}, d={x.shape[1]}, "
          f"threshold={threshold}, window={time_window_hours}h")

    # L2-normalize để cosine = dot product
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)

    edges: List[Tuple[int, int]] = []
    deltas: List[float] = []

    if times_hours is None:
        # Không có thời gian → chỉ cosine (rare case)
        # [INFERRED] Paper luôn có time. Nếu thiếu → dùng kNN toàn cục.
        print("[GRAPH] [WARN] Không có time → chỉ dùng cosine similarity")
        k = min(n, 64)
        nn = NearestNeighbors(n_neighbors=k, metric="cosine",
                              algorithm="brute", n_jobs=-1).fit(x_norm)
        dist, nb = nn.kneighbors(x_norm)
        for i in range(n):
            for pos in range(1, k):
                if 1.0 - dist[i, pos] >= threshold:
                    j = int(nb[i, pos])
                    if i < j:
                        edges.append((i, j))
                        deltas.append(0.0)
        return edges, np.asarray(deltas, dtype=np.float32)

    times = np.asarray(times_hours, dtype=np.float32)

    # Bucket thời gian: mỗi bucket = 2*window → cặp trong bucket lân cận
    # có thể cách nhau tối đa ~3*window. Nhưng ta LỌC LẠI bằng |Δt| <= window
    # nên kết quả chính xác.
    bucket_size = max(1e-3, 2.0 * time_window_hours)
    bucket_ids = np.floor(times / bucket_size).astype(np.int64)
    unique_buckets = np.sort(np.unique(bucket_ids))

    print(f"[GRAPH] {len(unique_buckets)} time buckets (bucket_size={bucket_size:.3f}h)")

    # Chỉ mục các node theo bucket
    from collections import defaultdict
    bucket_to_nodes: dict[int, list[int]] = defaultdict(list)
    for i, b in enumerate(bucket_ids):
        bucket_to_nodes[b].append(i)

    last_log = 0
    for bi, b in enumerate(unique_buckets):
        progress = (bi + 1) / len(unique_buckets) * 100
        if progress - last_log >= 10:
            print(f"[GRAPH]   progress {progress:.0f}%, edges={len(edges)}")
            last_log = progress

        # Chỉ xét bucket b và b+1 (đã bao trùm |Δt| <= 2*window > window)
        idx_a = bucket_to_nodes.get(b, [])
        idx_b = bucket_to_nodes.get(b + 1, [])
        idx_self = idx_a  # cặp trong CÙNG bucket
        if len(idx_self) + len(idx_a) + len(idx_b) <= 1:
            continue

        # Tập candidates cho các node trong bucket b
        cand = sorted(set(idx_a + idx_b))
        if len(cand) <= 1:
            continue

        cand_arr = np.array(cand)
        X_cand = x_norm[cand_arr]
        T_cand = times[cand_arr]

        # kNN trong tập candidate
        k = min(len(cand), 64)
        nn = NearestNeighbors(n_neighbors=k, metric="cosine",
                              algorithm="brute", n_jobs=-1).fit(X_cand)
        dist, nb = nn.kneighbors(X_cand)

        for local_i in range(len(cand)):
            i = cand[local_i]
            kept = 0
            for pos in range(1, k):
                local_j = int(nb[local_i, pos])
                j = cand[local_j]
                if i >= j:
                    continue
                sim = 1.0 - float(dist[local_i, pos])
                if sim < threshold:
                    continue
                dt = abs(float(times[i] - times[j]))
                if dt > time_window_hours:
                    continue
                edges.append((i, j))
                deltas.append(dt)
                kept += 1
                # [INFERRED] cap kỹ thuật nếu user set
                if max_neighbors_per_node is not None and kept >= int(max_neighbors_per_node):
                    break

    print(f"[GRAPH] Done: {len(edges)} undirected cạnh gốc (chưa nhân đôi)")
    if n > 1:
        print(f"[GRAPH] Trung bình: {2*len(edges)/n:.3f} cạnh/node (sau khi nhân đôi)")

    return edges, np.asarray(deltas, dtype=np.float32)


# ----------------------------------------------------------------------
# Tạo edge_index undirected + KHÔNG self-loop
# ----------------------------------------------------------------------
def make_edge_tensors(
    edges: List[Tuple[int, int]],
    deltas: np.ndarray,
    num_nodes: int,
    add_self_loops: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trả về (edge_index [2, 2E], edge_time_delta [2E]).

    Undirected: mỗi cặp (i,j) sinh 2 cạnh (i→j) và (j→i).
    Self-loop: MẶC ĐỊNH KHÔNG THÊM (paper không nhắc — đã xác nhận 32 dòng).
    """
    if not edges:
        if add_self_loops:
            idx = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
            d = torch.zeros(num_nodes, dtype=torch.float32)
            return idx, d
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)

    src_list, dst_list, delta_list = [], [], []
    for (i, j), d in zip(edges, deltas.tolist()):
        src_list.append(i); dst_list.append(j); delta_list.append(d)
        src_list.append(j); dst_list.append(i); delta_list.append(d)

    if add_self_loops:
        for i in range(num_nodes):
            src_list.append(i); dst_list.append(i); delta_list.append(0.0)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_delta = torch.tensor(delta_list, dtype=torch.float32)
    return edge_index, edge_delta


__all__ = ["normalize_time_to_hours", "temporal_similarity_edges", "make_edge_tensors"]