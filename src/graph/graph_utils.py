# ============================================================
# src/graph/graph_utils.py — Graph construction (Paper V-A-4):
#   "each transaction is a node ... we connect transactions that occur within
#    a 1-hour window and have a cosine similarity of their feature vectors
#    above a threshold of 0.9."
#
# Cài đặt TÌM KIẾM CHÍNH XÁC (không kNN xấp xỉ): sort theo thời gian, với mỗi
# chunk query chỉ so với các ứng viên trong [t-w, t+w] bằng phép nhân ma trận.
#   - Không có time (CC2023): chỉ dùng cosine, so với toàn bộ node.
#   - max_neighbors_per_node=None : đúng paper (không giới hạn).
#   - max_neighbors_per_node=k    : cap KỸ THUẬT (top-k theo cosine) để tránh nổ bộ nhớ
#                                   trên dataset rất lớn [DEVIATION — ghi rõ trong kết quả].
# Hướng cạnh: UNDIRECTED, không self-loop [INFERRED].
# ============================================================
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def normalize_time_to_hours(time_values: Optional[np.ndarray], time_unit: Optional[str]) -> Optional[np.ndarray]:
    """Chuẩn hoá thời gian về GIỜ. time_unit: second | minute | hour | None (không có thời gian)."""
    if time_values is None or time_unit is None:
        return None
    t = np.asarray(time_values, dtype=np.float64)
    div = {"second": 3600.0, "minute": 60.0, "hour": 1.0}.get(time_unit)
    if div is None:
        raise ValueError(f"time_unit '{time_unit}' không hỗ trợ (second|minute|hour).")
    return (t / div).astype(np.float64)


@torch.no_grad()
def cosine_neighbors(
    x: np.ndarray | torch.Tensor,
    times_hours: Optional[np.ndarray] = None,
    threshold: float = 0.9,
    window_hours: Optional[float] = 1.0,
    max_neighbors: Optional[int] = None,
    causal: bool = False,
    device: str = "cpu",
    chunk_elems: float = 2e7,
    max_query_chunk: int = 4096,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Với mỗi node i, tìm các j != i thoả: cos(x_i,x_j) >= threshold và |t_i-t_j| <= window.

    Returns (edge_index [2,E], delta_hours [E], sim [E]); cạnh j -> i (src=j, dst=i).
    causal=True: chỉ giữ j đứng TRƯỚC i theo thời gian (dùng cho soft edges).
    """
    x = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x, dtype=torch.float32)
    n = x.size(0)
    xn = torch.nn.functional.normalize(x, dim=1, eps=1e-12).to(device)

    use_time = times_hours is not None and window_hours is not None
    if use_time:
        t_np = np.asarray(times_hours, dtype=np.float64)
        order_np = np.argsort(t_np, kind="stable")
        ts_np = t_np[order_np]
        lo_arr = np.searchsorted(ts_np, ts_np - window_hours, side="left")
        hi_arr = np.searchsorted(ts_np, ts_np + window_hours, side="right")
        order = torch.as_tensor(order_np, device=device)
        xs = xn[order]
        ts = torch.as_tensor(ts_np, dtype=torch.float64, device=device)
    else:
        order = torch.arange(n, device=device)
        xs, ts = xn, None
        lo_arr, hi_arr = np.zeros(n, dtype=np.int64), np.full(n, n, dtype=np.int64)

    src_l, dst_l, dlt_l, sim_l = [], [], [], []
    a, last_pct = 0, -1
    while a < n:
        q = min(max_query_chunk, n - a)
        while True:
            b = a + q
            lo, hi = int(lo_arr[a]), int(hi_arr[b - 1])  # lo,hi đơn điệu vì đã sort
            if q * (hi - lo) <= chunk_elems or q == 1:
                break
            q = max(1, q // 2)
        sims = xs[a:b] @ xs[lo:hi].T                                  # [q, c]
        mask = sims >= threshold
        r = torch.arange(q, device=device)
        self_col = (a + r) - lo                                       # vị trí chính node i trong ứng viên
        mask[r, self_col] = False
        if use_time:
            dt = (ts[a:b, None] - ts[None, lo:hi]).abs()
            mask &= dt <= window_hours
        if causal:  # ứng viên có chỉ số sort nhỏ hơn => không muộn hơn
            cand_idx = torch.arange(lo, hi, device=device)[None, :]
            mask &= cand_idx < (a + r)[:, None]
        if max_neighbors is not None and mask.any():
            k = min(int(max_neighbors), sims.size(1))
            vals, cols = sims.masked_fill(~mask, float("-inf")).topk(k, dim=1)
            rr, kk = (vals > float("-inf")).nonzero(as_tuple=True)
            cc = cols[rr, kk]
        else:
            rr, cc = mask.nonzero(as_tuple=True)
        if rr.numel():
            dst_l.append(order[a + rr].cpu())
            src_l.append(order[lo + cc].cpu())
            sim_l.append(sims[rr, cc].cpu())
            dlt_l.append((ts[a + rr] - ts[lo + cc]).abs().float().cpu() if use_time
                         else torch.zeros(rr.numel()))
        a = b
        if verbose:
            pct = int(100 * a / n)
            if pct // 20 > last_pct // 20:
                last_pct = pct
                print(f"[GRAPH]   {pct}% nodes scanned, edges so far={sum(len(s) for s in src_l):,}")

    if not src_l:
        e = torch.empty(0, dtype=torch.long)
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0), torch.empty(0)
    return (torch.stack([torch.cat(src_l), torch.cat(dst_l)]).long(),
            torch.cat(dlt_l).float(), torch.cat(sim_l).float())


def dedupe_edges(edge_index: torch.Tensor, *attrs: torch.Tensor, num_nodes: int):
    """Loại cạnh trùng (giữ lần xuất hiện đầu) và các attr tương ứng."""
    if edge_index.size(1) == 0:
        return (edge_index, *attrs)
    key = edge_index[0] * num_nodes + edge_index[1]
    uniq, inv = torch.unique(key, return_inverse=True)
    first = torch.full((uniq.numel(),), edge_index.size(1), dtype=torch.long)
    first.scatter_reduce_(0, inv, torch.arange(edge_index.size(1)), reduce="amin")
    return (edge_index[:, first], *[a[first] for a in attrs])


def make_undirected(edge_index: torch.Tensor, delta: torch.Tensor, num_nodes: int):
    """Thêm cạnh ngược và loại trùng (cần khi dùng cap top-k vì quan hệ top-k không đối xứng)."""
    if edge_index.size(1) == 0:
        return edge_index, delta
    ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    d = torch.cat([delta, delta])
    ei, d = dedupe_edges(ei, d, num_nodes=num_nodes)
    keep = ei[0] != ei[1]  # không self-loop
    return ei[:, keep], d[keep]


__all__ = ["normalize_time_to_hours", "cosine_neighbors", "dedupe_edges", "make_undirected"]
