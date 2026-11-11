# ============================================================
# src/eval/evaluate.py — suy luận + lưu kết quả.
# ============================================================
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def iter_node_chunks(n: int, chunk: int, order: Optional[torch.Tensor] = None, shuffle: bool = False):
    idx = order if order is not None else torch.arange(n)
    if shuffle:
        starts = list(range(0, n, chunk))
        np.random.shuffle(starts)
    else:
        starts = range(0, n, chunk)
    for a in starts:
        yield idx[a:a + chunk]


@torch.no_grad()
def predict_scores(model, data, device: str = "cpu", timing: Optional[Dict[str, float]] = None,
                   chunk_nodes: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Trả (scores, labels). chunk_nodes=None: suy luận toàn đồ thị (đúng paper)."""
    model.eval()
    t0 = time.perf_counter()
    if chunk_nodes is None or data.num_nodes <= chunk_nodes:
        scores = torch.sigmoid(model(data.to(device))).cpu().numpy().ravel()
    else:  # cạnh xuyên chunk bị bỏ — chỉ dùng khi hết bộ nhớ
        out = np.zeros(data.num_nodes, dtype=np.float32)
        for idx in iter_node_chunks(data.num_nodes, chunk_nodes):
            sub = data.subgraph(idx).to(device)
            out[idx.numpy()] = torch.sigmoid(model(sub)).cpu().numpy().ravel()
        scores = out
    if timing is not None:
        timing["inference_sec"] = timing.get("inference_sec", 0.0) + time.perf_counter() - t0
    return scores, data.y.cpu().numpy().ravel()


@torch.no_grad()
def predict_embeddings(model, data, device: str = "cpu", chunk_nodes: Optional[int] = None) -> torch.Tensor:
    model.eval()
    if chunk_nodes is None or data.num_nodes <= chunk_nodes:
        return model.embeddings(data.to(device))
    out = []
    for idx in iter_node_chunks(data.num_nodes, chunk_nodes):
        out.append(model.embeddings(data.subgraph(idx).to(device)))
    return torch.cat(out)


def _ser(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, dict):
        return {k: _ser(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_ser(v) for v in o]
    return o


def save_metrics(metrics: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_ser(metrics), f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {path}")
