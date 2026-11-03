# ============================================================
# src/eval/evaluate.py
#
# predict_scores + save_metrics (giữ đơn giản, không làm mất dữ liệu scores).
# ============================================================

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


@torch.no_grad()
def predict_scores(
    model,
    data,
    device: str = "cpu",
    timing: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trả về (scores, labels)."""
    model.eval()
    data = data.to(device)
    t0 = time.perf_counter()
    logits = model(data)
    scores = torch.sigmoid(logits).detach().cpu().numpy().flatten()
    if timing is not None:
        timing["inference_sec"] = timing.get("inference_sec", 0.0) + (time.perf_counter() - t0)
    return scores, data.y.detach().cpu().numpy().flatten()


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def save_metrics(metrics: dict, path: str) -> None:
    """Lưu metrics ra JSON. Lưu toàn bộ scores/labels (không cắt ngắn)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(metrics), f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {path}")