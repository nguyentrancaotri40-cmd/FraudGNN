# src/eval/evaluate.py
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import torch


@torch.no_grad()
def predict_scores(model, data, device: str = "cpu", timing: dict | None = None):
    model.eval()
    data = data.to(device)
    start_time = time.perf_counter()
    logits = model(data)
    scores = torch.sigmoid(logits).detach().cpu().numpy()
    if timing is not None:
        timing["inference_sec"] = time.perf_counter() - start_time
    return scores, data.y.detach().cpu().numpy()


def save_metrics(metrics: dict, path: str) -> None:
    """Save metrics to JSON with proper formatting."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            # ✅ Giữ nguyên list, chỉ format đẹp khi in ra
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_metrics = convert_to_serializable(metrics)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, ensure_ascii=False, indent=2, default=str)