from __future__ import annotations

import json
from pathlib import Path
import time
import torch
from .metrics import classification_metrics


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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)