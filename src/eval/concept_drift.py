# ============================================================
# src/eval/concept_drift.py — Độ bền trước concept drift (Paper abstract / Section I).
#
# Paper nêu "maintaining high performance over extended periods" nhưng KHÔNG mô tả giao thức.
# [INFERRED]: sắp xếp tập test theo thời gian, chia thành `n_windows` cửa sổ bằng số mẫu, tính metric
# từng cửa sổ với CÙNG threshold; báo cáo suy giảm tương đối (cửa sổ đầu -> cuối) và độ dốc tuyến tính.
# Cửa sổ không có fraud: AUC không xác định -> bỏ qua (ghi `skipped_windows`).
# ============================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .metrics import classification_metrics


def evaluate_concept_drift(scores, labels, threshold: float = 0.5, n_windows: int = 5,
                           y_pred=None) -> Dict:
    scores, labels = np.asarray(scores), np.asarray(labels)
    n = len(scores)
    edges = np.linspace(0, n, n_windows + 1).astype(int)
    wins, skipped = [], 0
    for i in range(n_windows):
        a, b = edges[i], edges[i + 1]
        if b - a == 0 or labels[a:b].sum() == 0 or labels[a:b].sum() == b - a:
            skipped += 1
            continue
        m = classification_metrics(labels[a:b], scores[a:b], threshold, None if y_pred is None else np.asarray(y_pred)[a:b])
        wins.append({"window": i, "n": int(b - a), "fraud_rate": float(labels[a:b].mean()),
                     **{k: m[k] for k in ("auc_roc", "auc_pr", "f1", "recall", "precision", "fpr", "recall_at_1pct")}})
    out: Dict = {"n_windows": n_windows, "skipped_windows": skipped, "windows": wins}
    if len(wins) >= 2:
        for k in ("auc_roc", "auc_pr", "f1", "recall_at_1pct"):
            v = [w[k] for w in wins]
            out[f"{k}_decay_pct"] = float((v[0] - v[-1]) / max(v[0], 1e-8) * 100)
            out[f"{k}_slope"] = float(np.polyfit(np.arange(len(v)), v, 1)[0])
    return out


def run_concept_drift_test(scores, labels, threshold: float = 0.5, n_windows: int = 5,
                           y_pred=None, output_dir: Optional[str] = None) -> Dict:
    res = evaluate_concept_drift(scores, labels, threshold, n_windows, y_pred)
    if output_dir:
        p = Path(output_dir); p.mkdir(parents=True, exist_ok=True)
        (p / "concept_drift_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res
