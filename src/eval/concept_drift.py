# ============================================================
# src/eval/concept_drift.py
#
# Đánh giá độ bền vững với Concept Drift (Paper Section V-B).
# ============================================================

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from typing import Dict, Optional

from .metrics import classification_metrics


def evaluate_concept_drift(
    scores: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    window_size: int = 1000,
    threshold: float = 0.5,
) -> Dict:
    """Tính metrics drift theo từng time window."""
    n = len(scores)
    n_windows = max(1, n // window_size)
    window_metrics = []

    for i in range(n_windows):
        start = i * window_size
        end = min((i + 1) * window_size, n)
        
        w_scores = scores[start:end]
        w_labels = labels[start:end]
        if len(w_scores) == 0:
            continue
            
        m = classification_metrics(w_labels, w_scores, threshold=threshold)
        m['window_idx'] = i
        window_metrics.append(m)

    f1s = [m['f1'] for m in window_metrics]
    aucs = [m.get('auc_roc', 0) for m in window_metrics]
    recalls = [m['recall'] for m in window_metrics]

    if len(f1s) > 1:
        f1_decay = (f1s[0] - f1s[-1]) / max(f1s[0], 1e-8) * 100
        auc_decay = (aucs[0] - aucs[-1]) / max(aucs[0], 1e-8) * 100
        recall_decay = (recalls[0] - recalls[-1]) / max(recalls[0], 1e-8) * 100
    else:
        f1_decay = auc_decay = recall_decay = 0.0

    return {
        'f1_decay_percent': float(f1_decay),
        'auc_decay_percent': float(auc_decay),
        'recall_decay_percent': float(recall_decay),
        'num_windows': len(window_metrics),
        'initial_f1': float(f1s[0]) if f1s else 0.0,
        'final_f1': float(f1s[-1]) if f1s else 0.0,
        'window_metrics': window_metrics,
    }


def run_concept_drift_test(
    result_dict: Optional[Dict] = None,
    scores: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    window_size: int = 1000,
    output_dir: Optional[str] = None,
) -> Dict:
    """Hàm wrapper gọi từ pipeline hoặc file JSON."""
    
    # Nếu truyền vào dict kết quả pipeline
    if result_dict is not None:
        scores = result_dict.get('test_scores', [])
        labels = result_dict.get('test_labels', [])
        # Dùng index làm timestamp nếu không có
        timestamps = np.arange(len(scores))
        
    if scores is None or labels is None:
        print("[WARN] Không có scores/labels để chạy Concept Drift.")
        return {}

    scores = np.array(scores)
    labels = np.array(labels)
    if timestamps is None:
        timestamps = np.arange(len(scores))
    else:
        timestamps = np.array(timestamps)

    # Sort by time
    order = np.argsort(timestamps)
    scores = scores[order]
    labels = labels[order]
    timestamps = timestamps[order]

    result = evaluate_concept_drift(scores, labels, timestamps, window_size=window_size)

    if output_dir:
        out_path = Path(output_dir) / 'concept_drift_results.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[OK] Concept drift saved to {out_path}")

    return result