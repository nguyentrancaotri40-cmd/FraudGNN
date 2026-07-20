from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def recall_at_k_percent(y_true: np.ndarray, y_score: np.ndarray, k_percent: float = 1.0) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    k = max(1, int(np.ceil(len(y_true) * (k_percent / 100.0))))
    top_idx = np.argsort(-y_score)[:k]
    positives = np.sum(y_true == 1)
    if positives == 0:
        return 0.0
    return float(np.sum(y_true[top_idx] == 1) / positives)


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    out = {}
    out["threshold"] = float(threshold)
    out["auc_roc"] = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    out["auc_pr"] = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["recall_at_1pct"] = recall_at_k_percent(y_true, y_score, 1.0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["fpr"] = float(fp / max(1, fp + tn))
    out["fnr"] = float(fn / max(1, fn + tp))
    out["tp"] = int(tp); out["fp"] = int(fp); out["tn"] = int(tn); out["fn"] = int(fn)
    return out