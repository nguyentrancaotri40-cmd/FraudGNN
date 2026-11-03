# ============================================================
# src/eval/metrics.py
# ============================================================
from __future__ import annotations

import warnings
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def recall_at_k_percent(y_true: np.ndarray, y_score: np.ndarray, k_percent: float = 1.0) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    if n == 0:
        return 0.0
    k = max(1, int(np.ceil(n * (k_percent / 100.0))))
    top = np.argsort(-y_score)[:k]
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0
    return float(np.sum(y_true[top] == 1) / n_pos)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    valid = ~(np.isnan(y_score) | np.isinf(y_score))
    if not np.all(valid):
        y_true, y_score = y_true[valid], y_score[valid]
    if len(y_true) == 0:
        return 0.5, 0.0
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    if np.all(y_score == y_score[0]):
        return 0.5, float(np.mean(y_true))
    try:
        auc_roc = float(roc_auc_score(y_true, y_score))
        auc_pr = float(average_precision_score(y_true, y_score))
        if np.isnan(auc_roc): auc_roc = 0.5
        if np.isnan(auc_pr): auc_pr = 0.0
        return auc_roc, auc_pr
    except Exception:
        return 0.5, 0.0


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    valid = ~(np.isnan(y_score) | np.isinf(y_score))
    if not np.all(valid):
        y_true, y_score = y_true[valid], y_score[valid]
    if len(y_true) == 0:
        return {
            "threshold": float(threshold), "auc_roc": 0.5, "auc_pr": 0.0,
            "f1": 0.0, "precision": 0.0, "recall": 0.0, "recall_at_1pct": 0.0,
            "fpr": 0.0, "fnr": 0.0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "num_samples": 0, "num_fraud": 0, "num_normal": 0,
        }
    y_pred = (y_score >= threshold).astype(int)
    auc_roc, auc_pr = _safe_auc(y_true, y_score)
    out = {
        "threshold": float(threshold),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "recall_at_1pct": recall_at_k_percent(y_true, y_score, 1.0),
    }
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        if np.all(y_true == 0):
            tn, fp, fn, tp = len(y_true), 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, len(y_true)
    out["fpr"] = float(fp / max(1, fp + tn))
    out["fnr"] = float(fn / max(1, fn + tp))
    out["tp"] = int(tp); out["fp"] = int(fp)
    out["tn"] = int(tn); out["fn"] = int(fn)
    out["num_samples"] = int(len(y_true))
    out["num_fraud"] = int(np.sum(y_true == 1))
    out["num_normal"] = int(np.sum(y_true == 0))
    return out


__all__ = ["recall_at_k_percent", "classification_metrics"]