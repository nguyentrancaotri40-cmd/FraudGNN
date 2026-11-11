# ============================================================
# src/eval/metrics.py — AUC-ROC, AUC-PR, F1, Recall@k% (Paper V-A-3) + precision/FPR.
#   AUC-PR = average precision (sklearn).
#   Recall@k%: tỉ lệ fraud nằm trong top k% giao dịch có điểm nghi ngờ cao nhất.
# ============================================================
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def recall_at_k_percent(y_true, y_score, k_percent: float = 1.0) -> float:
    y_true, y_score = np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float)
    n, n_pos = len(y_true), int(y_true.sum())
    if n == 0 or n_pos == 0:
        return 0.0
    k = max(1, int(np.ceil(n * k_percent / 100.0)))
    top = np.argpartition(-y_score, min(k, n) - 1)[:k]
    return float(y_true[top].sum() / n_pos)


def _auc(y, s):
    if len(np.unique(y)) < 2:
        return 0.5, float(np.mean(y)) if len(y) else 0.0
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def classification_metrics(y_true, y_score, threshold: float = 0.5, y_pred=None) -> dict:
    """y_pred (tuỳ chọn) ghi đè quyết định (dùng khi threshold thay đổi theo batch)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    ok = np.isfinite(s)
    if not ok.all():
        y, s = y[ok], s[ok]
        y_pred = None if y_pred is None else np.asarray(y_pred)[ok]
    if len(y) == 0:
        return {k: 0.0 for k in ("auc_roc", "auc_pr", "f1", "precision", "recall", "recall_at_1pct", "fpr")}
    pred = (s >= threshold) if y_pred is None else np.asarray(y_pred).astype(bool)
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    auc_roc, auc_pr = _auc(y, s)
    return {
        "threshold": float(threshold), "auc_roc": auc_roc, "auc_pr": auc_pr,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0,
        "precision": prec, "recall": rec, "fpr": fp / max(1, fp + tn), "fnr": fn / max(1, fn + tp),
        "recall_at_1pct": recall_at_k_percent(y, s, 1.0),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "num_samples": int(len(y)), "num_fraud": int(y.sum()),
    }


def best_f1_threshold(y_true, y_score, grid=None) -> float:
    """Ngưỡng tối đa F1 trên một tập (dùng cho baseline không có RL)."""
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float))
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))]) if len(t) else 0.5


__all__ = ["recall_at_k_percent", "classification_metrics", "best_f1_threshold"]
