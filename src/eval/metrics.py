from __future__ import annotations

import numpy as np
import warnings
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


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """
    Safe AUC calculation - KHÔNG BAO GIỜ trả về NaN.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    
    # ✅ 1. Remove NaN/Inf
    valid_mask = ~(np.isnan(y_score) | np.isinf(y_score))
    if not np.all(valid_mask):
        n_invalid = np.sum(~valid_mask)
        warnings.warn(f"Found {n_invalid} invalid scores (NaN/Inf). Removing them.", UserWarning)
        y_true = y_true[valid_mask]
        y_score = y_score[valid_mask]
    
    # ✅ 2. Nếu không có dữ liệu → random
    if len(y_true) == 0:
        return 0.5, 0.0
    
    # ✅ 3. Nếu chỉ có 1 class
    unique_labels = np.unique(y_true)
    if len(unique_labels) == 1:
        if unique_labels[0] == 1:
            # Tất cả fraud → AUC = 1.0 (perfect)
            return 1.0, 1.0
        else:
            # Tất cả normal → AUC = 0.5 (random)
            return 0.5, 0.0
    
    # ✅ 4. Có cả 2 classes → tính bình thường
    try:
        # Kiểm tra scores có đa dạng không
        if np.all(y_score == y_score[0]):
            # Tất cả scores giống nhau → random
            return 0.5, float(np.mean(y_true))
        
        auc_roc = float(roc_auc_score(y_true, y_score))
        auc_pr = float(average_precision_score(y_true, y_score))
        
        # ✅ 5. Đảm bảo không bị NaN
        if np.isnan(auc_roc):
            auc_roc = 0.5
        if np.isnan(auc_pr):
            auc_pr = 0.0
            
        return auc_roc, auc_pr
        
    except Exception as e:
        warnings.warn(f"AUC calculation failed: {e}. Using fallback.", UserWarning)
        return 0.5, 0.0


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Classification metrics - ĐẢM BẢO KHÔNG BAO GIỜ CÓ NaN.
    
    Paper sử dụng metrics:
    - AUC-ROC
    - AUC-PR
    - F1
    - Recall
    - Recall@1%
    - FPR
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    
    # ✅ 1. Remove invalid scores
    valid_mask = ~(np.isnan(y_score) | np.isinf(y_score))
    if not np.all(valid_mask):
        n_invalid = np.sum(~valid_mask)
        warnings.warn(f"Found {n_invalid} invalid scores (NaN/Inf). Removing them.", UserWarning)
        y_true = y_true[valid_mask]
        y_score = y_score[valid_mask]
    
    # ✅ 2. Nếu không có dữ liệu
    if len(y_true) == 0:
        return {
            "threshold": float(threshold),
            "auc_roc": 0.5,
            "auc_pr": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "recall_at_1pct": 0.0,
            "fpr": 0.0,
            "fnr": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "num_samples": 0,
            "num_fraud": 0,
            "num_normal": 0,
        }
    
    # ✅ 3. Tính predictions
    y_pred = (y_score >= threshold).astype(int)
    out = {}
    out["threshold"] = float(threshold)
    
    # ✅ 4. AUC (KHÔNG BAO GIỜ NaN)
    auc_roc, auc_pr = _safe_auc(y_true, y_score)
    out["auc_roc"] = auc_roc
    out["auc_pr"] = auc_pr
    
    # ✅ 5. Classification metrics (zero_division=0 để không bị NaN)
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["recall_at_1pct"] = recall_at_k_percent(y_true, y_score, 1.0)
    
    # ✅ 6. Confusion matrix (handling cho 1 class)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        # Chỉ có 1 class
        if np.all(y_true == 0):
            tn = len(y_true)
            fp = 0
            fn = 0
            tp = 0
        else:  # all fraud
            tn = 0
            fp = 0
            fn = 0
            tp = len(y_true)
    
    out["fpr"] = float(fp / max(1, fp + tn))
    out["fnr"] = float(fn / max(1, fn + tp))
    out["tp"] = int(tp)
    out["fp"] = int(fp)
    out["tn"] = int(tn)
    out["fn"] = int(fn)
    out["num_samples"] = len(y_true)
    out["num_fraud"] = int(np.sum(y_true == 1))
    out["num_normal"] = int(np.sum(y_true == 0))
    
    # ✅ 7. Final sanity check - ĐẢM BẢO KHÔNG CÓ NaN
    for key, value in out.items():
        if isinstance(value, float) and np.isnan(value):
            warnings.warn(f"⚠️ Found NaN in {key}! Setting to 0.0", UserWarning)
            out[key] = 0.0
    
    return out