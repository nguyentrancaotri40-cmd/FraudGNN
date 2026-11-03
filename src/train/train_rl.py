# ============================================================
# src/train/train_rl.py
#
# Wrapper RL: huấn luyện NAF trên tập validation, greedy eval trên test.
#
# Paper (Section IV-B, IV-D):
#   - State  = graph embedding (từ TSSGC)
#   - Action = (threshold, feature importance weights) — NAF
#   - Reward = "combination of detection accuracy and FPR" → tự thiết kế
#
# [INFERRED] Paper KHÔNG nói rõ:
#   - RL train trên tập nào (train/val)?
#   → Nhóm chọn: train NAF trên VALIDATION (vì test phải giữ độc lập cho
#     benchmark). Đây là chuẩn mực "hyperparameter tuning trên val".
#   - Threshold cuối cùng để báo metric là gì?
#   → Nhóm chọn: mean threshold mà NAF policy chọn greedy trên val.
# ============================================================

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.models.naf_agent import (
    NAFAgent,
    train_naf_agent,
    apply_naf_policy,
)


# ======================================================================
# Train NAF trên val
# ======================================================================
def train_rl_on_validation(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    val_embeddings: Optional[np.ndarray],
    cfg: Dict[str, Any],
    device: Optional[str] = None,
    n_features: int = 64,
) -> Tuple[NAFAgent, Dict[str, Any]]:
    """
    Train NAF agent trên val scores/labels/embeddings.

    Returns:
        (agent, rl_history)
    """
    print(f"\n[RL] Training NAF on val set")
    print(f"[RL] val_samples={len(val_scores)}, n_features={n_features}")

    agent, rl_history = train_naf_agent(
        scores=val_scores,
        labels=val_labels,
        cfg=cfg,
        graph_embeddings=val_embeddings,
        device=device,
        n_features=n_features,
    )

    losses = rl_history.get("losses", [])
    if losses:
        print(f"[RL] NAF done: {len(losses)} updates, "
              f"avg_loss={np.mean(losses):.4f}, last_loss={losses[-1]:.4f}")
    else:
        print(f"[RL] [WARN] NAF không update (buffer chưa đủ)")

    return agent, rl_history


# ======================================================================
# Chọn threshold cuối cùng bằng NAF policy (greedy)
# ======================================================================
def select_threshold_with_rl(
    agent: NAFAgent,
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    val_embeddings: Optional[np.ndarray],
    cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    Greedy áp NAF trên val → trả về (mean_threshold, metrics).
    """
    threshold, metrics = apply_naf_policy(
        agent,
        scores=val_scores,
        labels=val_labels,
        cfg=cfg,
        graph_embeddings=val_embeddings,
    )
    print(f"[RL] Selected threshold (NAF greedy mean): {threshold:.4f}")
    return float(threshold), metrics


# ======================================================================
# Fallback grid search — CHỈ dùng khi NAF không chạy được
# ======================================================================
def choose_best_threshold_by_validation(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
    cfg: Dict[str, Any] | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    [FALLBACK — KHÔNG phải paper]
    Grid search threshold trên val theo metric cấu hình.
    Dùng khi NAF fail hoặc khi user chọn static threshold.

    Default metric: f1
    """
    from src.eval.metrics import classification_metrics

    cfg = cfg or {}
    sel = cfg.get("threshold_selection", {})
    metric = str(sel.get("metric", "f1"))
    f1_w = float(sel.get("f1_weight", 1.0))
    rec_w = float(sel.get("recall_weight", 0.25))
    prec_w = float(sel.get("precision_weight", 0.0))
    fpr_w = float(sel.get("fpr_weight", 0.25))
    max_fpr = sel.get("max_fpr", None)
    min_recall = sel.get("min_recall", None)

    all_metrics = [
        classification_metrics(labels, scores, threshold=th)
        for th in thresholds
    ]

    # Filter ràng buộc
    candidates = []
    for m in all_metrics:
        if max_fpr is not None and float(m["fpr"]) > float(max_fpr):
            continue
        if min_recall is not None and float(m["recall"]) < float(min_recall):
            continue
        candidates.append(m)
    if not candidates:
        candidates = all_metrics

    # Chọn best
    best_score = -np.inf
    best_m = None
    for m in candidates:
        if metric == "f1":
            s = float(m["f1"])
        elif metric == "auc_pr":
            s = float(m["auc_pr"])
        elif metric == "recall_under_fpr":
            s = float(m["recall"]) - fpr_w * float(m["fpr"])
        elif metric == "precision_under_recall":
            s = float(m["precision"]) + 0.25 * float(m["f1"])
        else:  # custom
            s = (f1_w * float(m["f1"])
                 + rec_w * float(m["recall"])
                 + prec_w * float(m["precision"])
                 - fpr_w * float(m["fpr"]))
        if s > best_score:
            best_score = s
            best_m = dict(m)

    assert best_m is not None
    best_m["threshold_selection_score"] = float(best_score)
    best_m["threshold_selection_metric"] = metric
    best_m["threshold_selection_candidates"] = len(candidates)
    return float(best_m["threshold"]), best_m


__all__ = [
    "train_rl_on_validation",
    "select_threshold_with_rl",
    "choose_best_threshold_by_validation",
]