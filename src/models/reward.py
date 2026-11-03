# ============================================================
# src/models/reward.py
# Reward function cho FraudGNN-RL (Section IV-B).
#
# [WARN] [INFERRED — PAPER KHÔNG CÔNG BỐ CÔNG THỨC]
# Paper chỉ mô tả ĐỊNH TÍNH: "Reward r_i: A combination of detection
# accuracy and false positive rate" (Section IV-B, State/Action/Reward).
# KHÔNG có công thức cụ thể trong paper.
#
# → Nhóm tự thiết kế công thức dưới đây dựa trên nguyên tắc paper mô tả:
#       reward = F1 + λ·Recall - μ·FPR
#   - F1 đại diện "detection accuracy"
#   - Recall đảm bảo không bỏ sót fraud
#   - FPR phạt false positive (paper nhấn mạnh "reduce false positives by 31%")
#   - λ, μ là hyperparameter (mặc định λ=0.25, μ=1.0 — [INFERRED])
#
# [WARN] ĐÂY LÀ PHẦN TỰ SUY LUẬN — KHÔNG PHẢI TRÍCH DẪN TỪ PAPER.
# ============================================================

from __future__ import annotations

import numpy as np


def compute_reward(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    fpr_penalty: float = 1.0,
    recall_weight: float = 0.25,
) -> float:
    """
    Reward = F1 + recall_weight * Recall - fpr_penalty * FPR.

    Args:
        tp, fp, tn, fn: confusion matrix components trong batch hiện tại.
        fpr_penalty: hệ số phạt FPR (mặc định 1.0 — [INFERRED]).
        recall_weight: hệ số thưởng Recall (mặc định 0.25 — [INFERRED]).

    Returns:
        reward (float).
    """
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        2 * precision * recall / max(1e-8, precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = fp / max(1, fp + tn)

    reward = f1 + recall_weight * recall - fpr_penalty * fpr
    return float(reward)


def fixed_pos_step(batch_size: int, remaining: int) -> int:
    """
    Bước nhảy cố định cho batch environment (không overlap).

    [INFERRED] Paper không mô tả cách chia batch khi RL chạy trên val/test.
    Nhóm chọn: chia không overlap, bước = batch_size (trừ batch cuối).
    """
    if remaining <= 0:
        return 0
    return int(min(batch_size, remaining))


__all__ = ["compute_reward", "fixed_pos_step"]