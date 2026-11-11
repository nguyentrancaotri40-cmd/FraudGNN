# ============================================================
# src/extensions/pruning.py — EXTENSION (không có trong paper)
# L1 magnitude pruning theo tầng, zero hoá trực tiếp trên tham số (không mask hook)
# => tương thích FedAvg/state_dict.
# ============================================================
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


def _prunable(model: nn.Module):
    for name, p in model.named_parameters():
        if p.dim() >= 2 and name.endswith("weight") and "type_emb" not in name:
            yield name, p


@torch.no_grad()
def magnitude_prune_(model: nn.Module, amount: float) -> nn.Module:
    """Zero `amount` (0..1) trọng số nhỏ nhất (theo |w|) trong từng tensor >=2 chiều."""
    if amount <= 0:
        return model
    for _, p in _prunable(model):
        k = int(round(amount * p.numel()))
        if k <= 0:
            continue
        if k >= p.numel():
            p.zero_()
            continue
        thr = torch.kthvalue(p.abs().flatten(), k).values
        p.mul_((p.abs() > thr).to(p.dtype))
    return model


def amount_schedule(epoch: int, start: int = 5, end: int = 25,
                    initial: float = 0.1, final: float = 0.3) -> float:
    """Tăng tuyến tính initial->final trong [start,end]; trước start: 0; sau end: giữ final."""
    if epoch < start:
        return 0.0
    if epoch >= end:
        return final
    return initial + (final - initial) * (epoch - start) / max(1, end - start)


def get_pruning_stats(model: nn.Module) -> Dict[str, Any]:
    tot = zero = 0
    for _, p in _prunable(model):
        tot += p.numel()
        zero += int((p == 0).sum().item())
    return {"total_params": tot, "pruned_params": zero, "pruning_ratio": zero / tot if tot else 0.0}
