# ============================================================
# src/eval/adversarial.py — Robustness trước tấn công đối kháng.
#
# Paper: abstract nêu "resilience to ... adversarial attacks"; Future Work (5) "enhancing adversarial
# robustness". Paper KHÔNG mô tả giao thức tấn công -> [INFERRED] 3 tấn công TEST-TIME chuẩn:
#   - fgsm            : x' = clip(x + eps * sign(grad_x BCE), 0, 1)   (white-box, đặc trưng đã min-max [0,1])
#   - random_noise    : x' = clip(x + N(0, eps^2), 0, 1)
#   - edge_perturb    : thêm/xoá ngẫu nhiên tỉ lệ `eps` cạnh (tấn công cấu trúc đồ thị)
# (Bản cũ "label flipping" lật điểm số đã dự đoán nên không phải tấn công thật — đã loại.)
# Ngưỡng quyết định giữ NGUYÊN threshold của pipeline để so sánh công bằng.
# ============================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .evaluate import predict_scores
from .metrics import classification_metrics


def _fgsm(model, data, device, eps):
    model.eval()
    d = data.clone().to(device)
    d.x = d.x.clone().requires_grad_(True)
    loss = F.binary_cross_entropy_with_logits(model(d), d.y.float())
    grad, = torch.autograd.grad(loss, d.x)
    d.x = (d.x.detach() + eps * grad.sign()).clamp(0.0, 1.0)
    return d


def _noise(model, data, device, eps):
    d = data.clone().to(device)
    d.x = (d.x + torch.randn_like(d.x) * eps).clamp(0.0, 1.0)
    return d


def _edge_perturb(model, data, device, eps):
    d = data.clone().to(device)
    E, N = d.edge_index.size(1), d.num_nodes
    if E == 0:
        return d
    keep = torch.rand(E, device=device) >= eps                      # xoá eps cạnh
    n_add = int(E * eps)
    add = torch.randint(0, N, (2, n_add), device=device)             # thêm eps cạnh ngẫu nhiên
    has_dt = d.edge_time_delta.numel() == E
    d.edge_index = torch.cat([d.edge_index[:, keep], add], dim=1)
    if has_dt:
        d.edge_time_delta = torch.cat([d.edge_time_delta[keep], torch.zeros(n_add, device=device)])
    return d


ATTACKS = {"fgsm": _fgsm, "random_noise": _noise, "edge_perturb": _edge_perturb}


def evaluate_adversarial_robustness(model, test_graph, threshold: float = 0.5, device: str = "cpu",
                                    attacks: Optional[List[str]] = None, epsilon: float = 0.1,
                                    seed: int = 42, output_dir: Optional[str] = None,
                                    eval_chunk: Optional[int] = None) -> Dict:
    attacks = attacks or list(ATTACKS)
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device).eval()
    s0, y = predict_scores(model, test_graph, device, chunk_nodes=eval_chunk)
    base = classification_metrics(y, s0, threshold)
    res: Dict = {"epsilon": epsilon, "threshold": threshold,
                 "clean": {k: base[k] for k in ("auc_roc", "auc_pr", "f1", "recall", "precision", "fpr", "recall_at_1pct")}}
    for name in attacks:
        try:
            adv = ATTACKS[name](model, test_graph, device, epsilon)
            s, _ = predict_scores(model, adv, device, chunk_nodes=eval_chunk)
            m = classification_metrics(y, s, threshold)
            res[name] = {k: m[k] for k in ("auc_roc", "auc_pr", "f1", "recall", "precision", "fpr", "recall_at_1pct")}
            for k in ("f1", "auc_roc", "auc_pr", "recall"):
                res[name][f"{k}_drop_pct"] = float((base[k] - m[k]) / max(base[k], 1e-8) * 100)
        except Exception as e:  # noqa
            res[name] = {"error": repr(e)}
    if output_dir:
        p = Path(output_dir); p.mkdir(parents=True, exist_ok=True)
        (p / "adversarial_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


run_adversarial_test = evaluate_adversarial_robustness
