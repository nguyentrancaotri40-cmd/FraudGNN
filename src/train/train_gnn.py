# ============================================================
# src/train/train_gnn.py — Huấn luyện TSSGC + classifier (chế độ centralized: "w/o FL").
#
# Paper V-A-4: Adam lr=1e-3, batch size 64, 100 epochs, early stopping patience 10.
#
# [INFERRED]
#  - Loss: BCE với pos_weight = n_neg/n_pos (cắt tại pos_weight_max) để xử lý mất cân bằng.
#  - Early stopping theo AUC-PR trên val (paper không nói tiêu chí).
#  - "batch size 64" không khả thi trên đồ thị giao dịch (mini-batch node cần láng giềng 3-hop):
#      batch_mode="subgraph": mỗi bước = 1 subgraph liên tục theo thời gian gồm `subgraph_nodes` node
#      batch_mode="full"    : toàn đồ thị (1 bước/epoch, có thể lặp `grad_steps_per_epoch` lần)
# ============================================================
from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.eval.evaluate import iter_node_chunks, predict_scores
from src.eval.metrics import classification_metrics
from src.models.fraudgnn_rl import FraudGNNRL
from src.utils.config import get_flags


def build_model(in_dim: int, cfg: Dict[str, Any], device: str) -> FraudGNNRL:
    m, flags = cfg.get("model", {}), get_flags(cfg)
    model = FraudGNNRL(
        in_dim=in_dim, hidden_dim=int(m.get("hidden_dim", 64)), num_layers=int(m.get("num_layers", 3)),
        num_node_types=int(m.get("num_node_types", 1)), dropout=float(m.get("dropout", 0.2)),
        use_gnn=bool(flags["gnn"]), shared_projection=bool(m.get("shared_projection", True)),
        temporal_max_seq_len=m.get("temporal_max_seq_len"),
    ).to(device)
    return model


def compute_pos_weight(y: torch.Tensor, cap: Optional[float] = 100.0) -> torch.Tensor:
    y = y.flatten()
    pos, neg = (y == 1).sum().float(), (y == 0).sum().float()
    w = (neg / pos.clamp_min(1.0)).clamp_min(1.0)
    return w.clamp_max(cap) if cap else w


def train_steps(model, data, optimizer, pos_w, tcfg: Dict[str, Any], device: str, epochs: int = 1,
                on_epoch_end=None) -> float:
    """Chạy `epochs` epoch trên `data`; trả loss trung bình. Dùng chung cho centralized và FL client."""
    mode = tcfg.get("batch_mode", "full")
    chunk = int(tcfg.get("subgraph_nodes", 4096))
    steps_full = int(tcfg.get("grad_steps_per_epoch", 1))
    clip = float(tcfg.get("grad_clip", 5.0))
    losses = []
    model.train()
    for ep in range(epochs):
        if mode == "subgraph" and data.num_nodes > chunk:
            batches = (data.subgraph(idx) for idx in iter_node_chunks(data.num_nodes, chunk, shuffle=True))
        else:
            batches = (data for _ in range(steps_full))
        for b in batches:
            b = b.to(device)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(b), b.y.float(), pos_weight=pos_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            losses.append(float(loss.item()))
        if on_epoch_end is not None:
            on_epoch_end(ep)
    return float(np.mean(losses)) if losses else 0.0


def train_tssgc_classifier(train_data, val_data, cfg: Dict[str, Any], device: str,
                           timing: Optional[dict] = None) -> Dict[str, Any]:
    t, flags = cfg.get("train", {}), get_flags(cfg)
    model = build_model(int(train_data.x.size(1)), cfg, device)
    print(f"[GNN] centralized | params={sum(p.numel() for p in model.parameters()):,} | gnn={flags['gnn']}")
    opt = torch.optim.Adam(model.parameters(), lr=float(t.get("learning_rate", 1e-3)),
                           weight_decay=float(t.get("weight_decay", 0.0)))
    pos_w = compute_pos_weight(train_data.y, t.get("pos_weight_max", 100.0)).to(device)
    epochs, patience = int(t.get("epochs", 100)), int(t.get("patience", 10))
    eval_chunk = t.get("eval_chunk_nodes")

    prune_cfg = cfg.get("pruning", {})
    if flags["pruning"]:
        from src.extensions.pruning import amount_schedule, magnitude_prune_

    best_state, best, bad, hist = None, -1.0, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        loss = train_steps(model, train_data, opt, pos_w, t, device, epochs=1)
        if flags["pruning"]:
            magnitude_prune_(model, amount_schedule(epoch, int(prune_cfg.get("start_epoch", 5)),
                                                    int(prune_cfg.get("end_epoch", 25)),
                                                    float(prune_cfg.get("initial_amount", 0.1)),
                                                    float(prune_cfg.get("final_amount", 0.3))))
        s, y = predict_scores(model, val_data, device, chunk_nodes=eval_chunk)
        m = classification_metrics(y, s, 0.5)
        hist.append({"epoch": epoch, "train_loss": loss, "val_auc_pr": m["auc_pr"], "val_auc_roc": m["auc_roc"]})
        if epoch == 1 or epoch % 10 == 0:
            print(f"[GNN] ep {epoch:3d} loss={loss:.4f} val AUC-PR={m['auc_pr']:.4f} AUC-ROC={m['auc_roc']:.4f}")
        if m["auc_pr"] > best:
            best, bad, best_state = m["auc_pr"], 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                print(f"[GNN] Early stop @ epoch {epoch} (best val AUC-PR={best:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    total = time.perf_counter() - t0
    if timing is not None:
        timing["gnn_training_sec"] = total
    return {"model": model, "history": hist, "best_val_auc_pr": best, "total_time_sec": total}


__all__ = ["build_model", "compute_pos_weight", "train_steps", "train_tssgc_classifier"]
