# ============================================================
# src/train/federated.py — Federated Learning (Paper IV-C, Algorithm 1).
#
#   1: Initialize θ0
#   2: for t = 1..T:
#   3:    Select a subset of clients C_t
#   4:    for each client i in C_t in parallel:
#   5:        θ_t^i <- LocalUpdate(θ_{t-1}, G_i)        (G_i = local transaction graph)
#   7:    θ_t <- (1/|C_t|) Σ θ_t^i                      <- trung bình ĐỀU, không trọng số n_i
#
# Paper rõ:  FedAvg, trung bình đều, mỗi client có đồ thị cục bộ G_i, LocalUpdate = vài bước SGD,
#            "graph alignment technique" để nhất quán giữa các graph cục bộ.
# [INFERRED] (paper không nói):
#   - num_clients=3; chọn TẤT CẢ client mỗi round (|C_t|=K, tuỳ chọn client_fraction<1).
#   - Chia client theo THỜI GIAN (shard liên tục) -> mỗi client là một "tổ chức" có đồ thị riêng;
#     cạnh xuyên shard bị loại (không chia sẻ dữ liệu thô).
#   - Graph alignment = schema đặc trưng chung (cùng preprocessor) + projection layer dùng chung
#     được FedAvg (model.shared_projection) — phương án B đã xác nhận.
#   - rounds=50 x local_epochs=2 (= 100 epoch-tương-đương như paper), early stopping patience 10 round
#     trên AUC-PR của global model ở val.
#   - Optimizer Adam cục bộ giữ trạng thái qua các round; đồng bộ cả buffer (BatchNorm running stats).
# ============================================================
from __future__ import annotations

import copy
import time
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
import torch

from src.eval.evaluate import predict_scores
from src.eval.metrics import classification_metrics
from src.train.train_gnn import build_model, compute_pos_weight, train_steps
from src.utils.config import get_flags


def fedavg_uniform(states: List[OrderedDict]) -> OrderedDict:
    """θ = (1/|C_t|) Σ θ_i  (Algorithm 1, dòng 7)."""
    n = len(states)
    out = OrderedDict()
    for k in states[0]:
        stacked = torch.stack([s[k].float() for s in states])
        out[k] = stacked.mean(0).to(states[0][k].dtype) if states[0][k].is_floating_point() else stacked.mean(0).round().to(states[0][k].dtype)
    return out


class FederatedClient:
    def __init__(self, cid: int, data, cfg: Dict[str, Any], device: str):
        self.cid, self.data, self.cfg, self.device = cid, data.to(device), cfg, device
        self.pos_w = compute_pos_weight(data.y, cfg.get("train", {}).get("pos_weight_max", 100.0)).to(device)
        self.opt = None
        print(f"[FL] Client {cid}: N={data.num_nodes:,} edges={data.edge_index.size(1):,} "
              f"fraud={float((data.y==1).float().mean())*100:.4f}%")

    def local_update(self, global_state: OrderedDict, model, epochs: int, lr: float) -> OrderedDict:
        model.load_state_dict(global_state)
        if self.opt is None:
            self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        else:  # giữ trạng thái Adam, trỏ lại vào tham số của model dùng chung
            self.opt.param_groups[0]["params"] = list(model.parameters())
        train_steps(model, self.data, self.opt, self.pos_w, self.cfg.get("train", {}), self.device, epochs=epochs)
        return copy.deepcopy(model.state_dict())


def create_federated_clients(train_data, cfg: Dict[str, Any], num_clients: int, device: str) -> List[FederatedClient]:
    n = train_data.num_nodes
    num_clients = max(1, min(num_clients, n))
    order = torch.argsort(train_data.node_time) if getattr(train_data, "has_time", False) else torch.arange(n)
    shard = n // num_clients
    clients = []
    for cid in range(num_clients):
        a, b = cid * shard, (cid + 1) * shard if cid < num_clients - 1 else n
        clients.append(FederatedClient(cid, train_data.subgraph(order[a:b].sort().values), cfg, device))
    return clients


def train_federated(train_data, val_data, cfg: Dict[str, Any], device: str, timing: dict | None = None) -> Dict[str, Any]:
    fl, t = cfg.get("federated", {}), cfg.get("train", {})
    flags = get_flags(cfg)
    rounds, local_epochs = int(fl.get("rounds", 50)), int(fl.get("local_epochs", 2))
    lr, patience = float(fl.get("learning_rate", t.get("learning_rate", 1e-3))), int(fl.get("patience", t.get("patience", 10)))
    frac = float(fl.get("client_fraction", 1.0))
    eval_chunk = t.get("eval_chunk_nodes")

    global_model = build_model(int(train_data.x.size(1)), cfg, device)
    worker = copy.deepcopy(global_model)               # model dùng chung cho LocalUpdate (tuần tự)
    clients = create_federated_clients(train_data, cfg, int(fl.get("num_clients", 3)), device)

    if flags["pruning"]:
        from src.extensions.pruning import amount_schedule, magnitude_prune_
        pc = cfg.get("pruning", {})

    best_state, best, bad, hist, round_times = None, -1.0, 0, [], []
    g_state = copy.deepcopy(global_model.state_dict())
    t_start = time.perf_counter()
    for r in range(1, rounds + 1):
        rt0 = time.perf_counter()
        k = max(1, int(round(frac * len(clients))))
        chosen = clients if k == len(clients) else [clients[i] for i in np.random.choice(len(clients), k, replace=False)]
        states = [c.local_update(g_state, worker, local_epochs, lr) for c in chosen]
        g_state = fedavg_uniform(states)
        global_model.load_state_dict(g_state)
        if flags["pruning"]:
            magnitude_prune_(global_model, amount_schedule(r, int(pc.get("start_epoch", 5)), int(pc.get("end_epoch", 25)),
                                                           float(pc.get("initial_amount", 0.1)), float(pc.get("final_amount", 0.3))))
            g_state = copy.deepcopy(global_model.state_dict())
        s, y = predict_scores(global_model, val_data, device, chunk_nodes=eval_chunk)
        m = classification_metrics(y, s, 0.5)
        round_times.append(time.perf_counter() - rt0)
        hist.append({"round": r, "val_auc_pr": m["auc_pr"], "val_auc_roc": m["auc_roc"], "round_time_sec": round_times[-1]})
        if r == 1 or r % 5 == 0:
            print(f"[FL] round {r:3d}/{rounds} val AUC-PR={m['auc_pr']:.4f} AUC-ROC={m['auc_roc']:.4f} ({round_times[-1]:.1f}s)")
        if m["auc_pr"] > best:
            best, bad, best_state = m["auc_pr"], 0, copy.deepcopy(g_state)
        else:
            bad += 1
            if bad >= patience:
                print(f"[FL] Early stop @ round {r} (best val AUC-PR={best:.4f})")
                break
    if best_state is not None:
        global_model.load_state_dict(best_state)
    total = time.perf_counter() - t_start
    if timing is not None:
        timing["gnn_training_sec"] = total
    return {"model": global_model, "history": hist, "best_val_auc_pr": best, "total_time_sec": total,
            "avg_round_time_sec": float(np.mean(round_times)) if round_times else 0.0,
            "num_clients": len(clients), "num_rounds": len(hist)}


__all__ = ["fedavg_uniform", "FederatedClient", "create_federated_clients", "train_federated"]
