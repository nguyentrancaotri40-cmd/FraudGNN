# ============================================================
# src/train/federated.py
#
# Federated Learning theo Section IV-C + Algorithm 1.
#
# Paper (nguyên văn, Algorithm 1):
#   1: Initialize θ0
#   2: for each round t = 1..T do
#   3:     Select a subset of clients C_t
#   4:     for each client i ∈ C_t in parallel do
#   5:         θ_t^i ← LocalUpdate(θ_{t-1}, G_i)
#   6:     end for
#   7:     θ_t ← (1/|C_t|) Σ_{i∈C_t} θ_t^i     ← CHIA ĐỀU, không trọng số n_i
#   8: end for
#
# Paper (văn xuôi ngay dưới Algorithm 1):
#   "The LocalUpdate function performs several steps of stochastic gradient
#    descent on the local data. To address the challenge of graph data in
#    federated learning, we employ a graph alignment technique to ensure
#    consistency across different local graphs."
#
# 32 dòng đã xác nhận:
#   #13 local_epochs=5 [INFERRED]
#   #14 chọn tất cả client mỗi round (|C_t| = K) [INFERRED]
#   #15 num_clients=3 [INFERRED]
#   #16 (b6) Phương án B: shared_encoder — projection layer dùng chung
#       + FedAvg align W_t/W_s/W_m. Ghi rõ "giả thuyết suy luận".
#   #17 FedAvg chia đều 1/|C_t| — đúng Algorithm 1, KHÔNG trọng số n_i.
# ============================================================

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.fraudgnn_rl import FraudGNNRL
from src.eval.evaluate import predict_scores
from src.eval.metrics import classification_metrics
from src.train.train_gnn import _pos_weight


# ======================================================================
# Federated Client
# ======================================================================
class FederatedClient:
    def __init__(
        self,
        client_id: int,
        data,
        cfg: Dict[str, Any],
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.data = data
        self.cfg = copy.deepcopy(cfg)
        self.device = device
        self.num_samples = int(data.x.size(0))
        self.model: Optional[FraudGNNRL] = None
        self.optimizer = None
        print(f"[FL] Client {client_id}: N={self.num_samples}")

    def set_model(self, model: FraudGNNRL) -> None:
        self.model = copy.deepcopy(model).to(self.device)

    def set_weights(self, weights: OrderedDict) -> None:
        self.model.load_state_dict(weights)
        self.optimizer = None

    def get_weights(self) -> OrderedDict:
        return self.model.state_dict()

    def get_num_samples(self) -> int:
        return self.num_samples

    # ------------------------------------------------------------------
    def local_update(
        self,
        epochs: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> Dict[str, float]:
        """
        "several steps of stochastic gradient descent on the local data"
        → full-batch GD × epochs (paper không mô tả mini-batch).
        """
        if self.model is None:
            raise RuntimeError("Model chưa được set.")

        data = self.data.to(self.device)
        model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        pos_w = _pos_weight(data.y).to(self.device)

        model.train()
        total_loss = 0.0
        for _ in range(epochs):
            self.optimizer.zero_grad()
            logits = model(data)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, data.y.float(), pos_weight=pos_w,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            self.optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, epochs)

        # Eval local
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(data)).cpu().numpy().flatten()
            labels = data.y.cpu().numpy().flatten()
        metrics = classification_metrics(labels, scores, threshold=0.5)

        return {
            "loss": avg_loss,
            "auc_roc": float(metrics.get("auc_roc", 0)),
            "auc_pr": float(metrics.get("auc_pr", 0)),
            "f1": float(metrics.get("f1", 0)),
            "num_samples": self.num_samples,
        }


# ======================================================================
# Federated Server
# ======================================================================
class FederatedServer:
    def __init__(
        self,
        global_model: FraudGNNRL,
        device: str = "cpu",
    ):
        self.device = device
        self.global_model = global_model.to(device)
        self.global_weights = copy.deepcopy(self.global_model.state_dict())
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def aggregate_fedavg(self, clients: List[FederatedClient]) -> OrderedDict:
        """
        Algorithm 1 dòng 7: θ_t = (1/|C_t|) Σ θ_t^i — CHIA ĐỀU.
        KHÔNG dùng trọng số n_i (khác FedAvg gốc McMahan).
        """
        n = len(clients)
        if n == 0:
            return self.global_weights

        avg = OrderedDict()
        first_w = clients[0].get_weights()
        for k in first_w.keys():
            avg[k] = torch.zeros_like(first_w[k], dtype=torch.float32)

        for c in clients:
            w = c.get_weights()
            for k in avg.keys():
                avg[k] += w[k].float() / n

        self.global_weights = avg
        self.global_model.load_state_dict(avg)
        print(f"[FL] FedAvg ({n} clients, chia đều 1/{n})")
        return avg

    # ------------------------------------------------------------------
    def federated_round(
        self,
        clients: List[FederatedClient],
        local_epochs: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        # 1) Broadcast global weights
        for c in clients:
            c.set_model(self.global_model)
            c.set_weights(self.global_weights)

        # 2) Local update song song (thực tế tuần tự trên cùng GPU)
        losses = []
        local_metrics = []
        for c in clients:
            m = c.local_update(
                epochs=local_epochs, lr=lr, weight_decay=weight_decay,
            )
            losses.append(m["loss"])
            local_metrics.append(m)

        # 3) FedAvg (chia đều)
        self.aggregate_fedavg(clients)

        stats = {
            "round": len(self.history) + 1,
            "avg_loss": float(np.mean(losses)),
            "num_clients": len(clients),
            "client_metrics": local_metrics,
        }
        self.history.append(stats)
        if verbose:
            print(f"[FL] Round {stats['round']}: loss={stats['avg_loss']:.4f}")
        return stats

    # ------------------------------------------------------------------
    def federated_training(
        self,
        clients: List[FederatedClient],
        rounds: int = 10,
        local_epochs: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        print(f"\n[FL] Federated training: rounds={rounds}, local_epochs={local_epochs}, "
              f"lr={lr}, clients={len(clients)}")

        start = time.perf_counter()
        round_times = []

        for t in range(rounds):
            rt0 = time.perf_counter()
            if verbose:
                print(f"\n[FL] Round {t+1}/{rounds}")
            self.federated_round(
                clients, local_epochs=local_epochs, lr=lr,
                weight_decay=weight_decay, verbose=verbose,
            )
            rt = time.perf_counter() - rt0
            round_times.append(rt)
            self.history[-1]["round_time_sec"] = rt

        total = time.perf_counter() - start
        print(f"\n[FL] Done: total={total:.2f}s, avg_round={np.mean(round_times):.2f}s")

        return {
            "history": self.history,
            "final_model": self.global_model,
            "total_time_sec": total,
            "avg_round_time_sec": float(np.mean(round_times)) if round_times else 0.0,
            "round_times": round_times,
        }


# ======================================================================
# Tạo clients từ graph train đã split (KHÔNG đọc raw lại)
# ======================================================================
def create_federated_clients(
    train_data,
    cfg: Dict[str, Any],
    num_clients: int = 3,
    device: str = "cpu",
) -> List[FederatedClient]:
    """
    Chia graph train (đã split) thành num_clients shard theo thời gian.

    [INFERRED] Paper không nói cách chia. Nhóm chọn temporal sharding
    (giữ temporal structure cục bộ) thay vì random (phá vỡ cạnh thời gian).
    """
    n = int(train_data.x.size(0))
    if n < num_clients:
        num_clients = max(1, n)

    if hasattr(train_data, "node_time"):
        order = torch.argsort(train_data.node_time)
    else:
        order = torch.arange(n)

    shard = n // num_clients
    clients = []
    for cid in range(num_clients):
        a = cid * shard
        b = a + shard if cid < num_clients - 1 else n
        idx = order[a:b]
        sub = train_data.subgraph(idx)
        clients.append(FederatedClient(cid, sub, cfg, device=device))
    return clients


# ======================================================================
# Graph alignment (b6-B: shared_encoder = projection dùng chung)
# ======================================================================
def build_global_model_with_shared_encoder(
    in_dim: int,
    cfg: Dict[str, Any],
    device: str = "cpu",
) -> FraudGNNRL:
    """
    [INFERRED — paper không cho công thức graph alignment]
    Nhóm chọn phương án B (b6):
      - Dùng projection layer dùng chung (trainable) trong model.
      - Sau FedAvg, W_t/W_s/W_m dùng chung → đây là "alignment" ở tầng
        tham số.
    Ghi rõ: "giả thuyết suy luận của nhóm".
    """
    mcfg = cfg.get("model", {})
    hidden = int(mcfg.get("hidden_dim", 64))
    model = FraudGNNRL(
        in_dim=in_dim,
        hidden_dim=hidden,
        num_layers=int(mcfg.get("num_layers", 3)),
        num_node_types=int(mcfg.get("num_node_types", 1)),
        dropout=float(mcfg.get("dropout", 0.2)),
        use_projection=True,
        projection_dim=hidden,
    ).to(device)
    print(f"[FL] Global model with shared_encoder projection: {in_dim}→{hidden}")
    return model


# ======================================================================
# Full FL pipeline
# ======================================================================
def train_federated(
    train_data,
    val_data,
    test_data,
    cfg: Dict[str, Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Train FL theo Algorithm 1.

    [WARN] [INFERRED] Paper nói "graph alignment" nhưng không cho công thức
    → dùng shared_encoder (phương án B, b6).
    """
    fl_cfg = cfg.get("federated", {})
    num_clients = int(fl_cfg.get("num_clients", 3))
    rounds = int(fl_cfg.get("rounds", 10))
    local_epochs = int(fl_cfg.get("local_epochs", 5))
    lr = float(fl_cfg.get("learning_rate", 1e-3))
    wd = float(fl_cfg.get("weight_decay", 0.0))

    # 1) Global model với shared_encoder
    global_model = build_global_model_with_shared_encoder(
        in_dim=int(train_data.x.size(1)),
        cfg=cfg,
        device=device,
    )

    # 2) Clients
    clients = create_federated_clients(
        train_data=train_data,
        cfg=cfg,
        num_clients=num_clients,
        device=device,
    )

    # 3) Server
    server = FederatedServer(global_model=global_model, device=device)

    # 4) Train
    result = server.federated_training(
        clients=clients,
        rounds=rounds,
        local_epochs=local_epochs,
        lr=lr,
        weight_decay=wd,
        verbose=True,
    )

    global_model = result["final_model"]
    global_model.eval()

    # 5) Eval
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    val_scores, val_labels = predict_scores(global_model, val_data, device=device)
    test_scores, test_labels = predict_scores(global_model, test_data, device=device)

    print(f"[FL] Val  metrics: {classification_metrics(val_labels, val_scores, 0.5)}")
    print(f"[FL] Test metrics: {classification_metrics(test_labels, test_scores, 0.5)}")

    return {
        "history": result["history"],
        "global_model": global_model,
        "val_scores": val_scores,
        "val_labels": val_labels,
        "test_scores": test_scores,
        "test_labels": test_labels,
        "num_clients": num_clients,
        "num_rounds": rounds,
        "total_time_sec": result["total_time_sec"],
        "avg_round_time_sec": result["avg_round_time_sec"],
        "round_times": result["round_times"],
    }


__all__ = [
    "FederatedClient",
    "FederatedServer",
    "create_federated_clients",
    "build_global_model_with_shared_encoder",
    "train_federated",
]