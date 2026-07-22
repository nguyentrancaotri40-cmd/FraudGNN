# ============================================================
# FILE: src/train/federated.py
# Federated Learning with FedAvg for FraudGNN-RL
# ============================================================

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import copy
import time
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class FederatedClient:
    """Client for federated learning."""
    
    def __init__(
        self,
        client_id: int,
        data,
        cfg: Dict[str, Any],
        model_class: nn.Module,
        device: str = "cpu"
    ):
        self.client_id = client_id
        self.data = data
        self.cfg = copy.deepcopy(cfg)
        
        # ✅ FIX: Luôn dùng device được truyền vào
        # Không tự động chuyển sang cuda để đảm bảo tất cả clients và server cùng device
        self.device = device
        print(f"[Federated] Client {client_id} on {self.device}")
        
        model_cfg = cfg.get("model", {})
        self.model = model_class(
            in_dim=data.x.size(-1),
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_layers=int(model_cfg.get("num_layers", 3)),
            num_node_types=int(model_cfg.get("num_node_types", 1)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        ).to(self.device)
        
        # ✅ Không khởi tạo optimizer ở đây
        self.optimizer = None
    
    def set_weights(self, weights: OrderedDict):
        """Set model weights from global model."""
        self.model.load_state_dict(weights)
        # ✅ Reset optimizer khi nhận weights mới
        self.optimizer = None
    
    def get_weights(self) -> OrderedDict:
        """Get model weights for aggregation."""
        return self.model.state_dict()
    
    def local_update(
        self,
        epochs: int = 5,
        lr: float = 0.001,
        batch_size: int = 64,
        current_round: int = 0,
        total_rounds: int = 1,
        use_pruning: bool = False,
    ) -> Dict[str, float]:
        """Perform local training on client data."""
        from src.train.train_gnn import _pos_weight
        from src.eval.evaluate import predict_scores
        from src.eval.metrics import classification_metrics
        
        device = self.device
        data = self.data.to(device)
        
        # ✅ Tạo optimizer MỚI mỗi round (đúng FedAvg)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=float(self.cfg.get("train", {}).get("weight_decay", 1e-4))
        )
        
        pos_weight = _pos_weight(data.y).to(device)
        self.model.train()
        
        # ============================================================
        # 🔧 PRUNING: Apply before training (if enabled)
        # ============================================================
        if use_pruning:
            from src.utils.pruning import apply_pruning_inplace, update_pruning_mask, get_pruning_stats
            
            pruning_cfg = self.cfg.get("pruning", {})
            initial_sparsity = pruning_cfg.get("initial_sparsity", 0.1)
            final_sparsity = pruning_cfg.get("final_sparsity", 0.3)
            
            progress = current_round / max(1, total_rounds - 1)
            amount = initial_sparsity + (final_sparsity - initial_sparsity) * progress
            
            from src.utils.pruning import _PRUNABLE
            has_mask = any(
                hasattr(module, 'weight_mask')
                for _, module in self.model.named_modules()
                if isinstance(module, _PRUNABLE)
            )
            
            if not has_mask:
                apply_pruning_inplace(self.model, amount=amount)
            else:
                update_pruning_mask(self.model, amount=amount)
            
            stats = get_pruning_stats(self.model)
            print(f"[Federated] Client {self.client_id}, Round {current_round}: "
                  f"pruned {stats['pruning_ratio']*100:.2f}% of weights")
        
        # ============================================================
        # TRAINING
        # ============================================================
        from torch_geometric.loader import NeighborLoader
        
        neighbor_samples = self.cfg.get("train", {}).get("neighbor_samples", [15, 10])
        
        total_loss = 0.0
        total_batches = 0
        
        for epoch in range(epochs):
            loader = NeighborLoader(
                data,
                num_neighbors=neighbor_samples,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
            )
            
            epoch_loss = 0.0
            epoch_batches = 0
            for batch in loader:
                batch = batch.to(device)
                self.optimizer.zero_grad()
                logits = self.model(batch)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    batch.y.float(),
                    pos_weight=pos_weight
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1
            
            total_loss += epoch_loss
            total_batches += epoch_batches
        
        # ✅ Tính avg_loss chính xác
        avg_loss = total_loss / max(1, total_batches) if total_batches > 0 else 0.0
        
        # ============================================================
        # 🔧 PRUNING: Remove masks before returning weights
        # ============================================================
        if use_pruning:
            from src.utils.pruning import remove_pruning
            self.model = remove_pruning(self.model)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(data)
            scores = torch.sigmoid(logits).cpu().numpy().flatten()
            labels = data.y.cpu().numpy().flatten()
            metrics = classification_metrics(labels, scores, threshold=0.5)
        
        return {
            "loss": avg_loss,
            "auc_roc": metrics.get("auc_roc", 0.0),
            "auc_pr": metrics.get("auc_pr", 0.0),
            "f1": metrics.get("f1", 0.0),
            "num_batches": total_batches,
        }
    
    def evaluate(self, data) -> Dict[str, float]:
        """Evaluate local model on given data."""
        from src.eval.evaluate import predict_scores
        from src.eval.metrics import classification_metrics
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(data.to(self.device))
            scores = torch.sigmoid(logits).cpu().numpy().flatten()
            labels = data.y.cpu().numpy().flatten()
            return classification_metrics(labels, scores, threshold=0.5)


class FederatedServer:
    """Federated learning server with FedAvg algorithm."""
    
    def __init__(
        self,
        model_class: nn.Module,
        model_args: Dict[str, Any],
        device: str = "cpu",
        seed: int = 42,
    ):
        self.model_class = model_class
        self.model_args = model_args
        self.device = device
        self.seed = seed
        
        self.global_model = model_class(**model_args).to(device)
        self.global_weights = self.global_model.state_dict()
        self.history = []
        
    def aggregate(
        self,
        clients: List[FederatedClient],
        method: str = "fedavg",
    ) -> OrderedDict:
        """Aggregate client weights."""
        if not clients:
            return self.global_weights
        
        if method == "fedavg":
            return self._fedavg_aggregate(clients)
        elif method == "median":
            return self._median_aggregate(clients)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def _fedavg_aggregate(self, clients: List[FederatedClient]) -> OrderedDict:
        """FedAvg: weighted average of client models."""
        avg_weights = OrderedDict()
        num_clients = len(clients)
        
        for key in self.global_weights.keys():
            avg_weights[key] = torch.zeros_like(self.global_weights[key], dtype=torch.float32)
        
        for client in clients:
            client_weights = client.get_weights()
            for key in avg_weights.keys():
                avg_weights[key] += client_weights[key].float() / num_clients
        
        self.global_weights = avg_weights
        self.global_model.load_state_dict(avg_weights)
        
        return avg_weights
    
    def _median_aggregate(self, clients: List[FederatedClient]) -> OrderedDict:
        """Median aggregation (robust to outliers)."""
        median_weights = OrderedDict()
        all_weights = [client.get_weights() for client in clients]
        
        for key in self.global_weights.keys():
            stacked = torch.stack([w[key].float() for w in all_weights], dim=0)
            median_weights[key] = torch.median(stacked, dim=0).values
        
        self.global_weights = median_weights
        self.global_model.load_state_dict(median_weights)
        
        return median_weights
    
    def federated_round(
        self,
        clients: List[FederatedClient],
        local_epochs: int = 5,
        lr: float = 0.001,
        batch_size: int = 64,
        current_round: int = 0,
        total_rounds: int = 1,
        use_pruning: bool = False,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """One round of federated learning."""
        for client in clients:
            client.set_weights(self.global_weights)
        
        losses = []
        local_metrics = []
        
        for client in clients:
            metrics = client.local_update(
                epochs=local_epochs,
                lr=lr,
                batch_size=batch_size,
                current_round=current_round,
                total_rounds=total_rounds,
                use_pruning=use_pruning,
            )
            losses.append(metrics["loss"])
            local_metrics.append(metrics)
        
        self.aggregate(clients)
        
        round_stats = {
            "round": len(self.history) + 1,
            "avg_loss": np.mean(losses),
            "num_clients": len(clients),
            "client_metrics": local_metrics,
        }
        self.history.append(round_stats)
        
        if verbose:
            print(f"[Federated] Round {round_stats['round']}: "
                  f"avg_loss={round_stats['avg_loss']:.4f}, "
                  f"num_clients={round_stats['num_clients']}")
        
        return round_stats
    
    def federated_training(
        self,
        clients: List[FederatedClient],
        rounds: int = 10,
        local_epochs: int = 5,
        lr: float = 0.001,
        batch_size: int = 64,
        use_pruning: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Full federated training loop with timing."""
        
        print(f"\n[Federated] Starting federated training with {len(clients)} clients")
        print(f"[Federated] Rounds: {rounds}, Local epochs: {local_epochs}, LR: {lr}")
        print(f"[Federated] Pruning: {'ENABLED' if use_pruning else 'DISABLED'}")
        
        start_time = time.perf_counter()
        round_times = []
        
        for round_idx in range(rounds):
            round_start = time.perf_counter()
            
            if verbose:
                print(f"\n[Federated] Round {round_idx + 1}/{rounds}")
            
            self.federated_round(
                clients=clients,
                local_epochs=local_epochs,
                lr=lr,
                batch_size=batch_size,
                current_round=round_idx,
                total_rounds=rounds,
                use_pruning=use_pruning,
                verbose=verbose,
            )
            
            round_time = time.perf_counter() - round_start
            round_times.append(round_time)
            
            if self.history:
                self.history[-1]["round_time_sec"] = round_time
            
            if verbose:
                print(f"[Federated] Round {round_idx + 1} completed in {round_time:.2f}s")
        
        total_time = time.perf_counter() - start_time
        
        print(f"\n[Federated] Federated training completed in {total_time:.2f}s")
        print(f"[Federated] Avg round time: {sum(round_times)/len(round_times):.2f}s")
        
        return {
            "history": self.history,
            "final_model": self.global_model,
            "total_time_sec": total_time,
            "avg_round_time_sec": sum(round_times) / len(round_times) if round_times else 0,
            "round_times": round_times,
            "num_rounds": rounds,
            "num_clients": len(clients),
        }


def create_federated_clients(
    data,
    cfg: Dict[str, Any],
    model_class: nn.Module,
    num_clients: int = 3,
    device: str = "cpu",
) -> List[FederatedClient]:
    """Create federated clients by splitting data into shards."""
    import numpy as np
    import torch
    
    # ✅ FIX: Strip hybrid_summary trước khi sharding
    if hasattr(data, "hybrid_summary"):
        print(f"[Federated] Removing hybrid_summary before sharding")
        delattr(data, "hybrid_summary")
    
    clients = []
    
    num_nodes = data.x.size(0)
    indices = np.random.permutation(num_nodes)
    shard_size = num_nodes // num_clients
    
    for client_id in range(num_clients):
        start = client_id * shard_size
        end = start + shard_size if client_id < num_clients - 1 else num_nodes
        client_indices = indices[start:end]
        
        client_indices_tensor = torch.tensor(client_indices, dtype=torch.long)
        client_data = data.subgraph(client_indices_tensor)
        
        client = FederatedClient(
            client_id=client_id,
            data=client_data,
            cfg=copy.deepcopy(cfg),
            model_class=model_class,
            device=device,  # ✅ Truyền device từ tham số
        )
        clients.append(client)
        print(f"[Federated] Created client {client_id}: {len(client_indices)} nodes")
    
    return clients


def train_federated(
    train_data,
    val_data,
    test_data,
    cfg: Dict[str, Any],
    model_class: nn.Module,
    device: str = "cpu",
    use_pruning: bool = False,
) -> Dict[str, Any]:
    """Full federated training pipeline."""
    from src.models.fraudgnn_rl import FraudGNNRL
    from src.eval.evaluate import predict_scores
    from src.eval.metrics import classification_metrics
    
    fed_cfg = cfg.get("federated", {})
    
    num_clients = int(fed_cfg.get("num_clients", 3))
    rounds = int(fed_cfg.get("rounds", 10))
    local_epochs = int(fed_cfg.get("local_epochs", 5))
    lr = float(fed_cfg.get("learning_rate", 0.001))
    batch_size = int(fed_cfg.get("batch_size", 64))
    
    clients = create_federated_clients(
        data=train_data,
        cfg=cfg,
        model_class=model_class,
        num_clients=num_clients,
        device=device,  # ✅ Truyền device
    )
    
    model_cfg = cfg.get("model", {})
    server = FederatedServer(
        model_class=model_class,
        model_args={
            "in_dim": train_data.x.size(-1),
            "hidden_dim": int(model_cfg.get("hidden_dim", 64)),
            "num_layers": int(model_cfg.get("num_layers", 3)),
            "num_node_types": int(model_cfg.get("num_node_types", 1)),
            "dropout": float(model_cfg.get("dropout", 0.2)),
        },
        device=device,  # ✅ Truyền device
    )
    
    result = server.federated_training(
        clients=clients,
        rounds=rounds,
        local_epochs=local_epochs,
        lr=lr,
        batch_size=batch_size,
        use_pruning=use_pruning,
        verbose=True,
    )
    
    global_model = result["final_model"]
    global_model.eval()
    
    val_scores, val_labels = predict_scores(global_model, val_data, device=device)
    test_scores, test_labels = predict_scores(global_model, test_data, device=device)
    
    val_metrics = classification_metrics(val_labels, val_scores, threshold=0.5)
    test_metrics = classification_metrics(test_labels, test_scores, threshold=0.5)
    
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
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }