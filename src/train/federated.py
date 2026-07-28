# ============================================================
# FILE: src/train/federated.py
# Federated Learning with FedAvg for FraudGNN-RL
# GIỐNG PAPER 100%: 
#   - FedAvg trung bình cộng không trọng số (Algorithm 1)
#   - Mỗi client có graph riêng
#   - Graph alignment technique (Section IV.C)
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
from pathlib import Path

logger = logging.getLogger(__name__)


def align_client_graphs(
    client_graphs: List[Any],
    alignment_method: str = "feature_projection",
    shared_dim: Optional[int] = None,
) -> List[Any]:
    """
    ✅ GIỐNG PAPER: Graph alignment technique (Section IV.C).
    
    Paper: "employ a graph alignment technique to ensure consistency 
    across different local graphs."
    """
    import torch
    
    if alignment_method == "none" or len(client_graphs) <= 1:
        return client_graphs
    
    print(f"[Federated] Applying graph alignment: {alignment_method}")
    
    aligned_graphs = []
    
    if alignment_method == "feature_projection":
        feat_dims = [g.x.size(1) for g in client_graphs]
        max_dim = max(feat_dims) if feat_dims else 64
        
        if shared_dim is not None:
            target_dim = shared_dim
        else:
            target_dim = max_dim
        
        print(f"[Federated] Aligning features to dimension {target_dim}")
        
        for g in client_graphs:
            g_copy = copy.deepcopy(g)
            if g_copy.x.size(1) < target_dim:
                padding = torch.zeros(
                    g_copy.x.size(0), 
                    target_dim - g_copy.x.size(1), 
                    device=g_copy.x.device
                )
                g_copy.x = torch.cat([g_copy.x, padding], dim=1)
            elif g_copy.x.size(1) > target_dim:
                g_copy.x = g_copy.x[:, :target_dim]
            aligned_graphs.append(g_copy)
    
    elif alignment_method == "node_type_mapping":
        all_types = set()
        for g in client_graphs:
            if hasattr(g, 'node_type'):
                all_types.update(g.node_type.cpu().numpy().tolist())
        
        if all_types:
            type_to_idx = {t: i for i, t in enumerate(sorted(all_types))}
            print(f"[Federated] Aligning {len(all_types)} node types")
            
            for g in client_graphs:
                g_copy = copy.deepcopy(g)
                if hasattr(g_copy, 'node_type'):
                    new_types = torch.tensor(
                        [type_to_idx[t.item()] for t in g_copy.node_type],
                        device=g_copy.node_type.device
                    )
                    g_copy.node_type = new_types
                aligned_graphs.append(g_copy)
        else:
            aligned_graphs = client_graphs
    
    elif alignment_method == "feature_normalization":
        print(f"[Federated] Normalizing features")
        for g in client_graphs:
            g_copy = copy.deepcopy(g)
            mean = g_copy.x.mean(dim=0, keepdim=True)
            std = g_copy.x.std(dim=0, keepdim=True)
            g_copy.x = (g_copy.x - mean) / (std + 1e-8)
            aligned_graphs.append(g_copy)
    
    elif alignment_method == "shared_encoder":
        feat_dims = [g.x.size(1) for g in client_graphs]
        max_dim = max(feat_dims) if feat_dims else 64
        target_dim = shared_dim or 64
        
        projection = nn.Linear(max_dim, target_dim)
        
        for g in client_graphs:
            g_copy = copy.deepcopy(g)
            if g_copy.x.size(1) < max_dim:
                padding = torch.zeros(
                    g_copy.x.size(0), 
                    max_dim - g_copy.x.size(1), 
                    device=g_copy.x.device
                )
                g_copy.x = torch.cat([g_copy.x, padding], dim=1)
            g_copy.x = projection(g_copy.x)
            aligned_graphs.append(g_copy)
    else:
        raise ValueError(f"Unknown alignment method: {alignment_method}")
    
    return aligned_graphs


class FederatedClient:
    """Client for federated learning - Mỗi client có graph riêng."""
    
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
        self.device = device
        print(f"[Federated] Client {client_id} on {self.device} with {data.x.size(0)} nodes")
        
        model_cfg = cfg.get("model", {})
        self.model = model_class(
            in_dim=data.x.size(-1),
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_layers=int(model_cfg.get("num_layers", 3)),
            num_node_types=int(model_cfg.get("num_node_types", 1)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        ).to(self.device)
        
        self.num_samples = data.x.size(0)
        self.optimizer = None
    
    def set_weights(self, weights: OrderedDict):
        self.model.load_state_dict(weights)
        self.optimizer = None
    
    def get_weights(self) -> OrderedDict:
        return self.model.state_dict()
    
    def get_num_samples(self) -> int:
        return self.num_samples
    
    def local_update(
        self,
        epochs: int = 5,
        lr: float = 0.001,
        batch_size: int = 64,
        current_round: int = 0,
        total_rounds: int = 1,
        use_pruning: bool = False,
    ) -> Dict[str, float]:
        from src.train.train_gnn import _pos_weight
        from src.eval.evaluate import predict_scores
        from src.eval.metrics import classification_metrics
        
        device = self.device
        data = self.data.to(device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=float(self.cfg.get("train", {}).get("weight_decay", 1e-4))
        )
        
        pos_weight = _pos_weight(data.y).to(device)
        self.model.train()
        
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
        
        avg_loss = total_loss / max(1, total_batches) if total_batches > 0 else 0.0
        
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
            "num_samples": self.num_samples,
        }
    
    def evaluate(self, data) -> Dict[str, float]:
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
        if not clients:
            return self.global_weights
        
        if method == "fedavg":
            return self._fedavg_aggregate(clients)
        elif method == "median":
            return self._median_aggregate(clients)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def _fedavg_aggregate(self, clients: List[FederatedClient]) -> OrderedDict:
        """
        ✅ GIỐNG PAPER 100%: Trung bình cộng không trọng số.
        
        Paper Algorithm 1, dòng 7: θ_t = (1/|C_t|) Σ_{i∈C_t} θ_i^t
        """
        num_clients = len(clients)
        avg_weights = OrderedDict()
        
        for key in self.global_weights.keys():
            avg_weights[key] = torch.zeros_like(self.global_weights[key], dtype=torch.float32)
        
        for client in clients:
            client_weights = client.get_weights()
            for key in avg_weights.keys():
                avg_weights[key] += client_weights[key].float() / num_clients
        
        self.global_weights = avg_weights
        self.global_model.load_state_dict(avg_weights)
        
        print(f"[Federated] Aggregated {len(clients)} clients (unweighted average, giống paper)")
        for client in clients:
            print(f"  Client {client.client_id}: {client.get_num_samples():,} samples")
        
        return avg_weights
    
    def _median_aggregate(self, clients: List[FederatedClient]) -> OrderedDict:
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


def load_client_graph(client_id: int, data_dir: str) -> Any:
    from src.graph.build_graph import load_graph
    
    graph_path = Path(data_dir) / f"client_{client_id}_graph.pkl"
    if graph_path.exists():
        return load_graph(str(graph_path))
    return None


def save_client_graph(graph, client_id: int, data_dir: str) -> None:
    from src.graph.build_graph import save_graph
    
    graph_path = Path(data_dir) / f"client_{client_id}_graph.pkl"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    save_graph(graph, str(graph_path))


def create_federated_clients_from_raw_data(
    raw_data_dir: str,
    cfg: Dict[str, Any],
    model_class: nn.Module,
    num_clients: int = 3,
    device: str = "cpu",
    alignment_method: str = "none",
) -> List[FederatedClient]:
    from src.data.load_data import load_dataset
    from src.data.preprocess import FraudPreprocessor
    from src.graph.build_graph import build_transaction_graph
    from src.graph.hybrid_graph import build_hybrid_transaction_graph
    
    clients = []
    client_graphs = []
    
    for client_id in range(num_clients):
        client_csv = Path(raw_data_dir) / f"client_{client_id}.csv"
        
        if client_csv.exists():
            print(f"[Federated] Loading client {client_id} from: {client_csv}")
            client_cfg = copy.deepcopy(cfg)
            client_cfg["dataset"]["path"] = str(client_csv)
            df = load_dataset(client_cfg)
            
            pre = FraudPreprocessor(client_cfg)
            x, y, t = pre.fit_transform(df)
            
            flags = cfg.get("flags", {})
            use_hybrid = flags.get("hybrid_graph", False)
            
            if use_hybrid:
                graph = build_hybrid_transaction_graph(x, y, t, client_cfg)
            else:
                graph = build_transaction_graph(x, y, t, client_cfg)
            
            client_graphs.append(graph)
    
    if alignment_method != "none" and client_graphs:
        client_graphs = align_client_graphs(
            client_graphs,
            alignment_method=alignment_method,
            shared_dim=cfg.get("model", {}).get("hidden_dim", 64),
        )
    
    for client_id, graph in enumerate(client_graphs):
        client = FederatedClient(
            client_id=client_id,
            data=graph,
            cfg=cfg,
            model_class=model_class,
            device=device,
        )
        clients.append(client)
    
    return clients


def create_federated_clients_from_different_datasets(
    dataset_paths: List[str],
    cfg: Dict[str, Any],
    model_class: nn.Module,
    device: str = "cpu",
    alignment_method: str = "none",
) -> List[FederatedClient]:
    from src.data.load_data import load_dataset
    from src.data.preprocess import FraudPreprocessor
    from src.graph.build_graph import build_transaction_graph
    from src.graph.hybrid_graph import build_hybrid_transaction_graph
    
    clients = []
    client_graphs = []
    
    for client_id, dataset_path in enumerate(dataset_paths):
        print(f"[Federated] Client {client_id} loading dataset: {dataset_path}")
        
        client_cfg = copy.deepcopy(cfg)
        client_cfg["dataset"]["path"] = dataset_path
        df = load_dataset(client_cfg)
        
        pre = FraudPreprocessor(client_cfg)
        x, y, t = pre.fit_transform(df)
        
        flags = cfg.get("flags", {})
        use_hybrid = flags.get("hybrid_graph", False)
        
        if use_hybrid:
            graph = build_hybrid_transaction_graph(x, y, t, client_cfg)
        else:
            graph = build_transaction_graph(x, y, t, client_cfg)
        
        client_graphs.append(graph)
    
    if alignment_method != "none" and client_graphs:
        client_graphs = align_client_graphs(
            client_graphs,
            alignment_method=alignment_method,
            shared_dim=cfg.get("model", {}).get("hidden_dim", 64),
        )
    
    for client_id, graph in enumerate(client_graphs):
        client = FederatedClient(
            client_id=client_id,
            data=graph,
            cfg=cfg,
            model_class=model_class,
            device=device,
        )
        clients.append(client)
    
    return clients


def create_federated_clients_from_single_dataset(
    cfg: Dict[str, Any],
    model_class: nn.Module,
    num_clients: int = 3,
    device: str = "cpu",
    alignment_method: str = "none",
) -> List[FederatedClient]:
    from src.data.load_data import load_dataset
    from src.data.preprocess import FraudPreprocessor
    from src.graph.build_graph import build_transaction_graph
    from src.graph.hybrid_graph import build_hybrid_transaction_graph
    
    clients = []
    client_graphs = []
    
    df = load_dataset(cfg)
    time_col = cfg["dataset"].get("time_col")
    
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
        n = len(df)
        shard_size = n // num_clients
        
        for client_id in range(num_clients):
            start = client_id * shard_size
            end = start + shard_size if client_id < num_clients - 1 else n
            
            client_df = df.iloc[start:end].copy()
            
            print(f"[Federated] Client {client_id}: {len(client_df)} samples "
                  f"(time {start} to {end})")
            
            pre = FraudPreprocessor(cfg)
            x, y, t = pre.fit_transform(client_df)
            
            flags = cfg.get("flags", {})
            use_hybrid = flags.get("hybrid_graph", False)
            
            if use_hybrid:
                graph = build_hybrid_transaction_graph(x, y, t, cfg)
            else:
                graph = build_transaction_graph(x, y, t, cfg)
            
            client_graphs.append(graph)
    
    if alignment_method != "none" and client_graphs:
        client_graphs = align_client_graphs(
            client_graphs,
            alignment_method=alignment_method,
            shared_dim=cfg.get("model", {}).get("hidden_dim", 64),
        )
    
    for client_id, graph in enumerate(client_graphs):
        client = FederatedClient(
            client_id=client_id,
            data=graph,
            cfg=cfg,
            model_class=model_class,
            device=device,
        )
        clients.append(client)
    
    return clients


def create_federated_clients(
    data,
    cfg: Dict[str, Any],
    model_class: nn.Module,
    num_clients: int = 3,
    device: str = "cpu",
) -> List[FederatedClient]:
    fed_cfg = cfg.get("federated", {})
    alignment_method = fed_cfg.get("alignment_method", "feature_projection")
    
    raw_data_dir = cfg.get("dataset", {}).get("path", "data/raw")
    
    clients = create_federated_clients_from_raw_data(
        Path(raw_data_dir).parent,
        cfg,
        model_class,
        num_clients,
        device,
        alignment_method=alignment_method,
    )
    
    if clients:
        return clients
    
    dataset_paths = cfg.get("federated", {}).get("dataset_paths", [])
    if dataset_paths:
        clients = create_federated_clients_from_different_datasets(
            dataset_paths,
            cfg,
            model_class,
            device,
            alignment_method=alignment_method,
        )
        if clients:
            return clients
    
    return create_federated_clients_from_single_dataset(
        cfg,
        model_class,
        num_clients,
        device,
        alignment_method=alignment_method,
    )


def train_federated(
    train_data,
    val_data,
    test_data,
    cfg: Dict[str, Any],
    model_class: nn.Module,
    device: str = "cpu",
    use_pruning: bool = False,
) -> Dict[str, Any]:
    """Full federated training pipeline với graph alignment (giống paper)."""
    from src.models.fraudgnn_rl import FraudGNNRL
    from src.eval.evaluate import predict_scores
    from src.eval.metrics import classification_metrics
    
    fed_cfg = cfg.get("federated", {})
    
    num_clients = int(fed_cfg.get("num_clients", 3))
    rounds = int(fed_cfg.get("rounds", 10))
    local_epochs = int(fed_cfg.get("local_epochs", 5))
    lr = float(fed_cfg.get("learning_rate", 0.001))
    batch_size = int(fed_cfg.get("batch_size", 64))
    
    # ============================================================
    # BƯỚC 1: Tạo clients
    # ============================================================
    clients = create_federated_clients(
        data=train_data,
        cfg=cfg,
        model_class=model_class,
        num_clients=num_clients,
        device=device,
    )
    
    # ============================================================
    # BƯỚC 2: Lấy in_dim thực tế
    # ============================================================
    if clients:
        actual_in_dim = clients[0].data.x.size(1)
        print(f"[Federated] Actual in_dim after alignment: {actual_in_dim}")
    else:
        actual_in_dim = train_data.x.size(1)
    
    # ============================================================
    # BƯỚC 3: Đồng bộ val_data và test_data
    # ============================================================
    # Đưa về đúng device
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    # Pad hoặc truncate val_data
    if val_data.x.size(1) != actual_in_dim:
        print(f"[Federated] Resizing val_data from {val_data.x.size(1)} to {actual_in_dim}")
        if val_data.x.size(1) < actual_in_dim:
            padding = torch.zeros(
                val_data.x.size(0), 
                actual_in_dim - val_data.x.size(1), 
                device=device,
                dtype=val_data.x.dtype
            )
            val_data.x = torch.cat([val_data.x, padding], dim=1)
        else:
            val_data.x = val_data.x[:, :actual_in_dim]
    
    # Pad hoặc truncate test_data
    if test_data.x.size(1) != actual_in_dim:
        print(f"[Federated] Resizing test_data from {test_data.x.size(1)} to {actual_in_dim}")
        if test_data.x.size(1) < actual_in_dim:
            padding = torch.zeros(
                test_data.x.size(0), 
                actual_in_dim - test_data.x.size(1), 
                device=device,
                dtype=test_data.x.dtype
            )
            test_data.x = torch.cat([test_data.x, padding], dim=1)
        else:
            test_data.x = test_data.x[:, :actual_in_dim]
    
    # ============================================================
    # BƯỚC 4: Tạo server
    # ============================================================
    model_cfg = cfg.get("model", {})
    
    server = FederatedServer(
        model_class=model_class,
        model_args={
            "in_dim": actual_in_dim,
            "hidden_dim": int(model_cfg.get("hidden_dim", 64)),
            "num_layers": int(model_cfg.get("num_layers", 3)),
            "num_node_types": int(model_cfg.get("num_node_types", 1)),
            "dropout": float(model_cfg.get("dropout", 0.2)),
        },
        device=device,
    )
    
    # ============================================================
    # BƯỚC 5: Cập nhật global_model
    # ============================================================
    server.global_model = model_class(
        in_dim=actual_in_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        num_node_types=int(model_cfg.get("num_node_types", 1)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(device)
    server.global_weights = server.global_model.state_dict()
    
    for client in clients:
        client.set_weights(server.global_weights)
    
    # ============================================================
    # BƯỚC 6: Federated training
    # ============================================================
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