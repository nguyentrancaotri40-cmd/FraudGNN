# src/train/train_gnn.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import copy
import time
import json
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import trange

from src.models.fraudgnn_rl import FraudGNNRL
from src.eval.evaluate import predict_scores
from src.eval.metrics import classification_metrics
from src.utils.pruning import get_pruning_stats, update_pruning_amount, remove_pruning


def _pos_weight(y: torch.Tensor) -> torch.Tensor:
    """Calculate positive weight for imbalanced binary classification."""
    pos = (y == 1).sum().float()
    neg = (y == 0).sum().float()
    return (neg / pos.clamp_min(1.0)).clamp_min(1.0)


def _save_hybrid_summary(data, name: str, output_dir: str = "data/graphs") -> None:
    """Save hybrid graph summary if available."""
    summary = getattr(data, "hybrid_summary", None)
    if summary is not None:
        out_path = Path(output_dir) / f"{name}_hybrid_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Saved hybrid_summary to {out_path}")
        del data.hybrid_summary


def train_tssgc_classifier(
    train_data,
    val_data,
    cfg: Dict[str, Any],
    output_dir: str = "outputs/checkpoints",
    timing: dict | None = None
):
    """Train TSSGC encoder + classifier with early stopping and timing."""
    
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    _save_hybrid_summary(train_data, "train")
    _save_hybrid_summary(val_data, "val")
    
    model = FraudGNNRL(
        in_dim=train_data.x.size(-1),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        num_node_types=int(model_cfg.get("num_node_types", 1)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(device)
    
    lr = float(train_cfg.get("learning_rate", 1e-3))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pos_weight = _pos_weight(train_data.y).to(device)
    
    epochs = int(train_cfg.get("epochs", 100))
    patience = int(train_cfg.get("patience", 10))
    best_state = None
    best_metric = -1.0
    bad = 0
    history = []
    
    # ============================================================
    # 🔧 PRUNING CONFIG
    # ============================================================
    pruning_cfg = cfg.get("pruning", {})
    use_pruning = pruning_cfg.get("enabled", False)
    pruning_start_epoch = pruning_cfg.get("start_epoch", 5)
    pruning_end_epoch = pruning_cfg.get("end_epoch", 25)
    pruning_initial_amount = pruning_cfg.get("initial_amount", 0.1)
    pruning_final_amount = pruning_cfg.get("final_amount", 0.3)
    remove_pruning_after_training = pruning_cfg.get("remove_after_training", True)
    
    if use_pruning:
        print(f"[PRUNING] Enabled: start_epoch={pruning_start_epoch}, end_epoch={pruning_end_epoch}")
    
    from torch_geometric.loader import NeighborLoader
    
    batch_size = int(train_cfg.get("batch_size", 64))
    neighbor_samples = train_cfg.get("neighbor_samples", [15, 10])
    
    train_loader = NeighborLoader(
        train_data,
        num_neighbors=neighbor_samples,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    
    val_loader = NeighborLoader(
        val_data,
        num_neighbors=neighbor_samples,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    
    # ✅ TIMING
    start_time = time.perf_counter()
    epoch_times = []
    
    for epoch in trange(1, epochs + 1, desc="Training TSSGC"):
        epoch_start = time.perf_counter()
        
        # ============================================================
        # 🔧 UPDATE PRUNING AMOUNT (gradual pruning)
        # ============================================================
        if use_pruning and epoch <= pruning_end_epoch and epoch >= pruning_start_epoch:
            model = update_pruning_amount(
                model,
                current_epoch=epoch,
                start_epoch=pruning_start_epoch,
                end_epoch=pruning_end_epoch,
                initial_amount=pruning_initial_amount,
                final_amount=pruning_final_amount,
            )
            stats = get_pruning_stats(model)
            print(f"[PRUNING] Epoch {epoch}: pruned {stats['pruning_ratio']*100:.2f}% of weights")
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                batch.y.float(),
                pos_weight=pos_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        model.eval()
        all_val_scores = []
        all_val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                scores = torch.sigmoid(model(batch)).cpu().numpy().flatten()
                all_val_scores.extend(scores)
                all_val_labels.extend(batch.y.cpu().numpy().flatten())
        
        val_metrics = classification_metrics(
            np.array(all_val_labels),
            np.array(all_val_scores),
            threshold=0.5
        )
        
        auc_pr = val_metrics.get("auc_pr", 0.0)
        
        # ✅ Lưu epoch time
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        
        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "epoch_time_sec": epoch_time,  # ✅ Thêm
            **val_metrics
        })
        
        if use_pruning and epoch <= pruning_end_epoch:
            stats = get_pruning_stats(model)
            history[-1]["pruning_ratio"] = stats["pruning_ratio"]
        
        if auc_pr > best_metric:
            best_metric = auc_pr
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # ============================================================
    # 🔧 REMOVE PRUNING (make permanent)
    # ============================================================
    if use_pruning and remove_pruning_after_training:
        model = remove_pruning(model)
        print("[PRUNING] Pruning masks removed (pruning made permanent)")
    
    # ✅ Tổng hợp timing
    total_training_time = time.perf_counter() - start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    
    if timing is not None:
        timing["tssgc_training_sec"] = total_training_time
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(output_dir) / "tssgc_classifier.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_dim": train_data.x.size(-1),
        "config": cfg,
        "history": history,
        "best_val_auc_pr": best_metric,
    }, ckpt_path)
    
    print(f"\n[TSSGC] Training completed in {total_training_time:.2f}s")
    print(f"[TSSGC] Avg epoch time: {avg_epoch_time:.2f}s")
    
    # ✅ Trả về timing chi tiết
    tssgc_timing = {
        "total_training_sec": total_training_time,
        "avg_epoch_time_sec": avg_epoch_time,
        "epoch_times": epoch_times,
        "num_epochs": len(epoch_times),
    }
    
    return model, history, str(ckpt_path), tssgc_timing