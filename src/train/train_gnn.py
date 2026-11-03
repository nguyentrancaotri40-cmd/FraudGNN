# ============================================================
# src/train/train_gnn.py
#
# Train TSSGC + classifier (centralized, khi flags.federated = False).
#
# Paper (Section V-A-4):
#   "We use the Adam optimizer with a learning rate of 0.001 and a
#    batch size of 64. We train the model for 100 epochs and use early
#    stopping with a patience of 10 epochs to prevent overfitting."
#
# → Adam lr=1e-3, batch_size=64, epochs=100, patience=10.
# → pos_weight = neg/pos (xử lý imbalanced) [INFERRED — paper không nói,
#   nhưng BCE cần pos_weight cho class imbalance 0.129%).
# → BỎ PRUNING HOÀN TOÀN (b9).
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from src.models.fraudgnn_rl import FraudGNNRL
from src.eval.evaluate import predict_scores
from src.eval.metrics import classification_metrics


def _pos_weight(y: torch.Tensor) -> torch.Tensor:
    """pos_weight = n_neg / n_pos (chuẩn BCE cho imbalanced)."""
    y = y.flatten()
    pos = (y == 1).sum().float()
    neg = (y == 0).sum().float()
    return (neg / pos.clamp_min(1.0)).clamp_min(1.0)


def train_tssgc_classifier(
    train_data,
    val_data,
    cfg: Dict[str, Any],
    output_dir: str = "outputs/checkpoints",
    timing: dict | None = None,
) -> Tuple[FraudGNNRL, list, str, dict]:
    """
    Train centralized TSSGC (dùng khi KHÔNG federated).

    Returns:
        (model, history, ckpt_path, tssgc_timing)
    """
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[TSSGC] Device={device}")
    print(f"[TSSGC] Train N={train_data.x.size(0)}, Val N={val_data.x.size(0)}")

    # ==========================================================
    # Model
    # ==========================================================
    model = FraudGNNRL(
        in_dim=int(train_data.x.size(-1)),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        num_node_types=int(model_cfg.get("num_node_types", 1)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        use_projection=False,   # centralized → không cần projection
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[TSSGC] Model params: {n_params:,}")

    # ==========================================================
    # Optimizer — paper: Adam lr=1e-3, batch=64
    # ==========================================================
    lr = float(train_cfg.get("learning_rate", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.0))  # paper không nói wd
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    pos_weight = _pos_weight(train_data.y).to(device)
    print(f"[TSSGC] pos_weight = {pos_weight.item():.4f}")

    epochs = int(train_cfg.get("epochs", 100))
    patience = int(train_cfg.get("patience", 10))
    batch_size = int(train_cfg.get("batch_size", 64))

    # [INFERRED] Paper nói "batch size of 64" nhưng graph nhỏ → full-batch
    # thực tế vẫn dùng được. Nếu graph lớn, ta chia mini-batch theo node index.
    # → Nhóm chọn full-batch cho đơn giản (paper không mô tả minibatch loop).
    print(f"[TSSGC] Full-batch training ({batch_size=} chỉ dùng cho tham khảo)")

    train_data = train_data.to(device)
    val_data = val_data.to(device)

    best_state = None
    best_metric = -1.0
    bad = 0
    history = []

    start_time = time.perf_counter()
    epoch_times = []

    for epoch in trange(1, epochs + 1, desc="Train TSSGC"):
        ep_start = time.perf_counter()

        # ================== TRAIN ==================
        model.train()
        optimizer.zero_grad()
        logits = model(train_data)
        loss = F.binary_cross_entropy_with_logits(
            logits, train_data.y.float(), pos_weight=pos_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        train_loss = float(loss.item())

        # ================== VAL ==================
        model.eval()
        with torch.no_grad():
            val_logits = model(val_data)
            val_scores = torch.sigmoid(val_logits).cpu().numpy().flatten()
            val_labels = val_data.y.cpu().numpy().flatten()

        val_metrics = classification_metrics(val_labels, val_scores, threshold=0.5)
        auc_pr = float(val_metrics.get("auc_pr", 0.0))

        ep_time = time.perf_counter() - ep_start
        epoch_times.append(ep_time)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_time_sec": ep_time,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        # ================== EARLY STOP ==================
        if auc_pr > best_metric:
            best_metric = auc_pr
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"\n[TSSGC] Early stop at epoch {epoch} "
                      f"(best AUC-PR={best_metric:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    total_time = time.perf_counter() - start_time
    avg_epoch = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0

    if timing is not None:
        timing["tssgc_training_sec"] = total_time

    # ================== SAVE ==================
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(output_dir) / "tssgc_classifier.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_dim": int(train_data.x.size(-1)),
        "config": cfg,
        "history": history,
        "best_val_auc_pr": best_metric,
    }, ckpt_path)
    print(f"[TSSGC] Saved: {ckpt_path}")

    tssgc_timing = {
        "total_training_sec": total_time,
        "avg_epoch_time_sec": avg_epoch,
        "epoch_times": epoch_times,
        "num_epochs": len(epoch_times),
        "best_val_auc_pr": best_metric,
    }
    print(f"[TSSGC] Done: total={total_time:.2f}s, "
          f"avg_epoch={avg_epoch:.2f}s, best_auc_pr={best_metric:.4f}")

    return model, history, str(ckpt_path), tssgc_timing


__all__ = ["train_tssgc_classifier", "_pos_weight"]