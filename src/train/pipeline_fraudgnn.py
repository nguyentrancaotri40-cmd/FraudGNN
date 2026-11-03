# ============================================================
# src/train/pipeline_fraudgnn.py
#
# Pipeline đầy đủ theo Fig. 2 (Section IV-D):
#   New Transactions → Update Graph → TSSGC Processing → RL Action Selection
#   → Classification → Feedback → Update RL Policy → (lặp lại)
#
# Trong benchmark:
#   - Transactions = toàn bộ split (train/val/test) đã cố định
#   - "Update Graph" = build graph 1 lần cho mỗi split
#   - "Feedback → Update RL Policy" = train NAF trên VAL
#   - "Classification" trên TEST = eval (KHÔNG feedback online trong benchmark)
#
# Bỏ toàn bộ: soft_edges, hybrid_graph, weighted_fusion, pruning (b9).
# ============================================================

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.data.load_data import load_dataset
from src.data.preprocess import FraudPreprocessor
from src.data.split import split_dataframe
from src.graph.build_graph import (
    build_transaction_graph,
    extract_entity_type,
)
from src.models.fraudgnn_rl import FraudGNNRL
from src.train.federated import train_federated
from src.train.train_gnn import train_tssgc_classifier
from src.train.train_rl import (
    train_rl_on_validation,
    select_threshold_with_rl,
    choose_best_threshold_by_validation,
)
from src.eval.evaluate import predict_scores, save_metrics
from src.eval.metrics import classification_metrics
from src.utils.seed import set_seed


# ======================================================================
# Flags
# ======================================================================
def resolve_flags(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Đọc flags, mặc định paper = full model."""
    flags = cfg.get("flags", {})
    defaults = {
        "federated": True,
        "rl": True,
    }
    resolved = {**defaults, **flags}
    print("=" * 60)
    print(f"[PIPELINE] flags = {resolved}")
    print("=" * 60)
    return resolved


# ======================================================================
# Build 3 graphs (train/val/test)
# ======================================================================
def _build_graphs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Dict[str, Any],
):
    """
    Preprocess → build graph cho 3 split.

    entity_type được extract từ MỖI split độc lập (dùng cột 'type'/'ProductCD'
    có sẵn trong df gốc — không rò rỉ nhãn).
    """
    pre = FraudPreprocessor(cfg)
    x_tr, y_tr, t_tr = pre.fit_transform(train_df)
    x_va, y_va, t_va = pre.transform(val_df)
    x_te, y_te, t_te = pre.transform(test_df)

    # entity_type per split
    et_tr = extract_entity_type(train_df, cfg)
    et_va = extract_entity_type(val_df, cfg)
    et_te = extract_entity_type(test_df, cfg)

    g_tr = build_transaction_graph(x_tr, y_tr, t_tr, cfg, entity_type=et_tr)
    g_va = build_transaction_graph(x_va, y_va, t_va, cfg, entity_type=et_va)
    g_te = build_transaction_graph(x_te, y_te, t_te, cfg, entity_type=et_te)
    return g_tr, g_va, g_te


# ======================================================================
# Pipeline on ONE split
# ======================================================================
def run_pipeline_on_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Dict[str, Any],
    fold_idx: Optional[int] = None,
) -> Dict[str, Any]:
    seed = int(cfg.get("dataset", {}).get("random_state", 42))
    set_seed(seed)

    flags = resolve_flags(cfg)
    use_fl = bool(flags.get("federated", True))
    use_rl = bool(flags.get("rl", True))

    timing = {
        "preprocessing_sec": 0.0,
        "graph_building_sec": 0.0,
        "gnn_training_sec": 0.0,
        "rl_training_sec": 0.0,
        "inference_sec": 0.0,
        "total_runtime_sec": 0.0,
    }
    total_start = time.perf_counter()

    # ==========================================================
    # 1. PREPROCESS + GRAPH
    # ==========================================================
    t0 = time.perf_counter()
    g_tr, g_va, g_te = _build_graphs(train_df, val_df, test_df, cfg)
    timing["graph_building_sec"] = time.perf_counter() - t0
    print(f"[PIPE] Graph built in {timing['graph_building_sec']:.2f}s")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================================
    # 2. GNN TRAINING (FL hoặc centralized)
    # ==========================================================
    t0 = time.perf_counter()
    if use_fl:
        fed = train_federated(
            train_data=g_tr, val_data=g_va, test_data=g_te,
            cfg=cfg, device=device,
        )
        model = fed["global_model"]
        val_scores = fed["val_scores"]
        val_labels = fed["val_labels"]
        test_scores = fed["test_scores"]
        test_labels = fed["test_labels"]
        gnn_history = fed["history"]
        fl_metrics = {
            "num_clients": fed["num_clients"],
            "num_rounds": fed["num_rounds"],
            "avg_round_time_sec": fed["avg_round_time_sec"],
            "total_time_sec": fed["total_time_sec"],
        }
    else:
        model, history, ckpt, tssgc_timing = train_tssgc_classifier(
            train_data=g_tr, val_data=g_va, cfg=cfg,
            output_dir="outputs/checkpoints",
            timing=timing,
        )
        val_scores, val_labels = predict_scores(model, g_va, device=device)
        test_scores, test_labels = predict_scores(model, g_te, device=device)
        gnn_history = history
        fl_metrics = {"centralized": True}

    timing["gnn_training_sec"] = time.perf_counter() - t0
    print(f"[PIPE] GNN training: {timing['gnn_training_sec']:.2f}s")

    # ==========================================================
    # 3. RL — chọn threshold
    # ==========================================================
    threshold = 0.5
    rl_info: Dict[str, Any] = {"enabled": use_rl}
    val_threshold_metrics: Dict[str, Any] = {}

    if use_rl:
        model.eval()
        with torch.no_grad():
            val_emb = model.embeddings(g_va).cpu().numpy()

        t0 = time.perf_counter()
        agent, rl_hist = train_rl_on_validation(
            val_scores=val_scores,
            val_labels=val_labels,
            val_embeddings=val_emb,
            cfg=cfg,
            device=device,
            n_features=val_emb.shape[1],
        )
        timing["rl_training_sec"] = time.perf_counter() - t0

        # Chọn threshold cuối
        threshold, rl_policy_metrics = select_threshold_with_rl(
            agent=agent,
            val_scores=val_scores,
            val_labels=val_labels,
            val_embeddings=val_emb,
            cfg=cfg,
        )

        # Áp feature weights cuối vào model để dùng cho inference
        try:
            fw = agent.get_feature_weights(val_emb[0] if len(val_emb) else np.zeros(val_emb.shape[1]))
            model.set_feature_weights(torch.tensor(fw, dtype=torch.float32))
        except Exception as e:
            print(f"[PIPE] [WARN] Không set được feature weights: {e}")

        rl_info = {
            "enabled": True,
            "num_updates": len(rl_hist.get("losses", [])),
            "final_loss": float(np.mean(rl_hist["losses"][-10:])) if rl_hist.get("losses") else 0.0,
            "policy_threshold": threshold,
            "policy_steps": rl_policy_metrics.get("num_steps", 0),
        }
        val_threshold_metrics = {
            "method": "naf_policy",
            "policy_threshold": threshold,
        }
    else:
        # Fallback static threshold
        thresholds_grid = [float(x) for x in
                           cfg.get("rl", {}).get("threshold_bins", [0.5])]
        threshold, grid_metrics = choose_best_threshold_by_validation(
            val_scores, val_labels, thresholds_grid, cfg=cfg,
        )
        rl_info = {"enabled": False, "selected_threshold": threshold}
        val_threshold_metrics = {
            "method": "static_grid",
            "threshold": threshold,
            "grid_f1": float(grid_metrics.get("f1", 0)),
        }

    print(f"[PIPE] Selected threshold = {threshold:.4f} ({val_threshold_metrics.get('method')})")

    # ==========================================================
    # 4. EVAL trên VAL và TEST (KHÔNG feedback tiếp)
    # ==========================================================
    val_metrics = classification_metrics(val_labels, val_scores, threshold=threshold)
    test_metrics = classification_metrics(test_labels, test_scores, threshold=threshold)

    timing["total_runtime_sec"] = time.perf_counter() - total_start

    print(f"\n[PIPE] TEST  → AUC-ROC={test_metrics['auc_roc']:.4f} | "
          f"AUC-PR={test_metrics['auc_pr']:.4f} | "
          f"F1={test_metrics['f1']:.4f} | "
          f"Recall@1%={test_metrics['recall_at_1pct']:.4f}")

    return {
        "selected_threshold": float(threshold),
        "threshold_selection_method": val_threshold_metrics.get("method", "unknown"),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "runtime": timing,
        "federated_metrics": fl_metrics,
        "rl_info": rl_info,
        "val_scores": val_scores.tolist(),
        "val_labels": val_labels.tolist(),
        "test_scores": test_scores.tolist(),
        "test_labels": test_labels.tolist(),
        "gnn_history_len": len(gnn_history),
    }


# ======================================================================
# Main pipeline (single split)
# ======================================================================
def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    df = load_dataset(cfg)
    train_df, val_df, test_df = split_dataframe(df, cfg)

    result = run_pipeline_on_split(train_df, val_df, test_df, cfg)

    # Save
    exp = cfg.get("experiment", {}).get("name", "experiment")
    fname = f"{exp}_metrics.json"
    save_metrics(result, str(Path("outputs/results") / fname))
    print(f"[PIPE] Saved: outputs/results/{fname}")
    return result


# ======================================================================
# 5-fold CV (paper V-A-4)
# ======================================================================
def run_cv_pipeline(cfg: Dict[str, Any], n_folds: int = 5) -> Dict[str, Any]:
    from src.data.split import get_cv_splits

    df = load_dataset(cfg)
    splits = get_cv_splits(df, cfg, n_folds)

    results = []
    for i, (tr_idx, va_idx) in enumerate(splits):
        tr_df = df.iloc[tr_idx].copy()
        va_df = df.iloc[va_idx].copy()
        # test = phần còn lại; nếu rỗng dùng 20% cuối val
        used = set(tr_idx) | set(va_idx)
        te_idx = [j for j in range(len(df)) if j not in used]
        if not te_idx:
            n_test = max(1, int(len(va_idx) * 0.2))
            te_idx = va_idx[-n_test:].tolist()
            va_idx = va_idx[:-n_test]
            va_df = df.iloc[va_idx].copy()
        te_df = df.iloc[te_idx].copy()

        print(f"\n[CV] Fold {i+1}/{n_folds}: "
              f"tr={len(tr_df)}, va={len(va_df)}, te={len(te_df)}")
        try:
            res = run_pipeline_on_split(tr_df, va_df, te_df, cfg, fold_idx=i)
            results.append({"fold": i + 1, **res})
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"fold": i + 1, "error": str(e)})

    # Summary
    valid = [r for r in results if "error" not in r]
    metrics = ["auc_roc", "auc_pr", "f1", "recall", "precision", "fpr", "recall_at_1pct"]
    summary = {"n_folds": n_folds, "n_valid": len(valid), "all_results": results}
    for m in metrics:
        vals = [r["test_metrics"].get(m, 0) for r in valid]
        if vals:
            summary[f"{m}_mean"] = float(np.mean(vals))
            summary[f"{m}_std"] = float(np.std(vals))
            summary[f"{m}_ci95"] = float(1.96 * np.std(vals) / np.sqrt(len(vals)))

    exp = cfg.get("experiment", {}).get("name", "experiment")
    fname = f"{exp}_cv_results.json"
    save_metrics(summary, str(Path("outputs/results") / fname))
    print(f"\n[CV] Saved: outputs/results/{fname}")

    print("\n[CV] Summary (mean ± std):")
    for m in metrics:
        if f"{m}_mean" in summary:
            print(f"  {m:<15}: {summary[f'{m}_mean']:.4f} ± {summary[f'{m}_std']:.4f}")

    return summary


__all__ = ["resolve_flags", "run_pipeline_on_split", "run_pipeline", "run_cv_pipeline"]