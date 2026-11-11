# ============================================================
# src/train/pipeline_fraudgnn.py — Pipeline đầy đủ (Fig. 2, Section IV-D):
#   New Transactions -> Update Graph -> TSSGC Processing -> RL Action Selection
#   -> Classification -> Feedback -> Update RL Policy
#
# Trong benchmark offline:
#   - Update Graph        : dựng graph 1 lần cho mỗi split (train / val / test)
#   - TSSGC Processing    : huấn luyện encoder+classifier (FL hoặc centralized)
#   - RL Action Selection : NAF/DQN huấn luyện trên VAL (state = embedding, action = threshold + feature weights)
#   - Classification      : áp (threshold, feature weights) lên TEST — KHÔNG feedback online (future work của paper)
#
# Flags (ablation của paper): gnn / rl / federated. Extension (ngoài paper): soft_edges / pruning.
# ============================================================
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.data.load_data import load_dataset
from src.data.preprocess import FraudPreprocessor
from src.data.split import get_cv_splits, resample_fraud_ratio, split_dataframe
from src.eval.evaluate import predict_embeddings, predict_scores, save_metrics
from src.eval.metrics import classification_metrics
from src.graph.build_graph import build_transaction_graph, extract_entity_type
from src.train.federated import train_federated
from src.train.train_gnn import train_tssgc_classifier
from src.train.train_rl import apply_policy, train_rl_on_validation
from src.utils.config import get_flags
from src.utils.model_definitions import resolve_model_name
from src.utils.seed import set_seed


def get_device(cfg: Dict[str, Any]) -> str:
    d = cfg.get("runtime", {}).get("device", "auto")
    return ("cuda" if torch.cuda.is_available() else "cpu") if d == "auto" else d


def prepare_splits(train_df, val_df, test_df, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess (fit trên train) + dựng 3 graph. Dùng chung cho FraudGNN-RL và baseline."""
    ds = cfg.get("dataset", {})
    vals = ds.get("entity_type_values")
    cfg.setdefault("model", {})["num_node_types"] = len(vals) if vals else 1

    timing = {}
    t0 = time.perf_counter()
    pre = FraudPreprocessor(cfg)
    x_tr, y_tr, t_tr = pre.fit_transform(train_df)
    x_va, y_va, t_va = pre.transform(val_df)
    x_te, y_te, t_te = pre.transform(test_df)
    timing["preprocessing_sec"] = time.perf_counter() - t0
    print(f"[PREP] features={x_tr.shape[1]} (train fit only) in {timing['preprocessing_sec']:.1f}s")

    t0 = time.perf_counter()
    graphs = [build_transaction_graph(x, y, t, cfg, extract_entity_type(df, cfg))
              for x, y, t, df in ((x_tr, y_tr, t_tr, train_df), (x_va, y_va, t_va, val_df), (x_te, y_te, t_te, test_df))]
    timing["graph_building_sec"] = time.perf_counter() - t0
    return {"pre": pre, "x": (x_tr, x_va, x_te), "y": (y_tr, y_va, y_te), "t": (t_tr, t_va, t_te),
            "graphs": tuple(graphs), "timing": timing}


def run_pipeline_on_split(train_df, val_df, test_df, cfg: Dict[str, Any], prepared: Optional[dict] = None,
                          seed: Optional[int] = None, return_artifacts: bool = False):
    seed = int(seed if seed is not None else cfg.get("seed", cfg.get("dataset", {}).get("random_state", 42)))
    set_seed(seed)
    flags, device = get_flags(cfg), get_device(cfg)
    print("=" * 60 + f"\n[PIPE] {resolve_model_name(flags)} | flags={flags} | device={device} | seed={seed}\n" + "=" * 60)
    t_all = time.perf_counter()
    prep = prepared or prepare_splits(train_df, val_df, test_df, cfg)
    g_tr, g_va, g_te = prep["graphs"]
    timing = {"preprocessing_sec": 0.0, "graph_building_sec": 0.0, "gnn_training_sec": 0.0,
              "rl_training_sec": 0.0, "inference_sec": 0.0, **prep["timing"]}
    eval_chunk = cfg.get("train", {}).get("eval_chunk_nodes")

    # ---- 1) TSSGC + classifier (FL | centralized) ----------------------
    if flags["federated"]:
        fed = train_federated(g_tr, g_va, cfg, device, timing)
        model, hist = fed["model"], fed["history"]
        fl_info = {k: fed[k] for k in ("num_clients", "num_rounds", "avg_round_time_sec")}
    else:
        out = train_tssgc_classifier(g_tr, g_va, cfg, device, timing)
        model, hist = out["model"], out["history"]
        fl_info = {"centralized": True}
    model.reset_feature_weights()

    # ---- 2) RL ---------------------------------------------------------
    rl_cfg, threshold, rl_info, y_pred_te, y_pred_va = cfg.get("rl", {}), 0.5, {"enabled": False, "threshold_method": "fixed_0.5"}, None, None
    if flags["rl"]:
        val_emb = predict_embeddings(model, g_va, device, eval_chunk)
        y_va = g_va.y.numpy()
        t0 = time.perf_counter()
        agent, rl_hist = train_rl_on_validation(val_emb, y_va, model.classifier, cfg, device)
        timing["rl_training_sec"] = time.perf_counter() - t0
        pol = apply_policy(agent, val_emb, y_va, model.classifier, cfg)
        model.set_feature_weights(pol["feature_weights"])      # trọng số RL thực sự đi vào điểm số
        threshold = pol["threshold"]
        rl_info = {"enabled": True, "agent": agent.kind, "num_updates": len(rl_hist["losses"]),
                   "final_loss": float(np.mean(rl_hist["losses"][-100:])) if rl_hist["losses"] else None,
                   "last_episode_reward": rl_hist["episode_rewards"][-1], "policy_threshold": threshold,
                   "threshold_method": f"{agent.kind}_policy_{rl_cfg.get('inference', 'mean')}",
                   "feature_weight_std": float(np.std(pol["feature_weights"]))}
    print(f"[PIPE] threshold={threshold:.4f} ({rl_info['threshold_method']})")

    # ---- 3) Đánh giá (không feedback tiếp) ---------------------------------
    t0 = time.perf_counter()
    val_scores, val_labels = predict_scores(model, g_va, device, timing, eval_chunk)
    test_scores, test_labels = predict_scores(model, g_te, device, timing, eval_chunk)
    if flags["rl"] and rl_cfg.get("inference", "mean") == "per_batch":
        te_emb = predict_embeddings(model, g_te, device, eval_chunk)
        tp = apply_policy(agent, te_emb, test_labels, model.classifier, cfg)   # action chỉ phụ thuộc state (không nhãn)
        y_pred_te = np.zeros(len(test_scores), dtype=bool)
        for (a, b), th in zip(tp["spans"], tp["thresholds"]):
            y_pred_te[a:b] = test_scores[a:b] >= th
    val_m = classification_metrics(val_labels, val_scores, threshold)
    test_m = classification_metrics(test_labels, test_scores, threshold, y_pred=y_pred_te)
    timing["eval_sec"] = time.perf_counter() - t0
    timing["total_runtime_sec"] = time.perf_counter() - t_all

    extra = {}
    if flags["pruning"]:
        from src.extensions.pruning import get_pruning_stats
        extra["pruning"] = get_pruning_stats(model)
    deviations = []
    g = cfg.get("graph", {})
    if g.get("max_neighbors_per_node") is not None:
        deviations.append(f"graph.max_neighbors_per_node={g['max_neighbors_per_node']} (cap kỹ thuật, paper không giới hạn)")
    if cfg.get("model", {}).get("temporal_max_seq_len") is not None:
        deviations.append(f"model.temporal_max_seq_len={cfg['model']['temporal_max_seq_len']} (cap kỹ thuật cho GRU)")
    if cfg.get("dataset", {}).get("sample_frac") not in (None, 1.0):
        deviations.append(f"dataset.sample_frac={cfg['dataset']['sample_frac']} (KHÔNG phải paper)")

    print(f"\n[PIPE] TEST AUC-ROC={test_m['auc_roc']:.4f} AUC-PR={test_m['auc_pr']:.4f} F1={test_m['f1']:.4f} "
          f"Recall@1%={test_m['recall_at_1pct']*100:.1f}% FPR={test_m['fpr']:.5f} | total {timing['total_runtime_sec']:.0f}s")
    result = {
        "model_name": resolve_model_name(flags), "flags": flags, "seed": seed,
        "selected_threshold": float(threshold), "val_metrics": val_m, "test_metrics": test_m,
        "runtime": timing, "federated_metrics": fl_info, "rl_info": rl_info,
        "graph_info": {"train_edges": int(g_tr.edge_index.size(1)), "val_edges": int(g_va.edge_index.size(1)),
                       "test_edges": int(g_te.edge_index.size(1)), "num_features": int(g_tr.x.size(1))},
        "history_len": len(hist), "deviations_from_paper": deviations, **extra,
    }
    if return_artifacts:
        return result, {"model": model, "graphs": prep["graphs"], "prepared": prep, "device": device,
                        "val_scores": val_scores, "val_labels": val_labels, "test_scores": test_scores,
                        "test_labels": test_labels, "threshold": threshold, "y_pred_test": y_pred_te}
    return result


def _save(result: dict, cfg: Dict[str, Any], suffix: str) -> None:
    out = Path(cfg.get("output", {}).get("dir", "outputs")) / "results"
    save_metrics(result, str(out / f"{cfg.get('experiment', {}).get('name', 'experiment')}_{suffix}.json"))


def run_pipeline(cfg: Dict[str, Any], return_artifacts: bool = False, save: bool = True):
    df = load_dataset(cfg)
    ds = cfg["dataset"]
    if ds.get("fraud_ratio_pct") is not None:   # thí nghiệm imbalance (Fig. 4)
        df = resample_fraud_ratio(df, float(ds["fraud_ratio_pct"]), ds["label_col"], ds.get("time_col"),
                                  int(ds.get("random_state", 42)))
        print(f"[PIPE] fraud ratio ép về {ds['fraud_ratio_pct']}% -> N={len(df):,}")
    train_df, val_df, test_df = split_dataframe(df, cfg)
    res = run_pipeline_on_split(train_df, val_df, test_df, cfg, return_artifacts=return_artifacts)
    result = res[0] if return_artifacts else res
    if save:
        _save(result, cfg, "metrics")
    return res


def run_cv_pipeline(cfg: Dict[str, Any], n_folds: int = 5) -> Dict[str, Any]:
    """5-fold CV (paper V-A-4)."""
    df = load_dataset(cfg)
    runs = []
    for i, (tr, va, te) in enumerate(get_cv_splits(df, cfg, n_folds), 1):
        print(f"\n[CV] Fold {i}/{n_folds}: train={len(tr):,} val={len(va):,} test={len(te):,}")
        try:
            r = run_pipeline_on_split(tr, va, te, copy.deepcopy(cfg))
            runs.append({"fold": i, **r})
        except Exception as e:  # noqa
            import traceback; traceback.print_exc()
            runs.append({"fold": i, "error": str(e)})
    ok = [r for r in runs if "error" not in r]
    summary = {"n_folds": n_folds, "n_valid": len(ok), "folds": runs}
    for m in ("auc_roc", "auc_pr", "f1", "recall", "precision", "fpr", "recall_at_1pct"):
        v = [r["test_metrics"][m] for r in ok]
        if v:
            summary[f"{m}_mean"], summary[f"{m}_std"] = float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0)
    _save(summary, cfg, "cv_results")
    for m in ("auc_roc", "auc_pr", "f1", "recall_at_1pct"):
        if f"{m}_mean" in summary:
            print(f"[CV] {m:<15}: {summary[m+'_mean']:.4f} +/- {summary[m+'_std']:.4f}")
    return summary


__all__ = ["get_device", "prepare_splits", "run_pipeline_on_split", "run_pipeline", "run_cv_pipeline"]
