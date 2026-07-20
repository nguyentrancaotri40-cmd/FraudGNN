# ============================================================
# src/train/pipeline_fraudgnn.py
# Pipeline: Graph → TSSGC → FedAvg → DQN
# Hỗ trợ 2 model: FraudGNN-RL (baseline) và FraudGNN-RL+ (hybrid)
# ============================================================

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch_geometric.data import Data

from src.data.load_data import load_dataset
from src.data.split import split_dataframe
from src.data.preprocess import FraudPreprocessor
from src.graph.build_graph import build_transaction_graph
from src.graph.hybrid_graph import build_hybrid_transaction_graph
from src.graph.soft_behavior_graph import build_soft_behavior_edges
from src.graph.graph_utils import normalize_time_to_hours, make_edge_tensors
from src.models.fraudgnn_rl import FraudGNNRL
from src.train.federated import train_federated
from src.train.train_rl import train_threshold_dqn, choose_best_threshold_by_validation
from src.eval.evaluate import predict_scores, save_metrics
from src.eval.metrics import classification_metrics
from src.utils.seed import set_seed
from src.utils.config import ensure_dirs


def resolve_flags(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve flags from config, with defaults."""
    flags = cfg.get("flags", {})
    
    defaults = {
        "hard_edges": True,
        "soft_edges": False,
        "hybrid_graph": False,
        "weighted_fusion": False,
        "federated": True,
        "dqn": True,
        "pruning": False,
    }
    
    resolved = {**defaults, **flags}
    
    pipeline = cfg.get("experiment", {}).get("pipeline", "fraudgnn_rl")
    model_name = "FraudGNN-RL" if pipeline == "fraudgnn_rl" else "FraudGNN-RL+"
    
    print("="*60)
    print(f"[MODEL] {model_name}")
    print("="*60)
    print("[ABLATION FLAGS]")
    print("-"*60)
    for key, value in resolved.items():
        if key != "pruning_params":
            print(f"  {key}: {value}")
    if resolved.get("pruning", False):
        params = resolved.get("pruning_params", {})
        print(f"  pruning_initial: {params.get('initial_sparsity', 0.1)}")
        print(f"  pruning_final: {params.get('final_sparsity', 0.3)}")
    print("="*60)
    
    return resolved


def build_graph_from_flags(x, y, t, cfg, flags):
    """Build graph based on flags."""
    
    use_soft_edges = flags.get("soft_edges", False)
    use_hard_edges = flags.get("hard_edges", True)
    use_hybrid_graph = flags.get("hybrid_graph", False)
    use_weighted_fusion = flags.get("weighted_fusion", False)
    
    # ============================================================
    # CASE 1: SOFT ONLY (không hard) — dùng cho ablation
    # ============================================================
    if use_soft_edges and not use_hard_edges:
        ds = cfg.get("dataset", {})
        times_hours = normalize_time_to_hours(t, ds.get("time_unit"))
        
        soft_edges, soft_delta, soft_weight = build_soft_behavior_edges(
            x=x,
            times_hours=times_hours,
            cfg=cfg,
        )
        
        edge_index = make_edge_tensors(soft_edges, num_nodes=x.shape[0], self_loops=True)
        edge_time_delta = np.concatenate([soft_delta, np.zeros(x.shape[0], dtype=np.float32)])
        
        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.long),
            edge_index=edge_index,
            edge_time_delta=torch.tensor(edge_time_delta, dtype=torch.float32),
            node_type=torch.zeros(x.shape[0], dtype=torch.long),
            edge_weight=torch.ones(edge_index.size(1), dtype=torch.float32),
        )
        
        if times_hours is not None:
            data.node_time = torch.tensor(times_hours, dtype=torch.float32)
        
        print("  [SOFT ONLY] no hard edges")
        return data
    
    # ============================================================
    # CASE 2: HYBRID (hard + soft) — FraudGNN-RL+
    # ============================================================
    if use_soft_edges and use_hard_edges and use_hybrid_graph:
        cfg_clone = copy.deepcopy(cfg)
        
        if "hybrid_graph" not in cfg_clone:
            cfg_clone["hybrid_graph"] = {}
        
        cfg_clone["hybrid_graph"]["enabled"] = True
        cfg_clone["hybrid_graph"]["merge_prefer"] = "min_delta"
        
        if use_weighted_fusion:
            cfg_clone["hybrid_graph"]["base_edge_weight"] = 1.0
            cfg_clone["hybrid_graph"]["soft_edge_weight"] = 0.4
            cfg_clone["hybrid_graph"]["overlap_edge_weight"] = 1.2
            cfg_clone["hybrid_graph"]["self_loop_edge_weight"] = 1.0
            print("  [HYBRID] WEIGHTED fusion")
        else:
            for key in ["base_edge_weight", "soft_edge_weight", "overlap_edge_weight", "self_loop_edge_weight"]:
                cfg_clone["hybrid_graph"].pop(key, None)
            print("  [HYBRID] UNWEIGHTED fusion")
        
        return build_hybrid_transaction_graph(x, y, t, cfg_clone)
    
    # ============================================================
    # CASE 3: BASELINE (hard edges only) — FraudGNN-RL
    # ============================================================
    print("  [BASELINE] hard edges only")
    return build_transaction_graph(x, y, t, cfg)


def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Main pipeline for FraudGNN-RL / FraudGNN-RL+."""
    
    seed = int(cfg.get("dataset", {}).get("random_state", 42))
    set_seed(seed)
    ensure_dirs("data/processed", "data/graphs", "outputs/checkpoints", "outputs/results")
    
    flags = resolve_flags(cfg)
    use_federated = flags.get("federated", True)
    use_dqn = flags.get("dqn", True)
    use_pruning = flags.get("pruning", False)
    
    # ============================================================
    # 1. LOAD & PREPROCESS
    # ============================================================
    timing = {
        "data_loading_sec": 0.0,
        "data_splitting_sec": 0.0,
        "preprocessing_sec": 0.0,
        "graph_building_sec": 0.0,
        "federated_training_sec": 0.0,
        "dqn_training_sec": 0.0,
        "inference_sec": 0.0,
        "total_runtime_sec": 0.0,
        "runtime_per_sample_sec": 0.0,
        "throughput_samples_per_sec": 0.0,
        # ✅ Thêm chi tiết
        "federated_avg_round_time_sec": 0.0,
        "tssgc_avg_epoch_time_sec": 0.0,
    }
    
    total_start = time.perf_counter()
    print(f"[TIMING] Pipeline started at: {total_start}")
    
    # Data loading
    start = time.perf_counter()
    print(f"[TIMING] Loading data...")
    df = load_dataset(cfg)
    timing["data_loading_sec"] = time.perf_counter() - start
    print(f"[TIMING] Data loaded in {timing['data_loading_sec']:.2f}s")
    
    # Split
    start = time.perf_counter()
    print(f"[TIMING] Splitting data...")
    train_df, val_df, test_df = split_dataframe(df, cfg)
    timing["data_splitting_sec"] = time.perf_counter() - start
    print(f"[TIMING] Data split in {timing['data_splitting_sec']:.2f}s")
    
    # Preprocess
    start = time.perf_counter()
    print(f"[TIMING] Preprocessing...")
    pre = FraudPreprocessor(cfg)
    x_train, y_train, t_train = pre.fit_transform(train_df)
    x_val, y_val, t_val = pre.transform(val_df)
    x_test, y_test, t_test = pre.transform(test_df)
    timing["preprocessing_sec"] = time.perf_counter() - start
    print(f"[TIMING] Preprocess done in {timing['preprocessing_sec']:.2f}s")
    
    # Build graph
    start = time.perf_counter()
    print(f"[TIMING] Building graphs...")
    train_graph = build_graph_from_flags(x_train, y_train, t_train, cfg, flags)
    val_graph = build_graph_from_flags(x_val, y_val, t_val, cfg, flags)
    test_graph = build_graph_from_flags(x_test, y_test, t_test, cfg, flags)
    timing["graph_building_sec"] = time.perf_counter() - start
    print(f"[TIMING] Graphs built in {timing['graph_building_sec']:.2f}s")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TIMING] Using device: {device}")
    
    # ============================================================
    # 2. FEDERATED LEARNING (hoặc local training)
    # ============================================================
    if use_federated:
        start = time.perf_counter()
        print(f"[TIMING] Starting Federated Learning...")
        
        fed_result = train_federated(
            train_data=train_graph,
            val_data=val_graph,
            test_data=test_graph,
            cfg=cfg,
            model_class=FraudGNNRL,
            device=device,
            use_pruning=use_pruning,
        )
        timing["federated_training_sec"] = time.perf_counter() - start
        print(f"[TIMING] Federated Learning done in {timing['federated_training_sec']:.2f}s")
        
        # ✅ Lấy timing chi tiết từ federated
        timing["federated_avg_round_time_sec"] = fed_result.get("avg_round_time_sec", 0)
        timing["federated_round_times"] = fed_result.get("round_times", [])
        
        global_model = fed_result["global_model"]
        val_scores = fed_result["val_scores"]
        val_labels = fed_result["val_labels"]
        test_scores = fed_result["test_scores"]
        test_labels = fed_result["test_labels"]
        fed_history = fed_result["history"]
    else:
        from src.train.train_gnn import train_tssgc_classifier
        
        start = time.perf_counter()
        print(f"[TIMING] Starting Local Training...")
        
        model, history, ckpt_path, tssgc_timing = train_tssgc_classifier(
            train_graph, val_graph, cfg,
            output_dir="outputs/checkpoints/ablation",
            timing=timing,
        )
        timing["federated_training_sec"] = time.perf_counter() - start
        print(f"[TIMING] Local Training done in {timing['federated_training_sec']:.2f}s")
        
        # ✅ Lấy timing chi tiết từ TSSGC
        timing["tssgc_avg_epoch_time_sec"] = tssgc_timing.get("avg_epoch_time_sec", 0)
        timing["tssgc_epoch_times"] = tssgc_timing.get("epoch_times", [])
        timing["tssgc_total_training_sec"] = tssgc_timing.get("total_training_sec", 0)
        
        global_model = model
        fed_history = history
        val_scores, val_labels = predict_scores(global_model, val_graph, device=device)
        test_scores, test_labels = predict_scores(global_model, test_graph, device=device)
    
    # ============================================================
    # 3. DQN THRESHOLD (hoặc fixed threshold)
    # ============================================================
    thresholds = [float(x) for x in cfg.get("rl", {}).get("threshold_bins", [0.5])]
    
    if use_dqn:
        start = time.perf_counter()
        print(f"[TIMING] Training DQN...")
        agent, rl_history = train_threshold_dqn(val_scores, val_labels, cfg, device=device)
        timing["dqn_training_sec"] = time.perf_counter() - start
        print(f"[TIMING] DQN trained in {timing['dqn_training_sec']:.2f}s")
        
        best_threshold, val_threshold_metrics = choose_best_threshold_by_validation(
            val_scores, val_labels, thresholds, cfg=cfg
        )
    else:
        best_threshold = 0.5
        val_threshold_metrics = None
    
    # ============================================================
    # 4. EVALUATION
    # ============================================================
    print(f"[TIMING] Evaluating...")
    start = time.perf_counter()
    
    val_metrics = classification_metrics(val_labels, val_scores, threshold=best_threshold)
    test_metrics = classification_metrics(test_labels, test_scores, threshold=best_threshold)
    
    timing["inference_sec"] = time.perf_counter() - start
    print(f"[TIMING] Evaluation done in {timing['inference_sec']:.2f}s")
    
    # ============================================================
    # 5. TOTAL RUNTIME
    # ============================================================
    timing["total_runtime_sec"] = time.perf_counter() - total_start
    num_samples = len(df)
    timing["runtime_per_sample_sec"] = timing["total_runtime_sec"] / max(1, num_samples)
    timing["throughput_samples_per_sec"] = num_samples / max(1, timing["total_runtime_sec"])
    
    print(f"\n[TIMING] ===== SUMMARY =====")
    print(f"[TIMING] Total runtime: {timing['total_runtime_sec']:.2f}s ({timing['total_runtime_sec']/60:.2f}m)")
    print(f"[TIMING] Throughput: {timing['throughput_samples_per_sec']:.2f} samples/s")
    print(f"[TIMING] Runtime per sample: {timing['runtime_per_sample_sec']*1000:.2f}ms")
    
    # ============================================================
    # 6. RESULT
    # ============================================================
    pipeline = cfg.get("experiment", {}).get("pipeline", "fraudgnn_rl")
    model_name = "FraudGNN-RL" if pipeline == "fraudgnn_rl" else "FraudGNN-RL+"
    
    result = {
        "model": model_name,
        "pipeline": pipeline,
        "flags": flags,
        "selected_threshold": best_threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "runtime": timing,
        "federated_history": fed_history,
        "notes": f"{model_name} with ablation flags",
    }
    
    exp_name = cfg.get("experiment", {}).get("name", "experiment")
    pipeline_name = cfg.get("experiment", {}).get("pipeline", "fraudgnn_rl")
    result_filename = f"{exp_name}_{pipeline_name}_metrics.json"
    save_metrics(result, str(Path("outputs/results") / result_filename))
    
    print(f"\n✅ Results saved to: {result_filename}")
    
    return result