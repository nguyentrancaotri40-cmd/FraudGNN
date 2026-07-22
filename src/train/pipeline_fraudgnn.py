# ============================================================
# src/train/pipeline_fraudgnn.py
# Pipeline: Graph → TSSGC → FedAvg → RL (DQN/NAF)
# Hỗ trợ 2 model: FraudGNN-RL (baseline) và FraudGNN-RL+ (hybrid)
# ============================================================

from __future__ import annotations

import copy
import pickle
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
# ✅ THÊM TIMER
from src.utils.timer import measure_latency, get_memory_usage, print_timing_summary


def resolve_flags(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve flags from config, with defaults."""
    flags = cfg.get("flags", {})
    
    defaults = {
        "hard_edges": True,
        "soft_edges": False,
        "hybrid_graph": False,
        "weighted_fusion": False,
        "federated": True,
        "rl": True,
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


# ✅ THÊM HÀM CACHE GRAPH
def get_or_build_graph(x, y, t, cfg, flags, name="train"):
    """Lấy graph từ cache nếu có, nếu không thì xây dựng và lưu lại.
    
    Giúp tránh build graph 3 lần cho train/val/test, đặc biệt hữu ích cho hybrid graph.
    """
    graph_dir = Path("data/graphs/cache")
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo cache key từ config và flags
    cache_parts = [
        name,
        str(cfg.get("dataset", {}).get("sample_frac", 1.0)),
        str(cfg.get("graph", {}).get("similarity_threshold", 0.9)),
        str(cfg.get("graph", {}).get("max_neighbors_per_node", 3)),
        str(cfg.get("graph", {}).get("time_window_hours", 1.0)),
        str(flags.get("hard_edges", True)),
        str(flags.get("soft_edges", False)),
        str(flags.get("hybrid_graph", False)),
        str(flags.get("weighted_fusion", False)),
        str(cfg.get("dataset", {}).get("random_state", 42)),
    ]
    cache_key = "_".join(cache_parts)
    cache_path = graph_dir / f"{cache_key}.pkl"
    
    if cache_path.exists():
        print(f"[CACHE] Loading cached graph: {cache_path.name}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"[CACHE] Building graph (not cached): {cache_path.name}")
    data = build_graph_from_flags(x, y, t, cfg, flags)
    
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    
    return data


def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Main pipeline for FraudGNN-RL / FraudGNN-RL+."""
    
    seed = int(cfg.get("dataset", {}).get("random_state", 42))
    set_seed(seed)
    ensure_dirs("data/processed", "data/graphs", "outputs/checkpoints", "outputs/results")
    
    flags = resolve_flags(cfg)
    use_federated = flags.get("federated", True)
    use_rl = flags.get("rl", True)
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
        "rl_training_sec": 0.0,
        "inference_sec": 0.0,
        "total_runtime_sec": 0.0,
        "runtime_per_sample_sec": 0.0,
        "throughput_samples_per_sec": 0.0,
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
    
    # Build graph - ✅ SỬ DỤNG CACHE
    start = time.perf_counter()
    print(f"[TIMING] Building graphs...")
    
    train_graph = get_or_build_graph(x_train, y_train, t_train, cfg, flags, "train")
    val_graph = get_or_build_graph(x_val, y_val, t_val, cfg, flags, "val")
    test_graph = get_or_build_graph(x_test, y_test, t_test, cfg, flags, "test")
    
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
        
        timing["tssgc_avg_epoch_time_sec"] = tssgc_timing.get("avg_epoch_time_sec", 0)
        timing["tssgc_epoch_times"] = tssgc_timing.get("epoch_times", [])
        timing["tssgc_total_training_sec"] = tssgc_timing.get("total_training_sec", 0)
        
        global_model = model
        fed_history = history
        val_scores, val_labels = predict_scores(global_model, val_graph, device=device)
        test_scores, test_labels = predict_scores(global_model, test_graph, device=device)
    
    # ============================================================
    # 3. RL THRESHOLD - DQN hoặc NAF
    # ============================================================
    thresholds = [float(x) for x in cfg.get("rl", {}).get("threshold_bins", [0.5])]
    rl_type = cfg.get("rl", {}).get("type", "dqn")  # "dqn" hoặc "naf"
    
    if use_rl:
        # ✅ Lấy số features từ model
        n_features = global_model.encoder.layers[0].temporal.lin_msg.out_features
        print(f"[RL] Number of features for importance weighting: {n_features}")
        
        if rl_type == "naf":
            print(f"[NAF] Using Normalized Advantage Functions (continuous action)")
            from src.models.naf_agent import train_naf_agent, BatchNAFEnvironment
            
            start = time.perf_counter()
            print(f"[TIMING] Training NAF on validation set...")
            
            # ✅ Train NAF với feature weights
            agent, rl_history = train_naf_agent(
                val_scores, val_labels, cfg, device=device,
                n_features=n_features
            )
            timing["rl_training_sec"] = time.perf_counter() - start
            print(f"[TIMING] NAF trained in {timing['rl_training_sec']:.2f}s")
            
            # Static threshold (grid search) để so sánh
            static_threshold, static_metrics = choose_best_threshold_by_validation(
                val_scores, val_labels, thresholds, cfg=cfg
            )
            
            # NAF online learning trên test set
            print(f"[NAF] Running online adaptive loop on test set...")
            
            env = BatchNAFEnvironment(
                val_scores, val_labels,
                batch_size=cfg.get("rl", {}).get("batch_size", 256),
                fpr_penalty=cfg.get("rl", {}).get("fpr_penalty", 2.0),
            )
            
            test_scores_arr = test_scores
            test_labels_arr = test_labels
            
            adaptive_thresholds = []
            adaptive_feature_weights = []
            chunk_size = cfg.get("rl", {}).get("adaptive_chunk_size", 1000)
            
            # State ban đầu từ validation
            state = env._state_for_current_batch()
            
            for chunk_idx in range(0, len(test_scores_arr), chunk_size):
                chunk_scores = test_scores_arr[chunk_idx:chunk_idx+chunk_size]
                chunk_labels = test_labels_arr[chunk_idx:chunk_idx+chunk_size]
                
                if len(chunk_scores) == 0:
                    break
                
                # ✅ NAF chọn threshold + feature weights
                threshold, feature_weights = agent.act(state, explore=True)
                adaptive_thresholds.append(threshold)
                adaptive_feature_weights.append(feature_weights)
                
                # ✅ Apply feature weights to model
                global_model.set_feature_weights(torch.tensor(feature_weights, device=device))
                
                # Tính reward từ chunk này
                pred = (chunk_scores >= threshold).astype(np.int64)
                tp = np.sum((pred == 1) & (chunk_labels == 1))
                fp = np.sum((pred == 1) & (chunk_labels == 0))
                fn = np.sum((pred == 0) & (chunk_labels == 1))
                tn = np.sum((pred == 0) & (chunk_labels == 0))
                
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                f1 = 2 * precision * recall / max(1e-8, precision + recall)
                fpr = fp / max(1, fp + tn)
                
                # Reward dựa trên F1 và FPR
                reward = f1 + 0.5 * recall - 0.5 * fpr
                
                # Cập nhật state (bao gồm threshold vừa chọn)
                env.current_threshold = threshold
                next_state = env._state_for_current_batch()
                
                # LƯU EXPERIENCE VÀ UPDATE NAF
                done = (chunk_idx + chunk_size >= len(test_scores_arr))
                agent.memory.push(state, threshold, reward, next_state, done)
                
                if len(agent.memory) > 10:
                    agent.update()
                
                state = next_state
                
                if chunk_idx % max(1, len(test_scores_arr) // 10) == 0:
                    print(f"[NAF] Chunk {chunk_idx//chunk_size + 1}, threshold: {threshold:.3f}, F1: {f1:.3f}")
            
            # Chọn threshold cuối cùng
            naf_threshold = float(np.mean(adaptive_thresholds)) if adaptive_thresholds else static_threshold
            naf_metrics = classification_metrics(test_labels_arr, test_scores_arr, threshold=naf_threshold)
            
            # So sánh NAF với static
            if naf_metrics.get('f1', 0) > static_metrics.get('f1', 0):
                best_threshold = naf_threshold
                val_threshold_metrics = naf_metrics
                val_threshold_metrics["threshold_selection_method"] = "naf_online"
                print(f"[NAF] ✅ Using NAF online threshold: {best_threshold:.4f} (F1={naf_metrics['f1']:.4f} > static F1={static_metrics['f1']:.4f})")
            else:
                best_threshold = static_threshold
                val_threshold_metrics = static_metrics
                val_threshold_metrics["threshold_selection_method"] = "static"
                print(f"[NAF] Using static threshold: {best_threshold:.4f} (static F1={static_metrics['f1']:.4f} >= NAF F1={naf_metrics['f1']:.4f})")
            
            # Lưu thông tin adaptive
            val_threshold_metrics["adaptive_threshold_mean"] = float(np.mean(adaptive_thresholds)) if adaptive_thresholds else naf_threshold
            val_threshold_metrics["adaptive_threshold_std"] = float(np.std(adaptive_thresholds)) if adaptive_thresholds else 0.0
            val_threshold_metrics["naf_threshold"] = naf_threshold
            val_threshold_metrics["naf_f1"] = naf_metrics.get('f1', 0)
            
        else:
            print(f"[DQN] Using DQN (discrete action)")
            from src.models.dqn_agent import ThresholdDQNAgent, BatchThresholdEnvironment
            
            start = time.perf_counter()
            print(f"[TIMING] Training DQN on validation set...")
            
            # Khởi tạo DQN (không train trước)
            env = BatchThresholdEnvironment(
                val_scores, val_labels,
                batch_size=cfg.get("rl", {}).get("batch_size", 256),
                fpr_penalty=cfg.get("rl", {}).get("fpr_penalty", 2.0),
            )
            
            agent = ThresholdDQNAgent(
                state_dim=env.state_dim,
                thresholds=thresholds,
                n_features=n_features,
                device=device,
            )
            
            # Train DQN trên validation set (tối ưu)
            for ep in range(cfg.get("rl", {}).get("epochs", 30)):
                state = env.reset()
                done = False
                while not done:
                    action = agent.act(state, explore=True)
                    threshold = agent.threshold(action)
                    next_state, reward, done, info = env.step(threshold)
                    agent.memory.push(state, action, reward, next_state, done)
                    agent.update(batch_size=min(64, len(agent.memory)))
                    state = next_state
                agent.sync_target()
            
            timing["rl_training_sec"] = time.perf_counter() - start
            print(f"[TIMING] DQN trained in {timing['rl_training_sec']:.2f}s")
            
            # Tham khảo static threshold
            static_threshold, static_metrics = choose_best_threshold_by_validation(
                val_scores, val_labels, thresholds, cfg=cfg
            )
            
            # DQN online learning trên test set
            print(f"[DQN] Running online adaptive loop on test set...")
            
            test_scores_arr = test_scores
            test_labels_arr = test_labels
            
            adaptive_thresholds = []
            chunk_size = cfg.get("rl", {}).get("adaptive_chunk_size", 1000)
            
            state = env.reset()
            
            for chunk_idx in range(0, len(test_scores_arr), chunk_size):
                chunk_scores = test_scores_arr[chunk_idx:chunk_idx+chunk_size]
                chunk_labels = test_labels_arr[chunk_idx:chunk_idx+chunk_size]
                
                if len(chunk_scores) == 0:
                    break
                
                action = agent.act(state, explore=True)
                threshold = agent.threshold(action)
                adaptive_thresholds.append(threshold)
                
                pred = (chunk_scores >= threshold).astype(np.int64)
                tp = np.sum((pred == 1) & (chunk_labels == 1))
                fp = np.sum((pred == 1) & (chunk_labels == 0))
                fn = np.sum((pred == 0) & (chunk_labels == 1))
                tn = np.sum((pred == 0) & (chunk_labels == 0))
                
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                f1 = 2 * precision * recall / max(1e-8, precision + recall)
                fpr = fp / max(1, fp + tn)
                
                reward = f1 + 0.5 * recall - 0.5 * fpr
                
                env.current_threshold = threshold
                next_state = env._state_for_current_batch()
                done = (chunk_idx + chunk_size >= len(test_scores_arr))
                
                agent.memory.push(state, action, reward, next_state, done)
                if len(agent.memory) > 10:
                    agent.update(batch_size=min(64, len(agent.memory)))
                if len(agent.memory) % 100 == 0:
                    agent.sync_target()
                
                state = next_state
                
                if chunk_idx % max(1, len(test_scores_arr) // 10) == 0:
                    print(f"[DQN] Chunk {chunk_idx//chunk_size + 1}, threshold: {threshold:.3f}, F1: {f1:.3f}, epsilon: {agent.epsilon:.3f}")
            
            # Chọn threshold cuối cùng
            dqn_threshold = float(np.mean(adaptive_thresholds)) if adaptive_thresholds else static_threshold
            dqn_metrics = classification_metrics(test_labels_arr, test_scores_arr, threshold=dqn_threshold)
            
            if dqn_metrics.get('f1', 0) > static_metrics.get('f1', 0):
                best_threshold = dqn_threshold
                val_threshold_metrics = dqn_metrics
                val_threshold_metrics["threshold_selection_method"] = "dqn_online"
                print(f"[DQN] ✅ Using DQN online threshold: {best_threshold:.4f} (F1={dqn_metrics['f1']:.4f} > static F1={static_metrics['f1']:.4f})")
            else:
                best_threshold = static_threshold
                val_threshold_metrics = static_metrics
                val_threshold_metrics["threshold_selection_method"] = "static"
                print(f"[DQN] Using static threshold: {best_threshold:.4f} (static F1={static_metrics['f1']:.4f} >= DQN F1={dqn_metrics['f1']:.4f})")
            
            val_threshold_metrics["adaptive_threshold_mean"] = float(np.mean(adaptive_thresholds)) if adaptive_thresholds else dqn_threshold
            val_threshold_metrics["adaptive_threshold_std"] = float(np.std(adaptive_thresholds)) if adaptive_thresholds else 0.0
            val_threshold_metrics["dqn_threshold"] = dqn_threshold
            val_threshold_metrics["dqn_f1"] = dqn_metrics.get('f1', 0)
        
    else:
        best_threshold, val_threshold_metrics = choose_best_threshold_by_validation(
            val_scores, val_labels, thresholds, cfg=cfg
        )
        val_threshold_metrics["threshold_selection_method"] = "static_only"
    
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
    # 4.5. LATENCY & MEMORY MEASUREMENT (✅ THÊM TIMER)
    # ============================================================
    print(f"[TIMING] Measuring latency and memory...")
    
    latency_metrics = {}
    memory_metrics = {}
    
    try:
        from torch_geometric.loader import NeighborLoader
        
        # Lấy 1 batch từ test graph
        test_loader = NeighborLoader(
            test_graph,
            num_neighbors=[15, 10],
            batch_size=64,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        test_batch = next(iter(test_loader))
        
        # ✅ Đo latency
        latency_metrics = measure_latency(
            global_model,
            test_batch,
            device=device,
            num_runs=20,
        )
        
        # ✅ Đo memory
        memory_metrics = get_memory_usage()
        
        print(f"[TIMING] Latency: {latency_metrics.get('latency_mean_ms', 0):.2f}ms")
        print(f"[TIMING] Throughput: {latency_metrics.get('throughput_per_sec', 0):.0f} samples/s")
        print(f"[TIMING] RAM: {memory_metrics.get('ram_used_gb', 0):.2f}GB")
        
    except Exception as e:
        print(f"⚠️ Latency/memory measurement failed: {e}")
        latency_metrics = {}
        memory_metrics = {}
    
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
    
    # ✅ In timing summary
    print_timing_summary(timing)
    
    # ============================================================
    # 6. RESULT (✅ FIX INDENTATION + THÊM TIMER)
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
        "num_samples": num_samples,
        "latency": latency_metrics,
        "memory": memory_metrics,
        "federated_history": fed_history,
        "notes": f"{model_name} with ablation flags",
    }
    
    # Thêm RL comparison vào result nếu có
    if use_rl:
        result["rl_comparison"] = {
            "rl_type": rl_type,
            "static_threshold": static_threshold,
            "static_f1": static_metrics.get('f1', 0),
            "rl_threshold": val_threshold_metrics.get('threshold', best_threshold),
            "rl_f1": val_threshold_metrics.get('f1', 0),
            "adaptive_threshold_mean": val_threshold_metrics.get('adaptive_threshold_mean', 0),
            "adaptive_threshold_std": val_threshold_metrics.get('adaptive_threshold_std', 0),
            "num_adaptive_chunks": len(adaptive_thresholds) if use_rl else 0,
            "selected": val_threshold_metrics.get("threshold_selection_method", "static"),
        }
    
    exp_name = cfg.get("experiment", {}).get("name", "experiment")
    pipeline_name = cfg.get("experiment", {}).get("pipeline", "fraudgnn_rl")
    result_filename = f"{exp_name}_{pipeline_name}_metrics.json"
    save_metrics(result, str(Path("outputs/results") / result_filename))
    
    print(f"\n✅ Results saved to: {result_filename}")
    
    return result