# ============================================================
# src/train/pipeline_fraudgnn.py
# Pipeline: Graph → TSSGC → FedAvg → RL (DQN/NAF)
# GIỐNG PAPER 100%: 
#   - Benchmark: RL chỉ train trên validation, test chỉ đánh giá
#   - Online Adaptation: thí nghiệm riêng minh họa Figure 2
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
from src.train.train_rl import choose_best_threshold_by_validation
from src.eval.evaluate import predict_scores, save_metrics
from src.eval.metrics import classification_metrics
from src.utils.seed import set_seed
from src.utils.config import ensure_dirs
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


def get_or_build_graph(x, y, t, cfg, flags, name="train"):
    """Lấy graph từ cache nếu có, nếu không thì xây dựng và lưu lại."""
    graph_dir = Path("data/graphs/cache")
    graph_dir.mkdir(parents=True, exist_ok=True)
    
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


def simulate_online_adaptation(
    agent,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    test_embeddings: np.ndarray,
    cfg: Dict[str, Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    ✅ ONLINE ADAPTATION SIMULATION (FIGURE 2)
    
    Đây là thí nghiệm RIÊNG để minh họa tính "adaptive" của hệ thống.
    KHÔNG dùng kết quả này cho benchmark (Table 2).
    """
    from src.models.dqn_agent import BatchThresholdEnvironment
    
    print(f"\n{'='*60}")
    print(f"[ONLINE] Simulating online adaptation (Figure 2)...")
    print(f"{'='*60}")
    
    chunk_size = cfg.get("rl", {}).get("adaptive_chunk_size", 1000)
    
    online_env = BatchThresholdEnvironment(
        test_scores, test_labels,
        graph_embeddings=test_embeddings,
        batch_size=chunk_size,
        fpr_penalty=cfg.get("rl", {}).get("fpr_penalty", 2.0),
    )
    
    state = online_env.reset()
    done = False
    step_count = 0
    
    adaptive_thresholds = []
    adaptive_rewards = []
    adaptive_accuracies = []
    adaptive_fprs = []
    
    while not done:
        # ✅ FIX: explore=False khi đánh giá
        action = agent.act(state, explore=False)
        threshold = agent.threshold(action)
        
        next_state, reward, done, info = online_env.step(threshold)
        
        agent.memory.push(state, action, reward, next_state, done)
        agent.update(batch_size=min(64, len(agent.memory)))
        
        adaptive_thresholds.append(threshold)
        adaptive_rewards.append(reward)
        adaptive_accuracies.append(info.get('accuracy', 0))
        adaptive_fprs.append(info.get('fpr', 0))
        
        state = next_state
        step_count += 1
        
        if step_count % 10 == 0:
            avg_reward = np.mean(adaptive_rewards[-10:]) if adaptive_rewards else 0
            print(f"  Step {step_count}: threshold={threshold:.3f}, "
                  f"avg_reward={avg_reward:.4f}, "
                  f"accuracy={info.get('accuracy', 0):.4f}")
    
    final_threshold = adaptive_thresholds[-1] if adaptive_thresholds else 0.5
    mean_threshold = np.mean(adaptive_thresholds) if adaptive_thresholds else 0.5
    
    online_metrics = classification_metrics(
        test_labels, test_scores, threshold=final_threshold
    )
    
    print(f"\n[ONLINE] Online adaptation results (Figure 2 simulation):")
    print(f"  Steps: {step_count}")
    print(f"  Final threshold: {final_threshold:.4f}")
    print(f"  Mean threshold: {mean_threshold:.4f}")
    print(f"  F1: {online_metrics.get('f1', 0):.4f}")
    print(f"  AUC-ROC: {online_metrics.get('auc_roc', 0):.4f}")
    print(f"  Accuracy: {online_metrics.get('accuracy', 0):.4f}")
    print(f"{'='*60}")
    
    return {
        "thresholds": adaptive_thresholds,
        "rewards": adaptive_rewards,
        "accuracies": adaptive_accuracies,
        "fprs": adaptive_fprs,
        "steps": step_count,
        "final_threshold": final_threshold,
        "mean_threshold": mean_threshold,
        "metrics": online_metrics,
        "note": "Online adaptation simulation (Figure 2) - NOT used for benchmark comparison",
    }


def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Main pipeline - GIỐNG PAPER 100%."""
    
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
    
    # Build graph
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
    # 3. ✅ RL THRESHOLD - CHỈ TRÊN VALIDATION (GIỐNG PAPER)
    # ============================================================
    thresholds = [float(x) for x in cfg.get("rl", {}).get("threshold_bins", [0.5])]
    rl_type = cfg.get("rl", {}).get("type", "dqn")
    agent = None
    val_embeddings = None
    test_embeddings = None
    rl_epochs = 0
    
    if use_rl:
        global_model.eval()
        with torch.no_grad():
            embed_dim = global_model.encoder.layers[0].temporal.lin_msg.out_features
            print(f"[RL] Embedding dimension: {embed_dim}")
            
            val_embeddings = global_model.encoder(
                val_graph.x.to(device),
                val_graph.edge_index.to(device),
                getattr(val_graph, "edge_time_delta", None),
                getattr(val_graph, "node_type", None),
                getattr(val_graph, "edge_weight", None),
            ).cpu().numpy()
            
            test_embeddings = global_model.encoder(
                test_graph.x.to(device),
                test_graph.edge_index.to(device),
                getattr(test_graph, "edge_time_delta", None),
                getattr(test_graph, "node_type", None),
                getattr(test_graph, "edge_weight", None),
            ).cpu().numpy()
        
        if rl_type == "naf":
            print(f"[NAF] Using Normalized Advantage Functions (continuous action)")
            from src.models.naf_agent import train_naf_agent, BatchNAFEnvironment
            
            start = time.perf_counter()
            print(f"[TIMING] Training NAF on validation set...")
            
            agent, rl_history = train_naf_agent(
                val_scores, val_labels, cfg, 
                graph_embeddings=val_embeddings,
                device=device,
                n_features=embed_dim
            )
            timing["rl_training_sec"] = time.perf_counter() - start
            print(f"[TIMING] NAF trained in {timing['rl_training_sec']:.2f}s")
            
            best_threshold, val_threshold_metrics = choose_best_threshold_by_validation(
                val_scores, val_labels, thresholds, cfg=cfg
            )
            val_threshold_metrics["threshold_selection_method"] = "naf_validation"
            
            print(f"[NAF] Best threshold from validation: {best_threshold:.4f}")
            print(f"[NAF] Val F1: {val_threshold_metrics.get('f1', 0):.4f}")
            
        else:
            print(f"[DQN] Using DQN (discrete action)")
            from src.models.dqn_agent import ThresholdDQNAgent, BatchThresholdEnvironment
            
            start = time.perf_counter()
            print(f"[TIMING] Training DQN on validation set...")
            
            env = BatchThresholdEnvironment(
                val_scores, val_labels,
                graph_embeddings=val_embeddings,
                batch_size=cfg.get("rl", {}).get("batch_size", 256),
                fpr_penalty=cfg.get("rl", {}).get("fpr_penalty", 2.0),
            )
            
            # ✅ FIX: Giảm learning rate, thêm gradient clipping
            agent = ThresholdDQNAgent(
                state_dim=env.state_dim,
                thresholds=thresholds,
                n_features=embed_dim,
                device=device,
                lr=0.0001,           # ✅ Giảm từ 0.001 xuống 0.0001
                grad_clip=0.5,       # ✅ Gradient clipping
                min_buffer_size=64,  # ✅ Buffer tối thiểu
            )
            
            rl_epochs = cfg.get("rl", {}).get("epochs", 100)
            for ep in range(rl_epochs):
                state = env.reset()
                done = False
                ep_loss = []
                while not done:
                    action = agent.act(state, explore=True)
                    threshold = agent.threshold(action)
                    next_state, reward, done, info = env.step(threshold)
                    agent.memory.push(state, action, reward, next_state, done)
                    loss = agent.update(batch_size=min(64, len(agent.memory)))
                    if loss is not None:
                        ep_loss.append(loss)
                    state = next_state
                agent.sync_target()
                
                if (ep + 1) % 10 == 0:
                    avg_loss = np.mean(ep_loss) if ep_loss else 0
                    print(f"  [DQN] Epoch {ep+1}/{rl_epochs}, avg_loss={avg_loss:.4f}")
            
            timing["rl_training_sec"] = time.perf_counter() - start
            print(f"[TIMING] DQN trained in {timing['rl_training_sec']:.2f}s")
            
            best_threshold, val_threshold_metrics = choose_best_threshold_by_validation(
                val_scores, val_labels, thresholds, cfg=cfg
            )
            val_threshold_metrics["threshold_selection_method"] = "dqn_validation"
            
            print(f"[DQN] Best threshold from validation: {best_threshold:.4f}")
            print(f"[DQN] Val F1: {val_threshold_metrics.get('f1', 0):.4f}")
    
    else:
        best_threshold, val_threshold_metrics = choose_best_threshold_by_validation(
            val_scores, val_labels, thresholds, cfg=cfg
        )
        val_threshold_metrics["threshold_selection_method"] = "static_only"
        print(f"[NO RL] Best threshold from validation: {best_threshold:.4f}")
    
    # ============================================================
    # 4. ✅ BENCHMARK - ĐÁNH GIÁ TRÊN TEST (CHỈ DÙNG THRESHOLD CỐ ĐỊNH)
    # ============================================================
    print(f"\n[TIMING] Evaluating on test set with fixed threshold...")
    start = time.perf_counter()
    
    val_metrics = classification_metrics(val_labels, val_scores, threshold=best_threshold)
    test_metrics = classification_metrics(test_labels, test_scores, threshold=best_threshold)
    
    baseline_metrics = classification_metrics(test_labels, test_scores, threshold=0.5)
    
    timing["inference_sec"] = time.perf_counter() - start
    print(f"[TIMING] Evaluation done in {timing['inference_sec']:.2f}s")
    
    # ============================================================
    # 5. ✅ ONLINE ADAPTATION SIMULATION (FIGURE 2)
    # ============================================================
    online_result = None
    if use_rl and agent is not None and test_embeddings is not None:
        online_result = simulate_online_adaptation(
            agent,
            test_scores,
            test_labels,
            test_embeddings,
            cfg,
            device=device,
        )
    
    # ============================================================
    # 6. ✅ PRINT COMPARISON (GIỐNG PAPER TABLE 2)
    # ============================================================
    print(f"\n{'='*80}")
    print(f"[BENCHMARK] FraudGNN-RL vs Baseline (giống paper Table 2)")
    print(f"{'='*80}")
    print(f"{'Metric':<15} {'Baseline':<20} {'FraudGNN-RL':<20} {'Delta':<15}")
    print(f"{'-'*80}")
    
    for metric in ['auc_roc', 'auc_pr', 'f1', 'recall', 'precision', 'fpr']:
        base_val = baseline_metrics.get(metric, 0)
        ours_val = test_metrics.get(metric, 0)
        delta = ours_val - base_val
        arrow = '↑' if delta > 0 else '↓'
        if metric == 'fpr':
            arrow = '↓' if delta < 0 else '↑'
            delta = -delta
        print(f"{metric:<15} {base_val:<20.4f} {ours_val:<20.4f} {arrow}{abs(delta):<14.4f}")
    
    print(f"{'='*80}")
    print(f"Selected threshold: {best_threshold:.4f} (from validation set)")
    
    # ============================================================
    # 7. LATENCY & MEMORY
    # ============================================================
    print(f"\n[TIMING] Measuring latency and memory...")
    latency_metrics = {}
    memory_metrics = {}
    
    try:
        from torch_geometric.loader import NeighborLoader
        test_loader = NeighborLoader(
            test_graph,
            num_neighbors=[15, 10],
            batch_size=64,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        test_batch = next(iter(test_loader))
        
        latency_metrics = measure_latency(global_model, test_batch, device=device, num_runs=20)
        memory_metrics = get_memory_usage()
        
        print(f"[TIMING] Latency: {latency_metrics.get('latency_mean_ms', 0):.2f}ms")
        print(f"[TIMING] Throughput: {latency_metrics.get('throughput_per_sec', 0):.0f} samples/s")
        print(f"[TIMING] RAM: {memory_metrics.get('ram_used_gb', 0):.2f}GB")
    except Exception as e:
        print(f"⚠️ Latency/memory measurement failed: {e}")
    
    # ============================================================
    # 8. TOTAL RUNTIME
    # ============================================================
    timing["total_runtime_sec"] = time.perf_counter() - total_start
    num_samples = len(df)
    timing["runtime_per_sample_sec"] = timing["total_runtime_sec"] / max(1, num_samples)
    timing["throughput_samples_per_sec"] = num_samples / max(1, timing["total_runtime_sec"])
    
    print(f"\n[TIMING] ===== SUMMARY =====")
    print(f"[TIMING] Total runtime: {timing['total_runtime_sec']:.2f}s")
    print(f"[TIMING] Throughput: {timing['throughput_samples_per_sec']:.2f} samples/s")
    print_timing_summary(timing)
    
    # ============================================================
    # 9. RESULT
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
        "baseline_metrics": baseline_metrics,
        "runtime": timing,
        "num_samples": num_samples,
        "latency": latency_metrics,
        "memory": memory_metrics,
        "federated_history": fed_history,
        "threshold_selection": "validation_set",
        "notes": "Benchmark: RL only on validation, test uses fixed threshold (giống paper Table 2)",
        "online_adaptation": online_result,
    }
    
    if use_rl:
        result["rl_info"] = {
            "type": rl_type,
            "epochs": rl_epochs if use_rl else 0,
            "threshold_from": "validation",
        }
    
    exp_name = cfg.get("experiment", {}).get("name", "experiment")
    pipeline_name = cfg.get("experiment", {}).get("pipeline", "fraudgnn_rl")
    result_filename = f"{exp_name}_{pipeline_name}_metrics.json"
    save_metrics(result, str(Path("outputs/results") / result_filename))
    
    print(f"\n✅ Results saved to: {result_filename}")
    
    return result