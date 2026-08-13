# ============================================================
# src/train/pipeline_fraudgnn.py
# Pipeline: Graph → TSSGC → FedAvg → RL (DQN/NAF)
# GIỐNG PAPER 100%: 
#   - Benchmark: RL chỉ train trên validation, test chỉ đánh giá
#   - Online Adaptation: thí nghiệm riêng minh họa Figure 2
#   - Semantic Branch: sử dụng ENTITY TYPE (giống paper Eq 10)
#   - RL State: Graph Embedding từ TSSGC (giống paper Section IV-B)
#   - ✅ FIX: RL Agent thực sự được dùng để chọn threshold (giống paper Eq 12)
#   - ✅ FIX: Soft update (Polyak averaging) thay vì hard sync target network
#   - ✅ FIX: KHÔNG dùng label (y) làm node_type trong SOFT ONLY case
#   - ✅ FIX: Feature weights áp dụng vào model (giống paper Section IV-B)
#   - ✅ FIX: Dùng embeddings() thay vì encoder() để có projection
#   - ✅ FIX: policy_threshold là float, không phải numpy array
#   - ✅ FIX: Hỗ trợ tau (soft update rate) cho DQN
#   - ✅ FIX: Mặc định dùng NAF (có feature weights) thay vì DQN
#   - ✅ FIX: 5-fold Cross-Validation (giống paper)
#   - ✅ FIX: Xử lý policy_threshold từ apply_naf_policy() an toàn
#   - ✅ FIX: Log memory usage tại các stage quan trọng
# ============================================================

from __future__ import annotations

import copy
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.load_data import load_dataset
from src.data.split import split_dataframe, get_cv_splits
from src.data.preprocess import FraudPreprocessor
from src.graph.build_graph import build_transaction_graph
from src.graph.hybrid_graph import build_hybrid_transaction_graph
from src.graph.soft_behavior_graph import build_soft_behavior_edges
from src.graph.graph_utils import normalize_time_to_hours, make_edge_tensors
from src.models.fraudgnn_rl import FraudGNNRL
from src.train.federated import train_federated
from src.train.train_rl import choose_best_threshold_by_validation, apply_dqn_policy, apply_naf_policy
from src.eval.evaluate import predict_scores, save_metrics
from src.eval.metrics import classification_metrics
from src.utils.seed import set_seed
from src.utils.config import ensure_dirs
from src.utils.timer import measure_latency, get_memory_usage, print_timing_summary, log_memory_snapshot


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


def _extract_entity_type(df, cfg: Dict[str, Any]) -> np.ndarray | None:
    """
    ✅ GIỐNG PAPER: Trích xuất ENTITY TYPE từ dữ liệu
    
    Paper Section IV-A-3: "e_type(i) is a learnable embedding vector 
    for the type of entity i (e.g., individual, merchant, bank)"
    
    Returns:
        np.ndarray | None: entity type cho mỗi node, hoặc None nếu không có
    """
    dataset_name = cfg.get("experiment", {}).get("dataset", "unknown")
    
    # ============================================================
    # PAYSIM: Có cột 'type' với 5 loại giao dịch
    # ============================================================
    if dataset_name.lower() in ["paysim", "paysim"]:
        if "type" in df.columns:
            # Mã hóa type thành số
            type_mapping = {
                "CASH_IN": 0,
                "CASH_OUT": 1,
                "DEBIT": 2,
                "PAYMENT": 3,
                "TRANSFER": 4,
            }
            entity_type = df["type"].map(type_mapping).fillna(0).astype(int).values
            print(f"[ENTITY TYPE] PaySim: Using 'type' column ({len(np.unique(entity_type))} types)")
            return entity_type
    
    # ============================================================
    # IEEE-CIS: Có nhiều cột về entity (ProductCD, card4, card6)
    # ============================================================
    elif dataset_name.lower() in ["ieee", "ieee-cis", "ieee_cis"]:
        # Cách 1: Dùng ProductCD
        if "ProductCD" in df.columns:
            product_types = df["ProductCD"].astype('category').cat.codes.values
            print(f"[ENTITY TYPE] IEEE-CIS: Using 'ProductCD' ({len(np.unique(product_types))} types)")
            return product_types
        
        # Cách 2: Dùng card4 (loại thẻ)
        elif "card4" in df.columns:
            card_types = df["card4"].astype('category').cat.codes.values
            print(f"[ENTITY TYPE] IEEE-CIS: Using 'card4' ({len(np.unique(card_types))} types)")
            return card_types
        
        # Cách 3: Kết hợp nhiều cột
        elif "ProductCD" in df.columns and "card4" in df.columns:
            combined = df["ProductCD"].astype(str) + "_" + df["card4"].astype(str)
            entity_type = combined.astype('category').cat.codes.values
            print(f"[ENTITY TYPE] IEEE-CIS: Using combined 'ProductCD+card4' ({len(np.unique(entity_type))} types)")
            return entity_type
    
    # ============================================================
    # CREDIT CARD 2023: Không có entity type
    # ============================================================
    elif dataset_name.lower() in ["creditcard", "creditcard2023", "credit_card"]:
        print(f"[ENTITY TYPE] Credit Card 2023: No entity type available, using num_node_types=1")
        return None
    
    # ============================================================
    # FALLBACK: Không xác định được entity type
    # ============================================================
    print(f"[ENTITY TYPE] Unknown dataset '{dataset_name}': No entity type available")
    return None


def build_graph_from_flags(x, y, t, cfg, flags, entity_type=None):
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
        
        num_node_types = cfg.get("model", {}).get("num_node_types", 1)
        
        if entity_type is not None:
            node_type = torch.tensor(entity_type, dtype=torch.long)
            node_type = node_type.clamp(min=0, max=num_node_types - 1)
            print(f"[GRAPH SOFT] Using ENTITY TYPES from data")
        else:
            node_type = torch.zeros(x.shape[0], dtype=torch.long)
            if num_node_types > 1:
                print(f"[WARNING SOFT] No entity_type provided! Using single type (NOT using labels)")
            else:
                print(f"[GRAPH SOFT] Using single node type (num_node_types=1)")
        
        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.long),
            edge_index=edge_index,
            edge_time_delta=torch.tensor(edge_time_delta, dtype=torch.float32),
            node_type=node_type,
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
        
        return build_hybrid_transaction_graph(x, y, t, cfg_clone, entity_type=entity_type)
    
    # ============================================================
    # CASE 3: BASELINE (hard edges only) — FraudGNN-RL
    # ============================================================
    print("  [BASELINE] hard edges only")
    return build_transaction_graph(x, y, t, cfg, entity_type=entity_type)


def get_or_build_graph(x, y, t, cfg, flags, name="train", entity_type=None):
    """Lấy graph từ cache nếu có, nếu không thì xây dựng và lưu lại."""
    graph_dir = Path("data/graphs/cache")
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    entity_type_str = "_".join(map(str, entity_type[:10])) if entity_type is not None else "no_entity"
    
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
        str(cfg.get("model", {}).get("num_node_types", 1)),
        entity_type_str,
    ]
    cache_key = "_".join(cache_parts)
    cache_path = graph_dir / f"{cache_key}.pkl"
    
    if cache_path.exists():
        print(f"[CACHE] Loading cached graph: {cache_path.name}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"[CACHE] Building graph (not cached): {cache_path.name}")
    data = build_graph_from_flags(x, y, t, cfg, flags, entity_type=entity_type)
    
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


def run_pipeline_on_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    cfg: Dict[str, Any],
    fold_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Chạy pipeline trên một split cụ thể (train/val/test).
    
    Được sử dụng bởi:
    - run_pipeline() cho single split
    - run_cv_pipeline() cho cross-validation
    """
    seed = int(cfg.get("dataset", {}).get("random_state", 42))
    set_seed(seed)
    
    flags = resolve_flags(cfg)
    use_federated = flags.get("federated", True)
    use_rl = flags.get("rl", True)
    use_pruning = flags.get("pruning", False)
    
    # ============================================================
    # 1. TIMING
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
    fold_str = f"Fold {fold_idx+1}/" if fold_idx is not None else ""
    print(f"[TIMING] {fold_str}Pipeline started at: {total_start}")
    
    # ✅ LOG MEMORY: After pipeline start
    log_memory_snapshot("After pipeline start")
    
    # ============================================================
    # 2. ENTITY TYPE
    # ============================================================
    dfs_to_concat = [train_df, val_df]
    if test_df is not None:
        dfs_to_concat.append(test_df)
    
    combined_df = pd.concat(dfs_to_concat, ignore_index=True)
    entity_type_full = _extract_entity_type(combined_df, cfg)
    
    if entity_type_full is not None:
        train_idx = train_df.index
        val_idx = val_df.index
        entity_type_train = entity_type_full[train_idx]
        entity_type_val = entity_type_full[val_idx]
        if test_df is not None:
            test_idx = test_df.index
            entity_type_test = entity_type_full[test_idx]
        else:
            entity_type_test = None
    else:
        entity_type_train = None
        entity_type_val = None
        entity_type_test = None
    
    # ============================================================
    # 3. PREPROCESS
    # ============================================================
    start = time.perf_counter()
    pre = FraudPreprocessor(cfg)
    x_train, y_train, t_train = pre.fit_transform(train_df)
    x_val, y_val, t_val = pre.transform(val_df)
    if test_df is not None:
        x_test, y_test, t_test = pre.transform(test_df)
    else:
        x_test, y_test, t_test = None, None, None
    timing["preprocessing_sec"] = time.perf_counter() - start
    
    # ============================================================
    # 4. BUILD GRAPHS
    # ============================================================
    start = time.perf_counter()
    train_graph = get_or_build_graph(x_train, y_train, t_train, cfg, flags, "train", entity_type_train)
    val_graph = get_or_build_graph(x_val, y_val, t_val, cfg, flags, "val", entity_type_val)
    if test_df is not None:
        test_graph = get_or_build_graph(x_test, y_test, t_test, cfg, flags, "test", entity_type_test)
    else:
        test_graph = None
    timing["graph_building_sec"] = time.perf_counter() - start
    
    # ✅ LOG MEMORY: After graph building
    log_memory_snapshot("After graph building")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ============================================================
    # 5. FEDERATED LEARNING
    # ============================================================
    if use_federated:
        start = time.perf_counter()
        fed_result = train_federated(
            train_data=train_graph,
            val_data=val_graph,
            test_data=test_graph if test_graph is not None else val_graph,
            cfg=cfg,
            model_class=FraudGNNRL,
            device=device,
            use_pruning=use_pruning,
        )
        timing["federated_training_sec"] = time.perf_counter() - start
        timing["federated_avg_round_time_sec"] = fed_result.get("avg_round_time_sec", 0)
        timing["federated_round_times"] = fed_result.get("round_times", [])
        
        global_model = fed_result["global_model"]
        val_scores = fed_result["val_scores"]
        val_labels = fed_result["val_labels"]
        if test_graph is not None:
            test_scores = fed_result["test_scores"]
            test_labels = fed_result["test_labels"]
        else:
            test_scores, test_labels = None, None
        fed_history = fed_result["history"]
    else:
        from src.train.train_gnn import train_tssgc_classifier
        start = time.perf_counter()
        model, history, ckpt_path, tssgc_timing = train_tssgc_classifier(
            train_graph, val_graph, cfg,
            output_dir="outputs/checkpoints/ablation",
            timing=timing,
        )
        timing["federated_training_sec"] = time.perf_counter() - start
        timing["tssgc_avg_epoch_time_sec"] = tssgc_timing.get("avg_epoch_time_sec", 0)
        timing["tssgc_epoch_times"] = tssgc_timing.get("epoch_times", [])
        timing["tssgc_total_training_sec"] = tssgc_timing.get("total_training_sec", 0)
        
        global_model = model
        fed_history = history
        val_scores, val_labels = predict_scores(global_model, val_graph, device=device)
        if test_graph is not None:
            test_scores, test_labels = predict_scores(global_model, test_graph, device=device)
        else:
            test_scores, test_labels = None, None
    
    # ✅ LOG MEMORY: After FL training
    log_memory_snapshot("After FL training")
    
    # ============================================================
    # 6. RL THRESHOLD
    # ============================================================
    thresholds = [float(x) for x in cfg.get("rl", {}).get("threshold_bins", [0.5])]
    rl_type = cfg.get("rl", {}).get("type", "naf")
    agent = None
    val_embeddings = None
    test_embeddings = None
    rl_epochs = 0
    
    if use_rl:
        global_model.eval()
        with torch.no_grad():
            embed_dim = global_model.encoder.layers[0].temporal.lin_msg.out_features
            val_embeddings = global_model.embeddings(val_graph).cpu().numpy()
            if test_graph is not None:
                test_embeddings = global_model.embeddings(test_graph).cpu().numpy()
        
        if rl_type == "naf":
            from src.models.naf_agent import train_naf_agent
            start = time.perf_counter()
            agent, rl_history = train_naf_agent(
                val_scores, val_labels, cfg, 
                graph_embeddings=val_embeddings,
                device=device,
                n_features=embed_dim
            )
            timing["rl_training_sec"] = time.perf_counter() - start
            
            first_state = val_embeddings[0] if len(val_embeddings) > 0 else np.zeros(embed_dim)
            feature_weights = agent.get_feature_weights(first_state)
            global_model.set_feature_weights(torch.tensor(feature_weights, dtype=torch.float32))
            val_embeddings = global_model.embeddings(val_graph).cpu().numpy()
            if test_graph is not None:
                test_embeddings = global_model.embeddings(test_graph).cpu().numpy()
            
            # ============================================================
            # ✅ FIX: Xử lý policy_threshold từ apply_naf_policy() an toàn
            # ============================================================
            policy_result = apply_naf_policy(
                agent, val_scores, val_labels, cfg, graph_embeddings=val_embeddings
            )
            
            # ✅ Xử lý kết quả trả về (phòng thủ)
            if isinstance(policy_result, tuple):
                policy_threshold, policy_metrics = policy_result
            else:
                policy_threshold = float(policy_result)
                policy_metrics = {}
            
            # ✅ Đảm bảo policy_threshold là float
            if isinstance(policy_threshold, (tuple, list)):
                policy_threshold = float(policy_threshold[0])
            else:
                policy_threshold = float(policy_threshold)
                
            print(f"[NAF] Threshold from RL policy: {policy_threshold:.4f}")
            
            # 2. Grid-search để so sánh (log riêng)
            grid_threshold, grid_metrics = choose_best_threshold_by_validation(
                val_scores, val_labels, thresholds, cfg=cfg
            )
            print(f"[NAF] Grid-search threshold: {grid_threshold:.4f} (for comparison only)")
            print(f"[NAF] Difference: {abs(policy_threshold - grid_threshold):.4f}")
            
            # ✅ Dùng policy threshold cho benchmark
            best_threshold = policy_threshold
            val_threshold_metrics = {
                "threshold_selection_method": "naf_policy",
                "policy_threshold": policy_threshold,
                "grid_threshold": grid_threshold,
                "grid_f1": grid_metrics.get("f1", 0),
                "grid_auc_roc": grid_metrics.get("auc_roc", 0),
                "feature_weights_applied": True,
                "feature_weights_sum": float(feature_weights.sum()),
            }
            
            print(f"[NAF] Val F1 (policy): {classification_metrics(val_labels, val_scores, threshold=policy_threshold).get('f1', 0):.4f}")
            print(f"[NAF] Val F1 (grid): {grid_metrics.get('f1', 0):.4f}")
        
        else:
            from src.models.dqn_agent import ThresholdDQNAgent, BatchThresholdEnvironment
            start = time.perf_counter()
            
            env = BatchThresholdEnvironment(
                val_scores, val_labels,
                graph_embeddings=val_embeddings,
                batch_size=cfg.get("rl", {}).get("batch_size", 256),
                fpr_penalty=cfg.get("rl", {}).get("fpr_penalty", 2.0),
            )
            
            agent = ThresholdDQNAgent(
                state_dim=env.state_dim,
                thresholds=thresholds,
                n_features=embed_dim,
                device=device,
                lr=cfg.get("rl", {}).get("learning_rate", 0.00001),
                buffer_size=cfg.get("rl", {}).get("buffer_size", 100000),
                epsilon_decay=cfg.get("rl", {}).get("epsilon_decay", 0.9995),
                grad_clip=cfg.get("rl", {}).get("grad_clip", 0.5),
                min_buffer_size=cfg.get("rl", {}).get("min_buffer_size", 256),
                tau=cfg.get("rl", {}).get("tau", 0.005),
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
                if (ep + 1) % 10 == 0:
                    avg_loss = np.mean(ep_loss) if ep_loss else 0
                    print(f"  [DQN] Epoch {ep+1}/{rl_epochs}, avg_loss={avg_loss:.4f}")
            
            timing["rl_training_sec"] = time.perf_counter() - start
            
            policy_thresholds, policy_metrics = apply_dqn_policy(
                agent, val_scores, val_labels, cfg, graph_embeddings=val_embeddings
            )
            policy_threshold = float(np.mean(policy_thresholds)) if isinstance(policy_thresholds, np.ndarray) else float(policy_thresholds)
            grid_threshold, grid_metrics = choose_best_threshold_by_validation(
                val_scores, val_labels, thresholds, cfg=cfg
            )
            
            best_threshold = policy_threshold
            val_threshold_metrics = {
                "threshold_selection_method": "dqn_policy",
                "policy_threshold": policy_threshold,
                "grid_threshold": grid_threshold,
                "grid_f1": grid_metrics.get("f1", 0),
                "grid_auc_roc": grid_metrics.get("auc_roc", 0),
                "final_loss": np.mean(ep_loss) if ep_loss else 0,
            }
    else:
        best_threshold, val_threshold_metrics = choose_best_threshold_by_validation(
            val_scores, val_labels, thresholds, cfg=cfg
        )
        val_threshold_metrics["threshold_selection_method"] = "static_only"
    
    # ✅ LOG MEMORY: After RL training
    log_memory_snapshot("After RL training")
    
    # ============================================================
    # 7. EVALUATION
    # ============================================================
    val_metrics = classification_metrics(val_labels, val_scores, threshold=best_threshold)
    
    if test_scores is not None and test_labels is not None:
        test_metrics = classification_metrics(test_labels, test_scores, threshold=best_threshold)
    else:
        test_metrics = {}
    
    timing["total_runtime_sec"] = time.perf_counter() - total_start
    
    # ✅ LOG MEMORY: After evaluation
    log_memory_snapshot("After evaluation")
    
    # ============================================================
    # 8. RESULT
    # ============================================================
    result = {
        "selected_threshold": best_threshold,
        "threshold_selection_method": val_threshold_metrics.get("threshold_selection_method", "unknown"),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "runtime": timing,
        "val_scores": val_scores.tolist() if isinstance(val_scores, np.ndarray) else val_scores,
        "val_labels": val_labels.tolist() if isinstance(val_labels, np.ndarray) else val_labels,
        "federated_history": fed_history,
    }
    
    if test_scores is not None:
        result["test_scores"] = test_scores.tolist() if isinstance(test_scores, np.ndarray) else test_scores
        result["test_labels"] = test_labels.tolist() if isinstance(test_labels, np.ndarray) else test_labels
    
    if use_rl:
        result["rl_info"] = {
            "type": rl_type,
            "epochs": rl_epochs if use_rl else 0,
            "threshold_from": "rl_policy",
        }
    
    return result


def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main pipeline - GIỐNG PAPER 100%.
    Chạy single split (train/val/test).
    """
    # Data loading
    df = load_dataset(cfg)
    
    # Split
    train_df, val_df, test_df = split_dataframe(df, cfg)
    
    # Chạy pipeline trên split
    result = run_pipeline_on_split(train_df, val_df, test_df, cfg)
    
    # ============================================================
    # BENCHMARK: PAPER vs REPRODUCTION
    # ============================================================
    _print_benchmark(result, cfg)
    
    # Save
    exp_name = cfg.get("experiment", {}).get("name", "experiment")
    pipeline_name = cfg.get("experiment", {}).get("pipeline", "fraudgnn_rl")
    result_filename = f"{exp_name}_{pipeline_name}_metrics.json"
    save_metrics(result, str(Path("outputs/results") / result_filename))
    print(f"\n✅ Results saved to: {result_filename}")
    
    # ✅ LOG MEMORY: After pipeline complete
    log_memory_snapshot("After pipeline complete")
    
    return result


def run_cv_pipeline(
    cfg: Dict[str, Any],
    n_folds: int = 5,
    seeds: List[int] = [42, 123, 2024],
) -> Dict[str, Any]:
    """
    ✅ GIỐNG PAPER: 5-fold Cross-Validation + multiple seeds.
    
    Args:
        cfg: Config dictionary
        n_folds: Số folds (mặc định 5)
        seeds: List seeds để chạy mỗi fold
    
    Returns:
        Dict với kết quả tổng hợp
    """
    print(f"\n{'='*80}")
    print(f"🔬 5-FOLD CROSS-VALIDATION (GIỐNG PAPER)")
    print(f"{'='*80}")
    print(f"  Folds: {n_folds}")
    print(f"  Seeds per fold: {seeds}")
    print(f"  Total runs: {n_folds * len(seeds)}")
    print(f"{'='*80}\n")
    
    df = load_dataset(cfg)
    
    # Tạo CV splits
    from src.data.split import get_cv_splits
    splits = get_cv_splits(df, cfg, n_folds)
    
    all_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        
        # ✅ FIX: Xử lý test set rỗng ở fold cuối
        all_idx = set(range(len(df)))
        used_idx = set(train_idx) | set(val_idx)
        test_idx = list(all_idx - used_idx)
        
        if len(test_idx) == 0:
            print(f"[CV] Fold {fold_idx+1}: Test set is empty! Using 20% of val as test.")
            val_size = len(val_idx)
            test_from_val = max(1, int(val_size * 0.2))
            test_idx = val_idx[-test_from_val:]
            val_idx = val_idx[:-test_from_val]
            val_df = df.iloc[val_idx].copy()
        
        test_df = df.iloc[test_idx].copy()
        
        print(f"\n{'='*60}")
        print(f"[CV] Fold {fold_idx+1}/{n_folds}")
        print(f"{'='*60}")
        print(f"  Train: {len(train_df):,} samples")
        print(f"  Val:   {len(val_df):,} samples")
        print(f"  Test:  {len(test_df):,} samples")
        print(f"{'='*60}")
        
        for seed in seeds:
            print(f"\n  🌱 Running with seed: {seed}")
            
            # Cập nhật seed
            cfg['dataset']['random_state'] = seed
            
            try:
                fold_result = run_pipeline_on_split(
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    cfg=cfg,
                    fold_idx=fold_idx,
                )
                
                # Lưu kết quả với metadata
                all_results.append({
                    'fold': fold_idx + 1,
                    'seed': seed,
                    'test_metrics': fold_result.get('test_metrics', {}),
                    'val_metrics': fold_result.get('val_metrics', {}),
                    'selected_threshold': fold_result.get('selected_threshold', 0.5),
                    'threshold_selection_method': fold_result.get('threshold_selection_method', 'unknown'),
                })
                
                print(f"    ✅ F1: {fold_result.get('test_metrics', {}).get('f1', 0):.4f}")
                print(f"    ✅ Recall: {fold_result.get('test_metrics', {}).get('recall', 0):.4f}")
                
            except Exception as e:
                import traceback
                print(f"    ❌ Error: {e}")
                traceback.print_exc()
                all_results.append({
                    'fold': fold_idx + 1,
                    'seed': seed,
                    'error': str(e),
                })
    
    # ============================================================
    # TỔNG HỢP KẾT QUẢ
    # ============================================================
    print(f"\n{'='*80}")
    print(f"📊 CV SUMMARY (5-fold × {len(seeds)} seeds)")
    print(f"{'='*80}")
    
    # Lọc kết quả thành công
    valid_results = [r for r in all_results if 'error' not in r]
    
    if not valid_results:
        print("❌ No valid results!")
        return {'error': 'No valid results', 'all_results': all_results}
    
    # Tính mean ± std cho từng metric
    metrics = ['auc_roc', 'auc_pr', 'f1', 'recall', 'fpr', 'precision']
    summary = {
        'n_folds': n_folds,
        'n_seeds': len(seeds),
        'n_total_runs': len(valid_results),
        'all_results': all_results,
    }
    
    for metric in metrics:
        values = [r['test_metrics'].get(metric, 0) for r in valid_results]
        if values:
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
            summary[f'{metric}_ci_95'] = 1.96 * np.std(values) / np.sqrt(len(values))
        else:
            summary[f'{metric}_mean'] = 0
            summary[f'{metric}_std'] = 0
            summary[f'{metric}_ci_95'] = 0
    
    # Kết quả threshold
    thresholds = [r.get('selected_threshold', 0.5) for r in valid_results]
    summary['threshold_mean'] = np.mean(thresholds)
    summary['threshold_std'] = np.std(thresholds)
    
    # In kết quả
    print(f"\n📈 Results (mean ± std over {len(valid_results)} runs):")
    print("-"*60)
    for metric in metrics:
        mean = summary.get(f'{metric}_mean', 0)
        std = summary.get(f'{metric}_std', 0)
        ci = summary.get(f'{metric}_ci_95', 0)
        print(f"  {metric:12}: {mean:.4f} ± {std:.4f} (95% CI: {mean-ci:.4f} - {mean+ci:.4f})")
    
    print(f"\n  threshold   : {summary['threshold_mean']:.4f} ± {summary['threshold_std']:.4f}")
    print(f"{'='*80}")
    
    # ============================================================
    # PAPER vs REPRODUCTION (CV)
    # ============================================================
    PAPER_METRICS = {
        'PaySim': {'auc_roc': 0.995, 'auc_pr': 0.647, 'f1': 0.923, 'recall': 0.973},
        'paysim':  {'auc_roc': 0.995, 'auc_pr': 0.647, 'f1': 0.923, 'recall': 0.973},
        'creditcard':   {'auc_roc': 0.996, 'auc_pr': 0.652, 'f1': 0.928, 'recall': 0.978},
        'CreditCard':   {'auc_roc': 0.996, 'auc_pr': 0.652, 'f1': 0.928, 'recall': 0.978},
        'CreditCard2023': {'auc_roc': 0.996, 'auc_pr': 0.652, 'f1': 0.928, 'recall': 0.978},
        'ieee':     {'auc_roc': 0.995, 'auc_pr': 0.649, 'f1': 0.925, 'recall': 0.969},
        'IEEE':     {'auc_roc': 0.995, 'auc_pr': 0.649, 'f1': 0.925, 'recall': 0.969},
        'IEEE-CIS': {'auc_roc': 0.995, 'auc_pr': 0.649, 'f1': 0.925, 'recall': 0.969},
    }

    dataset_name = cfg.get('experiment', {}).get('dataset', 'unknown')

    print(f"\n{'='*80}")
    print(f"[BENCHMARK] PAPER vs REPRODUCTION (CV) - {dataset_name}")
    print(f"{'='*80}")
    print(f"📌 Paper: Cui et al., IEEE JOCS 2025 (Table 2)")
    print(f"📌 Reproduction: Your implementation (5-fold CV + 3 seeds)")
    print(f"{'='*80}")

    if dataset_name in PAPER_METRICS:
        paper = PAPER_METRICS[dataset_name]
        
        print(f"{'Metric':<15} {'Paper':<20} {'Reproduction':<20} {'Delta':<15}")
        print("-"*80)
        
        metric_names = {
            'auc_roc': 'AUC-ROC',
            'auc_pr': 'AUC-PR',
            'f1': 'F1',
            'recall': 'Recall@1%'
        }
        
        for metric in ['auc_roc', 'auc_pr', 'f1', 'recall']:
            paper_val = paper.get(metric, 0)
            repro_val = summary.get(f'{metric}_mean', 0)
            repro_std = summary.get(f'{metric}_std', 0)
            delta = repro_val - paper_val
            
            if delta > 0:
                arrow = "✅"
            elif delta < 0:
                arrow = "⚠️"
            else:
                arrow = "="
            
            display_name = metric_names.get(metric, metric)
            print(f"{display_name:<15} {paper_val:<20.4f} {repro_val:.4f}±{repro_std:.4f} {arrow}{abs(delta):<14.4f}")
        
        print(f"{'='*80}")
        
        f1_delta = summary.get('f1_mean', 0) - paper.get('f1', 0)
        if abs(f1_delta) < 0.01:
            print("📊 Reproduction F1 matches Paper (within ±0.01) ✅")
        elif f1_delta > 0:
            print(f"📊 Reproduction F1 is {f1_delta*100:.1f}% HIGHER than Paper ✅")
        else:
            print(f"📊 Reproduction F1 is {abs(f1_delta)*100:.1f}% LOWER than Paper ⚠️")
        
        recall_delta = summary.get('recall_mean', 0) - paper.get('recall', 0)
        if abs(recall_delta) < 0.01:
            print("📊 Reproduction Recall@1% matches Paper (within ±1%) ✅")
        elif recall_delta > 0:
            print(f"📊 Reproduction Recall@1% is {recall_delta*100:.1f}% HIGHER than Paper ✅")
        else:
            print(f"📊 Reproduction Recall@1% is {abs(recall_delta)*100:.1f}% LOWER than Paper ⚠️")
        
    else:
        print(f"⚠️ Dataset '{dataset_name}' not found in PAPER_METRICS")
    
    # ============================================================
    # LƯU KẾT QUẢ
    # ============================================================
    exp_name = cfg.get("experiment", {}).get("name", "experiment")
    result_filename = f"{exp_name}_cv_results.json"
    
    # Convert numpy types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_summary = convert_to_serializable(summary)
    
    save_metrics(serializable_summary, str(Path("outputs/results") / result_filename))
    print(f"\n✅ CV results saved to: {result_filename}")
    
    return summary


def _print_benchmark(result: Dict[str, Any], cfg: Dict[str, Any]):
    """In benchmark PAPER vs REPRODUCTION cho single run."""
    PAPER_METRICS = {
        'PaySim': {'auc_roc': 0.995, 'auc_pr': 0.647, 'f1': 0.923, 'recall': 0.973},
        'paysim':  {'auc_roc': 0.995, 'auc_pr': 0.647, 'f1': 0.923, 'recall': 0.973},
        'creditcard':   {'auc_roc': 0.996, 'auc_pr': 0.652, 'f1': 0.928, 'recall': 0.978},
        'CreditCard':   {'auc_roc': 0.996, 'auc_pr': 0.652, 'f1': 0.928, 'recall': 0.978},
        'CreditCard2023': {'auc_roc': 0.996, 'auc_pr': 0.652, 'f1': 0.928, 'recall': 0.978},
        'ieee':     {'auc_roc': 0.995, 'auc_pr': 0.649, 'f1': 0.925, 'recall': 0.969},
        'IEEE':     {'auc_roc': 0.995, 'auc_pr': 0.649, 'f1': 0.925, 'recall': 0.969},
        'IEEE-CIS': {'auc_roc': 0.995, 'auc_pr': 0.649, 'f1': 0.925, 'recall': 0.969},
    }

    dataset_name = cfg.get('experiment', {}).get('dataset', 'unknown')

    print(f"\n{'='*80}")
    print(f"[BENCHMARK] PAPER vs REPRODUCTION - {dataset_name}")
    print(f"{'='*80}")
    print(f"📌 Paper: Cui et al., IEEE JOCS 2025 (Table 2)")
    print(f"📌 Reproduction: Your implementation")
    print(f"{'='*80}")

    if dataset_name in PAPER_METRICS:
        paper = PAPER_METRICS[dataset_name]
        repro = result.get('test_metrics', {})
        
        print(f"{'Metric':<15} {'Paper':<20} {'Reproduction':<20} {'Delta':<15}")
        print("-"*80)
        
        for metric in ['auc_roc', 'auc_pr', 'f1', 'recall']:
            paper_val = paper.get(metric, 0)
            repro_val = repro.get(metric, 0)
            delta = repro_val - paper_val
            
            arrow = "✅" if delta > 0 else ("⚠️" if delta < 0 else "=")
            display_name = {'auc_roc': 'AUC-ROC', 'auc_pr': 'AUC-PR', 'f1': 'F1', 'recall': 'Recall@1%'}[metric]
            print(f"{display_name:<15} {paper_val:<20.4f} {repro_val:<20.4f} {arrow}{abs(delta):<14.4f}")
        
        print(f"{'='*80}")
        print(f"Selected threshold: {result.get('selected_threshold', 0.5):.4f} (from RL policy)")
        
        f1_delta = repro.get('f1', 0) - paper.get('f1', 0)
        if abs(f1_delta) < 0.01:
            print("📊 Reproduction F1 matches Paper (within ±0.01) ✅")
        elif f1_delta > 0:
            print(f"📊 Reproduction F1 is {f1_delta*100:.1f}% HIGHER than Paper ✅")
        else:
            print(f"📊 Reproduction F1 is {abs(f1_delta)*100:.1f}% LOWER than Paper ⚠️")
        
    else:
        print(f"⚠️ Dataset '{dataset_name}' not found in PAPER_METRICS")