# src/train/train_rl.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np
import time
import torch

from src.models.dqn_agent import ThresholdDQNAgent, BatchThresholdEnvironment


def train_threshold_dqn(
    scores: np.ndarray,
    labels: np.ndarray,
    cfg: Dict[str, Any],
    device: str | None = None,
    timing: dict | None = None
):
    """Train DQN agent on validation scores."""
    rl_cfg = cfg.get("rl", {})
    thresholds = [float(x) for x in rl_cfg.get("threshold_bins", [0.3, 0.4, 0.5, 0.6, 0.7])]
    batch_size = int(rl_cfg.get("batch_size", 256))
    
    env = BatchThresholdEnvironment(
        scores, labels,
        batch_size=batch_size,
        fpr_penalty=float(rl_cfg.get("fpr_penalty", 2.0)),
    )
    
    agent = ThresholdDQNAgent(
        state_dim=env.state_dim,
        thresholds=thresholds,
        hidden_dim=128,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    
    epochs = int(rl_cfg.get("epochs", 30))
    losses = []
    infos = []
    
    start_time = time.perf_counter()
    
    for ep in range(epochs):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, explore=True)
            threshold = agent.threshold(action)
            next_state, reward, done, info = env.step(threshold)
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.update(batch_size=min(64, max(1, len(agent.memory))))
            if loss is not None:
                losses.append(loss)
            infos.append(info)
            state = next_state
        agent.sync_target()
    
    if timing is not None:
        timing["rl_training_sec"] = time.perf_counter() - start_time
    
    return agent, {"losses": losses, "infos": infos}


def apply_dqn_policy(
    agent: ThresholdDQNAgent,
    scores: np.ndarray,
    labels: np.ndarray,
    cfg: Dict[str, Any],
    device: str | None = None,
) -> tuple[np.ndarray, float]:
    """Apply trained DQN agent greedily to select thresholds per batch."""
    import numpy as np
    
    rl_cfg = cfg.get("rl", {})
    batch_size = int(rl_cfg.get("batch_size", 256))
    fpr_penalty = float(rl_cfg.get("fpr_penalty", 2.0))
    
    env = BatchThresholdEnvironment(
        scores=scores,
        labels=labels,
        batch_size=batch_size,
        fpr_penalty=fpr_penalty,
    )
    
    state = env.reset()
    done = False
    thresholds = []
    
    while not done:
        action = agent.act(state, explore=False)
        threshold = agent.threshold(action)
        thresholds.append(threshold)
        state, _reward, done, _info = env.step(threshold)
    
    thresholds = np.array(thresholds)
    mean_threshold = float(np.mean(thresholds))
    
    return thresholds, mean_threshold


def choose_best_threshold_by_validation(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
    cfg: Dict[str, Any] | None = None,
) -> tuple[float, dict]:
    """Choose threshold on validation scores using grid search."""
    from src.eval.metrics import classification_metrics
    
    cfg = cfg or {}
    sel_cfg = cfg.get("threshold_selection", {})
    
    metric = str(sel_cfg.get("metric", "custom"))
    f1_weight = float(sel_cfg.get("f1_weight", 1.0))
    recall_weight = float(sel_cfg.get("recall_weight", 0.25))
    precision_weight = float(sel_cfg.get("precision_weight", 0.0))
    fpr_weight = float(sel_cfg.get("fpr_weight", 0.25))
    max_fpr = sel_cfg.get("max_fpr", None)
    min_recall = sel_cfg.get("min_recall", None)
    
    all_metrics = []
    for th in thresholds:
        m = classification_metrics(labels, scores, threshold=th)
        all_metrics.append(m)
    
    candidates = []
    for m in all_metrics:
        if max_fpr is not None and float(m["fpr"]) > float(max_fpr):
            continue
        if min_recall is not None and float(m["recall"]) < float(min_recall):
            continue
        candidates.append(m)
    
    if not candidates:
        candidates = all_metrics
    
    best_score = None
    best_m = None
    
    for m in candidates:
        if metric == "f1":
            score = float(m["f1"])
        elif metric == "auc_pr":
            score = float(m["auc_pr"])
        elif metric == "recall_under_fpr":
            score = float(m["recall"]) - fpr_weight * float(m["fpr"])
        elif metric == "precision_under_recall":
            score = float(m["precision"]) + 0.25 * float(m["f1"])
        else:
            score = (
                f1_weight * float(m["f1"])
                + recall_weight * float(m["recall"])
                + precision_weight * float(m["precision"])
                - fpr_weight * float(m["fpr"])
            )
        
        if best_score is None or score > best_score:
            best_score = score
            best_m = m
    
    assert best_m is not None
    
    best_m = dict(best_m)
    best_m["threshold_selection_score"] = float(best_score)
    best_m["threshold_selection_metric"] = metric
    best_m["threshold_selection_candidates"] = len(candidates)
    
    return float(best_m["threshold"]), best_m