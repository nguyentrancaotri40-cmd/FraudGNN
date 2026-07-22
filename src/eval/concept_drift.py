# src/eval/concept_drift.py
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from .metrics import classification_metrics


def evaluate_concept_drift(
    scores: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    window_size: int = 1000,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate model resilience to concept drift (Paper Section V-B).
    
    Args:
        scores: Model prediction scores
        labels: True labels
        timestamps: Time ordering
        window_size: Size of each time window
        threshold: Classification threshold
    
    Returns:
        Dict with drift metrics
    """
    n = len(scores)
    n_windows = max(1, n // window_size)
    
    window_metrics = []
    
    for i in range(n_windows):
        start = i * window_size
        end = min((i + 1) * window_size, n)
        
        window_scores = scores[start:end]
        window_labels = labels[start:end]
        
        if len(window_scores) == 0:
            continue
        
        metrics = classification_metrics(window_labels, window_scores, threshold=threshold)
        metrics['window_idx'] = i
        metrics['start_idx'] = start
        metrics['end_idx'] = end
        window_metrics.append(metrics)
    
    # Calculate drift severity
    f1s = [m['f1'] for m in window_metrics]
    aucs = [m.get('auc_roc', 0) for m in window_metrics]
    recalls = [m['recall'] for m in window_metrics]
    recall_at_1pct = [m.get('recall_at_1pct', 0) for m in window_metrics]
    
    if len(f1s) > 1:
        # F1 decay rate
        initial_f1 = f1s[0]
        final_f1 = f1s[-1]
        f1_decay = (initial_f1 - final_f1) / max(initial_f1, 1e-8) * 100
        
        # AUC decay rate
        initial_auc = aucs[0]
        final_auc = aucs[-1]
        auc_decay = (initial_auc - final_auc) / max(initial_auc, 1e-8) * 100
        
        # Recall decay rate
        initial_recall = recalls[0]
        final_recall = recalls[-1]
        recall_decay = (initial_recall - final_recall) / max(initial_recall, 1e-8) * 100
        
        # Recall@1% decay rate
        initial_recall_at_1pct = recall_at_1pct[0] if recall_at_1pct else 0
        final_recall_at_1pct = recall_at_1pct[-1] if recall_at_1pct else 0
        recall_at_1pct_decay = (initial_recall_at_1pct - final_recall_at_1pct) / max(initial_recall_at_1pct, 1e-8) * 100 if initial_recall_at_1pct > 0 else 0
        
        # Stability (std of metrics)
        f1_std = float(np.std(f1s))
        auc_std = float(np.std(aucs))
        recall_std = float(np.std(recalls))
    else:
        f1_decay = 0.0
        auc_decay = 0.0
        recall_decay = 0.0
        recall_at_1pct_decay = 0.0
        f1_std = 0.0
        auc_std = 0.0
        recall_std = 0.0
    
    return {
        'f1_decay_percent': f1_decay,
        'auc_decay_percent': auc_decay,
        'recall_decay_percent': recall_decay,
        'recall_at_1pct_decay_percent': recall_at_1pct_decay,
        'f1_std': f1_std,
        'auc_std': auc_std,
        'recall_std': recall_std,
        'num_windows': len(window_metrics),
        'avg_f1': float(np.mean(f1s)) if f1s else 0,
        'avg_auc': float(np.mean(aucs)) if aucs else 0,
        'avg_recall': float(np.mean(recalls)) if recalls else 0,
        'initial_f1': f1s[0] if f1s else 0,
        'final_f1': f1s[-1] if f1s else 0,
        'initial_recall': recalls[0] if recalls else 0,
        'final_recall': recalls[-1] if recalls else 0,
        'initial_auc': aucs[0] if aucs else 0,
        'final_auc': aucs[-1] if aucs else 0,
        'window_metrics': window_metrics,
    }


def run_concept_drift_test(
    scores: np.ndarray,
    labels: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    window_size: int = 1000,
    threshold: float = 0.5,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run concept drift test from pipeline results.
    
    Args:
        scores: Model prediction scores
        labels: True labels
        timestamps: Time ordering (optional)
        window_size: Size of each time window
        threshold: Classification threshold
        output_dir: Directory to save results (optional)
    
    Returns:
        Dict with drift metrics
    """
    if timestamps is None:
        timestamps = np.arange(len(scores))
    
    # Sort by time
    order = np.argsort(timestamps)
    scores = scores[order]
    labels = labels[order]
    timestamps = timestamps[order]
    
    result = evaluate_concept_drift(scores, labels, timestamps, window_size, threshold)
    
    # Save results if output_dir provided
    if output_dir:
        import json
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / 'concept_drift_results.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_result = {}
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    serializable_result[k] = v.tolist()
                elif isinstance(v, (np.float32, np.float64)):
                    serializable_result[k] = float(v)
                else:
                    serializable_result[k] = v
            json.dump(serializable_result, f, indent=2, default=str)
        print(f"[ConceptDrift] Results saved to {output_path / 'concept_drift_results.json'}")
    
    return result