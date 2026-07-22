# src/eval/adversarial.py
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from .evaluate import predict_scores
from .metrics import classification_metrics


def evaluate_adversarial_robustness(
    model,
    test_graph,
    device: str = "cpu",
    attacks: Optional[List[str]] = None,
    epsilon: float = 0.1,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Đánh giá adversarial robustness của model (Paper Section V-B).
    
    Args:
        model: FraudGNNRL model
        test_graph: PyG Data object
        device: 'cpu' hoặc 'cuda'
        attacks: list of attack names
        epsilon: Perturbation magnitude
        output_dir: Directory to save results
    
    Returns:
        Dict với kết quả từng attack
    """
    if attacks is None:
        attacks = ['fgsm', 'random_noise', 'label_flipping']
    
    model.eval()
    model.to(device)
    test_graph = test_graph.to(device)
    
    # 1. Baseline
    baseline_scores, baseline_labels = predict_scores(model, test_graph, device)
    baseline = classification_metrics(baseline_labels, baseline_scores, threshold=0.5)
    baseline_f1 = baseline['f1']
    baseline_recall = baseline['recall']
    baseline_auc = baseline.get('auc_roc', 0)
    
    results = {
        'baseline': {
            'f1': float(baseline_f1),
            'recall': float(baseline_recall),
            'auc_roc': float(baseline_auc),
            'precision': float(baseline.get('precision', 0)),
            'fpr': float(baseline.get('fpr', 0)),
        }
    }
    
    # 2. Các attack
    for attack_name in attacks:
        try:
            if attack_name == 'fgsm':
                adv_scores = _fgsm_attack(model, test_graph, device, epsilon=epsilon)
            elif attack_name == 'random_noise':
                adv_scores = _random_noise_attack(model, test_graph, device, epsilon=epsilon)
            elif attack_name == 'label_flipping':
                adv_scores = _label_flipping_attack(model, test_graph, device, flip_rate=epsilon)
            else:
                continue
            
            adv_metrics = classification_metrics(baseline_labels, adv_scores, threshold=0.5)
            adv_f1 = adv_metrics['f1']
            adv_recall = adv_metrics['recall']
            adv_auc = adv_metrics.get('auc_roc', 0)
            
            results[attack_name] = {
                'f1': float(adv_f1),
                'recall': float(adv_recall),
                'auc_roc': float(adv_auc),
                'f1_drop_percent': float((baseline_f1 - adv_f1) / max(baseline_f1, 1e-8) * 100),
                'recall_drop_percent': float((baseline_recall - adv_recall) / max(baseline_recall, 1e-8) * 100),
                'auc_drop_percent': float((baseline_auc - adv_auc) / max(baseline_auc, 1e-8) * 100),
            }
        except Exception as e:
            results[attack_name] = {'error': str(e)}
    
    # Save results if output_dir provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / 'adversarial_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[Adversarial] Results saved to {output_path / 'adversarial_results.json'}")
    
    return results


def _fgsm_attack(model, data, device, epsilon: float = 0.1):
    """Fast Gradient Sign Method attack."""
    data = data.to(device)
    data.x.requires_grad = True
    
    model.zero_grad()
    logits = model(data)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, data.y.float()
    )
    
    loss.backward()
    
    with torch.no_grad():
        grad_sign = data.x.grad.sign()
        adv_x = data.x + epsilon * grad_sign
        adv_x = torch.clamp(adv_x, 0.0, 1.0)
    
    adv_data = data.clone()
    adv_data.x = adv_x
    adv_data.x.requires_grad = False
    
    adv_scores, _ = predict_scores(model, adv_data, device)
    return adv_scores


def _random_noise_attack(model, data, device, epsilon: float = 0.1):
    """Random noise attack."""
    data = data.to(device)
    noise = torch.randn_like(data.x) * epsilon
    adv_x = torch.clamp(data.x + noise, 0.0, 1.0)
    
    adv_data = data.clone()
    adv_data.x = adv_x
    
    adv_scores, _ = predict_scores(model, adv_data, device)
    return adv_scores


def _label_flipping_attack(model, data, device, flip_rate: float = 0.1):
    """Label flipping attack."""
    data = data.to(device)
    
    scores, _ = predict_scores(model, data, device)
    n = len(scores)
    flip_idx = np.random.choice(n, size=int(n * flip_rate), replace=False)
    adv_scores = scores.copy()
    adv_scores[flip_idx] = 1 - adv_scores[flip_idx]
    
    return adv_scores


def run_adversarial_test(
    model,
    data,
    device: str = "cpu",
    epsilon: float = 0.1,
    attacks: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run adversarial robustness test.
    
    Args:
        model: PyTorch model
        data: Graph data
        device: Device to run on
        epsilon: Maximum perturbation
        attacks: List of attack names
        output_dir: Directory to save results
    
    Returns:
        Dict with robustness metrics
    """
    return evaluate_adversarial_robustness(
        model=model,
        test_graph=data,
        device=device,
        attacks=attacks,
        epsilon=epsilon,
        output_dir=output_dir,
    )