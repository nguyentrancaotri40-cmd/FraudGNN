#!/usr/bin/env python3
"""
So sánh kết quả giữa 2 models: FraudGNN-RL vs FraudGNN-RL+
"""

import argparse
import json
from pathlib import Path


def load_metrics(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def print_comparison(baseline_path: str, hybrid_path: str, split: str = "test"):
    """In bảng so sánh baseline vs hybrid."""
    
    base = load_metrics(baseline_path)
    hybrid = load_metrics(hybrid_path)
    
    base_m = base.get(f"{split}_metrics", {})
    hybrid_m = hybrid.get(f"{split}_metrics", {})
    
    print("\n" + "="*70)
    print("📊 COMPARISON: FraudGNN-RL (Baseline) vs FraudGNN-RL+ (Hybrid)")
    print("="*70)
    print(f"{'Metric':<15} {'FraudGNN-RL':<15} {'FraudGNN-RL+':<15} {'Delta':<12} {'Improvement'}")
    print("-"*70)
    
    metrics = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall', 'recall_at_1pct', 'fpr']
    
    for key in metrics:
        base_val = base_m.get(key, 0)
        hybrid_val = hybrid_m.get(key, 0)
        delta = hybrid_val - base_val
        
        if key == 'fpr':
            improved = "✅ BETTER" if delta < 0 else "⚠️ WORSE"
            direction = "↓" if delta < 0 else "↑"
        else:
            improved = "✅ BETTER" if delta > 0 else "⚠️ WORSE"
            direction = "↑" if delta > 0 else "↓"
        
        print(f"{key:<15} {base_val:<15.4f} {hybrid_val:<15.4f} {direction}{abs(delta):<11.4f} {improved}")
    
    print("="*70)
    print(f"\n📌 Baseline: {baseline_path}")
    print(f"📌 Hybrid:   {hybrid_path}")
    print(f"📌 Split:    {split}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True, help='Path to baseline metrics JSON')
    parser.add_argument('--hybrid', type=str, required=True, help='Path to hybrid metrics JSON')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    args = parser.parse_args()
    
    print_comparison(args.baseline, args.hybrid, args.split)


if __name__ == '__main__':
    main()