#!/usr/bin/env python3
"""
📊 COMPARE: FraudGNN-RL (Baseline) vs FraudGNN-RL+ (Hybrid)

SO SÁNH:
    FraudGNN-RL (Baseline - Paper gốc)
        - Hard Edges (transaction graph)
        - RL (DQN/NAF)
        - Federated Learning (FL)
        - TSSGC (Temporal-Spatial-Semantic)
        - KHÔNG có Soft Edges
        - KHÔNG có Pruning
    
    FraudGNN-RL+ (Hybrid - Của bạn)
        - Hard Edges (transaction graph)
        - Soft Edges (behavioral graph)  ← CẢI TIẾN
        - Hybrid Graph (hard + soft)     ← CẢI TIẾN
        - Weighted Fusion                 ← CẢI TIẾN
        - Pruning                         ← CẢI TIẾN
        - RL (DQN/NAF)
        - Federated Learning (FL)
        - TSSGC

MỤC ĐÍCH:
    Chứng minh các cải tiến (Soft Edges + Pruning) có cải thiện 
    hiệu suất so với baseline của paper hay không.

CÁCH DÙNG:
    python scripts/eval/compare.py \
        --baseline outputs/results/baseline_metrics.json \
        --hybrid outputs/results/hybrid_metrics.json \
        --split test
"""

import argparse
import json
from pathlib import Path


def load_metrics(path: str) -> dict:
    """Load metrics từ file JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def print_comparison(baseline_path: str, hybrid_path: str, split: str = "test"):
    """
    In bảng so sánh Baseline vs Hybrid.
    
    Args:
        baseline_path: Đường dẫn đến file JSON của FraudGNN-RL (Baseline)
        hybrid_path: Đường dẫn đến file JSON của FraudGNN-RL+ (Hybrid)
        split: 'val' hoặc 'test' - tập dữ liệu để so sánh
    """
    
    # Load kết quả
    base = load_metrics(baseline_path)      # FraudGNN-RL (Hard only)
    hybrid = load_metrics(hybrid_path)      # FraudGNN-RL+ (Hard + Soft + Prune)
    
    # Lấy metrics từ split tương ứng
    base_m = base.get(f"{split}_metrics", {})
    hybrid_m = hybrid.get(f"{split}_metrics", {})
    
    print("\n" + "="*70)
    print("📊 COMPARISON: FraudGNN-RL (Baseline) vs FraudGNN-RL+ (Hybrid)")
    print("="*70)
    print("  FraudGNN-RL  : Hard + RL + FL + TSSGC (Paper gốc)")
    print("  FraudGNN-RL+ : Hard + Soft + Weighted + Prune + RL + FL + TSSGC (Của bạn)")
    print("-"*70)
    
    # Danh sách metric cần so sánh
    metrics = ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall', 'recall_at_1pct', 'fpr']
    
    print(f"{'Metric':<15} {'FraudGNN-RL':<15} {'FraudGNN-RL+':<15} {'Delta':<12} {'Improvement'}")
    print("-"*70)
    
    for key in metrics:
        base_val = base_m.get(key, 0)
        hybrid_val = hybrid_m.get(key, 0)
        delta = hybrid_val - base_val
        
        # Đánh giá cải thiện
        if key == 'fpr':
            # FPR càng thấp càng tốt
            improved = "✅ BETTER" if delta < 0 else "⚠️ WORSE"
            direction = "↓" if delta < 0 else "↑"
        else:
            # Các metric khác càng cao càng tốt
            improved = "✅ BETTER" if delta > 0 else "⚠️ WORSE"
            direction = "↑" if delta > 0 else "↓"
        
        print(f"{key:<15} {base_val:<15.4f} {hybrid_val:<15.4f} {direction}{abs(delta):<11.4f} {improved}")
    
    print("="*70)
    print(f"\n📌 Baseline (FraudGNN-RL): {baseline_path}")
    print(f"📌 Hybrid (FraudGNN-RL+):  {hybrid_path}")
    print(f"📌 Split:                  {split}")
    print("\n💡 Kết luận: Nếu FraudGNN-RL+ tốt hơn FraudGNN-RL → Cải tiến (Soft + Prune) có hiệu quả!")


def main():
    parser = argparse.ArgumentParser(
        description="So sánh FraudGNN-RL (Baseline) vs FraudGNN-RL+ (Hybrid)",
        epilog="Ví dụ: python scripts/eval/compare.py --baseline outputs/results/baseline.json --hybrid outputs/results/hybrid.json"
    )
    parser.add_argument(
        '--baseline', 
        type=str, 
        required=True, 
        help='Đường dẫn đến file JSON của FraudGNN-RL (Baseline - Hard only)'
    )
    parser.add_argument(
        '--hybrid', 
        type=str, 
        required=True, 
        help='Đường dẫn đến file JSON của FraudGNN-RL+ (Hybrid - Hard + Soft + Prune)'
    )
    parser.add_argument(
        '--split', 
        type=str, 
        default='test', 
        choices=['val', 'test'],
        help='Tập dữ liệu để so sánh (val hoặc test), mặc định: test'
    )
    args = parser.parse_args()
    
    print_comparison(args.baseline, args.hybrid, args.split)


if __name__ == '__main__':
    main()