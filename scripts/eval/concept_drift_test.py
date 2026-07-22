#!/usr/bin/env python3
"""
Concept Drift Resilience Test

Paper Section V-B: "remarkable resilience to concept drift"
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

from src.eval.metrics import classification_metrics


def evaluate_by_time_window(
    scores: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    window_size: int = 1000,
) -> List[Dict]:
    """Đánh giá performance theo từng time window."""
    results = []
    n = len(scores)
    
    if n == 0:
        return results
    
    # Tìm threshold tối ưu trên toàn bộ data
    thresholds = np.arange(0.05, 0.95, 0.05)
    best_threshold = 0.5
    best_f1 = 0
    for th in thresholds:
        m = classification_metrics(labels, scores, threshold=th)
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_threshold = th
    
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        window_scores = scores[start:end]
        window_labels = labels[start:end]
        
        # Dùng threshold tối ưu chung
        metrics = classification_metrics(window_labels, window_scores, threshold=best_threshold)
        metrics['window_idx'] = start // window_size
        metrics['start_idx'] = start
        metrics['end_idx'] = end
        metrics['threshold'] = best_threshold
        metrics['num_samples'] = end - start
        
        # ✅ FIX: Tính recall@1% với kiểm tra division by zero
        total_fraud = int(np.sum(window_labels == 1))
        if total_fraud > 0:
            k = max(1, int(np.ceil(len(window_scores) * 0.01)))
            top_idx = np.argsort(-window_scores)[:k]
            recall_at_1pct = np.sum(window_labels[top_idx] == 1) / total_fraud
            metrics['recall_at_1pct'] = float(recall_at_1pct)
        else:
            metrics['recall_at_1pct'] = 0.0
            metrics['recall_at_1pct_note'] = 'No fraud in window'
        
        results.append(metrics)
    
    return results


def calculate_drift_metrics(results: List[Dict]) -> Dict:
    """Tính các metrics về concept drift."""
    if len(results) < 2:
        return {}
    
    f1s = [r['f1'] for r in results]
    recalls = [r['recall'] for r in results]
    aucs = [r.get('auc_roc', 0) for r in results]
    recall_at_1pct = [r.get('recall_at_1pct', 0) for r in results]
    
    # Decay rates
    initial_f1 = f1s[0]
    final_f1 = f1s[-1]
    f1_decay = (initial_f1 - final_f1) / max(initial_f1, 1e-8) * 100
    
    initial_recall = recalls[0]
    final_recall = recalls[-1]
    recall_decay = (initial_recall - final_recall) / max(initial_recall, 1e-8) * 100
    
    initial_recall_at_1pct = recall_at_1pct[0]
    final_recall_at_1pct = recall_at_1pct[-1]
    recall_at_1pct_decay = (initial_recall_at_1pct - final_recall_at_1pct) / max(initial_recall_at_1pct, 1e-8) * 100
    
    # Average and std
    avg_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    
    return {
        'initial_f1': float(initial_f1),
        'final_f1': float(final_f1),
        'avg_f1': float(avg_f1),
        'std_f1': float(std_f1),
        'f1_decay_percent': float(f1_decay),
        'initial_recall': float(initial_recall),
        'final_recall': float(final_recall),
        'recall_decay_percent': float(recall_decay),
        'initial_recall_at_1pct': float(initial_recall_at_1pct),
        'final_recall_at_1pct': float(final_recall_at_1pct),
        'recall_at_1pct_decay_percent': float(recall_at_1pct_decay),
        'num_windows': len(results),
        'f1s': f1s,
        'recalls': recalls,
        'aucs': aucs,
        'recall_at_1pct': recall_at_1pct,
    }


def plot_drift(results: List[Dict], output_dir: str = "outputs/figures") -> None:
    """Vẽ biểu đồ concept drift."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(results) == 0:
        print("⚠️ No results to plot")
        return
    
    windows = [r['window_idx'] for r in results]
    f1s = [r['f1'] for r in results]
    recalls = [r['recall'] for r in results]
    aucs = [r.get('auc_roc', 0) for r in results]
    recall_at_1pct = [r.get('recall_at_1pct', 0) for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # F1
    axes[0, 0].plot(windows, f1s, marker='o', color='blue')
    axes[0, 0].axhline(y=f1s[0], color='gray', linestyle='--', label='Initial F1')
    axes[0, 0].set_title('F1-score over Time')
    axes[0, 0].set_xlabel('Time Window')
    axes[0, 0].set_ylabel('F1-score')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Recall
    axes[0, 1].plot(windows, recalls, marker='o', color='green')
    axes[0, 1].axhline(y=recalls[0], color='gray', linestyle='--', label='Initial Recall')
    axes[0, 1].set_title('Recall over Time')
    axes[0, 1].set_xlabel('Time Window')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # AUC-ROC
    axes[0, 2].plot(windows, aucs, marker='o', color='orange')
    axes[0, 2].axhline(y=aucs[0], color='gray', linestyle='--', label='Initial AUC')
    axes[0, 2].set_title('AUC-ROC over Time')
    axes[0, 2].set_xlabel('Time Window')
    axes[0, 2].set_ylabel('AUC-ROC')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Recall@1%
    axes[1, 0].plot(windows, recall_at_1pct, marker='o', color='purple')
    axes[1, 0].axhline(y=recall_at_1pct[0] if recall_at_1pct else 0, color='gray', linestyle='--', label='Initial Recall@1%')
    axes[1, 0].set_title('Recall@1% over Time')
    axes[1, 0].set_xlabel('Time Window')
    axes[1, 0].set_ylabel('Recall@1%')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 Decay
    if f1s[0] > 0:
        decay = [(f1s[0] - f) / f1s[0] * 100 for f in f1s]
        axes[1, 1].plot(windows, decay, marker='o', color='red')
        axes[1, 1].axhline(y=0, color='gray', linestyle='--')
        axes[1, 1].set_title('F1 Decay (%)')
        axes[1, 1].set_xlabel('Time Window')
        axes[1, 1].set_ylabel('F1 Decay (%)')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'No decay (F1=0)', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('F1 Decay (%)')
    
    # Recall@1% Decay
    if recall_at_1pct and recall_at_1pct[0] > 0:
        decay_1pct = [(recall_at_1pct[0] - r) / recall_at_1pct[0] * 100 for r in recall_at_1pct]
        axes[1, 2].plot(windows, decay_1pct, marker='o', color='brown')
        axes[1, 2].axhline(y=0, color='gray', linestyle='--')
        axes[1, 2].set_title('Recall@1% Decay (%)')
        axes[1, 2].set_xlabel('Time Window')
        axes[1, 2].set_ylabel('Recall@1% Decay (%)')
        axes[1, 2].grid(True)
    else:
        axes[1, 2].text(0.5, 0.5, 'No decay (Recall@1%=0)', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Recall@1% Decay (%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'concept_drift_analysis.png', dpi=200)
    plt.close()
    print(f"✅ Saved: {output_dir / 'concept_drift_analysis.png'}")


def run_concept_drift_test(
    result_json: Optional[str] = None,
    result: Optional[Dict] = None,
    scores: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    window_size: int = 1000,
    output_dir: str = "outputs/results/concept_drift",
) -> Dict:
    """
    Chạy concept drift test từ nhiều nguồn dữ liệu.
    
    Args:
        result_json: Path đến file JSON kết quả từ pipeline
        result: Dict kết quả từ pipeline
        scores: Array scores trực tiếp
        labels: Array labels trực tiếp
        timestamps: Array timestamps trực tiếp
        window_size: Kích thước time window
        output_dir: Thư mục lưu kết quả
    
    Returns:
        Dict kết quả concept drift test
    """
    
    # ============================================================
    # 1. LẤY DỮ LIỆU
    # ============================================================
    
    # Từ file JSON
    if result_json is not None:
        with open(result_json, 'r') as f:
            data = json.load(f)
        scores = data.get('val_scores', [])
        labels = data.get('val_labels', [])
        timestamps = data.get('val_timestamps', np.arange(len(scores)))
    
    # Từ dict result
    elif result is not None and isinstance(result, dict):
        scores = result.get('val_scores', [])
        labels = result.get('val_labels', [])
        timestamps = result.get('val_timestamps', np.arange(len(scores)))
    
    # Từ array trực tiếp
    elif scores is not None and labels is not None:
        pass  # đã có
    
    else:
        raise ValueError("Need result_json path, result dict, or scores/labels arrays")
    
    if len(scores) == 0:
        print("⚠️ No scores/labels found. Run pipeline with save_scores=True")
        return {'error': 'No scores/labels found'}
    
    scores = np.array(scores)
    labels = np.array(labels)
    timestamps = np.array(timestamps) if timestamps is not None else np.arange(len(scores))
    
    # Sắp xếp theo thời gian
    if len(timestamps) == len(scores):
        order = np.argsort(timestamps)
        scores = scores[order]
        labels = labels[order]
        timestamps = timestamps[order]
    
    # ============================================================
    # 2. ĐÁNH GIÁ THEO TIME WINDOW
    # ============================================================
    results = evaluate_by_time_window(scores, labels, timestamps, window_size=window_size)
    drift_metrics = calculate_drift_metrics(results)
    
    # ============================================================
    # 3. VẼ BIỂU ĐỒ
    # ============================================================
    plot_drift(results)
    
    # ============================================================
    # 4. LƯU KẾT QUẢ
    # ============================================================
    output = {
        'drift_metrics': drift_metrics,
        'window_results': results,
        'num_windows': len(results),
        'window_size': window_size,
        'num_samples': len(scores),
    }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'concept_drift_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Concept drift results saved to: {output_path}")
    
    # ============================================================
    # 5. PRINT SUMMARY
    # ============================================================
    d = drift_metrics
    if d:
        print("\n📊 Concept Drift Summary:")
        print(f"  Initial F1:           {d.get('initial_f1', 0):.4f}")
        print(f"  Final F1:             {d.get('final_f1', 0):.4f}")
        print(f"  F1 Decay:             {d.get('f1_decay_percent', 0):.2f}%")
        print(f"  Avg F1:               {d.get('avg_f1', 0):.4f} ± {d.get('std_f1', 0):.4f}")
        print(f"  Initial Recall:       {d.get('initial_recall', 0):.4f}")
        print(f"  Final Recall:         {d.get('final_recall', 0):.4f}")
        print(f"  Recall Decay:         {d.get('recall_decay_percent', 0):.2f}%")
        print(f"  Initial Recall@1%:    {d.get('initial_recall_at_1pct', 0):.4f}")
        print(f"  Final Recall@1%:      {d.get('final_recall_at_1pct', 0):.4f}")
        print(f"  Recall@1% Decay:      {d.get('recall_at_1pct_decay_percent', 0):.2f}%")
        print(f"  Num Windows:          {d.get('num_windows', 0)}")
    
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, required=True, help='Path to result JSON')
    parser.add_argument('--window', type=int, default=1000, help='Window size')
    parser.add_argument('--output', type=str, default='outputs/results/concept_drift')
    args = parser.parse_args()
    
    run_concept_drift_test(
        result_json=args.result,
        window_size=args.window,
        output_dir=args.output,
    )