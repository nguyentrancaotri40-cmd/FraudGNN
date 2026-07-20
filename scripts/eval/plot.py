#!/usr/bin/env python3
"""
Vẽ biểu đồ so sánh ablation study
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_ablation_results(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def plot_ablation_comparison(results: dict, output_dir: str = "outputs/figures"):
    """Vẽ biểu đồ so sánh các ablation variants."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = []
    for name, result in results.items():
        if result and 'test_metrics' in result:
            m = result['test_metrics']
            data.append({
                'name': name,
                'auc_roc': m.get('auc_roc', 0),
                'auc_pr': m.get('auc_pr', 0),
                'f1': m.get('f1', 0),
                'recall': m.get('recall', 0),
                'precision': m.get('precision', 0),
                'fpr': m.get('fpr', 0),
            })
    
    if not data:
        print("No data to plot")
        return
    
    df = pd.DataFrame(data)
    df = df.set_index('name')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [('auc_roc', 'AUC-ROC'), ('auc_pr', 'AUC-PR'), ('f1', 'F1'), ('recall', 'Recall')]
    
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        df[metric].plot(kind='bar', ax=ax)
        ax.set_title(f'{title} Comparison')
        ax.set_ylabel(title)
        ax.set_xlabel('Ablation Variant')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend().remove()
        
        for i, v in enumerate(df[metric]):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_comparison.png', dpi=200)
    plt.close()
    
    print(f"✅ Figure saved to: {output_dir / 'ablation_comparison.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True, help='Path to ablation_summary.json')
    parser.add_argument('--output', type=str, default='outputs/figures')
    args = parser.parse_args()
    
    results = load_ablation_results(args.results)
    plot_ablation_comparison(results, args.output)


if __name__ == '__main__':
    main()