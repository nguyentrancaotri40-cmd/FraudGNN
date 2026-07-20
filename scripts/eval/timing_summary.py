#!/usr/bin/env python3
"""
Tạo bảng tổng hợp timing metrics từ ablation results
"""

import json
import pandas as pd
from pathlib import Path

def load_results(results_path: str) -> pd.DataFrame:
    """Load và parse ablation results"""
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    rows = []
    for name, result in data.items():
        if 'error' in result:
            continue
        
        runtime = result.get('runtime', {})
        latency = result.get('latency', {})
        memory = result.get('memory', {})
        m = result.get('test_metrics', {})
        
        rows.append({
            'variant': name,
            'dataset': result.get('experiment', {}).get('dataset', 'unknown'),
            'auc_roc': m.get('auc_roc', 0),
            'f1': m.get('f1', 0),
            'recall': m.get('recall', 0),
            
            # ✅ TIMING BREAKDOWN
            'data_loading_s': runtime.get('data_loading_sec', 0),
            'preprocessing_s': runtime.get('preprocessing_sec', 0),
            'graph_building_s': runtime.get('graph_building_sec', 0),
            'federated_training_s': runtime.get('federated_training_sec', 0),
            'dqn_training_s': runtime.get('dqn_training_sec', 0),
            'inference_s': runtime.get('inference_sec', 0),
            
            # ✅ TOTAL
            'total_runtime_s': runtime.get('total_runtime_sec', 0),
            'total_runtime_min': runtime.get('total_runtime_sec', 0) / 60,
            'total_runtime_h': runtime.get('total_runtime_sec', 0) / 3600,
            
            'runtime_per_sample_ms': runtime.get('runtime_per_sample_sec', 0) * 1000,
            'throughput_samples_s': runtime.get('throughput_samples_per_sec', 0),
            
            # Latency
            'latency_mean_ms': latency.get('latency_mean_ms', 0),
            'latency_p95_ms': latency.get('latency_p95_ms', 0),
            
            # Memory
            'ram_gb': memory.get('ram_used_gb', 0),
            'vram_gb': memory.get('vram_allocated_gb', 0),
        })
    
    return pd.DataFrame(rows)

def print_timing_table(df: pd.DataFrame):
    """In bảng timing breakdown"""
    
    print("\n" + "="*120)
    print("⏱️ TIMING BREAKDOWN SUMMARY")
    print("="*120)
    
    # Group by dataset and variant
    summary = df.groupby(['dataset', 'variant']).agg({
        'data_loading_s': ['mean', 'std'],
        'preprocessing_s': ['mean', 'std'],
        'graph_building_s': ['mean', 'std'],
        'federated_training_s': ['mean', 'std'],
        'dqn_training_s': ['mean', 'std'],
        'inference_s': ['mean', 'std'],
        'total_runtime_s': ['mean', 'std'],
        'total_runtime_h': ['mean', 'std'],
        'runtime_per_sample_ms': ['mean', 'std'],
        'throughput_samples_s': ['mean', 'std'],
    }).round(2)
    
    print(summary.to_string())
    
    return summary

def print_total_comparison(df: pd.DataFrame):
    """In bảng so sánh total runtime"""
    
    print("\n" + "="*80)
    print("📊 TOTAL RUNTIME COMPARISON (hours)")
    print("="*80)
    
    pivot = df.pivot_table(
        index='variant',
        columns='dataset',
        values='total_runtime_h',
        aggfunc='mean'
    ).round(2)
    
    print(pivot.to_string())
    
    return pivot

def print_breakdown_pie_chart(df: pd.DataFrame, dataset: str, variant: str):
    """Vẽ biểu đồ breakdown cho 1 config"""
    import matplotlib.pyplot as plt
    
    row = df[(df['dataset'] == dataset) & (df['variant'] == variant)]
    if row.empty:
        print(f"No data for {dataset}/{variant}")
        return
    
    row = row.iloc[0]
    labels = ['Data Loading', 'Preprocessing', 'Graph Building', 'FL Training', 'DQN Training', 'Inference']
    sizes = [
        row['data_loading_s'],
        row['preprocessing_s'],
        row['graph_building_s'],
        row['federated_training_s'],
        row['dqn_training_s'],
        row['inference_s'],
    ]
    
    # Loại bỏ thành phần = 0
    labels = [l for l, s in zip(labels, sizes) if s > 0]
    sizes = [s for s in sizes if s > 0]
    
    if not sizes:
        return
    
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'Timing Breakdown: {dataset.upper()} - {variant}')
    plt.tight_layout()
    
    output_dir = Path('outputs/figures/timing')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{dataset}_{variant}_breakdown.png', dpi=150)
    plt.close()
    print(f"✅ Saved: {output_dir / f'{dataset}_{variant}_breakdown.png'}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True, help='Path to ablation_summary.json')
    parser.add_argument('--plot', action='store_true', help='Generate breakdown charts')
    args = parser.parse_args()
    
    df = load_results(args.results)
    
    print_timing_table(df)
    print_total_comparison(df)
    
    if args.plot:
        for dataset in df['dataset'].unique():
            for variant in df['variant'].unique():
                print_breakdown_pie_chart(df, dataset, variant)

if __name__ == '__main__':
    main()