#!/usr/bin/env python3
"""
Chạy ablation study với timing metrics đầy đủ
- 6 variants × 3 seeds × 3 datasets
- Latency, throughput, memory
- Statistical analysis (mean, std, CI, p-value)
"""

import subprocess
import json
import time
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
import sys

# ============================================================
# CONFIGURATION
# ============================================================
DATASETS = ['creditcard2023', 'paysim', 'ieee_cis']
VARIANTS = ['baseline', 'soft_only', 'hybrid_unweighted', 'hybrid_weighted', 'pruning_only', 'full']
SEEDS = [42, 123, 2024]

VARIANT_FLAGS = {
    'baseline': {'hard_edges': True, 'soft_edges': False, 'hybrid_graph': False, 'weighted_fusion': False, 'pruning': False},
    'soft_only': {'hard_edges': False, 'soft_edges': True, 'hybrid_graph': False, 'weighted_fusion': False, 'pruning': False},
    'hybrid_unweighted': {'hard_edges': True, 'soft_edges': True, 'hybrid_graph': True, 'weighted_fusion': False, 'pruning': False},
    'hybrid_weighted': {'hard_edges': True, 'soft_edges': True, 'hybrid_graph': True, 'weighted_fusion': True, 'pruning': False},
    'pruning_only': {'hard_edges': True, 'soft_edges': False, 'hybrid_graph': False, 'weighted_fusion': False, 'pruning': True},
    'full': {'hard_edges': True, 'soft_edges': True, 'hybrid_graph': True, 'weighted_fusion': True, 'pruning': True},
}


def create_ablation_config(dataset, variant, seed, base_config):
    """Tạo config với flags + seed + timing."""
    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['flags'] = VARIANT_FLAGS[variant]
    config['experiment']['name'] = f"{dataset}_{variant}_seed{seed}"
    config['experiment']['seed'] = seed
    config['experiment']['ablation'] = variant
    config['dataset']['random_state'] = seed
    
    # Thêm timing config
    config['timing'] = {
        'measure_latency': True,
        'latency_runs': 50,
        'log_memory': True,
    }
    
    output_path = Path(f"configs/ablation/{dataset}/{variant}_seed{seed}.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f)
    
    return output_path


def run_experiment(config_path):
    """Chạy experiment và thu thập timing metrics."""
    
    start_time = time.perf_counter()
    cmd = [sys.executable, '-m', 'src.main_pipeline', '--config', str(config_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    total_time = time.perf_counter() - start_time
    
    try:
        output = result.stdout
        start = output.find('{')
        end = output.rfind('}') + 1
        if start >= 0 and end > start:
            data = json.loads(output[start:end])
            data['_total_time_sec'] = total_time
            data['_returncode'] = result.returncode
            return data
    except Exception as e:
        return {'error': str(e), 'stderr': result.stderr[:500]}
    
    return {'error': 'No JSON found', 'stdout': result.stdout[:500]}


def extract_timing_metrics(result):
    """Trích xuất tất cả timing metrics từ kết quả."""
    
    runtime = result.get('runtime', {})
    latency = result.get('latency', {})
    memory = result.get('memory', {})
    
    return {
        # Runtime breakdown
        'data_loading_s': runtime.get('data_loading_sec', 0),
        'data_splitting_s': runtime.get('data_splitting_sec', 0),
        'preprocessing_s': runtime.get('preprocessing_sec', 0),
        'graph_building_s': runtime.get('graph_building_sec', 0),
        'federated_training_s': runtime.get('federated_training_sec', 0),
        'dqn_training_s': runtime.get('dqn_training_sec', 0),
        'inference_s': runtime.get('inference_sec', 0),
        'total_runtime_s': runtime.get('total_runtime_sec', 0),
        'runtime_per_sample_ms': runtime.get('runtime_per_sample_sec', 0) * 1000,
        'throughput_samples_s': runtime.get('throughput_samples_per_sec', 0),
        
        # Federated details
        'federated_avg_round_s': runtime.get('federated_avg_round_time_sec', 0),
        'federated_num_rounds': len(runtime.get('federated_round_times', [])),
        
        # Latency
        'latency_mean_ms': latency.get('latency_mean_ms', 0),
        'latency_p50_ms': latency.get('latency_p50_ms', 0),
        'latency_p95_ms': latency.get('latency_p95_ms', 0),
        'latency_p99_ms': latency.get('latency_p99_ms', 0),
        'throughput_latency_s': latency.get('throughput_per_sec', 0),
        
        # Memory
        'ram_used_gb': memory.get('ram_used_gb', 0),
        'vram_allocated_gb': memory.get('vram_allocated_gb', 0),
        'vram_reserved_gb': memory.get('vram_reserved_gb', 0),
    }


def print_timing_table(df: pd.DataFrame):
    """In bảng timing breakdown."""
    
    print("\n" + "="*120)
    print("⏱️ TIMING BREAKDOWN (Mean ± Std over 3 seeds)")
    print("="*120)
    
    for dataset in df['dataset'].unique():
        print(f"\n📁 {dataset.upper()}")
        print("-"*110)
        
        subset = df[df['dataset'] == dataset]
        
        print(f"{'Variant':<20} {'Graph(s)':<12} {'FL(s)':<12} {'DQN(s)':<12} {'Total(s)':<12} {'Total(h)':<10} {'Latency(ms)':<12} {'Throughput':<12}")
        print("-"*110)
        
        for variant in VARIANTS:
            rows = subset[subset['variant'] == variant]
            if rows.empty:
                continue
            
            mean = rows.mean()
            std = rows.std()
            
            print(f"{variant:<20} "
                  f"{mean['graph_building_s']:.0f}±{std['graph_building_s']:.0f} "
                  f"{mean['federated_training_s']:.0f}±{std['federated_training_s']:.0f} "
                  f"{mean['dqn_training_s']:.0f}±{std['dqn_training_s']:.0f} "
                  f"{mean['total_runtime_s']:.0f}±{std['total_runtime_s']:.0f} "
                  f"{mean['total_runtime_s']/3600:.2f}±{std['total_runtime_s']/3600:.2f} "
                  f"{mean['latency_mean_ms']:.2f}±{std['latency_mean_ms']:.2f} "
                  f"{mean['throughput_samples_s']:.0f}±{std['throughput_samples_s']:.0f}")


def print_summary_table(df: pd.DataFrame):
    """In bảng ablation summary với timing."""
    
    print("\n" + "="*120)
    print("📊 ABLATION SUMMARY WITH TIMING (Mean ± Std over 3 seeds)")
    print("="*120)
    
    for dataset in df['dataset'].unique():
        print(f"\n📁 {dataset.upper()}")
        print("-"*120)
        
        subset = df[df['dataset'] == dataset]
        
        print(f"{'Variant':<20} {'AUC-ROC':<12} {'F1':<12} {'Recall':<12} {'FPR':<12} {'Total(h)':<10} {'Latency(ms)':<12}")
        print("-"*120)
        
        for variant in VARIANTS:
            rows = subset[subset['variant'] == variant]
            if rows.empty:
                continue
            
            mean = rows.mean()
            std = rows.std()
            
            print(f"{variant:<20} "
                  f"{mean['auc_roc']:.4f}±{std['auc_roc']:.4f} "
                  f"{mean['f1']:.4f}±{std['f1']:.4f} "
                  f"{mean['recall']:.4f}±{std['recall']:.4f} "
                  f"{mean['fpr']:.4f}±{std['fpr']:.4f} "
                  f"{mean['total_runtime_s']/3600:.2f}±{std['total_runtime_s']/3600:.2f} "
                  f"{mean['latency_mean_ms']:.2f}±{std['latency_mean_ms']:.2f}")


def print_baseline_vs_full(df: pd.DataFrame):
    """In bảng so sánh Baseline vs Full với statistical test."""
    
    print("\n" + "="*100)
    print("📊 BASELINE vs FULL - STATISTICAL COMPARISON")
    print("="*100)
    
    for dataset in df['dataset'].unique():
        print(f"\n📁 {dataset.upper()}")
        print("-"*80)
        
        base = df[(df['dataset'] == dataset) & (df['variant'] == 'baseline')]
        full = df[(df['dataset'] == dataset) & (df['variant'] == 'full')]
        
        if base.empty or full.empty:
            continue
        
        print(f"{'Metric':<20} {'Baseline':<20} {'Full':<20} {'Improvement':<15} {'p-value':<12}")
        print("-"*80)
        
        for metric in ['auc_roc', 'f1', 'recall', 'fpr', 'total_runtime_s', 'latency_mean_ms']:
            base_mean = base[metric].mean()
            base_std = base[metric].std()
            full_mean = full[metric].mean()
            full_std = full[metric].std()
            
            # T-test
            _, p_value = stats.ttest_ind(base[metric], full[metric])
            
            delta = full_mean - base_mean
            if metric == 'fpr':
                delta = -delta  # FPR càng thấp càng tốt
            if metric in ['total_runtime_s', 'latency_mean_ms']:
                delta = -delta  # Thời gian càng thấp càng tốt
            
            print(f"{metric:<20} "
                  f"{base_mean:.4f}±{base_std:.4f} "
                  f"{full_mean:.4f}±{full_std:.4f} "
                  f"{delta:+.4f} "
                  f"{p_value:.4f}")


def main():
    print("="*80)
    print("🔬 ABLATION STUDY WITH TIMING METRICS")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    all_results = []
    
    for dataset in DATASETS:
        print(f"\n📁 DATASET: {dataset.upper()}")
        print("-"*40)
        
        base_config = Path(f"configs/{dataset}.yaml")
        if not base_config.exists():
            print(f"❌ Config not found: {base_config}")
            continue
        
        for variant in VARIANTS:
            print(f"\n  🔬 Variant: {variant}")
            
            for seed in SEEDS:
                print(f"    🌱 Seed: {seed}", end=' ')
                
                config_path = create_ablation_config(dataset, variant, seed, base_config)
                result = run_experiment(config_path)
                
                if 'error' in result:
                    print(f"❌ ERROR: {result.get('error', 'Unknown')[:50]}")
                    continue
                
                # Lưu kết quả
                m = result.get('test_metrics', {})
                timing = extract_timing_metrics(result)
                
                row = {
                    'dataset': dataset,
                    'variant': variant,
                    'seed': seed,
                    # Detection metrics
                    'auc_roc': m.get('auc_roc', 0),
                    'auc_pr': m.get('auc_pr', 0),
                    'f1': m.get('f1', 0),
                    'recall': m.get('recall', 0),
                    'precision': m.get('precision', 0),
                    'fpr': m.get('fpr', 0),
                    'recall_at_1pct': m.get('recall_at_1pct', 0),
                    **timing,
                }
                all_results.append(row)
                
                print(f"✅ F1={row['f1']:.4f}, "
                      f"Total={row['total_runtime_s']/60:.1f}m, "
                      f"Latency={row['latency_mean_ms']:.2f}ms")
    
    # ============================================================
    # PHÂN TÍCH KẾT QUẢ
    # ============================================================
    
    if not all_results:
        print("❌ No results collected!")
        return
    
    df = pd.DataFrame(all_results)
    
    # Lưu raw data
    output_dir = Path("outputs/results/ablation_timing")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / 'ablation_timing_raw.csv', index=False)
    
    # In bảng
    print_summary_table(df)
    print_timing_table(df)
    print_baseline_vs_full(df)
    
    print("\n" + "="*80)
    print("✅ ABLATION STUDY WITH TIMING COMPLETE")
    print(f"📁 Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()