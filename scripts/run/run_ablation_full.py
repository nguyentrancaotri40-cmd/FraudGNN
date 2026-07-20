#!/usr/bin/env python3
"""
Ablation Study Full với:
- 6 variants × 3 seeds × 3 datasets
- Epoch-level logging (loss, auc, f1)
- Latency & Inference time
- Statistical analysis (mean, std, CI)
- Timing breakdown
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
    """Tạo config với flags + seed."""
    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['flags'] = VARIANT_FLAGS[variant]
    config['experiment']['name'] = f"{dataset}_{variant}_seed{seed}"
    config['experiment']['seed'] = seed
    config['experiment']['ablation'] = variant
    config['dataset']['random_state'] = seed
    
    # Thêm logging chi tiết
    config['logging'] = {
        'log_epoch_metrics': True,
        'log_interval': 1,
    }
    
    output_path = Path(f"configs/ablation/{dataset}/{variant}_seed{seed}.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f)
    
    return output_path


def run_experiment(config_path):
    """Chạy experiment và thu thập đầy đủ metrics."""
    
    start_time = time.perf_counter()
    cmd = [sys.executable, '-m', 'src.main_pipeline', '--config', str(config_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    total_time = time.perf_counter() - start_time
    
    # ✅ LUÔN TRẢ VỀ DICT, không trả về None
    try:
        output = result.stdout
        start = output.find('{')
        end = output.rfind('}') + 1
        if start >= 0 and end > start:
            data = json.loads(output[start:end])
            
            runtime = data.get('runtime', {})
            num_samples = data.get('num_samples', 0)
            inference_time = runtime.get('inference_sec', 0)
            data['_inference_latency_ms'] = (inference_time / num_samples * 1000) if num_samples > 0 else 0
            data['_total_time_sec'] = total_time
            data['_returncode'] = result.returncode
            
            # ✅ Trích xuất epoch history
            if 'federated_history' in data:
                epoch_data = []
                for round_info in data['federated_history']:
                    for client in round_info.get('client_metrics', []):
                        epoch_data.append({
                            'round': round_info['round'],
                            'loss': client.get('loss', 0),
                            'auc_roc': client.get('auc_roc', 0),
                            'auc_pr': client.get('auc_pr', 0),
                            'f1': client.get('f1', 0),
                        })
                data['_epoch_history'] = epoch_data
            
            # ✅ Thêm memory metrics nếu có
            if 'memory' in data:
                memory = data['memory']
                data['_ram_used_gb'] = memory.get('ram_used_gb', 0)
                data['_vram_allocated_gb'] = memory.get('vram_allocated_gb', 0)
            
            return data
    except Exception as e:
        return {'error': str(e), 'stderr': result.stderr, 'stdout': result.stdout[:500]}
    
    # ✅ Nếu không parse được JSON, trả về dict
    return {'error': 'No JSON found', 'stdout': result.stdout[:500], 'stderr': result.stderr[:500]}


def analyze_results(results_df):
    """Phân tích thống kê và tạo báo cáo."""
    
    # ✅ Thêm throughput và runtime_per_sample vào summary
    summary = results_df.groupby(['dataset', 'variant']).agg({
        'auc_roc': ['mean', 'std', lambda x: stats.sem(x).item()],
        'f1': ['mean', 'std', lambda x: stats.sem(x).item()],
        'recall': ['mean', 'std', lambda x: stats.sem(x).item()],
        'fpr': ['mean', 'std', lambda x: stats.sem(x).item()],
        'total_runtime_sec': ['mean', 'std'],
        'inference_latency_ms': ['mean', 'std'],
        'runtime_per_sample_ms': ['mean', 'std'],
        'throughput_samples_s': ['mean', 'std'],
    }).round(4)
    
    # Confidence Interval (95%)
    for metric in ['auc_roc', 'f1', 'recall', 'fpr']:
        mean = results_df.groupby(['dataset', 'variant'])[metric].mean()
        sem = results_df.groupby(['dataset', 'variant'])[metric].apply(lambda x: stats.sem(x))
        ci_lower = mean - 1.96 * sem
        ci_upper = mean + 1.96 * sem
        summary[(metric, 'ci_lower')] = ci_lower.round(4)
        summary[(metric, 'ci_upper')] = ci_upper.round(4)
    
    return summary


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
                  f"{mean['graph_building_sec']:.0f}±{std['graph_building_sec']:.0f} "
                  f"{mean['federated_training_sec']:.0f}±{std['federated_training_sec']:.0f} "
                  f"{mean['dqn_training_sec']:.0f}±{std['dqn_training_sec']:.0f} "
                  f"{mean['total_runtime_sec']:.0f}±{std['total_runtime_sec']:.0f} "
                  f"{mean['total_runtime_sec']/3600:.2f}±{std['total_runtime_sec']/3600:.2f} "
                  f"{mean['inference_latency_ms']:.2f}±{std['inference_latency_ms']:.2f} "
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
                  f"{mean['total_runtime_sec']/3600:.2f}±{std['total_runtime_sec']/3600:.2f} "
                  f"{mean['inference_latency_ms']:.2f}±{std['inference_latency_ms']:.2f}")


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
            print(f"  ⚠️ Missing data for {dataset}")
            continue
        
        print(f"{'Metric':<20} {'Baseline':<20} {'Full':<20} {'Improvement':<15} {'p-value':<12}")
        print("-"*80)
        
        for metric in ['auc_roc', 'f1', 'recall', 'fpr', 'total_runtime_sec', 'inference_latency_ms']:
            base_mean = base[metric].mean()
            base_std = base[metric].std()
            full_mean = full[metric].mean()
            full_std = full[metric].std()
            
            # T-test
            _, p_value = stats.ttest_ind(base[metric], full[metric])
            
            delta = full_mean - base_mean
            
            # Điều chỉnh delta cho metrics càng thấp càng tốt
            if metric in ['fpr', 'total_runtime_sec', 'inference_latency_ms']:
                delta = -delta
                improvement = "↓" if delta > 0 else "↑"
            else:
                improvement = "↑" if delta > 0 else "↓"
            
            print(f"{metric:<20} "
                  f"{base_mean:.4f}±{base_std:.4f} "
                  f"{full_mean:.4f}±{full_std:.4f} "
                  f"{improvement}{abs(delta):.4f} "
                  f"{p_value:.4f}")


def plot_epoch_curves(epoch_data, dataset, variant):
    """Vẽ learning curves từ epoch data."""
    import matplotlib.pyplot as plt
    
    df = pd.DataFrame(epoch_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(df['round'], df['loss'], marker='o')
    axes[0, 0].set_title('Loss over Rounds')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # AUC-ROC
    axes[0, 1].plot(df['round'], df['auc_roc'], marker='o', color='green')
    axes[0, 1].set_title('AUC-ROC over Rounds')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('AUC-ROC')
    axes[0, 1].grid(True)
    axes[0, 1].set_ylim(0.8, 1.0)
    
    # AUC-PR
    axes[1, 0].plot(df['round'], df['auc_pr'], marker='o', color='orange')
    axes[1, 0].set_title('AUC-PR over Rounds')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('AUC-PR')
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim(0.8, 1.0)
    
    # F1
    axes[1, 1].plot(df['round'], df['f1'], marker='o', color='red')
    axes[1, 1].set_title('F1 over Rounds')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('F1')
    axes[1, 1].grid(True)
    axes[1, 1].set_ylim(0.8, 1.0)
    
    plt.suptitle(f'{dataset.upper()} - {variant} - Learning Curves')
    plt.tight_layout()
    
    output_dir = Path(f'outputs/figures/epoch_curves/{dataset}')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{variant}_curves.png', dpi=150)
    plt.close()
    
    print(f"✅ Saved: {output_dir / f'{variant}_curves.png'}")


def main():
    print("="*80)
    print("🔬 ABLATION STUDY FULL - COMPLETE ANALYSIS")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    all_results = []
    all_epoch_data = []
    
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
                    print("❌ ERROR")
                    continue
                
                # Lưu kết quả
                m = result.get('test_metrics', {})
                runtime = result.get('runtime', {})
                
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
                    # Runtime breakdown
                    'graph_building_sec': runtime.get('graph_building_sec', 0),
                    'federated_training_sec': runtime.get('federated_training_sec', 0),
                    'dqn_training_sec': runtime.get('dqn_training_sec', 0),
                    'inference_sec': runtime.get('inference_sec', 0),
                    'total_runtime_sec': runtime.get('total_runtime_sec', 0),
                    'runtime_per_sample_ms': runtime.get('runtime_per_sample_sec', 0) * 1000,
                    'throughput_samples_s': runtime.get('throughput_samples_per_sec', 0),
                    # Latency
                    'inference_latency_ms': result.get('_inference_latency_ms', 0),
                }
                all_results.append(row)
                
                # Lưu epoch history
                if '_epoch_history' in result:
                    for ep in result['_epoch_history']:
                        ep['dataset'] = dataset
                        ep['variant'] = variant
                        ep['seed'] = seed
                        all_epoch_data.append(ep)
                
                print(f"✅ F1={row['f1']:.4f}, "
                      f"Recall={row['recall']:.4f}, "
                      f"Total={row['total_runtime_sec']/60:.1f}m, "
                      f"Latency={row['inference_latency_ms']:.2f}ms")
    
    # ============================================================
    # PHÂN TÍCH KẾT QUẢ
    # ============================================================
    
    if not all_results:
        print("❌ No results collected!")
        return
    
    # 1. DataFrame
    df = pd.DataFrame(all_results)
    df.to_csv('outputs/results/ablation_full_raw.csv', index=False)
    
    # 2. Thống kê
    summary = analyze_results(df)
    summary.to_csv('outputs/results/ablation_full_summary.csv')
    
    # 3. In bảng
    print_summary_table(df)
    print_timing_table(df)
    print_baseline_vs_full(df)
    
    # 4. Vẽ epoch curves
    print("\n" + "="*80)
    print("📈 GENERATING EPOCH CURVES")
    print("="*80)
    
    epoch_df = pd.DataFrame(all_epoch_data)
    for dataset in DATASETS:
        for variant in VARIANTS:
            ep_data = epoch_df[(epoch_df['dataset'] == dataset) & (epoch_df['variant'] == variant)]
            if len(ep_data) > 0:
                plot_epoch_curves(ep_data, dataset, variant)
    
    print("\n" + "="*80)
    print("✅ ABLATION STUDY COMPLETE")
    print(f"📁 Results saved to: outputs/results/")
    print("="*80)


if __name__ == "__main__":
    main()