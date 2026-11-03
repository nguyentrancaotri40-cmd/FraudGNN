#!/usr/bin/env python3
"""
Chạy 5-fold cross-validation (Paper V-A-4).
"""
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.train.pipeline_fraudgnn import run_cv_pipeline

def main():
    # Map tên dataset sang file config tương ứng
    datasets = {
        'paysim': 'configs/paysim.yaml',
        'creditcard2023': 'configs/creditcard2023.yaml',
        'ieee_cis': 'configs/ieee_cis.yaml',
    }
    results = {}

    for dataset, config_path in datasets.items():
        if not Path(config_path).exists():
            print(f"[WARN] Config không tồn tại: {config_path}, bỏ qua.")
            continue

        print(f"\n{'='*80}")
        print(f"[CHART] 5-FOLD CV: {dataset.upper()}")
        print(f"{'='*80}")

        cfg = load_config(config_path)
        
        # Hàm mới chỉ nhận (cfg, n_folds)
        result = run_cv_pipeline(cfg, n_folds=5)
        results[dataset] = result

        # In kết quả
        print(f"\n[UP] {dataset.upper()} (5-fold CV):")
        for metric in ['auc_roc', 'auc_pr', 'f1', 'recall', 'fpr']:
            mean = result.get(f'{metric}_mean', 0)
            std = result.get(f'{metric}_std', 0)
            ci = result.get(f'{metric}_ci95', 0) # Code mới dùng key _ci95
            print(f"  {metric}: {mean:.4f} ± {std:.4f} (95% CI: {mean-ci:.4f} - {mean+ci:.4f})")

    Path('outputs/results').mkdir(parents=True, exist_ok=True)
    with open('outputs/results/cv_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Kết quả lưu tại: outputs/results/cv_results.json")

if __name__ == '__main__':
    main()