# scripts/run/run_cv.py
#!/usr/bin/env python3
"""
Chạy 5-fold cross-validation + 3 seeds cho tất cả datasets.
✅ GIỐNG PAPER 100%
"""

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.train.pipeline_fraudgnn import run_cv_pipeline

def main():
    datasets = ['paysim', 'creditcard2023', 'ieee_cis']
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"📊 5-FOLD CV: {dataset.upper()}")
        print(f"{'='*80}")
        
        cfg = load_config(f'configs/{dataset}.yaml')
        
        # ✅ Bật CV
        cfg['split']['strategy'] = 'temporal'  # TimeSeriesSplit
        cfg['split']['n_folds'] = 5
        
        result = run_cv_pipeline(cfg, n_folds=5, seeds=[42, 123, 2024])
        results[dataset] = result
        
        # In kết quả
        print(f"\n📈 {dataset.upper()} (5-fold CV + 3 seeds):")
        for metric in ['auc_roc', 'auc_pr', 'f1', 'recall', 'fpr']:
            mean = result[f'{metric}_mean']
            std = result[f'{metric}_std']
            ci = result[f'{metric}_ci_95']
            print(f"  {metric}: {mean:.4f} ± {std:.4f} (95% CI: {mean-ci:.4f} - {mean+ci:.4f})")
    
    # Lưu kết quả
    Path('outputs/results').mkdir(parents=True, exist_ok=True)
    with open('outputs/results/cv_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Kết quả lưu tại: outputs/results/cv_results.json")

if __name__ == '__main__':
    main()