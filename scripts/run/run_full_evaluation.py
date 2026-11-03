#!/usr/bin/env python3
"""
Run pipeline + Concept Drift test.
(Đã bỏ Adversarial vì chưa có trong clean-room)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
from src.utils.config import load_config
from src.train.pipeline_fraudgnn import run_pipeline
from src.eval.concept_drift import run_concept_drift_test


def run_full_evaluation(config_path: str, output_dir: str = "outputs/results/full"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("[LAB] FULL EVALUATION (Pipeline + Concept Drift)")
    print("="*60)

    # 1. CHẠY PIPELINE
    print("\n[1/2] Running pipeline...")
    cfg = load_config(config_path)
    result = run_pipeline(cfg)

    with open(output_dir / "pipeline_result.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[OK] Pipeline result saved: {output_dir / 'pipeline_result.json'}")

    # 2. CONCEPT DRIFT TEST
    print("\n[2/2] Running concept drift test...")
    drift_results = {}
    try:
        # [OK] FIX: Dùng test_scores thay vì val_scores
        scores = result.get('test_scores', [])
        labels = result.get('test_labels', [])
        
        if len(scores) > 0:
            drift_results = run_concept_drift_test(
                scores=np.array(scores),
                labels=np.array(labels),
                timestamps=np.arange(len(scores)), # Dùng index nếu không có timestamp
                output_dir=str(output_dir)
            )
            with open(output_dir / "concept_drift_results.json", 'w') as f:
                json.dump(drift_results, f, indent=2)
            print(f"[OK] Concept drift saved: {output_dir / 'concept_drift_results.json'}")
        else:
            print("[WARN] Không có test_scores, bỏ qua Concept Drift.")
    except Exception as e:
        print(f"[WARN] Concept drift test failed: {e}")

    # 3. PRINT SUMMARY
    print("\n" + "="*60)
    print("[CHART] EVALUATION SUMMARY")
    print("="*60)
    
    m = result.get('test_metrics', {})
    print(f"\n[UP] Detection Performance:")
    print(f"  AUC-ROC: {m.get('auc_roc', 0):.4f}")
    print(f"  AUC-PR:  {m.get('auc_pr', 0):.4f}")
    print(f"  F1:      {m.get('f1', 0):.4f}")
    print(f"  Recall:  {m.get('recall', 0):.4f}")
    print(f"  FPR:     {m.get('fpr', 0):.4f}")

    if drift_results and 'f1_decay_percent' in drift_results:
         print(f"\n[DOWN] Concept Drift (test set):")
         print(f"  Initial F1: {drift_results.get('initial_f1', 0):.4f}")
         print(f"  Final F1:   {drift_results.get('final_f1', 0):.4f}")
         print(f"  F1 Decay:   {drift_results.get('f1_decay_percent', 0):.2f}%")
    
    print("\n" + "="*60)
    print("[OK] Full evaluation complete!")
    print("="*60)

    return {"pipeline": result, "concept_drift": drift_results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs/results/full')
    args = parser.parse_args()
    run_full_evaluation(args.config, args.output)