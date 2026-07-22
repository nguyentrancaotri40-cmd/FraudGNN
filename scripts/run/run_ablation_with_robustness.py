# scripts/run/run_ablation_with_robustness.py
#!/usr/bin/env python3
"""
Full evaluation pipeline: Detection + Concept Drift + Adversarial Robustness
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Thêm root path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import classification_metrics
from src.eval.adversarial import evaluate_adversarial_robustness
from scripts.eval.concept_drift_test import run_concept_drift_test


def run_pipeline(config_path: str, output_dir: str) -> dict:
    """Run main pipeline and return results."""
    import subprocess
    import json
    import tempfile
    import os
    
    cmd = [
        sys.executable,
        '-m', 'src.main_pipeline',
        '--config', config_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse JSON từ output
    output = result.stdout
    start = output.find('{')
    end = output.rfind('}') + 1
    
    if start >= 0 and end > start:
        data = json.loads(output[start:end])
        
        # Lưu kết quả
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / 'pipeline_result.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        return data
    
    return {'error': 'No JSON found', 'stdout': output}


def run_full_evaluation(config_path: str, output_dir: str):
    """Run full evaluation: pipeline + concept drift + adversarial."""
    
    print("="*60)
    print("🔬 FULL EVALUATION")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # ============================================================
    # 1. Run pipeline
    # ============================================================
    print("\n[1/3] Running pipeline...")
    pipeline_result = run_pipeline(config_path, str(output_dir))
    results['pipeline'] = pipeline_result
    
    if 'error' in pipeline_result:
        print(f"❌ Pipeline failed: {pipeline_result.get('error')}")
        return results
    
    # ============================================================
    # 2. Concept drift test
    # ============================================================
    print("\n[2/3] Running concept drift test...")
    try:
        # Lấy scores và labels từ pipeline result
        test_scores = pipeline_result.get('test_scores', [])
        test_labels = pipeline_result.get('test_labels', [])
        test_timestamps = pipeline_result.get('test_timestamps', [])
        
        if test_scores and test_labels:
            concept_drift_result = run_concept_drift_test(
                scores=test_scores,
                labels=test_labels,
                timestamps=test_timestamps,
                output_dir=str(output_dir / 'concept_drift')
            )
            results['concept_drift'] = concept_drift_result
            print(f"✅ Concept drift test completed")
        else:
            print("⚠️ No test scores/labels found, skipping concept drift test")
            results['concept_drift'] = {'error': 'No test data'}
            
    except Exception as e:
        print(f"⚠️ Concept drift test failed: {e}")
        results['concept_drift'] = {'error': str(e)}
    
    # ============================================================
    # 3. Adversarial robustness test
    # ============================================================
    print("\n[3/3] Running adversarial robustness test...")
    try:
        checkpoint_path = Path("outputs/checkpoints/tssgc_classifier.pt")
        graph_path = Path("data/graphs/test_graph.pkl")
        
        if checkpoint_path.exists() and graph_path.exists():
            adversarial_result = evaluate_adversarial_robustness(
                checkpoint_path=str(checkpoint_path),
                graph_path=str(graph_path),
                output_dir=str(output_dir / 'adversarial')
            )
            results['adversarial'] = adversarial_result
            print(f"✅ Adversarial robustness test completed")
        else:
            print(f"⚠️ Checkpoint or graph not found: {checkpoint_path} or {graph_path}")
            results['adversarial'] = {'error': 'Checkpoint or graph not found'}
            
    except Exception as e:
        print(f"⚠️ Adversarial robustness test failed: {e}")
        results['adversarial'] = {'error': str(e)}
    
    # ============================================================
    # 4. Summary
    # ============================================================
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    
    test_metrics = pipeline_result.get('test_metrics', {})
    if test_metrics:
        print(f"\n📈 Detection Performance:")
        print(f"  AUC-ROC: {test_metrics.get('auc_roc', 0):.4f}")
        print(f"  AUC-PR:  {test_metrics.get('auc_pr', 0):.4f}")
        print(f"  F1:      {test_metrics.get('f1', 0):.4f}")
        print(f"  Recall:  {test_metrics.get('recall', 0):.4f}")
        print(f"  FPR:     {test_metrics.get('fpr', 0):.4f}")
        print(f"  TP:      {test_metrics.get('tp', 0)}")
        print(f"  FP:      {test_metrics.get('fp', 0)}")
        print(f"  TN:      {test_metrics.get('tn', 0)}")
        print(f"  FN:      {test_metrics.get('fn', 0)}")
    
    # ✅ FIX: Kiểm tra concept_drift có tồn tại và có recall_at_1pct không
    concept_drift = results.get('concept_drift', {})
    if concept_drift and 'recall_at_1pct' in concept_drift:
        print(f"\n🔄 Concept Drift:")
        print(f"  Recall@1%: {concept_drift.get('recall_at_1pct', 0):.4f}")
        print(f"  Drift severity: {concept_drift.get('drift_severity', 0):.4f}")
    
    adversarial = results.get('adversarial', {})
    if adversarial and 'error' not in adversarial:
        print(f"\n🛡️ Adversarial Robustness:")
        print(f"  Clean accuracy: {adversarial.get('clean_accuracy', 0):.4f}")
        print(f"  Perturbed accuracy: {adversarial.get('perturbed_accuracy', 0):.4f}")
        print(f"  Robustness drop: {adversarial.get('robustness_drop', 0):.4f}")
    
    # ============================================================
    # 5. Save full results
    # ============================================================
    full_result = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'config': config_path,
        'results': results,
    }
    
    with open(output_dir / 'full_evaluation.json', 'w') as f:
        json.dump(full_result, f, indent=2)
    
    print(f"\n✅ Full evaluation saved: {output_dir / 'full_evaluation.json'}")
    print("="*60)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, default='outputs/results/full_evaluation')
    args = parser.parse_args()
    
    run_full_evaluation(args.config, args.output)


if __name__ == '__main__':
    main()