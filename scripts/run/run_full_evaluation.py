#!/usr/bin/env python3
"""
Run full evaluation: pipeline + concept drift + adversarial robustness
"""

import sys
from pathlib import Path

# Thêm root vào path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import torch

from src.utils.config import load_config
from src.train.pipeline_fraudgnn import run_pipeline
from src.eval.adversarial import evaluate_adversarial_robustness


def run_concept_drift_test_from_result(result: dict) -> dict:
    """Chạy concept drift test từ kết quả pipeline."""
    try:
        from scripts.eval.concept_drift_test import run_concept_drift_test
        
        val_scores = result.get('val_scores', [])
        val_labels = result.get('val_labels', [])
        val_timestamps = result.get('val_timestamps', np.arange(len(val_scores)))
        
        if len(val_scores) == 0:
            print("⚠️ No val_scores found, skipping concept drift test")
            return {}
        
        return run_concept_drift_test(
            scores=np.array(val_scores),
            labels=np.array(val_labels),
            timestamps=np.array(val_timestamps),
        )
    except ImportError as e:
        print(f"⚠️ Concept drift test not available: {e}")
        return {}
    except Exception as e:
        print(f"⚠️ Concept drift test failed: {e}")
        return {}


def run_full_evaluation(config_path: str, output_dir: str = "outputs/results/full"):
    """
    Chạy toàn bộ evaluation:
    1. Pipeline (train + test)
    2. Concept drift test
    3. Adversarial robustness test
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🔬 FULL EVALUATION")
    print("="*60)
    
    # ============================================================
    # 1. CHẠY PIPELINE
    # ============================================================
    print("\n[1/3] Running pipeline...")
    cfg = load_config(config_path)
    result = run_pipeline(cfg)
    
    # Lưu kết quả pipeline
    pipeline_path = output_dir / "pipeline_result.json"
    with open(pipeline_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✅ Pipeline result saved: {pipeline_path}")
    
    # ============================================================
    # 2. CONCEPT DRIFT TEST
    # ============================================================
    print("\n[2/3] Running concept drift test...")
    drift_results = run_concept_drift_test_from_result(result)
    
    if drift_results:
        drift_path = output_dir / "concept_drift_results.json"
        with open(drift_path, 'w') as f:
            json.dump(drift_results, f, indent=2)
        print(f"✅ Concept drift results saved: {drift_path}")
    else:
        print("⚠️ Concept drift test skipped or failed")
    
    # ============================================================
    # 3. ADVERSARIAL ROBUSTNESS TEST
    # ============================================================
    print("\n[3/3] Running adversarial robustness test...")
    
    # Lấy model từ result (cần lưu model trong pipeline)
    # Cách 1: Nếu pipeline đã lưu model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adv_results = {}
    
    # Cách 2: Load checkpoint nếu có
    checkpoint_path = Path("outputs/checkpoints/tssgc_classifier.pt")
    if checkpoint_path.exists():
        try:
            from src.models.fraudgnn_rl import FraudGNNRL
            from src.graph.build_graph import load_graph
            
            ckpt = torch.load(checkpoint_path, map_location=device)
            model = FraudGNNRL(
                in_dim=ckpt['in_dim'],
                hidden_dim=64,
                num_layers=3,
            ).to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            
            # Load graph
            test_graph = load_graph("data/graphs/test_graph.pkl")
            
            adv_results = evaluate_adversarial_robustness(
                model=model,
                test_graph=test_graph,
                device=device,
                attacks=['fgsm', 'random_noise', 'label_flipping'],
            )
            
            adv_path = output_dir / "adversarial_results.json"
            with open(adv_path, 'w') as f:
                json.dump(adv_results, f, indent=2)
            print(f"✅ Adversarial results saved: {adv_path}")
        except Exception as e:
            print(f"⚠️ Adversarial test failed: {e}")
    else:
        print("⚠️ No checkpoint found, skipping adversarial test")
    
    # ============================================================
    # 4. TỔNG HỢP KẾT QUẢ
    # ============================================================
    full_result = {
        'pipeline': result,
        'concept_drift': drift_results,
        'adversarial_robustness': adv_results,
    }
    
    full_path = output_dir / "full_evaluation.json"
    with open(full_path, 'w') as f:
        json.dump(full_result, f, indent=2)
    print(f"\n✅ Full evaluation saved: {full_path}")
    
    # ============================================================
    # 5. PRINT SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    
    m = result.get('test_metrics', {})
    print(f"\n📈 Detection Performance:")
    print(f"  AUC-ROC: {m.get('auc_roc', 0):.4f}")
    print(f"  AUC-PR:  {m.get('auc_pr', 0):.4f}")
    print(f"  F1:      {m.get('f1', 0):.4f}")
    print(f"  Recall:  {m.get('recall', 0):.4f}")
    print(f"  FPR:     {m.get('fpr', 0):.4f}")
    
    if drift_results:
        d = drift_results.get('drift_metrics', {})
        print(f"\n📉 Concept Drift:")
        print(f"  Initial F1: {d.get('initial_f1', 0):.4f}")
        print(f"  Final F1:   {d.get('final_f1', 0):.4f}")
        print(f"  F1 Decay:   {d.get('f1_decay_percent', 0):.2f}%")
        print(f"  Windows:    {d.get('num_windows', 0)}")
    
    if adv_results:
        print(f"\n🛡️ Adversarial Robustness (F1 drop %):")
        for attack, res in adv_results.items():
            if attack != 'baseline':
                print(f"  {attack}: {res.get('f1_drop_percent', 0):.2f}%")
    
    print("\n" + "="*60)
    print("✅ Full evaluation complete!")
    print("="*60)
    
    return full_result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, default='outputs/results/full')
    args = parser.parse_args()
    
    run_full_evaluation(args.config, args.output)