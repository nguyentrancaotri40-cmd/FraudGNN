"""Concept drift trên mô hình vừa huấn luyện: python scripts/eval/concept_drift_test.py --config configs/paysim.yaml --windows 5"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.common import write_json  # noqa
from src.eval.concept_drift import run_concept_drift_test
from src.train.pipeline_fraudgnn import run_pipeline
from src.utils.config import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--windows", type=int, default=5)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if not cfg["dataset"].get("time_col"):
        print("[WARN] Dataset không có thời gian: cửa sổ theo thứ tự hàng không phản ánh drift thật.")
    res, art = run_pipeline(cfg, return_artifacts=True)
    d = run_concept_drift_test(art["test_scores"], art["test_labels"], art["threshold"], a.windows, art["y_pred_test"])
    for w in d["windows"]:
        print(f"window {w['window']}: n={w['n']} fraud={w['fraud_rate']:.4%} AUC-ROC={w['auc_roc']:.4f} AUC-PR={w['auc_pr']:.4f} F1={w['f1']:.4f}")
    write_json(d, f"outputs/results/{cfg['experiment']['name']}_concept_drift.json")
