"""Đánh giá đầy đủ theo paper cho 1 dataset: FraudGNN-RL + 8 baseline (Table 2) + (tuỳ chọn) imbalance (Fig. 4).
   python scripts/run/run_full_evaluation.py --config configs/paysim.yaml [--baselines xgboost gcn] [--imbalance]"""
import argparse
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.common import metrics_row, print_table, write_json  # noqa
from src.data.load_data import load_dataset
from src.data.split import resample_fraud_ratio, split_dataframe
from src.models.baselines import BASELINES, run_baseline
from src.train.pipeline_fraudgnn import get_device, prepare_splits, run_pipeline_on_split
from src.utils.config import load_config

RATIOS = [0.1, 0.5, 1.0, 2.0]  # paper Fig. 4


def one_round(df, cfg, models, seed):
    tr, va, te = split_dataframe(df, cfg)
    prep = prepare_splits(tr, va, te, cfg)
    out = {"FraudGNN-RL": run_pipeline_on_split(tr, va, te, cfg, prepared=prep, seed=seed)["test_metrics"]}
    for b in models:
        try:
            out[b] = run_baseline(b, prep, get_device(cfg), seed)["test"]
        except Exception as e:  # noqa
            print(f"[WARN] baseline {b} lỗi: {e!r}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--baselines", nargs="*", default=BASELINES)
    ap.add_argument("--imbalance", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = load_config(a.config)
    ds, name = cfg["dataset"], cfg["experiment"]["name"]
    df = load_dataset(cfg)
    res = one_round(df, cfg, a.baselines, a.seed)
    print_table([metrics_row(k, v) for k, v in res.items()])
    report = {"table2": res}
    if a.imbalance:  # Fig. 4: AUC-PR theo tỉ lệ fraud + degradation = (S(2%) - S(x%)) / S(2%)
        curves = {}
        for r in RATIOS:
            sub = resample_fraud_ratio(df, r, ds["label_col"], ds.get("time_col"), a.seed)
            curves[r] = {k: v["auc_pr"] for k, v in one_round(sub, copy.deepcopy(cfg), a.baselines, a.seed).items()}
        ref = curves[2.0]
        report["imbalance"] = {"auc_pr": curves,
                               "degradation": {r: {k: (ref[k] - curves[r][k]) / max(ref[k], 1e-8) for k in ref} for r in RATIOS}}
    write_json(report, f"outputs/results/{name}_full_evaluation.json")
