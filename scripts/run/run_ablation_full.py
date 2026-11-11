"""Ablation trên cả 3 dataset x nhiều seed -> bảng mean +/- std.
   python scripts/run/run_ablation_full.py --seeds 42 43 44 [--datasets paysim creditcard2023 ieee_cis]"""
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.run.run_ablation import run_ablation  # noqa
from scripts.utils.common import CONFIGS, write_json  # noqa
from src.utils.config import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(CONFIGS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    ap.add_argument("--config-dir-override", default=None, help="vd: configs/*_sample.yaml để chạy mẫu")
    a = ap.parse_args()
    summary = {}
    for ds in a.datasets:
        path = CONFIGS[ds] if not a.config_dir_override else a.config_dir_override.replace("*", ds)
        per_seed = [run_ablation(load_config(path), seed=s, out_dir=f"outputs/ablation/seed{s}")[0] for s in a.seeds]
        agg = {}
        for i, r in enumerate(per_seed[0]):
            agg[r["name"]] = {m: [float(np.mean([ps[i][m] for ps in per_seed])), float(np.std([ps[i][m] for ps in per_seed]))]
                              for m in ("auc_roc", "auc_pr", "f1", "recall_at_1pct", "fpr")}
        summary[ds] = agg
    write_json(summary, "outputs/ablation/ablation_full_summary.json")
