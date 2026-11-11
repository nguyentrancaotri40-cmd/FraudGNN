"""Thời gian từng giai đoạn cho mỗi biến thể ablation (paper không báo latency chi tiết -> đây là phần đo thêm).
   python scripts/run/run_ablation_timing.py --config configs/paysim.yaml"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.run.run_ablation import run_ablation  # noqa
from scripts.utils.common import write_json  # noqa
from src.utils.config import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = load_config(a.config)
    _, full = run_ablation(cfg, seed=a.seed, out_dir="outputs/ablation_timing")
    keys = ("preprocessing_sec", "graph_building_sec", "gnn_training_sec", "rl_training_sec", "inference_sec", "total_runtime_sec")
    print(f"{'variant':<14}" + "".join(f"{k[:-4]:>18}" for k in keys))
    out = {}
    for n, r in full.items():
        out[n] = {k: r["runtime"].get(k, 0.0) for k in keys}
        print(f"{n:<14}" + "".join(f"{out[n][k]:>18.2f}" for k in keys))
    write_json(out, f"outputs/ablation_timing/{cfg['experiment']['name']}_timing.json")
