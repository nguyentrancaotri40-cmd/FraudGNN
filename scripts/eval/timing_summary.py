"""Tổng hợp runtime từ các file kết quả: python scripts/eval/timing_summary.py outputs/results/*_metrics.json"""
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    keys = ("preprocessing_sec", "graph_building_sec", "gnn_training_sec", "rl_training_sec", "inference_sec", "total_runtime_sec")
    print(f"{'file':<44}" + "".join(f"{k[:-4]:>16}" for k in keys))
    for f in sys.argv[1:]:
        rt = json.load(open(f)).get("runtime", {})
        print(f"{Path(f).name:<44}" + "".join(f"{rt.get(k, 0):>16.2f}" for k in keys))
