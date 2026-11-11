# src/main_pipeline.py — entry point (cũng dùng được qua: python -m src.main_pipeline --config ...)
from __future__ import annotations

import argparse
import json

from src.train.pipeline_fraudgnn import run_cv_pipeline, run_pipeline
from src.utils.config import deep_update, load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="FraudGNN-RL reproduction")
    ap.add_argument("--config", default="configs/paysim.yaml")
    ap.add_argument("--cv", action="store_true", help="5-fold cross validation (paper V-A-4)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="override, ví dụ flags.rl=false graph.max_neighbors_per_node=32")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for kv in args.set:
        k, v = kv.split("=", 1)
        import yaml
        d: dict = {}
        cur = d
        parts = k.split(".")
        for p in parts[:-1]:
            cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = yaml.safe_load(v)
        cfg = deep_update(cfg, d)
    if args.seed is not None:
        cfg["seed"] = args.seed

    if args.cv:
        run_cv_pipeline(cfg, 5)
    else:
        res = run_pipeline(cfg)
        print(json.dumps({k: res[k] for k in ("model_name", "selected_threshold", "test_metrics")}, indent=2, default=float))


if __name__ == "__main__":
    main()
