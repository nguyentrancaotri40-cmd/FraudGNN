# ============================================================
# src/main_pipeline.py
# Entry point cho FraudGNN-RL pipeline
# ============================================================

from __future__ import annotations

import argparse
import json
from src.utils.config import load_config
from src.train.pipeline_fraudgnn import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FraudGNN-RL pipeline with ablation flags"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/paysim.yaml", 
        help="Path to YAML config"
    )
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    result = run_pipeline(cfg)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()