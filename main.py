# main.py
import torch
print(f"[GPU] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[GPU] Device: {torch.cuda.get_device_name(0)}")
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.utils.config import load_config
from src.train.pipeline_fraudgnn import run_pipeline, run_cv_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--cv", action="store_true", help="Chạy 5-fold Cross Validation")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.cv:
        print("[START] Running 5-Fold Cross Validation...")
        run_cv_pipeline(cfg, n_folds=5)
    else:
        print("[START] Running Single Split Pipeline...")
        run_pipeline(cfg)

if __name__ == "__main__":
    main()