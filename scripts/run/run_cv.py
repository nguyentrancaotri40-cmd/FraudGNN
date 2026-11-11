"""5-fold cross-validation (paper V-A-4).  python scripts/run/run_cv.py --config configs/paysim.yaml"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.common import ROOT  # noqa
from src.train.pipeline_fraudgnn import run_cv_pipeline
from src.utils.config import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    run_cv_pipeline(load_config(a.config), a.folds)
