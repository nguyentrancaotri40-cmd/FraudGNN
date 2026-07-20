#!/usr/bin/env python3
"""
Sweep threshold để tìm ngưỡng tối ưu cho FraudGNN-RL / FraudGNN-RL+
"""

import argparse
import pickle
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fraudgnn_rl import FraudGNNRL
from src.eval.evaluate import predict_scores
from src.eval.metrics import classification_metrics


def load_graph(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def load_model(checkpoint_path: Path, device: str):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    
    model = FraudGNNRL(
        in_dim=int(ckpt["in_dim"]),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        num_node_types=int(model_cfg.get("num_node_types", 1)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def fmt(x, digits=4):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def print_table(rows):
    headers = ["thr", "auc_pr", "f1", "precision", "recall", "fpr"]
    
    formatted = []
    for r in rows:
        formatted.append({
            "thr": fmt(r["threshold"], 3),
            "auc_pr": fmt(r["auc_pr"], 4),
            "f1": fmt(r["f1"], 4),
            "precision": fmt(r["precision"], 4),
            "recall": fmt(r["recall"], 4),
            "fpr": fmt(r["fpr"], 5),
        })
    
    widths = {h: max(len(h), max(len(row[h]) for row in formatted)) for h in headers}
    
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in formatted:
        print(" | ".join(row[h].ljust(widths[h]) for h in headers))


def best_views(rows):
    def best_by(key, reverse=True):
        return sorted(rows, key=lambda r: float(r[key]), reverse=reverse)[0]
    
    print("\n🎯 Best threshold views:")
    best_f1 = best_by("f1")
    best_recall = best_by("recall")
    best_auc_pr = best_by("auc_pr")
    lowest_fpr = best_by("fpr", reverse=False)
    
    print(f"  Best F1:      thr={best_f1['threshold']:.3f}, f1={best_f1['f1']:.4f}")
    print(f"  Best Recall:  thr={best_recall['threshold']:.3f}, recall={best_recall['recall']:.4f}")
    print(f"  Best AUC-PR:  thr={best_auc_pr['threshold']:.3f}, auc_pr={best_auc_pr['auc_pr']:.4f}")
    print(f"  Lowest FPR:   thr={lowest_fpr['threshold']:.3f}, fpr={lowest_fpr['fpr']:.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pt file')
    parser.add_argument('--graph', type=str, required=True, help='Path to graph .pkl file')
    parser.add_argument('--start', type=float, default=0.05)
    parser.add_argument('--end', type=float, default=0.60)
    parser.add_argument('--step', type=float, default=0.005)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = load_model(Path(args.checkpoint), device=device)
    graph = load_graph(Path(args.graph))
    
    scores, labels = predict_scores(model, graph, device=device)
    
    thresholds = np.arange(args.start, args.end + 1e-9, args.step)
    rows = [classification_metrics(labels, scores, threshold=float(th)) for th in thresholds]
    
    print("\n🔍 THRESHOLD SWEEP")
    print_table(rows)
    best_views(rows)


if __name__ == "__main__":
    main()