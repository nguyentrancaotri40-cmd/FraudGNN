"""Vẽ Fig. 3-style (ablation) hoặc Fig. 4-style (imbalance) từ file JSON kết quả.
   python scripts/eval/plot.py --ablation outputs/ablation/<exp>_ablation.json
   python scripts/eval/plot.py --imbalance outputs/results/<exp>_full_evaluation.json"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation"); ap.add_argument("--imbalance")
    ap.add_argument("--out", default="outputs/figures")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.ablation:
        rows = json.load(open(a.ablation))["rows"]
        fig, ax = plt.subplots(1, 4, figsize=(18, 4))
        for axi, k in zip(ax, ("auc_roc", "auc_pr", "f1", "recall_at_1pct")):
            axi.bar([r["name"] for r in rows], [r[k] for r in rows]); axi.set_title(k); axi.tick_params(axis="x", rotation=60)
        fig.tight_layout(); fig.savefig(f"{a.out}/ablation.png", dpi=150); print(f"[SAVE] {a.out}/ablation.png")
    if a.imbalance:
        d = json.load(open(a.imbalance))["imbalance"]
        ratios = sorted(d["auc_pr"], key=float)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for m in d["auc_pr"][ratios[0]]:
            ax[0].plot([float(r) for r in ratios], [d["auc_pr"][r][m] for r in ratios], marker="o", label=m)
            ax[1].plot([float(r) for r in ratios], [d["degradation"][r][m] for r in ratios], marker="o", label=m)
        ax[0].set_title("AUC-PR vs fraud ratio (%)"); ax[1].set_title("Degradation vs 2.0%"); ax[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{a.out}/imbalance.png", dpi=150); print(f"[SAVE] {a.out}/imbalance.png")
