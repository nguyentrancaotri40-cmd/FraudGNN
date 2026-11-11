"""Tiện ích dùng chung cho scripts (đảm bảo import được `src` khi chạy từ bất kỳ đâu)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIGS = {"paysim": "configs/paysim.yaml", "creditcard2023": "configs/creditcard2023.yaml",
           "ieee_cis": "configs/ieee_cis.yaml"}


def metrics_row(name, m):
    return {"name": name, **{k: m[k] for k in ("auc_roc", "auc_pr", "f1", "recall_at_1pct", "precision", "recall", "fpr")}}


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"[SAVE] {path}")


def print_table(rows, cols=("auc_roc", "auc_pr", "f1", "recall_at_1pct", "fpr")):
    print(f"{'name':<26}" + "".join(f"{c:>16}" for c in cols))
    for r in rows:
        print(f"{r['name']:<26}" + "".join(f"{r.get(c, float('nan')):>16.4f}" for c in cols))
