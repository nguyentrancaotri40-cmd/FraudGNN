"""Ablation của paper (V-B-2): Full, w/o GNN, w/o RL, w/o FL, w/o GNN&RL, w/o GNN&FL, w/o RL&FL.
   python scripts/run/run_ablation.py --config configs/paysim.yaml [--variants "Full model" "w/o RL"]
Mỗi biến thể dùng CÙNG seed/split; preprocess + graph chỉ dựng một lần khi mọi biến thể dùng chung đồ thị."""
import argparse
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.common import metrics_row, print_table, write_json  # noqa
from src.data.load_data import load_dataset
from src.data.split import split_dataframe
from src.train.pipeline_fraudgnn import prepare_splits, run_pipeline_on_split
from src.utils.config import deep_update, load_config
from src.utils.model_definitions import PAPER_ABLATIONS


def run_ablation(cfg, variants=None, seed=42, out_dir="outputs/ablation", prepared_cache=None):
    df = load_dataset(cfg)
    tr, va, te = split_dataframe(df, cfg)
    names = variants or list(PAPER_ABLATIONS)
    rows, full = [], {}
    cache = {}  # (gnn flag) -> prepared ; w/o GNN không cần cạnh nên dựng riêng
    for name in names:
        c = deep_update(cfg, {"flags": PAPER_ABLATIONS[name]})
        key = bool(c["flags"]["gnn"])
        if key not in cache:
            cache[key] = prepare_splits(tr, va, te, copy.deepcopy(c))
        res = run_pipeline_on_split(tr, va, te, c, prepared=cache[key], seed=seed)
        full[name] = res
        rows.append(metrics_row(name, res["test_metrics"]))
    print_table(rows)
    exp = cfg["experiment"]["name"]
    write_json({"rows": rows, "details": full}, f"{out_dir}/{exp}_ablation.json")
    return rows, full


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run_ablation(load_config(a.config), a.variants, a.seed)
