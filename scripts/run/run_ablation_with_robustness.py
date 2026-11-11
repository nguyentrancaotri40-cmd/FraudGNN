"""Ablation + adversarial + concept drift cho từng biến thể.
   python scripts/run/run_ablation_with_robustness.py --config configs/paysim.yaml --epsilon 0.1"""
import argparse
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.common import write_json  # noqa
from src.data.load_data import load_dataset
from src.data.split import split_dataframe
from src.eval.adversarial import evaluate_adversarial_robustness
from src.eval.concept_drift import run_concept_drift_test
from src.train.pipeline_fraudgnn import prepare_splits, run_pipeline_on_split
from src.utils.config import deep_update, load_config
from src.utils.model_definitions import PAPER_ABLATIONS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = load_config(a.config)
    tr, va, te = split_dataframe(load_dataset(cfg), cfg)
    cache, out = {}, {}
    for name, fl in PAPER_ABLATIONS.items():
        c = deep_update(cfg, {"flags": fl})
        k = bool(c["flags"]["gnn"])
        if k not in cache:
            cache[k] = prepare_splits(tr, va, te, copy.deepcopy(c))
        res, art = run_pipeline_on_split(tr, va, te, c, prepared=cache[k], seed=a.seed, return_artifacts=True)
        out[name] = {"test_metrics": res["test_metrics"],
                     "adversarial": evaluate_adversarial_robustness(art["model"], art["graphs"][2], art["threshold"],
                                                                    art["device"], epsilon=a.epsilon),
                     "concept_drift": run_concept_drift_test(art["test_scores"], art["test_labels"], art["threshold"],
                                                             a.windows, art["y_pred_test"])}
    write_json(out, f"outputs/ablation_robustness/{cfg['experiment']['name']}.json")
