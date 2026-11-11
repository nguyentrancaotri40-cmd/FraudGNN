"""So sánh kết quả reproduction với Table 2 của paper và/hoặc Baseline vs Augmented.
   python scripts/eval/compare.py --results outputs/results/paysim_paper_repro_full_evaluation.json --dataset paysim
   python scripts/eval/compare.py --a outputs/results/x_baseline.json --b outputs/results/x_augmented.json"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Table 2 của paper (AUC-ROC, AUC-PR, F1, Recall@1% [%]) — chép từ PDF, CHỈ để tham chiếu
PAPER = {
 "paysim": {"xgboost": (.948, .412, .756, 69.8), "isolation_forest": (.928, .385, .722, 65.4), "lof": (.915, .362, .698, 62.5),
            "deepae": (.965, .524, .839, 79.2), "gcn": (.978, .578, .879, 83.7), "graphsemi": (.985, .612, .901, 88.9),
            "autogcn": (.990, .638, .925, 91.8), "fedfraud": (.988, .632, .914, 90.5), "FraudGNN-RL": (.995, .647, .923, 97.3)},
 "creditcard2023": {"xgboost": (.957, .438, .783, 72.1), "isolation_forest": (.935, .392, .745, 67.2), "lof": (.922, .375, .712, 64.8),
            "deepae": (.972, .538, .852, 81.5), "gcn": (.983, .585, .885, 84.9), "graphsemi": (.988, .625, .912, 89.8),
            "autogcn": (.992, .645, .932, 92.5), "fedfraud": (.990, .638, .922, 91.2), "FraudGNN-RL": (.996, .652, .928, 97.8)},
 "ieee_cis": {"xgboost": (.952, .425, .768, 70.5), "isolation_forest": (.932, .388, .735, 66.3), "lof": (.918, .368, .705, 63.2),
            "deepae": (.968, .532, .845, 80.4), "gcn": (.980, .582, .882, 84.2), "graphsemi": (.986, .618, .908, 89.2),
            "autogcn": (.991, .642, .928, 92.1), "fedfraud": (.989, .635, .918, 90.8), "FraudGNN-RL": (.995, .649, .925, 96.9)},
}
KEYS = ("auc_roc", "auc_pr", "f1", "recall_at_1pct")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results")
    ap.add_argument("--dataset", choices=list(PAPER))
    ap.add_argument("--a"); ap.add_argument("--b")
    a = ap.parse_args()
    if a.results:
        res = json.load(open(a.results))["table2"]
        print(f"{'model':<18}{'metric':<16}{'paper':>9}{'ours':>9}{'delta':>9}")
        for m, vals in PAPER[a.dataset].items():
            if m not in res:
                continue
            for k, pv in zip(KEYS, vals):
                ov = res[m][k] * (100 if k == "recall_at_1pct" else 1)
                print(f"{m:<18}{k:<16}{pv:>9.3f}{ov:>9.3f}{ov-pv:>+9.3f}")
    if a.a and a.b:
        A, B = json.load(open(a.a))["test_metrics"], json.load(open(a.b))["test_metrics"]
        print(f"{'metric':<16}{'A':>10}{'B':>10}{'B-A':>10}")
        for k in ("auc_roc", "auc_pr", "f1", "precision", "recall", "fpr", "recall_at_1pct"):
            print(f"{k:<16}{A[k]:>10.4f}{B[k]:>10.4f}{B[k]-A[k]:>+10.4f}")
