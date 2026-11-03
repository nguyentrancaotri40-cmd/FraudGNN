# ============================================================
# verify_all.py
# Script kiểm chứng 4 tầng: Data, Model, Training, Result.
# Chạy: python verify_all.py --config configs/paysim.yaml
# ============================================================

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.utils.config import load_config
from src.data.load_data import load_dataset
from src.data.split import split_dataframe
from src.data.preprocess import FraudPreprocessor
from src.graph.build_graph import build_transaction_graph, extract_entity_type
from src.models.tssgc import TemporalAggregator, SemanticEncoder, TSSGCLayer
from src.models.fraudgnn_rl import FraudGNNRL


PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️ WARN"


def check(cond, name, detail=""):
    """In kết quả check."""
    if cond:
        print(f"  {PASS}: {name} {detail}")
        return True
    else:
        print(f"  {FAIL}: {name} {detail}")
        return False


# ============================================================
# TẦNG 1: DATA CONTRACT
# ============================================================
def verify_data(cfg):
    print("\n" + "="*60)
    print("TẦNG 1: DATA CONTRACT (Paper Sec. III-A & V-A)")
    print("="*60)
    ok = True

    df = load_dataset(cfg)
    label_col = cfg["dataset"]["label_col"]
    time_col = cfg["dataset"]["time_col"]

    ok &= check(label_col in df.columns, "Có label_col")
    ok &= check(df[label_col].isin([0, 1]).all(), "Label là binary")

    train_df, val_df, test_df = split_dataframe(df, cfg)

    # Temporal order
    if cfg["split"]["strategy"] == "temporal":
        ok &= check(
            train_df[time_col].max() <= val_df[time_col].min() + 1e-6,
            "Temporal: train → val không leak"
        )
        ok &= check(
            val_df[time_col].max() <= test_df[time_col].min() + 1e-6,
            "Temporal: val → test không leak"
        )

    # Preprocess
    pre = FraudPreprocessor(cfg)
    x_tr, y_tr, t_tr = pre.fit_transform(train_df)
    ok &= check(x_tr.min() >= -1e-6 and x_tr.max() <= 1 + 1e-6,
                "MinMaxScaler: x ∈ [0,1]")
    ok &= check("__hour_of_day" in pre._temporal_cols,
                "Temporal features có __hour_of_day")
    ok &= check("__delta_t" in pre._temporal_cols,
                "Temporal features có __delta_t")

    # Build graph
    et = extract_entity_type(train_df, cfg)
    g = build_transaction_graph(x_tr, y_tr, t_tr, cfg, entity_type=et)

    times = g.node_time
    src, dst = g.edge_index

    if times is not None and g.edge_index.size(1) > 0:
        dt = (times[src] - times[dst]).abs()
        ok &= check((dt <= 1.0 + 1e-3).all(),
                    "Mọi cạnh có |Δt| ≤ 1h", f"(max={dt.max():.3f}h)")

        # Undirected
        edge_set = set(zip(src.tolist(), dst.tolist()))
        undirected = all((j, i) in edge_set for i, j in edge_set)
        ok &= check(undirected, "Đồ thị undirected")

        # Self-loop
        ok &= check((src == dst).sum() == 0, "Không có self-loop")

        # Cosine ≥ 0.9 (sample 1000 cạnh để không tốn thời gian)
        import torch.nn.functional as F
        x_norm = F.normalize(g.x, dim=1)
        n_sample = min(1000, src.numel())
        idx = torch.randperm(src.numel())[:n_sample]
        cos = (x_norm[src[idx]] * x_norm[dst[idx]]).sum(dim=1)
        ok &= check((cos >= 0.9 - 1e-3).all(),
                    f"Cosine ≥ 0.9 (sample {n_sample} cạnh)",
                    f"(min={cos.min():.4f})")
    else:
        print(f"  {WARN}: Đồ thị có ít/không có cạnh, bỏ qua check temporal/cosine")

    return ok


# ============================================================
# TẦNG 2: MODEL CONTRACT
# ============================================================
def verify_model(cfg):
    print("\n" + "="*60)
    print("TẦNG 2: MODEL CONTRACT (Paper Sec. IV-A, Eq. 5-11)")
    print("="*60)
    ok = True

    # TEMP (Eq. 6-7)
    temp = TemporalAggregator(in_dim=10, out_dim=32)
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    delta = torch.tensor([0.5, 1.0, 0.3, 0.8])
    out = temp(x, edge_index, delta)
    ok &= check(out.shape == (5, 32), "TEMP shape [N, out_dim]", str(tuple(out.shape)))
    ok &= check(not torch.isnan(out).any(), "TEMP không NaN")

    # SEM (Eq. 10)
    sem = SemanticEncoder(in_dim=10, out_dim=32, num_node_types=5)
    nt = torch.tensor([0, 1, 2, 3, 4])
    out_sem = sem(x, nt)
    ok &= check(out_sem.shape == (5, 32), "SEM shape [N, out_dim]")

    # TSSGC Layer
    layer = TSSGCLayer(in_dim=10, out_dim=32, num_node_types=5)
    out_layer = layer(x, edge_index, delta, nt)
    ok &= check(out_layer.shape == (5, 32), "TSSGC layer shape [N, out_dim]")
    ok &= check(not torch.isnan(out_layer).any(), "TSSGC không NaN")

    # FraudGNNRL + feature weights
    model = FraudGNNRL(in_dim=10, hidden_dim=32, num_layers=1,
                       num_node_types=5, use_projection=False)
    fw = torch.ones(32) / 32
    model.set_feature_weights(fw)
    ok &= check(abs(model._feature_weights.sum().item() - 1.0) < 1e-5,
                "Feature weights tổng = 1")

    return ok


# ============================================================
# TẦNG 3: TRAINING CONTRACT
# ============================================================
def verify_training(cfg):
    print("\n" + "="*60)
    print("TẦNG 3: TRAINING CONTRACT (Paper Sec. V-A-4)")
    print("="*60)
    ok = True

    expected = {
        ("model", "hidden_dim"): 64,
        ("model", "num_layers"): 3,
        ("train", "epochs"): 100,
        ("train", "patience"): 10,
        ("train", "learning_rate"): 0.001,
        ("train", "batch_size"): 64,
        ("graph", "similarity_threshold"): 0.9,
        ("graph", "time_window_hours"): 1.0,
    }
    for (section, key), val in expected.items():
        actual = cfg.get(section, {}).get(key, None)
        ok &= check(actual == val, f"{section}.{key} = {val}", f"(actual={actual})")

    return ok


# ============================================================
# TẦNG 4: RESULT CONTRACT
# ============================================================
def verify_result(cfg):
    print("\n" + "="*60)
    print("TẦNG 4: RESULT CONTRACT (Paper Table 2)")
    print("="*60)

    dataset_name = cfg.get("experiment", {}).get("dataset", "unknown")
    PAPER_METRICS = {
        "paysim": {"auc_roc": 0.995, "auc_pr": 0.647, "f1": 0.923, "recall_at_1pct": 0.973},
        "ieee": {"auc_roc": 0.995, "auc_pr": 0.649, "f1": 0.925, "recall_at_1pct": 0.969},
        "creditcard": {"auc_roc": 0.996, "auc_pr": 0.652, "f1": 0.928, "recall_at_1pct": 0.978},
    }

    if dataset_name not in PAPER_METRICS:
        print(f"  {WARN}: Dataset '{dataset_name}' không có số liệu paper.")
        return True

    result_file = Path("outputs/results") / f"{cfg['experiment']['name']}_metrics.json"
    if not result_file.exists():
        print(f"  {WARN}: Chưa có file kết quả {result_file}.")
        print(f"  → Hãy chạy: python main.py --config {cfg.get('_config_path', '<config>')}")
        return True

    with open(result_file) as f:
        result = json.load(f)
    m = result.get("test_metrics", {})
    paper = PAPER_METRICS[dataset_name]

    ok = True
    for key, target in paper.items():
        actual = m.get(key, 0)
        delta = abs(actual - target)
        if delta < 0.05:
            status = PASS
        elif delta < 0.10:
            status = WARN
        else:
            status = FAIL
            ok = False
        print(f"  {status}: {key}: actual={actual:.4f}, paper={target:.4f}, delta={delta:.4f}")

    return ok


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    args = parser.parse_args()

    # Fix UTF-8 trên Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    cfg = load_config(args.config)
    cfg["_config_path"] = args.config

    print("\n" + "="*60)
    print(f"🔍 VERIFY ALL: {args.config}")
    print("="*60)

    results = {}
    try:
        results["data"] = verify_data(cfg)
    except Exception as e:
        print(f"  {FAIL}: Data check lỗi: {e}")
        results["data"] = False

    try:
        results["model"] = verify_model(cfg)
    except Exception as e:
        print(f"  {FAIL}: Model check lỗi: {e}")
        results["model"] = False

    try:
        results["training"] = verify_training(cfg)
    except Exception as e:
        print(f"  {FAIL}: Training check lỗi: {e}")
        results["training"] = False

    try:
        results["result"] = verify_result(cfg)
    except Exception as e:
        print(f"  {FAIL}: Result check lỗi: {e}")
        results["result"] = False

    # ===== TỔNG KẾT =====
    print("\n" + "="*60)
    print("📊 TỔNG KẾT")
    print("="*60)
    all_ok = True
    for k, v in results.items():
        status = PASS if v else FAIL
        print(f"  {status}: Tầng {k.upper()}")
        if not v:
            all_ok = False

    print("\n" + ("="*60))
    if all_ok:
        print("🎉 TẤT CẢ 4 TẦNG PASS — Code đúng logic paper!")
    else:
        print("⚠️ Có tầng FAIL — cần debug thêm.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()