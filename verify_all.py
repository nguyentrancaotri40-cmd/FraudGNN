# ============================================================
# verify_all.py — kiểm chứng reproduction.
#
#   python verify_all.py --unit                      # kiểm tra toán/shape bằng tensor nhỏ (KHÔNG dùng để báo kết quả)
#   python verify_all.py --config configs/paysim.yaml  # kiểm tra data contract trên DỮ LIỆU THẬT của bạn
#   python verify_all.py --config ... --train        # + chạy pipeline và kiểm tra kết quả hợp lệ
#
# Lưu ý: --unit dùng số ngẫu nhiên CHỈ để test đúng/sai của code (so với cài đặt tham chiếu brute-force).
# Chúng không sinh ra bất kỳ metric nào và không bao giờ được dùng làm kết quả thí nghiệm.
# ============================================================
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

OK, BAD = "[PASS]", "[FAIL]"
_fails = 0


def check(cond: bool, name: str, detail: str = "") -> bool:
    global _fails
    print(f"  {OK if cond else BAD} {name} {detail}")
    if not cond:
        _fails += 1
    return cond


# ----------------------------------------------------------------------
def unit_tests() -> None:
    from torch_geometric.data import Data

    from src.eval.metrics import classification_metrics, recall_at_k_percent
    from src.extensions.pruning import get_pruning_stats, magnitude_prune_
    from src.graph.graph_utils import cosine_neighbors, make_undirected
    from src.models.fraudgnn_rl import FraudGNNRL
    from src.models.naf_agent import NAFAgent, NAFNetwork
    from src.models.reward import BatchFraudEnv, compute_reward, weights_from_action
    from src.models.tssgc import TemporalAggregator, TSSGCEncoder
    from src.train.federated import fedavg_uniform

    torch.manual_seed(0)
    np.random.seed(0)

    print("\n[1] Graph construction == brute force (cosine>=thr AND |dt|<=window)")
    n, d = 300, 8
    x = np.abs(np.random.randn(n, d)).astype(np.float32) + 0.2
    t = np.sort(np.random.rand(n) * 5.0)
    thr, win = 0.93, 0.7
    ei, dl, _ = cosine_neighbors(x, t, thr, win, None, chunk_elems=5000, max_query_chunk=37)
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    sim = xn @ xn.T
    dt = np.abs(t[:, None] - t[None, :])
    ref = (sim >= thr) & (dt <= win) & ~np.eye(n, dtype=bool)
    got = np.zeros((n, n), dtype=bool)
    got[ei[1].numpy(), ei[0].numpy()] = True
    check((ref == got).all(), "exact neighbor set", f"(edges={int(ref.sum())})")
    ei2, _, _ = cosine_neighbors(x, None, thr, None, None, chunk_elems=5000, max_query_chunk=37)
    ref2 = (sim >= thr) & ~np.eye(n, dtype=bool)
    got2 = np.zeros((n, n), dtype=bool); got2[ei2[1].numpy(), ei2[0].numpy()] = True
    check((ref2 == got2).all(), "exact (không có time)")
    ei3, dl3, _ = cosine_neighbors(x, t, thr, win, 3)
    ei3u, _ = make_undirected(ei3, dl3, n)
    check(ei3u.size(1) % 2 == 0 and (ei3u[0] != ei3u[1]).all(), "cap top-k + undirected, không self-loop")

    print("\n[2] Temporal aggregator (Eq. 6-7) vs tham chiếu vòng lặp")
    D = 6
    agg = TemporalAggregator(5, D).eval()
    xx = torch.randn(7, 5)
    e_src = torch.tensor([1, 2, 3, 0, 4, 5, 6, 2]); e_dst = torch.tensor([0, 0, 0, 1, 1, 1, 1, 3])
    e_dt = torch.tensor([0.9, 0.1, 0.5, 0.3, 0.2, 0.8, 0.0, 0.4])
    out = agg(xx, torch.stack([e_src, e_dst]), e_dt)
    beta = torch.nn.functional.softplus(agg.beta_raw)
    ok = True
    for node in (0, 1, 3):
        m = (e_dst == node).nonzero().flatten()
        a = torch.softmax(-beta * e_dt[m], 0)
        seq = (agg.lin_msg(xx[e_src[m]]) * a[:, None])
        order = torch.argsort(e_dt[m], descending=True)      # cũ -> mới
        h = torch.zeros(1, 1, D)
        _, h = agg.gru(seq[order][None], h)
        ok &= torch.allclose(h.flatten(), out[node], atol=1e-5)
    check(bool(ok) and bool(out[2].abs().sum() == 0), "khớp GRU tuần tự; node không cạnh = 0")
    agg2 = TemporalAggregator(5, D, max_seq_len=2).eval()
    check(agg2(xx, torch.stack([e_src, e_dst]), e_dt).shape == (7, D), "max_seq_len cap chạy được")

    print("\n[3] TSSGC encoder / FraudGNNRL")
    N = 40
    data = Data(x=torch.rand(N, 12), y=(torch.rand(N) > 0.8).long(),
                edge_index=torch.randint(0, N, (2, 150)), edge_time_delta=torch.rand(150),
                node_type=torch.randint(0, 5, (N,)))
    model = FraudGNNRL(12, 64, 3, num_node_types=5)
    out = model(data)
    check(out.shape == (N,), "logits shape", str(tuple(out.shape)))
    n_layers = sum(1 for _ in model.encoder.layers)
    check(n_layers == 3 and model.encoder.layers[0].norm.num_features == 64, "3 lớp, 64 hidden, BatchNorm")
    torch.nn.functional.binary_cross_entropy_with_logits(out, data.y.float()).backward()
    check(all(p.grad is not None for p in model.encoder.layers[0].temporal.parameters()), "gradient chảy qua nhánh Temporal")
    check(all(p.grad is not None for p in model.encoder.layers[0].semantic.parameters()), "gradient chảy qua nhánh Semantic")
    model.eval()
    s0 = model(data)
    model.set_feature_weights(torch.linspace(0.1, 2.0, 64))
    check(not torch.allclose(s0, model(data)), "feature weights của RL thay đổi điểm số")
    mlp = FraudGNNRL(12, 64, 3, use_gnn=False).eval()
    d2 = data.clone(); d2.edge_index = torch.randint(0, N, (2, 150))
    check(torch.allclose(mlp(data), mlp(d2)), "w/o GNN: kết quả không phụ thuộc cạnh")

    print("\n[4] FedAvg (Algorithm 1, dòng 7)")
    a, b = FraudGNNRL(12, 8, 1).state_dict(), FraudGNNRL(12, 8, 1).state_dict()
    avg = fedavg_uniform([a, b])
    k = "projection.weight"
    check(torch.allclose(avg[k], (a[k] + b[k]) / 2), "trung bình ĐỀU 1/|C_t|")

    print("\n[5] NAF (Gu et al. 2016)")
    net = NAFNetwork(64, 65, 128)
    s = torch.randn(4, 64)
    _, mu, _ = net(s)
    _, _, q = net(s, mu)
    v, _, _ = net(s)
    check(torch.allclose(q, v, atol=1e-5), "A(s, mu(s)) = 0  =>  Q = V")
    _, _, q2 = net(s, (mu + 0.3).clamp(0, 1))
    check(bool((q2 <= v + 1e-6).all()), "A <= 0 (P positive semi-definite)")
    check(bool(((mu >= 0) & (mu <= 1)).all()), "mu trong [0,1]")
    agent = NAFAgent(64, 64, device="cpu", batch_size=8)
    for _ in range(20):
        agent.remember(np.random.randn(64), np.random.rand(65), 0.1, np.random.randn(64), 0.0)
    check(agent.update() is not None, "update() chạy")
    w = weights_from_action(np.random.rand(64))
    check(abs(w.mean() - 1.0) < 1e-5, "w = D*softmax-like có mean = 1")

    print("\n[6] RL environment")
    emb = torch.rand(500, 64)
    clf = model.classifier
    lab = (np.random.rand(500) > 0.9).astype(int)
    env = BatchFraudEnv(emb, lab, clf, batch_size=128)
    st, done, steps = env.reset(), False, 0
    while not done:
        st, r, done, info = env.step(0.5, np.ones(64))
        steps += 1
    check(steps == 4 and info["end"] == 500, "quét batch không chồng lấn", f"(steps={steps})")
    check(abs(compute_reward(5, 0, 90, 5, 1.0, 0.25) - (2 * 1 * 0.5 / 1.5 + 0.25 * 0.5)) < 1e-9, "công thức reward")

    print("\n[7] Metrics")
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0]); sc = np.array([.9, .1, .2, .3, .4, .1, .1, .2, .1, .1])
    check(recall_at_k_percent(y, sc, 10) == 0.5, "Recall@10% = 0.5")
    m = classification_metrics(y, sc, 0.5)
    check(m["tp"] == 1 and m["fn"] == 1 and m["fp"] == 0, "confusion matrix")

    print("\n[8] Pruning (extension)")
    pm = FraudGNNRL(12, 16, 1)
    magnitude_prune_(pm, 0.3)
    check(0.25 <= get_pruning_stats(pm)["pruning_ratio"] <= 0.4, "pruning ratio ~0.3")

    print("\n[9] Preprocess: mode-imputation giữ NaN, temporal features được đưa vào model")
    import pandas as pd

    from src.data.preprocess import FraudPreprocessor
    df = pd.DataFrame({"isFraud": [0, 1, 0, 0, 1, 0], "TransactionDT": [10, 3600, 7300, 9000, 12000, 15000],
                       "amt": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0], "cat": ["a", None, "a", "b", "a", None]})
    cfg = {"dataset": {"label_col": "isFraud", "time_col": "TransactionDT", "time_unit": "second"},
           "preprocess": {"drop_cols": []}}
    pp = FraudPreprocessor(cfg)
    X, yy, tt = pp.fit_transform(df)
    names = pp.get_feature_names_out()
    check(all(c in names for c in ("__hour_of_day", "__day_of_week", "__delta_t")), "temporal features có trong X")
    check("cat_nan" not in names and "cat_a" in names, "NaN categorical được thay bằng mode (không thành lớp 'nan')")
    check(np.isfinite(X).all() and X.min() >= 0 and X.max() <= 1, "min-max trong [0,1], không NaN")


# ----------------------------------------------------------------------
def data_contract(cfg) -> None:
    from src.data.load_data import load_dataset
    from src.data.split import split_dataframe
    ds = cfg["dataset"]
    print("\n[DATA] Data contract trên dữ liệu thật")
    df = load_dataset(cfg)
    y = ds["label_col"]
    check(set(df[y].unique()) <= {0, 1}, "nhãn nhị phân")
    tr, va, te = split_dataframe(df, cfg)
    tc = ds.get("time_col")
    if cfg["split"]["strategy"] == "temporal" and tc:
        check(tr[tc].max() <= va[tc].min() and va[tc].max() <= te[tc].min(), "temporal: train <= val <= test (không rò rỉ tương lai)")
    check(len(tr) + len(va) + len(te) == len(df), "split phủ toàn bộ dữ liệu")
    for n, p in (("train", tr), ("val", va), ("test", te)):
        check(p[y].nunique() == 2, f"{n} có cả 2 lớp")


def real_run(cfg) -> None:
    from src.train.pipeline_fraudgnn import run_pipeline
    print("\n[RUN] Chạy pipeline trên dữ liệu thật")
    res = run_pipeline(cfg)
    m = res["test_metrics"]
    check(0.0 <= m["auc_roc"] <= 1.0 and 0.0 <= m["auc_pr"] <= 1.0, "metrics trong [0,1]")
    check(m["tp"] + m["fp"] + m["tn"] + m["fn"] == m["num_samples"], "confusion matrix khớp số mẫu")
    check(0.0 <= res["selected_threshold"] <= 1.0, "threshold hợp lệ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    if not a.unit and not a.config:
        ap.error("cần --unit và/hoặc --config")
    if a.unit:
        unit_tests()
    if a.config:
        from src.utils.config import load_config
        c = load_config(a.config)
        data_contract(c)
        if a.train:
            real_run(c)
    print(f"\n{'ALL PASS' if not _fails else str(_fails) + ' FAIL'}")
    sys.exit(1 if _fails else 0)
