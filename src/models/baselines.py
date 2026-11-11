# ============================================================
# src/models/baselines.py — 8 baseline của Table 2 (Paper V-A-2).
# (File bổ sung ngoài cây thư mục gốc: cần để tái hiện bảng so sánh.)
#
# Paper chỉ nêu TÊN + mô tả 1 dòng, KHÔNG cho siêu tham số -> mọi chi tiết dưới đây là [INFERRED]:
#   XGBoost           : XGBClassifier hist, 300 cây, depth 6, scale_pos_weight = n_neg/n_pos
#   Isolation Forest  : 100 cây, điểm = -score_samples (unsupervised, fit trên train)
#   LOF               : novelty=True, fit trên tối đa `max_fit` mẫu train (O(N^2))
#   DeepAE            : AE 64-32-16 huấn luyện trên mẫu NORMAL, điểm = lỗi tái tạo
#   GCN               : 3 lớp GCNConv 64, BN, ReLU — cùng đồ thị cosine/1h với FraudGNN-RL
#   GraphSemi         : GraphSAGE (paper: SAGE trên đồ thị tương đồng Pearson) — ở đây dùng cùng đồ thị
#                       giao dịch để so sánh công bằng; KHÔNG có label propagation  [ĐƠN GIẢN HOÁ]
#   AutoGCN           : AE(x->64->x) + 2 GCN, loss = BCE + lambda*MSE tái tạo (multi-task như paper mô tả)
#   FedFraud          : CNN-1D (biến thể tốt nhất theo paper) + FedAvg trên cùng các client theo thời gian
# Ngưỡng quyết định F1 của baseline: chọn trên VAL (max F1) vì không có RL.
# ============================================================
from __future__ import annotations

import copy
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

from src.eval.metrics import best_f1_threshold, classification_metrics
from src.train.federated import fedavg_uniform
from src.train.train_gnn import compute_pos_weight


def _norm01(s_tr, s):
    lo, hi = float(np.min(s_tr)), float(np.max(s_tr))
    return np.clip((s - lo) / max(hi - lo, 1e-12), 0.0, None)


# ------------------------------ sklearn / xgboost ----------------------------
def run_xgboost(X, y, **kw):
    from xgboost import XGBClassifier
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
                      scale_pos_weight=float((y[0] == 0).sum() / max(1, (y[0] == 1).sum())),
                      eval_metric="aucpr", random_state=kw.get("seed", 42), n_jobs=-1)
    m.fit(X[0], y[0])
    return [m.predict_proba(x)[:, 1] for x in X[1:]]


def run_isolation_forest(X, y, **kw):
    from sklearn.ensemble import IsolationForest
    m = IsolationForest(n_estimators=100, random_state=kw.get("seed", 42), n_jobs=-1).fit(X[0])
    tr = -m.score_samples(X[0])
    return [_norm01(tr, -m.score_samples(x)) for x in X[1:]]


def run_lof(X, y, max_fit: int = 100000, **kw):
    from sklearn.neighbors import LocalOutlierFactor
    rng = np.random.default_rng(kw.get("seed", 42))
    idx = rng.choice(len(X[0]), size=min(max_fit, len(X[0])), replace=False)
    m = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1).fit(X[0][idx])
    tr = -m.score_samples(X[0][idx])
    return [_norm01(tr, -m.score_samples(x)) for x in X[1:]]


# ------------------------------ neural baselines -----------------------------
class _AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
        self.dec = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def run_deepae(X, y, device="cpu", epochs=30, **kw):
    torch.manual_seed(kw.get("seed", 42))
    xt = torch.tensor(X[0][y[0] == 0], dtype=torch.float32)
    m = _AE(xt.size(1)).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            b = xt[perm[i:i + 256]].to(device)
            loss = F.mse_loss(m(b), b)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()

    @torch.no_grad()
    def err(x):
        out = []
        for i in range(0, len(x), 4096):
            b = torch.tensor(x[i:i + 4096], dtype=torch.float32, device=device)
            out.append(((m(b) - b) ** 2).mean(1).cpu().numpy())
        return np.concatenate(out)

    tr = err(X[0])
    return [_norm01(tr, err(x)) for x in X[1:]]


class _GNN(nn.Module):
    def __init__(self, d, kind="gcn", hidden=64, layers=3, dropout=0.2, recon=False):
        super().__init__()
        conv = (lambda i, o: GCNConv(i, o)) if kind != "sage" else (lambda i, o: SAGEConv(i, o))
        self.recon = recon
        self.enc = nn.Linear(d, hidden) if recon else None
        self.dec = nn.Linear(hidden, d) if recon else None
        dims = [hidden if recon else d] + [hidden] * layers
        self.convs = nn.ModuleList([conv(dims[i], dims[i + 1]) for i in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.out = nn.Linear(hidden, 1)
        self.dp = dropout

    def forward(self, data):
        x = data.x
        rec = None
        if self.recon:
            z = F.relu(self.enc(x)); rec = self.dec(z); x = z
        for c, b in zip(self.convs, self.bns):
            x = F.dropout(F.relu(b(c(x, data.edge_index))), self.dp, self.training)
        return self.out(x).squeeze(-1), rec


def run_graph_baseline(graphs, kind: str, device="cpu", epochs=100, patience=10, lr=1e-3, recon_lambda=0.1, **kw):
    from src.eval.evaluate import predict_scores  # noqa
    torch.manual_seed(kw.get("seed", 42))
    g_tr, g_va, g_te = [g.to(device) for g in graphs]
    model = _GNN(g_tr.x.size(1), "sage" if kind == "graphsemi" else "gcn", recon=(kind == "autogcn")).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pw = compute_pos_weight(g_tr.y, kw.get("pos_weight_max", 100.0)).to(device)

    @torch.no_grad()
    def score(g):
        model.eval()
        return torch.sigmoid(model(g)[0]).cpu().numpy()

    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logit, rec = model(g_tr)
        loss = F.binary_cross_entropy_with_logits(logit, g_tr.y.float(), pos_weight=pw)
        if rec is not None:
            loss = loss + recon_lambda * F.mse_loss(rec, g_tr.x)
        loss.backward(); opt.step()
        m = classification_metrics(g_va.y.cpu().numpy(), score(g_va), 0.5)
        if m["auc_pr"] > best:
            best, bad, best_state = m["auc_pr"], 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return [score(g_va), score(g_te)]


class _CNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Conv1d(1, 16, 3, padding=1)
        self.fc = nn.Sequential(nn.Linear(16 * (d // 2), 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h = F.max_pool1d(F.relu(self.conv(x.unsqueeze(1))), 2)
        return self.fc(h.flatten(1)).squeeze(-1)


def run_fedfraud(X, y, device="cpu", rounds=30, num_clients=3, lr=1e-3, **kw):
    torch.manual_seed(kw.get("seed", 42))
    d = X[0].shape[1]
    d_eff = d if d % 2 == 0 else d + 1
    pad = lambda a: np.pad(a, ((0, 0), (0, d_eff - d))) if d_eff != d else a
    Xs = [torch.tensor(pad(a), dtype=torch.float32) for a in X]
    ys = torch.tensor(y[0], dtype=torch.float32)
    shards = np.array_split(np.arange(len(Xs[0])), num_clients)       # shard theo thứ tự (đã sort thời gian)
    glob = _CNN(d_eff).to(device)
    worker = copy.deepcopy(glob)
    pw = compute_pos_weight(ys.long(), 100.0).to(device)
    state = copy.deepcopy(glob.state_dict())
    for _ in range(rounds):
        sts = []
        for sh in shards:
            worker.load_state_dict(state); worker.train()
            opt = torch.optim.Adam(worker.parameters(), lr=lr)
            for i in range(0, len(sh), 512):
                b = sh[i:i + 512]
                loss = F.binary_cross_entropy_with_logits(worker(Xs[0][b].to(device)), ys[b].to(device), pos_weight=pw)
                opt.zero_grad(); loss.backward(); opt.step()
            sts.append(copy.deepcopy(worker.state_dict()))
        state = fedavg_uniform(sts)
    glob.load_state_dict(state); glob.eval()

    @torch.no_grad()
    def sc(a):
        return np.concatenate([torch.sigmoid(glob(a[i:i + 4096].to(device))).cpu().numpy() for i in range(0, len(a), 4096)])

    return [sc(Xs[1]), sc(Xs[2])]


BASELINES = ["xgboost", "isolation_forest", "lof", "deepae", "gcn", "graphsemi", "autogcn", "fedfraud"]


def run_baseline(name: str, prepared: Dict[str, Any], device: str = "cpu", seed: int = 42) -> Dict[str, Any]:
    """prepared = output của prepare_splits(). Trả {'val':metrics,'test':metrics,'runtime_sec':..}."""
    X, y, graphs = prepared["x"], prepared["y"], prepared["graphs"]
    t0 = time.perf_counter()
    if name in ("gcn", "graphsemi", "autogcn"):
        sv, st = run_graph_baseline(graphs, name, device, seed=seed)
    else:
        fn = {"xgboost": run_xgboost, "isolation_forest": run_isolation_forest, "lof": run_lof,
              "deepae": run_deepae, "fedfraud": run_fedfraud}[name]
        sv, st = fn(X, y, device=device, seed=seed)
    th = best_f1_threshold(y[1], sv)
    return {"model": name, "threshold": th, "val": classification_metrics(y[1], sv, th),
            "test": classification_metrics(y[2], st, th), "runtime_sec": time.perf_counter() - t0}
