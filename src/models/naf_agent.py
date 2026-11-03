# ============================================================
# src/models/naf_agent.py
#
# NAF (Normalized Advantage Functions) — Gu et al. 2016 [29].
#
# Paper (Section IV-B, nguyên văn):
#   "The action space is continuous, representing the fraud detection
#    threshold and feature importance weights. We discretize this space
#    and use a variant of DQN called Deep Q-Learning with Normalized
#    Advantage Functions (NAF) [29] to handle the continuous action space
#    more effectively."
#
# → DÙNG NAF cho cả threshold + feature weights.
#
# NAF (Gu et al. 2016):
#   Q(s,a) = V(s) + A(s,a)
#   A(s,a) = -½ (a - μ(s))^T P(s) (a - μ(s))
#   P(s)   = L(s) L(s)^T  (L tam giác dưới, đường chéo = exp)
#
# State (b5-A) = mean pooling embedding h_i (trước classifier, đã nhân
# feature weights) của batch hiện tại.
#
# Action:
#   - action[0]      : threshold ∈ [0,1] qua sigmoid
#   - action[1:]     : feature weights qua softmax
#
# [INFERRED] Paper không cho:
#   - Công thức reward → src/models/reward.py (tự thiết kế)
#   - Cách mean pooling state → nhóm chọn mean
#   - Noise exploration → Gaussian noise, decay 0.995
#   - Target update → soft (Polyak) τ=0.005
#   - γ = 0.99, lr = 1e-3, buffer = 10000 (chuẩn DQN)
# ============================================================

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple, Dict, Any, Optional
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.reward import compute_reward, fixed_pos_step


# ======================================================================
# Replay Buffer
# ======================================================================
class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        # (state, action, reward, next_state, done)
        # action là vector [1 + n_features]
        self.buffer: Deque = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d) -> None:
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.asarray(s), dtype=torch.float32),
            torch.tensor(np.asarray(a), dtype=torch.float32),
            torch.tensor(np.asarray(r), dtype=torch.float32),
            torch.tensor(np.asarray(s2), dtype=torch.float32),
            torch.tensor(np.asarray(d), dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ======================================================================
# NAF Network
# ======================================================================
class NAFNetwork(nn.Module):
    """
    Q(s,a) = V(s) + A(s,a)
    A(s,a) = -0.5 * (a - μ(s))^T P(s) (a - μ(s))
    P(s)   = L(s) L(s)^T
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.v  = nn.Linear(hidden_dim, 1)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.l1 = nn.Linear(hidden_dim, action_dim)
        # Số phần tử tam giác dưới (không kể đường chéo)
        self._tril_n = action_dim * (action_dim + 1) // 2
        self.l2 = nn.Linear(hidden_dim, self._tril_n)

    def forward(self, s: torch.Tensor, a: torch.Tensor | None = None):
        h = self.shared(s)
        v = self.v(h)                         # [B, 1]
        mu = self.mu(h)                       # [B, A]

        B = s.size(0)
        A_ = self.action_dim

        l1 = self.l1(h)                       # [B, A]
        l2 = self.l2(h)                       # [B, A(A+1)/2]

        L = torch.zeros(B, A_, A_, device=s.device)
        # Đường chéo = exp(l1) (đảm bảo dương)
        diag_idx = torch.arange(A_, device=s.device)
        L[:, diag_idx, diag_idx] = torch.exp(l1)

        # Tam giác dưới
        tril = torch.tril_indices(A_, A_, offset=-1, device=s.device)
        L[:, tril[0], tril[1]] = l2[:, :tril[0].shape[0]]

        P = torch.bmm(L, L.transpose(1, 2))   # [B, A, A]

        if a is not None:
            delta = (a - mu).unsqueeze(2)     # [B, A, 1]
            p_delta = torch.bmm(torch.bmm(delta.transpose(1, 2), P), delta)
            adv = -0.5 * p_delta.squeeze(-1).squeeze(-1)
            q = v.squeeze(-1) + adv
        else:
            q = v.squeeze(-1)

        return q, mu


# ======================================================================
# NAF Agent
# ======================================================================
@dataclass
class NAFAgent:
    state_dim: int
    action_dim: int = 1        # threshold
    n_features: int = 64       # feature weights (hidden_dim TSSGC)
    hidden_dim: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    tau: float = 0.005         # soft update
    buffer_size: int = 10000
    batch_size: int = 64
    device: str = "cpu"

    def __post_init__(self):
        self.device = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.total_action_dim = self.action_dim + self.n_features

        self.policy_net = NAFNetwork(
            state_dim=self.state_dim,
            action_dim=self.total_action_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        self.target_net = NAFNetwork(
            state_dim=self.state_dim,
            action_dim=self.total_action_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(self.buffer_size)

        # Exploration noise
        self.noise_std = 0.10
        self.noise_decay = 0.995
        self.noise_min = 0.01

    # ------------------------------------------------------------------
    def act(self, state: np.ndarray, explore: bool = True):
        """
        Returns: (action_vector [A+1+n_features], threshold, feature_weights)
        """
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, mu = self.policy_net(s, None)
            raw = mu[0]

            threshold = float(torch.sigmoid(raw[0]).item())
            fw = torch.softmax(raw[1:], dim=0).cpu().numpy()

        if explore:
            threshold = float(np.clip(threshold + np.random.normal(0, self.noise_std), 0.0, 1.0))
            fw = np.clip(fw + np.random.normal(0, 0.05, size=fw.shape), 0.0, None)
            if fw.sum() > 0:
                fw = fw / fw.sum()

        action = np.zeros(self.action_dim + self.n_features, dtype=np.float32)
        action[0] = threshold
        action[1:] = fw
        return action, threshold, fw

    def get_feature_weights(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, mu = self.policy_net(s, None)
            fw = torch.softmax(mu[0, 1:], dim=0).cpu().numpy()
        return fw

    # ------------------------------------------------------------------
    def update(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None

        s, a, r, s2, d = self.memory.sample(self.batch_size)
        s, a, r, s2, d = (
            s.to(self.device), a.to(self.device), r.to(self.device),
            s2.to(self.device), d.to(self.device),
        )

        q, _ = self.policy_net(s, a)

        with torch.no_grad():
            target_q, _ = self.target_net(s2, None)
            target = r + self.gamma * target_q * (1.0 - d)

        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        # Soft update target
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

        self.noise_std = max(self.noise_min, self.noise_std * self.noise_decay)
        return float(loss.item())


# ======================================================================
# Batch environment (mỗi step = 1 batch scores)
# ======================================================================
class BatchNAFEnvironment:
    """
    [INFERRED] Paper không mô tả cách chia batch khi RL chạy trên val/test.
    Nhóm chọn: chia không overlap, batch_size cố định (trừ batch cuối).

    State = mean pooling embedding h_i của batch hiện tại (b5-A).
    """

    def __init__(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        graph_embeddings: Optional[np.ndarray] = None,
        batch_size: int = 256,
        fpr_penalty: float = 1.0,
        recall_weight: float = 0.25,
    ):
        self.scores = np.asarray(scores, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.graph_embeddings = graph_embeddings
        self.batch_size = int(batch_size)
        self.fpr_penalty = float(fpr_penalty)
        self.recall_weight = float(recall_weight)

        self.pos = 0
        self.current_threshold = 0.5
        self._emb_dim = (
            graph_embeddings.shape[1] if graph_embeddings is not None else 64
        )

    @property
    def state_dim(self) -> int:
        return self._emb_dim

    def reset(self) -> np.ndarray:
        self.pos = 0
        self.current_threshold = 0.5
        return self._state()

    def _batch(self):
        a = self.pos
        b = min(len(self.scores), a + self.batch_size)
        return self.scores[a:b], self.labels[a:b]

    def _state(self) -> np.ndarray:
        if self.graph_embeddings is not None:
            a = self.pos
            b = min(len(self.scores), a + self.batch_size)
            if b <= a:
                return np.zeros(self.state_dim, dtype=np.float32)
            return np.mean(self.graph_embeddings[a:b], axis=0).astype(np.float32)
        # Fallback: state = thống kê batch (KHÔNG phải paper — chỉ để không crash)
        s, _ = self._batch()
        if len(s) == 0:
            return np.zeros(self.state_dim, dtype=np.float32)
        return np.array(
            [float(np.mean(s)), float(np.std(s)), float(np.min(s)),
             float(np.max(s)), float(self.current_threshold)],
            dtype=np.float32,
        )

    def step(self, threshold: float):
        self.current_threshold = float(threshold)
        s, y = self._batch()
        if len(s) == 0:
            return np.zeros(self.state_dim, dtype=np.float32), 0.0, True, {"done": True}

        pred = (s >= threshold).astype(np.int64)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        tn = int(np.sum((pred == 0) & (y == 0)))

        acc = (tp + tn) / max(1, tp + fp + fn + tn)
        fpr = fp / max(1, fp + tn)
        reward = compute_reward(tp, fp, tn, fn,
                                fpr_penalty=self.fpr_penalty,
                                recall_weight=self.recall_weight)

        step_sz = fixed_pos_step(self.batch_size, len(self.scores) - self.pos)
        self.pos += step_sz
        done = self.pos >= len(self.scores)
        next_state = self._state() if not done else np.zeros(self.state_dim, dtype=np.float32)

        info = dict(accuracy=float(acc), fpr=float(fpr), threshold=float(threshold),
                    tp=tp, fp=fp, tn=tn, fn=fn, pos=int(self.pos))
        return next_state, float(reward), done, info


# ======================================================================
# Train NAF
# ======================================================================
def train_naf_agent(
    scores: np.ndarray,
    labels: np.ndarray,
    cfg: Dict[str, Any],
    graph_embeddings: Optional[np.ndarray] = None,
    device: str | None = None,
    n_features: int = 64,
) -> Tuple[NAFAgent, Dict[str, Any]]:
    """
    Train NAF trên (scores, labels) — thường là val set (paper: RL train
    trên val, test chỉ eval).
    """
    rl_cfg = cfg.get("rl", {})
    batch_size = int(rl_cfg.get("batch_size", 256))

    env = BatchNAFEnvironment(
        scores, labels,
        graph_embeddings=graph_embeddings,
        batch_size=batch_size,
        fpr_penalty=float(rl_cfg.get("fpr_penalty", 1.0)),
        recall_weight=float(rl_cfg.get("recall_weight", 0.25)),
    )

    agent = NAFAgent(
        state_dim=env.state_dim,
        action_dim=1,
        n_features=n_features,
        hidden_dim=int(rl_cfg.get("hidden_dim", 128)),
        gamma=float(rl_cfg.get("gamma", 0.99)),
        lr=float(rl_cfg.get("learning_rate", 1e-3)),
        tau=float(rl_cfg.get("tau", 0.005)),
        buffer_size=int(rl_cfg.get("buffer_size", 10000)),
        batch_size=int(rl_cfg.get("batch_size", 64)),  # DQN update batch
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    epochs = int(rl_cfg.get("epochs", 100))
    losses: List[float] = []
    infos: List[dict] = []

    for ep in range(epochs):
        state = env.reset()
        done = False
        while not done:
            action, threshold, _fw = agent.act(state, explore=True)
            next_state, reward, done, info = env.step(threshold)
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            infos.append(info)
            state = next_state

    return agent, {"losses": losses, "infos": infos}


# ======================================================================
# Apply NAF (greedy) — chọn threshold
# ======================================================================
def apply_naf_policy(
    agent: NAFAgent,
    scores: np.ndarray,
    labels: np.ndarray,
    cfg: Dict[str, Any],
    graph_embeddings: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Greedy (explore=False) → trả về mean threshold + metrics.

    [WARN] Paper không nói rõ:
      - Có lấy threshold mean hay threshold cuối? → nhóm chọn mean.
      - Có train online trên test không? → KHÔNG (benchmark chỉ eval).
    """
    rl_cfg = cfg.get("rl", {})
    batch_size = int(rl_cfg.get("batch_size", 256))

    env = BatchNAFEnvironment(
        scores, labels,
        graph_embeddings=graph_embeddings,
        batch_size=batch_size,
        fpr_penalty=float(rl_cfg.get("fpr_penalty", 1.0)),
        recall_weight=float(rl_cfg.get("recall_weight", 0.25)),
    )

    state = env.reset()
    done = False
    thresholds: List[float] = []
    all_info: List[dict] = []

    while not done:
        _action, threshold, _fw = agent.act(state, explore=False)
        thresholds.append(float(threshold))
        state, _r, done, info = env.step(threshold)
        all_info.append(info)

    mean_th = float(np.mean(thresholds)) if thresholds else 0.5
    return mean_th, {
        "thresholds": thresholds,
        "mean": mean_th,
        "num_steps": len(thresholds),
        "info": all_info,
    }


__all__ = [
    "ReplayBuffer",
    "NAFNetwork",
    "NAFAgent",
    "BatchNAFEnvironment",
    "train_naf_agent",
    "apply_naf_policy",
]