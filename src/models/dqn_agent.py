# ============================================================
# src/models/dqn_agent.py — Vanilla DQN (Eq. 3-4, 12) trên không gian hành động ĐÃ RỜI RẠC HOÁ.
# Dùng khi rl.type = "dqn": action = chỉ số bin của threshold; feature weights giữ đều (w=1).
# Mặc định của paper là NAF (naf_agent.py) vì "continuous action space".
# ============================================================
from __future__ import annotations

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNAgent:
    def __init__(self, state_dim: int, n_features: int, n_bins: int = 19, hidden_dim: int = 128,
                 gamma: float = 0.99, lr: float = 1e-3, tau: float = 0.005, buffer_size: int = 10000,
                 batch_size: int = 64, eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: float = 0.995,
                 device: str = "cpu"):
        self.kind = "dqn"
        self.device, self.n_features = device, n_features
        self.bins = np.linspace(0.05, 0.95, n_bins).astype(np.float32)
        mk = lambda: nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, n_bins)).to(device)
        self.policy, self.target = mk(), mk()
        self.target.load_state_dict(self.policy.state_dict())
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self.gamma, self.tau, self.bs = gamma, tau, batch_size
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.n_bins = n_bins

    @torch.no_grad()
    def act(self, state: np.ndarray, explore: bool = True) -> Tuple[np.ndarray, float, np.ndarray]:
        if explore and random.random() < self.eps:
            k = random.randrange(self.n_bins)
        else:
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            k = int(self.policy(s).argmax(1).item())
        fw = np.full(self.n_features, 1.0 / self.n_features, dtype=np.float32)
        return np.array([k], dtype=np.float32), float(self.bins[k]), fw

    def remember(self, s, a, r, s2, d):
        self.memory.append((s, a, r, s2, d))

    def update(self):
        if len(self.memory) < self.bs:
            return None
        s, a, r, s2, d = zip(*random.sample(self.memory, self.bs))
        f = lambda z, dt=torch.float32: torch.as_tensor(np.asarray(z), dtype=dt, device=self.device)
        s, s2, r, d = f(s), f(s2), f(r), f(d)
        a = f(a, torch.long).view(-1)
        q = self.policy(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            y = r + self.gamma * self.target(s2).max(1).values * (1.0 - d)   # Eq. 12
        loss = F.mse_loss(q, y)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.opt.step()
        with torch.no_grad():
            for tp, pp in zip(self.target.parameters(), self.policy.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * pp)
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        return float(loss.item())


__all__ = ["DQNAgent"]
