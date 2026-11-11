# ============================================================
# src/models/naf_agent.py — Normalized Advantage Functions (Gu et al., 2016) [29].
#
# Paper IV-B: "The action space is continuous, representing the fraud detection threshold
# and feature importance weights. We discretize this space and use a variant of DQN called
# Deep Q-Learning with Normalized Advantage Functions (NAF) to handle the continuous action
# space more effectively."   Loss = Eq. 12 (target network theta^-).
#
#   Q(s,a) = V(s) + A(s,a),  A = -1/2 (a-mu(s))^T P(s) (a-mu(s)),  P = L L^T
#   Action a in [0,1]^(1+D): a[0]=threshold, a[1:]=raw feature weights (mu qua sigmoid => cùng không gian).
#
# Siêu tham số paper nêu: 2 hidden layers x 128, Adam lr=1e-3.
# [INFERRED]: gamma=0.99, replay 10k, Polyak tau=0.005, Gaussian noise decay 0.995, update-batch 64.
# ============================================================
from __future__ import annotations

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, n: int, device):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        f = lambda z: torch.as_tensor(np.asarray(z), dtype=torch.float32, device=device)
        return f(s), f(a), f(r), f(s2), f(d)

    def __len__(self):
        return len(self.buf)


class NAFNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.A = action_dim
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.v = nn.Linear(hidden, 1)
        self.mu = nn.Linear(hidden, action_dim)
        self.l_diag = nn.Linear(hidden, action_dim)
        self.n_off = action_dim * (action_dim - 1) // 2
        self.l_off = nn.Linear(hidden, self.n_off)
        self.register_buffer("tril", torch.tril_indices(action_dim, action_dim, offset=-1))

    def forward(self, s: torch.Tensor, a: torch.Tensor | None = None):
        h = self.trunk(s)
        v, mu = self.v(h).squeeze(-1), torch.sigmoid(self.mu(h))
        if a is None:
            return v, mu, None
        B = s.size(0)
        L = s.new_zeros(B, self.A, self.A)
        idx = torch.arange(self.A, device=s.device)
        L[:, idx, idx] = torch.exp(self.l_diag(h).clamp(-4.0, 2.0))
        if self.n_off:
            L[:, self.tril[0], self.tril[1]] = self.l_off(h)
        y = torch.bmm(L.transpose(1, 2), (a - mu).unsqueeze(-1)).squeeze(-1)   # L^T (a-mu)
        adv = -0.5 * (y ** 2).sum(-1)                                           # = -1/2 d^T P d
        return v, mu, v + adv


class NAFAgent:
    def __init__(self, state_dim: int, n_features: int, hidden_dim: int = 128, gamma: float = 0.99,
                 lr: float = 1e-3, tau: float = 0.005, buffer_size: int = 10000, batch_size: int = 64,
                 noise_std: float = 0.10, noise_decay: float = 0.995, noise_min: float = 0.01,
                 device: str = "cpu"):
        self.kind = "naf"
        self.device = device
        self.n_features = n_features
        self.A = 1 + n_features
        self.gamma, self.tau, self.bs = gamma, tau, batch_size
        self.policy = NAFNetwork(state_dim, self.A, hidden_dim).to(device)
        self.target = NAFNetwork(state_dim, self.A, hidden_dim).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.noise_std, self.noise_decay, self.noise_min = noise_std, noise_decay, noise_min

    def remember(self, s, a, r, s2, d):
        self.memory.push(s, a, r, s2, d)

    @torch.no_grad()
    def act(self, state: np.ndarray, explore: bool = True) -> Tuple[np.ndarray, float, np.ndarray]:
        """Returns (action_vector, threshold, fw_raw)."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, mu, _ = self.policy(s)
        a = mu[0].cpu().numpy()
        if explore:
            a = np.clip(a + np.random.normal(0, self.noise_std, size=a.shape), 0.0, 1.0)
        a = a.astype(np.float32)
        return a, float(a[0]), a[1:]

    def update(self):
        if len(self.memory) < self.bs:
            return None
        s, a, r, s2, d = self.memory.sample(self.bs, self.device)
        _, _, q = self.policy(s, a)
        with torch.no_grad():
            v2, _, _ = self.target(s2)
            y = r + self.gamma * v2 * (1.0 - d)               # Eq. 12 với V(s') = max_a' Q(s',a')
        loss = F.mse_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.opt.step()
        with torch.no_grad():
            for tp, pp in zip(self.target.parameters(), self.policy.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * pp)
        self.noise_std = max(self.noise_min, self.noise_std * self.noise_decay)
        return float(loss.item())


__all__ = ["ReplayBuffer", "NAFNetwork", "NAFAgent"]
