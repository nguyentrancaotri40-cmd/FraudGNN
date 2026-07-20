# src/models/dqn_agent.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.asarray(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.asarray(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ThresholdDQNAgent:
    state_dim: int
    thresholds: List[float]
    hidden_dim: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    buffer_size: int = 10000
    device: str = "cpu"

    def __post_init__(self):
        self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        self.policy_net = DQN(self.state_dim, len(self.thresholds), self.hidden_dim).to(self.device)
        self.target_net = DQN(self.state_dim, len(self.thresholds), self.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(self.buffer_size)
        self.epsilon = self.epsilon_start

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randrange(len(self.thresholds))
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.policy_net(s).argmax(dim=1).item())

    def threshold(self, action: int) -> float:
        return float(self.thresholds[action])

    def update(self, batch_size: int = 64) -> float | None:
        if len(self.memory) < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * next_q * (1.0 - dones)
        
        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())


class BatchThresholdEnvironment:
    def __init__(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 256,
        fpr_penalty: float = 2.0,
    ):
        self.scores = np.asarray(scores, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.fpr_penalty = float(fpr_penalty)
        self.pos = 0
        self.current_threshold = 0.5
        self.history = []
        self.memory_size = 10

    @property
    def state_dim(self) -> int:
        return 7

    def reset(self) -> np.ndarray:
        self.pos = 0
        self.current_threshold = 0.5
        self.history = []
        return self._state_for_current_batch()

    def _batch(self):
        start = self.pos
        end = min(len(self.scores), start + self.batch_size)
        return self.scores[start:end], self.labels[start:end]

    def _state_for_current_batch(self) -> np.ndarray:
        s, y = self._batch()
        if len(s) == 0:
            return np.zeros(self.state_dim, dtype=np.float32)
        
        current_fraud_ratio = float(np.mean(y))
        
        if len(self.history) > 0:
            trend = current_fraud_ratio - np.mean(self.history[-5:])
        else:
            trend = 0.0
        
        return np.array([
            float(np.mean(s)),
            float(np.std(s)),
            float(np.min(s)),
            float(np.max(s)),
            current_fraud_ratio,
            float(len(s) / max(1, len(self.scores))),
            float(self.current_threshold),
        ], dtype=np.float32)

    def step(self, threshold: float):
        old_threshold = self.current_threshold
        self.current_threshold = float(threshold)
        
        s, y = self._batch()
        pred = (s >= threshold).astype(np.int64)
        
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        fpr = fp / max(1, fp + tn)
        
        threshold_change_penalty = abs(threshold - old_threshold) * 0.1
        
        reward = float(f1 + 0.5 * recall - self.fpr_penalty * fpr - threshold_change_penalty)
        
        self.pos += self.batch_size
        
        self.history.append(f1)
        if len(self.history) > self.memory_size:
            self.history.pop(0)
        
        done = self.pos >= len(self.scores)
        
        next_state = self._state_for_current_batch() if not done else np.zeros(self.state_dim, dtype=np.float32)
        
        info = {
            "f1": float(f1),
            "recall": float(recall),
            "fpr": float(fpr),
            "threshold": float(threshold),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }
        return next_state, reward, done, info