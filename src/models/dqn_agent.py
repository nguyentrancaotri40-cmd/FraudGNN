# src/models/dqn_agent.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, List, Tuple, Optional
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
    n_features: int = 10
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
        self.feature_weights = torch.ones(self.n_features, device=self.device) / self.n_features

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randrange(len(self.thresholds))
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.policy_net(s).argmax(dim=1).item())

    def threshold(self, action: int) -> float:
        return float(self.thresholds[action])

    def update(self, batch_size: int = 64) -> float | None:
        """
        ✅ GIỐNG PAPER: Vanilla DQN update (Eq 12)
        
        Paper: target = r + γ * max_a' Q(s', a'; θ⁻)
        Dùng target network để chọn AND đánh giá action.
        """
        if len(self.memory) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # ✅ GIỐNG PAPER: Vanilla DQN
        # Dùng target network để tính max Q cho next state
        with torch.no_grad():
            # Chọn và đánh giá bằng target_net (giống paper Eq 12)
            next_q = self.target_net(next_states).max(dim=1)[0]
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
    """
    Offline RL environment cho adaptive threshold selection.
    
    ✅ GIỐNG PAPER: State = graph embedding từ TSSGC
    ✅ GIỐNG PAPER: Reward = combination of accuracy and FPR
    """
    
    def __init__(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        graph_embeddings: Optional[np.ndarray] = None,
        batch_size: int = 256,
        fpr_penalty: float = 2.0,
    ):
        self.scores = np.asarray(scores, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.graph_embeddings = graph_embeddings
        self.batch_size = int(batch_size)
        self.fpr_penalty = float(fpr_penalty)
        self.pos = 0
        self.current_threshold = 0.5
        self.history = []
        self.threshold_history = []
        self.memory_size = 10
        
        if self.graph_embeddings is not None:
            self.embedding_dim = self.graph_embeddings.shape[1]
        else:
            self.embedding_dim = 64

    @property
    def state_dim(self) -> int:
        return self.embedding_dim

    def reset(self) -> np.ndarray:
        self.pos = 0
        self.current_threshold = 0.5
        self.history = []
        self.threshold_history = []
        return self._state_for_current_batch()

    def _batch(self):
        start = self.pos
        end = min(len(self.scores), start + self.batch_size)
        return self.scores[start:end], self.labels[start:end]

    def _state_for_current_batch(self) -> np.ndarray:
        """
        ✅ GIỐNG PAPER: State = graph embedding từ TSSGC
        """
        s, y = self._batch()
        if len(s) == 0:
            return np.zeros(self.state_dim, dtype=np.float32)
        
        if self.graph_embeddings is not None:
            start = self.pos
            end = min(len(self.scores), start + self.batch_size)
            batch_embeddings = self.graph_embeddings[start:end]
            state = np.mean(batch_embeddings, axis=0)
            return state.astype(np.float32)
        
        # Fallback (không khuyến nghị)
        return np.array([
            float(np.mean(s)),
            float(np.std(s)),
            float(np.min(s)),
            float(np.max(s)),
            float(np.mean(y)),
            float(len(s) / max(1, len(self.scores))),
            float(self.current_threshold),
        ], dtype=np.float32)

    def _calculate_pos_step(self, threshold: float, scores: np.ndarray, labels: np.ndarray) -> int:
        if len(scores) == 0:
            return 1
        
        fraud_ratio = float(np.mean(labels))
        threshold_factor = 1.0 + (0.5 - threshold) * 1.5
        threshold_factor = max(0.3, min(2.0, threshold_factor))
        fraud_factor = 0.5 + fraud_ratio * 1.0
        
        base_step = self.batch_size
        pos_step = int(base_step * threshold_factor * fraud_factor)
        
        remaining = len(self.scores) - self.pos
        pos_step = min(pos_step, remaining)
        
        return max(1, pos_step)

    def step(self, threshold: float):
        old_threshold = self.current_threshold
        self.current_threshold = float(threshold)
        self.threshold_history.append(float(threshold))
        
        s, y = self._batch()
        if len(s) == 0:
            next_state = np.zeros(self.state_dim, dtype=np.float32)
            return next_state, 0.0, True, {"done": True}
        
        pred = (s >= threshold).astype(np.int64)
        
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        
        # ============================================================
        # ✅ GIỐNG PAPER: Reward = combination of accuracy and FPR
        # Paper: "Reward rt: A combination of detection accuracy and false positive rate"
        # ============================================================
        accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
        fpr = fp / max(1, fp + tn)
        
        # ✅ Reward = accuracy - fpr_penalty * fpr (giống paper)
        reward = float(accuracy - self.fpr_penalty * fpr)
        
        pos_step = self._calculate_pos_step(threshold, s, y)
        self.pos += pos_step
        
        self.history.append(accuracy)
        if len(self.history) > self.memory_size:
            self.history.pop(0)
        if len(self.threshold_history) > self.memory_size:
            self.threshold_history.pop(0)
        
        done = self.pos >= len(self.scores)
        next_state = self._state_for_current_batch() if not done else np.zeros(self.state_dim, dtype=np.float32)
        
        info = {
            "accuracy": float(accuracy),
            "fpr": float(fpr),
            "threshold": float(threshold),
            "pos_step": int(pos_step),
            "pos": int(self.pos),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }
        return next_state, reward, done, info