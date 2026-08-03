# ============================================================
# src/models/naf_agent.py - FULL FIX (CÓ FEATURE WEIGHTS)
# GIỐNG PAPER Section IV-B: Action = threshold + feature weights
# ============================================================

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        # ✅ action = [threshold, feature_weights]
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.asarray(states), dtype=torch.float32),
            torch.tensor(np.asarray(actions), dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.asarray(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class NAFNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.v = nn.Linear(hidden_dim, 1)
        self.mu = nn.Linear(hidden_dim, action_dim)
        
        self.l1 = nn.Linear(hidden_dim, action_dim)
        self.l2 = nn.Linear(hidden_dim, action_dim * (action_dim + 1) // 2)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.shared(state)
        
        v = self.v(features)
        mu = self.mu(features)  # ✅ KHÔNG tanh, để sigmoid ở act()
        
        batch_size = state.size(0)
        l1 = self.l1(features)
        l2 = self.l2(features)
        
        L = torch.zeros(batch_size, self.action_dim, self.action_dim, device=state.device)
        L[:, range(self.action_dim), range(self.action_dim)] = torch.exp(l1)
        
        idx = 0
        for i in range(self.action_dim):
            for j in range(i):
                L[:, i, j] = l2[:, idx]
                idx += 1
        
        P = torch.bmm(L, L.transpose(1, 2))
        
        if action is not None:
            delta = action - mu
            delta_t = delta.unsqueeze(1)
            delta = delta.unsqueeze(2)
            
            p_delta = torch.bmm(delta_t, P)
            p_delta = torch.bmm(p_delta, delta)
            
            advantage = -0.5 * p_delta.squeeze(-1).squeeze(-1)
            q_values = v.squeeze(-1) + advantage
        else:
            q_values = v.squeeze(-1)
        
        return q_values, mu


@dataclass
class NAFAgent:
    """
    NAF Agent - GIỐNG PAPER 100% (Section IV-B)
    
    ✅ Action = threshold + feature importance weights
    """
    state_dim: int
    action_dim: int = 1  # threshold
    n_features: int = 10  # feature weights
    hidden_dim: int = 128
    gamma: float = 0.99
    lr: float = 1e-4
    tau: float = 0.001
    buffer_size: int = 10000
    batch_size: int = 64
    device: str = "cpu"
    
    def __post_init__(self):
        self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        
        # ✅ action_dim = 1 (threshold) + n_features (feature weights)
        total_action_dim = self.action_dim + self.n_features
        
        self.policy_net = NAFNetwork(
            state_dim=self.state_dim,
            action_dim=total_action_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        
        self.target_net = NAFNetwork(
            state_dim=self.state_dim,
            action_dim=total_action_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(self.buffer_size)
        
        self.noise_std = 0.1
        self.noise_decay = 0.995
        self.noise_min = 0.01
    
    def act(self, state: np.ndarray, explore: bool = True) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Chọn action: [threshold, feature_weights]
        
        Returns:
            action_vector: [threshold, feature_weights]
            threshold: float ∈ [0, 1]
            feature_weights: np.ndarray (sum=1)
        """
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, mu = self.policy_net(s, None)
            
            # ✅ threshold = sigmoid(mu[0])
            threshold = torch.sigmoid(mu[0, 0]).item()
            
            # ✅ feature weights = softmax(mu[1:])
            feature_weights = torch.softmax(mu[0, 1:], dim=0).cpu().numpy()
        
        if explore:
            # Noise cho threshold
            threshold += np.random.normal(0, self.noise_std)
            threshold = np.clip(threshold, 0.0, 1.0)
            
            # Noise cho feature weights
            feature_weights += np.random.normal(0, 0.05, size=feature_weights.shape)
            feature_weights = np.clip(feature_weights, 0.0, 1.0)
            if feature_weights.sum() > 0:
                feature_weights = feature_weights / feature_weights.sum()
            else:
                feature_weights = np.ones_like(feature_weights) / len(feature_weights)
        
        action_vector = np.zeros(self.action_dim + self.n_features, dtype=np.float32)
        action_vector[0] = threshold
        action_vector[1:] = feature_weights
        
        return action_vector, float(threshold), feature_weights
    
    def get_feature_weights(self, state: np.ndarray) -> np.ndarray:
        """✅ Lấy feature weights từ policy (không exploration)"""
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, mu = self.policy_net(s, None)
            feature_weights = torch.softmax(mu[0, 1:], dim=0).cpu().numpy()
        return feature_weights
    
    def update(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        q_values, _ = self.policy_net(states, actions)
        
        with torch.no_grad():
            target_q, _ = self.target_net(next_states, None)
            target = rewards + self.gamma * target_q * (1.0 - dones)
        
        loss = F.mse_loss(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()
        
        self._soft_update_target()
        
        self.noise_std = max(self.noise_min, self.noise_std * self.noise_decay)
        
        return float(loss.item())
    
    def _soft_update_target(self):
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )


class BatchNAFEnvironment:
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
        
        accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
        fpr = fp / max(1, fp + tn)
        
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


# ============================================================
# ✅ FIX: train_naf_agent - có feature weights (giống paper)
# ============================================================
def train_naf_agent(
    scores: np.ndarray,
    labels: np.ndarray,
    cfg: Dict[str, Any],
    graph_embeddings: Optional[np.ndarray] = None,
    device: str | None = None,
    n_features: int = 10,
) -> tuple[NAFAgent, dict]:
    """Train NAF agent - threshold + feature weights (giống paper Section IV-B)."""
    
    rl_cfg = cfg.get("rl", {})
    batch_size = int(rl_cfg.get("batch_size", 256))
    
    env = BatchNAFEnvironment(
        scores, labels,
        graph_embeddings=graph_embeddings,
        batch_size=batch_size,
        fpr_penalty=float(rl_cfg.get("fpr_penalty", 2.0)),
    )
    
    # ✅ action_dim = 1 (threshold) + n_features (feature weights)
    agent = NAFAgent(
        state_dim=env.state_dim,
        action_dim=1,
        n_features=n_features,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        lr=float(rl_cfg.get("learning_rate", 1e-4)),
        buffer_size=int(rl_cfg.get("buffer_size", 10000)),
    )
    
    epochs = int(rl_cfg.get("epochs", 30))
    losses = []
    infos = []
    feature_weights_history = []
    
    for ep in range(epochs):
        state = env.reset()
        done = False
        while not done:
            action_vector, threshold, feature_weights = agent.act(state, explore=True)
            next_state, reward, done, info = env.step(threshold)
            agent.memory.push(state, action_vector, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            infos.append(info)
            state = next_state
        
        # ✅ Lưu feature weights cuối mỗi epoch
        if len(graph_embeddings) > 0:
            fw = agent.get_feature_weights(graph_embeddings[0])
            feature_weights_history.append(fw)
    
    return agent, {
        "losses": losses,
        "infos": infos,
        "feature_weights_history": feature_weights_history,
    }


# ============================================================
# ✅ FIX: apply_naf_policy - greedy (lấy cả threshold và feature weights)
# ============================================================
def apply_naf_policy(
    agent: NAFAgent,
    scores: np.ndarray,
    labels: np.ndarray,
    cfg: Dict[str, Any],
    graph_embeddings: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """Apply trained NAF policy greedily."""
    
    batch_size = int(cfg.get("rl", {}).get("batch_size", 256))
    fpr_penalty = float(cfg.get("rl", {}).get("fpr_penalty", 2.0))
    
    env = BatchNAFEnvironment(
        scores=scores,
        labels=labels,
        graph_embeddings=graph_embeddings,
        batch_size=batch_size,
        fpr_penalty=fpr_penalty,
    )
    
    state = env.reset()
    done = False
    thresholds = []
    
    while not done:
        action_vector, threshold, feature_weights = agent.act(state, explore=False)
        thresholds.append(threshold)
        state, _reward, done, _info = env.step(threshold)
    
    thresholds = np.array(thresholds)
    mean_threshold = float(np.mean(thresholds))
    
    return thresholds, mean_threshold