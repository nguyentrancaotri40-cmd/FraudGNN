# ============================================================
# src/train/train_rl.py — Huấn luyện / áp dụng RL agent (NAF mặc định | DQN rời rạc).
#
# [INFERRED] Paper không nói:
#   - RL train trên tập nào  -> VALIDATION (test giữ độc lập; Fig.2 'feedback' dùng nhãn của val).
#   - Threshold cuối cùng    -> trung bình threshold của policy greedy trên val
#                               (rl.inference="mean"), hoặc theo từng batch ("per_batch").
#   - Feature weights cuối   -> trung bình trọng số greedy trên các batch val.
#   - Không update online trên test (online learning là FUTURE WORK của paper).
# ============================================================
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from src.models.dqn_agent import DQNAgent
from src.models.naf_agent import NAFAgent
from src.models.reward import BatchFraudEnv, weights_from_action


def _env(emb: torch.Tensor, labels, classifier, rl: Dict[str, Any]) -> BatchFraudEnv:
    return BatchFraudEnv(emb, labels, classifier, batch_size=int(rl.get("env_batch_size", 256)),
                         fpr_penalty=float(rl.get("fpr_penalty", 1.0)), recall_weight=float(rl.get("recall_weight", 0.25)))


def make_agent(state_dim: int, n_features: int, rl: Dict[str, Any], device: str):
    common = dict(hidden_dim=int(rl.get("hidden_dim", 128)), gamma=float(rl.get("gamma", 0.99)),
                  lr=float(rl.get("learning_rate", 1e-3)), tau=float(rl.get("tau", 0.005)),
                  buffer_size=int(rl.get("buffer_size", 10000)), batch_size=int(rl.get("update_batch_size", 64)),
                  device=device)
    if str(rl.get("type", "naf")).lower() == "dqn":
        return DQNAgent(state_dim, n_features, n_bins=int(rl.get("n_threshold_bins", 19)), **common)
    return NAFAgent(state_dim, n_features, **common)


def train_rl_on_validation(val_emb: torch.Tensor, val_labels: np.ndarray, classifier, cfg: Dict[str, Any],
                           device: str) -> Tuple[Any, Dict[str, Any]]:
    rl = cfg.get("rl", {})
    env = _env(val_emb, val_labels, classifier, rl)
    agent = make_agent(env.state_dim, val_emb.size(1), rl, device)
    epochs = int(rl.get("epochs", 100))
    losses, ep_rewards = [], []
    print(f"[RL] {agent.kind.upper()} | val N={env.n:,} | env_batch={env.bs} | epochs={epochs}")
    for ep in range(epochs):
        s, done, tot = env.reset(), False, 0.0
        while not done:
            a, thr, fw = agent.act(s, explore=True)
            s2, r, done, _ = env.step(thr, weights_from_action(fw))
            agent.remember(s, a, r, s2, float(done))
            l = agent.update()
            if l is not None:
                losses.append(l)
            s, tot = s2, tot + r
        ep_rewards.append(tot)
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"[RL] epoch {ep+1:3d}/{epochs} episode_reward={tot:.3f} "
                  f"loss={np.mean(losses[-100:]) if losses else float('nan'):.5f}")
    return agent, {"losses": losses, "episode_rewards": ep_rewards}


@torch.no_grad()
def apply_policy(agent, emb: torch.Tensor, labels, classifier, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Chạy policy greedy trên (emb, labels) -> thresholds theo batch, trọng số trung bình."""
    env = _env(emb, labels, classifier, cfg.get("rl", {}))
    s, done = env.reset(), False
    thr_list, ws, spans = [], [], []
    while not done:
        _, thr, fw = agent.act(s, explore=False)
        w = weights_from_action(fw)
        s, _, done, info = env.step(thr, w)
        thr_list.append(thr); ws.append(w); spans.append((info["start"], info["end"]))
    w_mean = np.mean(ws, axis=0).astype(np.float32)
    return {"threshold": float(np.mean(thr_list)), "thresholds": thr_list, "spans": spans,
            "feature_weights": w_mean, "num_steps": len(thr_list)}


__all__ = ["make_agent", "train_rl_on_validation", "apply_policy"]
