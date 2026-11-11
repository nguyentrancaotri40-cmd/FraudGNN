# ============================================================
# src/models/reward.py — Reward + môi trường RL (Section IV-B / IV-D).
#
# Paper: State = graph embedding (TSSGC); Action = (fraud threshold, feature importance
# weights); Reward = "combination of detection accuracy and false positive rate".
# Paper KHÔNG cho công thức -> [INFERRED]:
#       reward = F1 + recall_weight * Recall - fpr_penalty * FPR
#   (accuracy thuần bị chi phối bởi lớp normal khi fraud ~0.1% nên dùng F1/Recall làm "detection accuracy")
#
# Môi trường (Fig. 2: Update Graph -> TSSGC -> RL action -> Classification -> Feedback -> Update policy):
#   - 1 episode = quét tuần tự các batch KHÔNG chồng lấn trên tập (theo thứ tự thời gian).
#   - state  = mean pooling embedding (trước feature weights) của batch hiện tại.
#   - action = [threshold, raw feature weights] ; w = D * a_fw / sum(a_fw).
#   - score  = sigmoid(classifier(embedding * w)) => feature weights THỰC SỰ ảnh hưởng điểm số/reward.
# ============================================================
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


def compute_reward(tp: int, fp: int, tn: int, fn: int, fpr_penalty: float = 1.0,
                   recall_weight: float = 0.25) -> float:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    fpr = fp / max(1, fp + tn)
    return float(f1 + recall_weight * recall - fpr_penalty * fpr)


def weights_from_action(fw_raw: np.ndarray) -> np.ndarray:
    """a_fw in [0,1]^D  ->  w, mean(w)=1."""
    fw = np.clip(np.asarray(fw_raw, dtype=np.float64), 1e-6, None)
    return (fw.size * fw / fw.sum()).astype(np.float32)


class BatchFraudEnv:
    """
    embeddings: [N, D] (tensor, đã ở thiết bị tính toán) ; labels: [N] ;
    classifier: FraudClassifier (eval, đóng băng) dùng để chấm điểm batch.
    """

    def __init__(self, embeddings: torch.Tensor, labels: np.ndarray, classifier: torch.nn.Module,
                 batch_size: int = 256, fpr_penalty: float = 1.0, recall_weight: float = 0.25):
        self.emb = embeddings
        self.labels = np.asarray(labels, dtype=np.int64)
        self.clf = classifier.eval()
        self.bs, self.fpr_penalty, self.recall_weight = int(batch_size), float(fpr_penalty), float(recall_weight)
        self.n, self.state_dim = embeddings.size(0), embeddings.size(1)
        self.pos = 0

    def _state(self) -> np.ndarray:
        if self.pos >= self.n:
            return np.zeros(self.state_dim, dtype=np.float32)
        return self.emb[self.pos:self.pos + self.bs].mean(0).float().cpu().numpy()

    def reset(self) -> np.ndarray:
        self.pos = 0
        return self._state()

    @torch.no_grad()
    def step(self, threshold: float, weights: Optional[np.ndarray] = None
             ) -> Tuple[np.ndarray, float, bool, Dict]:
        a, b = self.pos, min(self.n, self.pos + self.bs)
        e = self.emb[a:b]
        if weights is not None:
            e = e * torch.as_tensor(weights, dtype=e.dtype, device=e.device)
        s = torch.sigmoid(self.clf(e)).cpu().numpy()
        y = self.labels[a:b]
        pred = s >= threshold
        tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
        r = compute_reward(tp, fp, tn, fn, self.fpr_penalty, self.recall_weight)
        self.pos = b
        done = self.pos >= self.n
        info = dict(tp=tp, fp=fp, tn=tn, fn=fn, threshold=float(threshold), start=a, end=b)
        return self._state(), r, done, info


__all__ = ["compute_reward", "weights_from_action", "BatchFraudEnv"]
