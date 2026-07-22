# tests/test_adversarial.py
import pytest
import numpy as np
import torch
from src.eval.metrics import classification_metrics
from src.models.fraudgnn_rl import FraudGNNRL


class AdversarialAttacker:
    @staticmethod
    def random_noise(x: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        noise = np.random.uniform(-epsilon, epsilon, x.shape)
        return x + noise
    
    @staticmethod
    def gradient_attack(model, data, epsilon: float = 0.1):
        # FGSM attack
        ...
    
    @staticmethod
    def label_flipping(labels: np.ndarray, flip_rate: float = 0.1) -> np.ndarray:
        flip_idx = np.random.choice(len(labels), size=int(len(labels) * flip_rate), replace=False)
        adv_labels = labels.copy()
        adv_labels[flip_idx] = 1 - adv_labels[flip_idx]
        return adv_labels


def test_random_noise_attack(): ...
def test_label_flipping_attack(): ...
def test_fgsm_attack(): ...
def test_adversarial_robustness_of_ensemble(): ...