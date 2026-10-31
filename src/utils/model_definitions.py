"""
Định nghĩa chính thức các biến thể model trong dự án.
NGUỒN SỰ THẬT DUY NHẤT — mọi nơi khác phải suy ra tên model từ flags qua resolve_model_name().
"""

MODEL_DEFINITIONS = {
    "baseline": {
        "display_name": "FraudGNN-RL (Baseline)",
        "graph_type": "Transaction Similarity Graph (hard edges only)",
        "description": "Baseline — đúng paper (Cui et al., 2025) 100%",
        "flags": {"hard_edges": True, "soft_edges": False, "pruning": False},
    },
    "augmented_graph": {
        "display_name": "Augmented Graph (FraudGNN-RL+)",
        "graph_type": "Behavior-Augmented Transaction Graph",
        "description": (
            "Model cải tiến: Hard Edges (causal) + TSSGC + DQN/NAF + "
            "Federated Learning + Soft Edges (causal) + Pruning"
        ),
        "flags": {"hard_edges": True, "soft_edges": True, "pruning": True},
    },
    "ablation_soft_only": {
        "display_name": "Ablation: Soft Only",
        "graph_type": "Behavioral Similarity Graph (soft edges only)",
        "description": "hard_edges=False, soft_edges=True — kiểm chứng causal structure",
        "flags": {"hard_edges": False, "soft_edges": True, "pruning": False},
    },
    "ablation_pruning_only": {
        "display_name": "Ablation: Pruning Only",
        "graph_type": "Transaction Similarity Graph (hard edges only)",
        "description": "hard_edges=True, pruning=True — kiểm chứng pruning tự nó có tác dụng",
        "flags": {"hard_edges": True, "soft_edges": False, "pruning": True},
    },
}

def resolve_model_name(flags: dict) -> str:
    """Suy ra display_name từ flags thực tế."""
    hard = flags.get("hard_edges", True)
    soft = flags.get("soft_edges", False)
    prune = flags.get("pruning", False)

    if hard and not soft:
        return "FraudGNN-RL (Baseline)" if not prune else "FraudGNN-RL (Pruned)"
    if soft and not hard:
        return "Ablation: Soft Only"
    if hard and soft:
        return "Augmented Graph" if prune else "Augmented Graph (chưa nén)"
    return "Invalid (no edges enabled)"
