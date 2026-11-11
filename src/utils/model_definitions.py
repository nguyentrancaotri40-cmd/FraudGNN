"""
Định nghĩa các biến thể model — NGUỒN SỰ THẬT DUY NHẤT.

1) PAPER_ABLATIONS : 7 biến thể ablation của paper (Section V-B-2).
2) MODEL_DEFINITIONS: baseline (= paper) và các biến thể mở rộng của riêng bạn.
"""

# ----------------------------------------------------------------------
# Ablation của paper: tên -> flags (gnn / rl / federated)
# ----------------------------------------------------------------------
PAPER_ABLATIONS = {
    "Full model": {"gnn": True, "rl": True, "federated": True},
    "w/o GNN": {"gnn": False, "rl": True, "federated": True},
    "w/o RL": {"gnn": True, "rl": False, "federated": True},
    "w/o FL": {"gnn": True, "rl": True, "federated": False},
    "w/o GNN&RL": {"gnn": False, "rl": False, "federated": True},
    "w/o GNN&FL": {"gnn": False, "rl": True, "federated": False},
    "w/o RL&FL": {"gnn": True, "rl": False, "federated": False},
}

# ----------------------------------------------------------------------
# Biến thể đồ thị (ngoài paper)
# ----------------------------------------------------------------------
MODEL_DEFINITIONS = {
    "baseline": {
        "display_name": "FraudGNN-RL (Baseline)",
        "graph_type": "Transaction Similarity Graph (hard edges only)",
        "description": "Baseline — reproduction theo paper (Cui et al., 2025)",
        "flags": {"hard_edges": True, "soft_edges": False, "pruning": False},
    },
    "augmented_graph": {
        "display_name": "Augmented Graph (FraudGNN-RL+)",
        "graph_type": "Behavior-Augmented Transaction Graph",
        "description": "Hard edges + Soft edges (causal) + Pruning — NGOÀI paper",
        "flags": {"hard_edges": True, "soft_edges": True, "pruning": True},
    },
    "ablation_soft_only": {
        "display_name": "Ablation: Soft Only",
        "graph_type": "Behavioral Similarity Graph (soft edges only)",
        "description": "hard_edges=False, soft_edges=True",
        "flags": {"hard_edges": False, "soft_edges": True, "pruning": False},
    },
    "ablation_pruning_only": {
        "display_name": "Ablation: Pruning Only",
        "graph_type": "Transaction Similarity Graph (hard edges only)",
        "description": "hard_edges=True, pruning=True",
        "flags": {"hard_edges": True, "soft_edges": False, "pruning": True},
    },
}


def resolve_model_name(flags: dict) -> str:
    """Suy ra tên model từ flags thực tế."""
    hard = flags.get("hard_edges", True)
    soft = flags.get("soft_edges", False)
    prune = flags.get("pruning", False)

    if hard and not soft:
        return "FraudGNN-RL (Pruned)" if prune else "FraudGNN-RL (Baseline)"
    if soft and not hard:
        return "Ablation: Soft Only"
    if hard and soft:
        return "Augmented Graph" if prune else "Augmented Graph (chưa nén)"
    return "Invalid (no edges enabled)"
