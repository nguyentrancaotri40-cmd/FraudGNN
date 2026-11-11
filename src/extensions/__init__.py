"""Extensions NGOÀI paper (mặc định TẮT). Xem EXTENSIONS.md."""
from .soft_behavior_graph import (
    build_soft_behavior_edges,
    merge_hard_soft_edges,
    summarize_soft_edges,
)
from .pruning import (
    amount_schedule,
    get_pruning_stats,
    magnitude_prune_,
)

__all__ = [
    "build_soft_behavior_edges", "merge_hard_soft_edges", "summarize_soft_edges",
    "amount_schedule", "get_pruning_stats", "magnitude_prune_",
]
