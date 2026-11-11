# src/utils/config.py
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_FLAGS: Dict[str, bool] = {
    # --- Thành phần của PAPER (dùng cho ablation: w/o GNN, w/o RL, w/o FL) ---
    "gnn": True,
    "rl": True,
    "federated": True,
    # --- Mở rộng NGOÀI paper (OFF = bản reproduction 100% paper) ---
    "hard_edges": True,
    "soft_edges": False,
    "pruning": False,
}


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config (UTF-8) và điền flags mặc định."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["flags"] = {**DEFAULT_FLAGS, **(cfg.get("flags") or {})}
    return cfg


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Trả về bản sao của base được cập nhật đệ quy bởi upd."""
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_flags(cfg: Dict[str, Any]) -> Dict[str, bool]:
    return {**DEFAULT_FLAGS, **(cfg.get("flags") or {})}


def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
