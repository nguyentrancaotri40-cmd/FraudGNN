# src/utils/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config file with UTF-8 encoding."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths: str | Path) -> None:
    """Ensure all directories exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)