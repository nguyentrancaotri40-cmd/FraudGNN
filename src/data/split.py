# ============================================================
# src/data/split.py
#
# Paper V-A-1-c: "we maintain the original train-test splits where provided,
# and for PaySim, we use an 80-20 random split while preserving the temporal
# order of transactions."
#
# [INFERRED] Diễn giải thống nhất cho cả 3 dataset (80/20 ngoài, val lấy từ 80%):
#   train 0.64 / val 0.16 / test 0.20
#   - temporal           : cắt theo time_col (PaySim, IEEE-CIS) — "random" của
#                          paper mâu thuẫn với "preserving temporal order";
#                          chọn temporal để không rò rỉ tương lai.
#   - stratified_random  : CC2023 (không có cột thời gian)
#   IEEE-CIS: Kaggle không công bố nhãn test -> chỉ có train_transaction -> temporal.
#
# CV 5-fold (V-A-4): temporal -> expanding-window TimeSeriesSplit;
#                    không có thời gian -> StratifiedKFold.
# ============================================================
from __future__ import annotations

from typing import Any, Dict, Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split


def _validate_split(name: str, df: pd.DataFrame, label_col: str) -> None:
    if len(df) == 0:
        raise ValueError(f"{name} split rỗng.")
    m = df[label_col].mean()
    if m in (0.0, 1.0):
        raise ValueError(f"{name} split chỉ có một lớp (fraud rate={m:.0%}). Kiểm tra strategy/time_col.")


def _temporal_cut(df: pd.DataFrame, time_col: str, train_ratio: float, val_ratio: float):
    df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    n = len(df)
    a, b = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def split_dataframe(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds, sp = cfg["dataset"], cfg.get("split", {})
    label_col, time_col = ds["label_col"], ds.get("time_col")
    strategy = sp.get("strategy", "temporal")
    tr, va = float(sp.get("train_ratio", 0.64)), float(sp.get("val_ratio", 0.16))
    seed = int(ds.get("random_state", 42))
    if not (0 < tr < 1 and 0 < va < 1 and tr + va < 1):
        raise ValueError("train_ratio/val_ratio không hợp lệ.")

    df = df.reset_index(drop=True)
    if strategy == "temporal":
        if not time_col:
            raise ValueError("strategy='temporal' cần dataset.time_col.")
        train_df, val_df, test_df = _temporal_cut(df, time_col, tr, va)
    elif strategy in ("stratified_random", "random"):
        strat = df[label_col] if strategy == "stratified_random" and df[label_col].nunique() > 1 else None
        train_df, rest = train_test_split(df, train_size=tr, random_state=seed, stratify=strat)
        s2 = rest[label_col] if strat is not None else None
        val_df, test_df = train_test_split(rest, train_size=va / (1 - tr), random_state=seed, stratify=s2)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    for name, part in (("Train", train_df), ("Val", val_df), ("Test", test_df)):
        _validate_split(name, part, label_col)

    print(f"[SPLIT] {strategy}: train={len(train_df):,} ({train_df[label_col].mean()*100:.4f}% fraud) | "
          f"val={len(val_df):,} ({val_df[label_col].mean()*100:.4f}%) | "
          f"test={len(test_df):,} ({test_df[label_col].mean()*100:.4f}%)")
    return tuple(p.reset_index(drop=True) for p in (train_df, val_df, test_df))  # type: ignore


def get_cv_splits(df: pd.DataFrame, cfg: Dict[str, Any], n_folds: int = 5
                  ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Sinh (train_df, val_df, test_df) cho mỗi fold. Val = phần cuối/ngẫu nhiên 20% của train-fold."""
    ds, sp = cfg["dataset"], cfg.get("split", {})
    label_col, time_col = ds["label_col"], ds.get("time_col")
    seed = int(ds.get("random_state", 42))
    val_frac = float(sp.get("cv_val_frac", 0.2))
    df = df.reset_index(drop=True)

    if sp.get("strategy", "temporal") == "temporal" and time_col:
        df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
        print(f"[CV] TimeSeriesSplit({n_folds}) — expanding window, giữ thứ tự thời gian")
        for tr_idx, te_idx in TimeSeriesSplit(n_splits=n_folds).split(df):
            n_va = max(1, int(len(tr_idx) * val_frac))
            yield (df.iloc[tr_idx[:-n_va]].copy().reset_index(drop=True),
                   df.iloc[tr_idx[-n_va:]].copy().reset_index(drop=True),
                   df.iloc[te_idx].copy().reset_index(drop=True))
    else:
        print(f"[CV] StratifiedKFold({n_folds})")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr_idx, te_idx in skf.split(df, df[label_col]):
            tr_df = df.iloc[tr_idx]
            train_df, val_df = train_test_split(tr_df, test_size=val_frac, random_state=seed,
                                                stratify=tr_df[label_col])
            yield (train_df.reset_index(drop=True), val_df.reset_index(drop=True),
                   df.iloc[te_idx].copy().reset_index(drop=True))


def resample_fraud_ratio(df: pd.DataFrame, ratio_pct: float, label_col: str,
                         time_col: str | None, seed: int = 42) -> pd.DataFrame:
    """
    Thí nghiệm class-imbalance (V-B-3, Fig. 4): ép tỉ lệ fraud về ratio_pct (%) bằng
    random sampling và GIỮ thứ tự thời gian.
      - ratio < hiện tại : giữ toàn bộ normal, bớt fraud.
      - ratio > hiện tại : giữ toàn bộ fraud, bớt normal.
    [INFERRED] Paper chỉ nói "random sampling preserving temporal order".
    """
    r = ratio_pct / 100.0
    rng = np.random.default_rng(seed)
    pos, neg = df[df[label_col] == 1], df[df[label_col] == 0]
    cur = len(pos) / max(1, len(df))
    if r >= cur:
        n_neg = int(round(len(pos) * (1 - r) / r))
        neg = neg.iloc[np.sort(rng.choice(len(neg), size=min(n_neg, len(neg)), replace=False))]
    else:
        n_pos = int(round(len(neg) * r / (1 - r)))
        pos = pos.iloc[np.sort(rng.choice(len(pos), size=min(n_pos, len(pos)), replace=False))]
    out = pd.concat([pos, neg])
    out = out.sort_values(time_col, kind="stable") if time_col else out.sort_index()
    return out.reset_index(drop=True)


__all__ = ["split_dataframe", "get_cv_splits", "resample_fraud_ratio"]
