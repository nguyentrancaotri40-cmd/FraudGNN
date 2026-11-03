# ============================================================
# src/data/split.py
#
# Theo paper (Section V-A-1):
#   "For reproducibility, we maintain the original train-test splits where
#    provided, and for PaySim, we use an 80-20 random split while preserving
#    the temporal order of transactions."
#
# Theo b7 đã xác nhận:
#   - IEEE-CIS  → temporal split theo TransactionDT
#   - PaySim    → temporal 80/20 (giữ thứ tự thời gian; "random" trong paper
#                 được hiểu là chọn mẫu phân tầng ngẫu nhiên, nhưng THỨ TỰ
#                 thời gian trong từng tập phải giữ — nhưng để CHẮC CHẮN
#                 không temporal leakage, nhóm chọn temporal split 80/20
#                 theo step, ghi rõ [INFERRED].)
#   - CC2023    → stratified_random 70/15/15 [INFERRED — paper không cho
#                 split gốc cho CC2023]
#
# [WARN] [INFERRED — PaySim] Paper viết "80-20 random split while preserving
# temporal order". Đây là câu mâu thuẫn nội tại (random ≠ preserve order).
# Nhóm diễn giải: 80% giao dịch ĐẦU theo thời gian làm train, 20% CUỐI làm
# test. Không shuffle. → Ghi rõ đây là diễn giải.
# ============================================================

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split


# ----------------------------------------------------------------------
# Hard validation: không cho phép split 100% một class
# ----------------------------------------------------------------------
def _validate_split(name: str, df: pd.DataFrame, label_col: str) -> None:
    if len(df) == 0:
        raise ValueError(f"[ERROR] {name} split rỗng.")
    mean = df[label_col].mean()
    if mean == 0.0 or mean == 1.0:
        raise ValueError(
            f"[ERROR] CRITICAL: {name} split có {mean*100:.0f}% "
            f"{'fraud' if mean == 1.0 else 'normal'}.\n"
            f"→ Split sai chiến lược. Kiểm tra lại time_col / strategy."
        )


# ----------------------------------------------------------------------
# Public: split_dataframe
# ----------------------------------------------------------------------
def split_dataframe(
    df: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Trả về (train_df, val_df, test_df).

    Strategies hỗ trợ:
      - "temporal"           : cắt theo time_col, 1 mốc chung
      - "stratified_random"  : stratified 70/15/15
      - "random"             : random 70/15/15
    """
    ds = cfg["dataset"]
    sp = cfg.get("split", {})
    label_col = ds["label_col"]
    strategy = sp.get("strategy", "temporal")
    train_ratio = float(sp.get("train_ratio", 0.70))
    val_ratio = float(sp.get("val_ratio", 0.15))
    random_state = int(ds.get("random_state", 42))
    time_col = ds.get("time_col")

    if not (0 < train_ratio < 1) or not (0 <= val_ratio < 1) or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio / val_ratio không hợp lệ.")

    df = df.copy().reset_index(drop=True)

    # =========================================================
    # TEMPORAL — 1 mốc cắt chung theo time_col
    # =========================================================
    if strategy == "temporal":
        if not time_col or time_col not in df.columns:
            raise ValueError(
                f"strategy='temporal' nhưng không có time_col='{time_col}'. "
                f"Hãy dùng 'stratified_random' cho dataset không có timestamp."
            )
        df = df.sort_values(time_col).reset_index(drop=True)
        print(f"[SPLIT] Temporal split by '{time_col}' (giữ thứ tự thời gian)")

        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val:].copy()

        # Paper PaySim: 80/20 → không có val riêng, nhưng pipeline cần val
        # → [INFERRED] Nếu test quá nhỏ, tách 10% train cuối làm val
        if len(val_df) == 0 and len(test_df) > 0:
            n_val = max(1, int(len(test_df) * 0.5))
            val_df = test_df.iloc[:n_val].copy()
            test_df = test_df.iloc[n_val:].copy()
            print("[SPLIT] val rỗng → tách 50% test làm val [INFERRED]")

    # =========================================================
    # STRATIFIED RANDOM — CC2023 (b7)
    # =========================================================
    elif strategy == "stratified_random":
        print(f"[SPLIT] Stratified random {train_ratio:.0%}/{val_ratio:.0%}/{1-train_ratio-val_ratio:.0%}")
        strat = df[label_col] if df[label_col].nunique() > 1 else None

        train_df, temp_df = train_test_split(
            df, train_size=train_ratio, random_state=random_state, stratify=strat,
        )
        val_size_rel = val_ratio / (1 - train_ratio)
        strat_temp = temp_df[label_col] if temp_df[label_col].nunique() > 1 else None
        val_df, test_df = train_test_split(
            temp_df, train_size=val_size_rel, random_state=random_state,
            stratify=strat_temp,
        )

    # =========================================================
    # RANDOM — fallback (không stratify)
    # =========================================================
    elif strategy == "random":
        print(f"[SPLIT] Random split {train_ratio:.0%}/{val_ratio:.0%}")
        train_df, temp_df = train_test_split(
            df, train_size=train_ratio, random_state=random_state,
        )
        val_size_rel = val_ratio / (1 - train_ratio)
        val_df, test_df = train_test_split(
            temp_df, train_size=val_size_rel, random_state=random_state,
        )

    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    # =========================================================
    # HARD VALIDATION
    # =========================================================
    _validate_split("Train", train_df, label_col)
    _validate_split("Val", val_df, label_col)
    _validate_split("Test", test_df, label_col)

    print(f"\n{'='*60}")
    print(f"[SPLIT] Strategy: {strategy}")
    print(f"  Total : {len(df):,}")
    print(f"  Train : {len(train_df):,} (fraud {train_df[label_col].mean()*100:.4f}%)")
    print(f"  Val   : {len(val_df):,} (fraud {val_df[label_col].mean()*100:.4f}%)")
    print(f"  Test  : {len(test_df):,} (fraud {test_df[label_col].mean()*100:.4f}%)")
    print(f"{'='*60}")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ----------------------------------------------------------------------
# CV splits — paper: 5-fold (Section V-A-4)
# ----------------------------------------------------------------------
def get_cv_splits(df: pd.DataFrame, cfg: Dict[str, Any], n_folds: int = 5):
    """
    5-fold CV. TimeSeriesSplit nếu có time_col, StratifiedKFold nếu không.
    """
    ds = cfg["dataset"]
    label_col = ds["label_col"]
    random_state = int(ds.get("random_state", 42))
    time_col = ds.get("time_col")

    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
        tscv = TimeSeriesSplit(n_splits=n_folds)
        splits = list(tscv.split(df))
        print(f"[CV] TimeSeriesSplit ({n_folds} folds) — giữ temporal order")
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(skf.split(df, df[label_col]))
        print(f"[CV] StratifiedKFold ({n_folds} folds)")

    return splits


__all__ = ["split_dataframe", "get_cv_splits"]