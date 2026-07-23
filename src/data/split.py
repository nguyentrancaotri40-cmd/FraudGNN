from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataframe(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    ORIGINAL KAGGLE SPLIT - GIỐNG PAPER 100%
    Paper: temporal split based on time index
    - Train: 70% (first)
    - Val: 15% (next)
    - Test: 15% (last)
    """
    ds = cfg["dataset"]
    sp = cfg.get("split", {})
    label_col = ds["label_col"]
    
    # ✅ FORCE temporal split - giống paper
    strategy = "temporal"  # Luôn là temporal, không cho phép khác
    train_ratio = float(sp.get("train_ratio", 0.70))
    val_ratio = float(sp.get("val_ratio", 0.15))
    random_state = int(ds.get("random_state", 42))

    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid train/validation ratios.")

    df = df.copy().reset_index(drop=True)
    
    # ✅ BẮT BUỘC: Sort theo thời gian (giống paper)
    time_col = ds.get("time_col")
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
        print(f"[SPLIT] Sorting by time_col: {time_col} (giống paper)")
    else:
        raise ValueError(f"time_col '{time_col}' not found! Paper yêu cầu temporal split.")
    
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    # ✅ ORIGINAL: Train = first 70%, Val = next 15%, Test = last 15%
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train+n_val].copy()
    test_df = df.iloc[n_train+n_val:].copy()
    
    # ✅ LOG CHI TIẾT
    print(f"\n{'='*60}")
    print(f"[SPLIT] ORIGINAL KAGGLE TEMPORAL SPLIT (giống paper)")
    print(f"{'='*60}")
    print(f"  Total samples: {n:,}")
    print(f"  Train: {len(train_df):,} ({len(train_df)/n*100:.1f}%) - first {n_train:,} samples")
    print(f"  Val:   {len(val_df):,} ({len(val_df)/n*100:.1f}%) - next {n_val:,} samples")
    print(f"  Test:  {len(test_df):,} ({len(test_df)/n*100:.1f}%) - last {len(test_df):,} samples")
    
    print(f"\n[SPLIT] Label distribution (giống paper):")
    print(f"  Train - fraud: {train_df[label_col].sum():,} ({train_df[label_col].mean()*100:.4f}%)")
    print(f"  Val   - fraud: {val_df[label_col].sum():,} ({val_df[label_col].mean()*100:.4f}%)")
    print(f"  Test  - fraud: {test_df[label_col].sum():,} ({test_df[label_col].mean()*100:.4f}%)")
    
    # ✅ ĐẢM BẢO: Mỗi split có ít nhất 1 fraud sample
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if split_df[label_col].sum() == 0:
            print(f"  ⚠️ WARNING: {name} split has ZERO fraud samples!")
            print(f"     → This will cause AUC-ROC = NaN!")
    
    print(f"{'='*60}")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)