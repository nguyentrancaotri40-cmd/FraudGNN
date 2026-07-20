from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataframe(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds = cfg["dataset"]
    sp = cfg.get("split", {})
    label_col = ds["label_col"]
    strategy = sp.get("strategy", "temporal")
    train_ratio = float(sp.get("train_ratio", 0.70))
    val_ratio = float(sp.get("val_ratio", 0.15))
    random_state = int(ds.get("random_state", 42))

    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid train/validation ratios.")

    df = df.copy().reset_index(drop=True)
    
    if strategy == "temporal":
        time_col = ds.get("time_col")
        if time_col and time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)
        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train+n_val].copy()
        test_df = df.iloc[n_train+n_val:].copy()
        
    elif strategy == "stratified_random":
        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=random_state,
            stratify=df[label_col] if df[label_col].nunique() > 1 else None,
        )
        val_size_rel = val_ratio / (1 - train_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size_rel,
            random_state=random_state,
            stratify=temp_df[label_col] if temp_df[label_col].nunique() > 1 else None,
        )
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    # 🔧 THÊM LOG ĐỂ DEBUG
    print(f"\n[SPLIT] Strategy: {strategy}")
    print(f"[SPLIT] Label distribution:")
    print(f"  Train: {train_df[label_col].value_counts().sort_index().to_dict()}")
    print(f"  Val:   {val_df[label_col].value_counts().sort_index().to_dict()}")
    print(f"  Test:  {test_df[label_col].value_counts().sort_index().to_dict()}")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)