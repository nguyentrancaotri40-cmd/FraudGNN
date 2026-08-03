# ============================================================
# src/data/split.py - FULL FIX (không temporal leakage)
# ============================================================

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
    
    # ============================================================
    # ✅ TEMPORAL SPLIT (giống paper) - 1 mốc cắt chung
    # ============================================================
    if strategy == "temporal":
        time_col = ds.get("time_col")
        if time_col and time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)
            print(f"[SPLIT] Sorting by time_col: {time_col} (giống paper)")
        else:
            raise ValueError(f"time_col '{time_col}' not found! Paper yêu cầu temporal split.")
        
        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        # ✅ 1 mốc cắt chung cho toàn bộ df
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train+n_val].copy()
        test_df = df.iloc[n_train+n_val:].copy()
    
    # ============================================================
    # ⚠️ STRATIFIED_TEMPORAL - CÓ THỂ GÂY TEMPORAL LEAKAGE
    # ============================================================
    elif strategy == "stratified_temporal":
        print("[WARNING] stratified_temporal may cause temporal leakage! Use 'temporal' instead.")
        time_col = ds.get("time_col")
        if time_col and time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)
        else:
            raise ValueError(f"time_col '{time_col}' not found!")
        
        # ✅ 1 mốc cắt chung, giữ tỷ lệ fraud tự nhiên
        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train+n_val].copy()
        test_df = df.iloc[n_train+n_val:].copy()
        
        # Log tỷ lệ fraud trong từng split
        print(f"[SPLIT] Fraud ratio in Train: {train_df[label_col].mean():.4f}")
        print(f"[SPLIT] Fraud ratio in Val: {val_df[label_col].mean():.4f}")
        print(f"[SPLIT] Fraud ratio in Test: {test_df[label_col].mean():.4f}")
    
    # ============================================================
    # ✅ STRATIFIED RANDOM (cho Credit Card 2023)
    # ============================================================
    elif strategy == "stratified_random":
        print(f"[SPLIT] Using stratified random split (giống paper)")
        
        # 1. Split train + temp
        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=random_state,
            stratify=df[label_col] if df[label_col].nunique() > 1 else None,
        )
        
        # 2. Split temp → val + test
        val_size_rel = val_ratio / (1 - train_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size_rel,
            random_state=random_state,
            stratify=temp_df[label_col] if temp_df[label_col].nunique() > 1 else None,
        )
    
    # ============================================================
    # ✅ RANDOM SPLIT (fallback)
    # ============================================================
    elif strategy == "random":
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

    # ============================================================
    # ✅ VALIDATION CỨNG: KHÔNG cho phép split 100% 1 class
    # ============================================================
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        label_mean = split_df[label_col].mean()
        if label_mean == 0.0 or label_mean == 1.0:
            raise ValueError(
                f"❌ CRITICAL: {name} split has {label_mean*100:.0f}% {'fraud' if label_mean == 1.0 else 'normal'} samples!\n"
                f"  This means temporal split is WRONG for this dataset.\n"
                f"  For datasets without real timestamps (e.g., CreditCard2023), use 'stratified_random' strategy.\n"
                f"  Current strategy: {strategy}\n"
                f"  Time col: {ds.get('time_col', 'None')}\n"
                f"  Dataset: {cfg.get('experiment', {}).get('dataset', 'unknown')}"
            )

    # ✅ LOG CHI TIẾT
    print(f"\n{'='*60}")
    print(f"[SPLIT] Strategy: {strategy}")
    print(f"{'='*60}")
    print(f"  Total samples: {len(df):,}")
    print(f"  Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    print(f"\n[SPLIT] Label distribution:")
    print(f"  Train - fraud: {train_df[label_col].sum():,} ({train_df[label_col].mean()*100:.4f}%)")
    print(f"  Val   - fraud: {val_df[label_col].sum():,} ({val_df[label_col].mean()*100:.4f}%)")
    print(f"  Test  - fraud: {test_df[label_col].sum():,} ({test_df[label_col].mean()*100:.4f}%)")
    print(f"{'='*60}")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)