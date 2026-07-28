# src/data/split.py
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
    # ✅ TEMPORAL SPLIT (giống paper) - không shuffle
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
        
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train+n_val].copy()
        test_df = df.iloc[n_train+n_val:].copy()
    
    # ============================================================
    # ✅ STRATIFIED TEMPORAL (cho CreditCard - vẫn giữ temporal)
    # ============================================================
    elif strategy == "stratified_temporal":
        time_col = ds.get("time_col")
        if time_col and time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)
            print(f"[SPLIT] Sorting by time_col: {time_col} (giống paper)")
        else:
            raise ValueError(f"time_col '{time_col}' not found!")
        
        # ✅ Lấy fraud và normal riêng, giữ thứ tự thời gian trong từng class
        fraud_df = df[df[label_col] == 1]
        normal_df = df[df[label_col] == 0]
        
        n_total = len(df)
        fraud_ratio = len(fraud_df) / n_total
        
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Tính số fraud trong từng split theo tỷ lệ
        n_fraud_train = int(n_train * fraud_ratio)
        n_fraud_val = int(n_val * fraud_ratio)
        
        # Lấy fraud (theo thứ tự thời gian)
        fraud_train = fraud_df.iloc[:n_fraud_train]
        fraud_remaining = fraud_df.iloc[n_fraud_train:]
        fraud_val = fraud_remaining.iloc[:n_fraud_val]
        fraud_test = fraud_remaining.iloc[n_fraud_val:]
        
        # Lấy normal theo số lượng tương ứng
        n_normal_train = n_train - len(fraud_train)
        n_normal_val = n_val - len(fraud_val)
        
        normal_train = normal_df.iloc[:n_normal_train]
        normal_remaining = normal_df.iloc[n_normal_train:]
        normal_val = normal_remaining.iloc[:n_normal_val]
        normal_test = normal_remaining.iloc[n_normal_val:]
        
        # Ghép lại và sort theo thời gian
        train_df = pd.concat([fraud_train, normal_train]).sort_values(time_col).reset_index(drop=True)
        val_df = pd.concat([fraud_val, normal_val]).sort_values(time_col).reset_index(drop=True)
        test_df = pd.concat([fraud_test, normal_test]).sort_values(time_col).reset_index(drop=True)
    
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
    
    # ✅ Cảnh báo nếu split bị 100% fraud
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if split_df[label_col].mean() == 1.0:
            print(f"  ⚠️ WARNING: {name} split has 100% fraud samples!")
    print(f"{'='*60}")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)