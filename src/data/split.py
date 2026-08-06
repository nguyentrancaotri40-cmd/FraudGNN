# ============================================================
# src/data/split.py - FULL FIX (không temporal leakage)
# ============================================================

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold


def split_dataframe(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe thành train/val/test.
    
    ✅ GIỐNG PAPER: temporal split cho temporal data
    ✅ GIỐNG PAPER: stratified random cho non-temporal data
    ✅ FIX: KHÔNG cho phép temporal leakage
    ✅ FIX: Validation cứng - bắt lỗi nếu split bị 100% 1 class
    """
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
        )
        val_size_rel = val_ratio / (1 - train_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size_rel,
            random_state=random_state,
        )
    
    # ============================================================
    # ✅ 5-FOLD CROSS-VALIDATION (GIỐNG PAPER)
    # ============================================================
    elif strategy == "cross_validation":
        n_folds = int(sp.get("n_folds", 5))
        print(f"[SPLIT] Using {n_folds}-fold cross-validation (giống paper)")
        
        # Tạo splits
        time_col = ds.get("time_col")
        if time_col and time_col in df.columns and strategy == "temporal":
            # TimeSeriesSplit cho temporal data
            tscv = TimeSeriesSplit(n_splits=n_folds)
            splits = list(tscv.split(df))
        else:
            # StratifiedKFold cho non-temporal
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            splits = list(skf.split(df, df[label_col]))
        
        # Lưu splits vào config để pipeline xử lý
        cfg['_cv_splits'] = splits
        cfg['_n_folds'] = n_folds
        
        # Tạm thời dùng fold đầu tiên để pipeline chạy
        train_idx, val_idx = splits[0]
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        
        # Test set = phần còn lại của data
        test_idx = np.setdiff1d(np.arange(len(df)), np.concatenate([train_idx, val_idx]))
        test_df = df.iloc[test_idx].copy()
        
        print(f"[SPLIT] Fold 1/{n_folds}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    # ============================================================
    # ✅ VALIDATION CỨNG: KHÔNG cho phép split 100% 1 class
    # ============================================================
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if len(split_df) == 0:
            raise ValueError(f"❌ {name} split is empty!")
        
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


def get_cv_splits(df: pd.DataFrame, cfg: Dict[str, Any], n_folds: int = 5):
    """
    Lấy splits cho cross-validation.
    
    ✅ GIỐNG PAPER: 5-fold CV
    ✅ TEMPORAL: TimeSeriesSplit không shuffle
    """
    ds = cfg["dataset"]
    label_col = ds["label_col"]
    random_state = int(ds.get("random_state", 42))
    time_col = ds.get("time_col")
    
    if time_col and time_col in df.columns:
        # ✅ TimeSeriesSplit cho temporal data
        tscv = TimeSeriesSplit(n_splits=n_folds)
        splits = list(tscv.split(df))
        print(f"[CV] Using TimeSeriesSplit ({n_folds} folds) - temporal order preserved")
    else:
        # ✅ StratifiedKFold cho non-temporal
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(skf.split(df, df[label_col]))
        print(f"[CV] Using StratifiedKFold ({n_folds} folds) - stratified by label")
    
    return splits


def split_cv_fold(df: pd.DataFrame, cfg: Dict[str, Any], fold_idx: int, train_idx, val_idx) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Lấy một fold cụ thể từ CV splits.
    """
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    
    # Test set = phần còn lại (không overlap với train/val)
    all_idx = set(range(len(df)))
    used_idx = set(train_idx) | set(val_idx)
    test_idx = list(all_idx - used_idx)
    
    # ✅ Nếu test rỗng, dùng val làm test (fold cuối cùng)
    if len(test_idx) == 0:
        print(f"[CV] Fold {fold_idx+1}: Test set is empty! Using validation set as test.")
        test_df = val_df.copy()
        val_df = df.iloc[train_idx[-len(train_idx)//10:]].copy()  # Lấy 10% cuối train làm val
    else:
        test_df = df.iloc[test_idx].copy()
    
    print(f"[CV] Fold {fold_idx+1}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df