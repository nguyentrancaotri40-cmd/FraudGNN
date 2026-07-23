from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import pandas as pd


def _sample(df: pd.DataFrame, n: int | None, random_state: int = 42) -> pd.DataFrame:
    if n is None or n <= 0 or n >= len(df):
        return df.reset_index(drop=True)

    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


def _sample_by_class(
    df: pd.DataFrame,
    label_col: str,
    normal_sample_size: int | None = None,
    fraud_sample_size: int | str | None = None,
    random_state: int = 42,
    time_col: str | None = None,
) -> pd.DataFrame:
    """Sample normal and fraud rows separately.

    Expected fraud label:
      0 = normal
      1 = fraud

    Config examples:
      normal_sample_size: 30000
      fraud_sample_size: all
    """
    if normal_sample_size is None and fraud_sample_size is None:
        return df.reset_index(drop=True)

    normal_df = df[df[label_col] == 0]
    fraud_df = df[df[label_col] == 1]

    if normal_sample_size is None:
        sampled_normal = normal_df
    else:
        normal_sample_size = int(normal_sample_size)

        if normal_sample_size <= 0 or normal_sample_size >= len(normal_df):
            sampled_normal = normal_df
        else:
            sampled_normal = normal_df.sample(
                n=normal_sample_size,
                random_state=random_state,
            )

    if fraud_sample_size is None or str(fraud_sample_size).lower() == "all":
        sampled_fraud = fraud_df
    else:
        fraud_sample_size = int(fraud_sample_size)

        if fraud_sample_size <= 0 or fraud_sample_size >= len(fraud_df):
            sampled_fraud = fraud_df
        else:
            sampled_fraud = fraud_df.sample(
                n=fraud_sample_size,
                random_state=random_state,
            )

    out = pd.concat([sampled_normal, sampled_fraud], axis=0)

    # Với temporal split, sau khi sample phải sắp xếp lại theo thời gian.
    # Nếu không sort, sample random sẽ làm vỡ thứ tự thời gian.
    if time_col and time_col in out.columns:
        out = out.sort_values(time_col)

    return out.reset_index(drop=True)


def load_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load PaySim, CreditCard2023, or IEEE-CIS style data.

    Expected config keys:
    - dataset.path for single CSV datasets.
    - dataset.transaction_path + dataset.identity_path for IEEE-CIS.

    Optional sampling keys:
    - dataset.sample_rows
    - dataset.normal_sample_size
    - dataset.fraud_sample_size
    - dataset.sample_frac: fraction of data to sample (applied to ALL data before split)
    """
    ds = cfg["dataset"]

    random_state = int(ds.get("random_state", 42))
    sample_rows = ds.get("sample_rows")

    if "transaction_path" in ds:
        tx_path = Path(ds["transaction_path"])
        id_path = Path(ds.get("identity_path", ""))
        key_col = ds.get("key_col", "TransactionID")

        if not tx_path.exists():
            raise FileNotFoundError(f"Transaction file not found: {tx_path}")

        tx = pd.read_csv(tx_path)

        if id_path and id_path.exists():
            ident = pd.read_csv(id_path)
            df = tx.merge(ident, how="left", on=key_col)
        else:
            df = tx

    else:
        path = Path(ds["path"])

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        df = pd.read_csv(path)
        df = df.fillna(0)

    label_col = ds["label_col"]

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. "
            f"Available columns: {list(df.columns)[:30]}..."
        )

    # ============================================================
    # ✅ FIX: TẠO CỘT time_idx NẾU CHƯA CÓ (cho IEEE-CIS)
    # ============================================================
    time_col = ds.get("time_col")
    
    # Nếu config yêu cầu time_idx nhưng không có, tạo từ TransactionDT
    if time_col == "time_idx" and time_col not in df.columns:
        if "TransactionDT" in df.columns:
            df["time_idx"] = df["TransactionDT"]
            print("[INFO] Created 'time_idx' from 'TransactionDT'")
        elif "time" in df.columns:
            df["time_idx"] = df["time"]
            print("[INFO] Created 'time_idx' from 'time'")
        else:
            # Tạo index nếu không có cột thời gian nào
            df["time_idx"] = range(len(df))
            print("[INFO] Created 'time_idx' as row index (no time column found)")

    # ============================================================
    # ✅ GIỐNG PAPER: STRATIFIED RANDOM SAMPLING
    # ============================================================
    # Paper: random sampling để tạo tập con với tỷ lệ fraud kiểm soát
    # Nhưng vẫn giữ nguyên thứ tự thời gian (sort after sampling)
    sample_frac = ds.get("sample_frac", 1.0)
    use_stratified_sampling = ds.get("stratified_sampling", True)
    
    if sample_frac is not None and sample_frac < 1.0 and use_stratified_sampling:
        print(f"[INFO] Stratified random sampling: {sample_frac*100:.0f}%")
        
        # ✅ Giữ nguyên thứ tự thời gian TRƯỚC KHI SAMPLING
        if time_col and time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)
            print(f"[INFO] Sorted by time_col: {time_col} before sampling")
        
        # ✅ Tách fraud và normal, lấy mẫu ngẫu nhiên từng class
        fraud_df = df[df[label_col] == 1]
        normal_df = df[df[label_col] == 0]
        
        original_fraud = len(fraud_df)
        original_normal = len(normal_df)
        
        # Lấy mẫu với tỷ lệ sample_frac cho từng class
        sampled_fraud = fraud_df.sample(frac=sample_frac, random_state=random_state)
        sampled_normal = normal_df.sample(frac=sample_frac, random_state=random_state)
        
        # Ghép lại và SORT theo thời gian (GIỮ NGUYÊN THỨ TỰ THỜI GIAN)
        df = pd.concat([sampled_normal, sampled_fraud]).sort_values(time_col).reset_index(drop=True)
        
        print(f"[INFO] Sampled {sample_frac*100:.0f}% of data:")
        print(f"  Fraud: {original_fraud:,} -> {len(sampled_fraud):,}")
        print(f"  Normal: {original_normal:,} -> {len(sampled_normal):,}")
        print(f"  Total: {original_fraud + original_normal:,} -> {len(df):,}")
    else:
        # Fallback: sample thông thường
        original_len = len(df)
        if sample_frac is not None and sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=random_state)
            print(f"[INFO] Sampled {sample_frac*100:.0f}% of data: {original_len:,} -> {len(df):,} rows")

    # Ưu tiên sample theo class nếu config có normal_sample_size/fraud_sample_size
    normal_sample_size = ds.get("normal_sample_size")
    fraud_sample_size = ds.get("fraud_sample_size")
    time_col = ds.get("time_col")

    if normal_sample_size is not None or fraud_sample_size is not None:
        df = _sample_by_class(
            df=df,
            label_col=label_col,
            normal_sample_size=normal_sample_size,
            fraud_sample_size=fraud_sample_size,
            random_state=random_state,
            time_col=time_col,
        )
    else:
        df = _sample(df, sample_rows, random_state=random_state)

    print("\nLoaded dataset:")
    print(f"  shape: {df.shape}")
    print(f"  labels: {df[label_col].value_counts().sort_index().to_dict()}")
    
    # Log time column info
    if time_col and time_col in df.columns:
        print(f"  time_col: {time_col} (min: {df[time_col].min()}, max: {df[time_col].max()})")
    else:
        print(f"  ⚠️ time_col '{time_col}' not found!")

    return df