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
    """Sample normal and fraud rows separately."""
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

    if time_col and time_col in out.columns:
        out = out.sort_values(time_col)

    return out.reset_index(drop=True)


def load_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load dataset với random sampling (giống paper)."""
    ds = cfg["dataset"]

    random_state = int(ds.get("random_state", 42))
    sample_rows = ds.get("sample_rows")

    # ============================================================
    # 1. LOAD DATA
    # ============================================================
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
    # 2. TẠO CỘT time_idx NẾU CHƯA CÓ (cho IEEE-CIS)
    # ============================================================
    time_col = ds.get("time_col")
    
    if time_col == "time_idx" and time_col not in df.columns:
        if "TransactionDT" in df.columns:
            df["time_idx"] = df["TransactionDT"]
            print("[INFO] Created 'time_idx' from 'TransactionDT'")
        elif "time" in df.columns:
            df["time_idx"] = df["time"]
            print("[INFO] Created 'time_idx' from 'time'")
        else:
            df["time_idx"] = range(len(df))
            print("[INFO] Created 'time_idx' as row index")

    # ============================================================
    # 3. ✅ GIỐNG PAPER: STRATIFIED RANDOM SAMPLING
    # ============================================================
    sample_frac = ds.get("sample_frac", 1.0)
    use_stratified_sampling = ds.get("stratified_sampling", True)

    if sample_frac is not None and sample_frac < 1.0 and use_stratified_sampling:
        print(f"[INFO] Stratified random sampling: {sample_frac*100:.0f}% (giống paper)")

        fraud_df = df[df[label_col] == 1]
        normal_df = df[df[label_col] == 0]

        original_fraud = len(fraud_df)
        original_normal = len(normal_df)

        # ✅ Random sampling từng class (giống paper)
        sampled_fraud = fraud_df.sample(frac=sample_frac, random_state=random_state)
        sampled_normal = normal_df.sample(frac=sample_frac, random_state=random_state)

        # ✅ Ghép lại và SORT theo thời gian (GIỮ NGUYÊN THỨ TỰ THỜI GIAN)
        df = pd.concat([sampled_normal, sampled_fraud])

        if time_col and time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)
            print(f"[INFO] Preserved temporal ordering by {time_col} (giống paper)")

        print(f"[INFO] Sampled {sample_frac*100:.0f}% of data:")
        print(f"  Fraud: {original_fraud:,} -> {len(sampled_fraud):,}")
        print(f"  Normal: {original_normal:,} -> {len(sampled_normal):,}")
        print(f"  Total: {original_fraud + original_normal:,} -> {len(df):,}")
    else:
        original_len = len(df)
        if sample_frac is not None and sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=random_state)
            print(f"[INFO] Sampled {sample_frac*100:.0f}% of data: {original_len:,} -> {len(df):,} rows")

    # ============================================================
    # 4. SAMPLE THEO CLASS (nếu config yêu cầu)
    # ============================================================
    normal_sample_size = ds.get("normal_sample_size")
    fraud_sample_size = ds.get("fraud_sample_size")

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

    # ============================================================
    # 5. LOG
    # ============================================================
    print("\nLoaded dataset:")
    print(f"  shape: {df.shape}")
    print(f"  labels: {df[label_col].value_counts().sort_index().to_dict()}")
    if time_col and time_col in df.columns:
        print(f"  time_col: {time_col} (min: {df[time_col].min()}, max: {df[time_col].max()})")
    else:
        print(f"  ⚠️ time_col '{time_col}' not found!")

    return df