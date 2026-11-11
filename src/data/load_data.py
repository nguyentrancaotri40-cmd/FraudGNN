# ============================================================
# src/data/load_data.py
#
# Paper V-A-1: 3 dataset — PaySim, Credit Card 2023, IEEE-CIS.
#   IEEE-CIS: merge transaction + identity bằng TransactionID,
#             "handling cases where identity information is missing"
#             -> LEFT JOIN, giữ NaN (sẽ được impute ở bước preprocess).
#
# QUY TẮC: không tạo/giả lập dữ liệu. File thiếu -> báo lỗi rõ ràng.
# [INFERRED] sample_frac chỉ để chạy thử nhanh (KHÔNG phải paper); mặc định null.
# ============================================================
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _need(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {path}\n"
            f"  -> {hint}\n"
            f"  (Project này KHÔNG tự sinh dữ liệu giả; hãy đặt file thật vào data/raw/.)"
        )


def load_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
    ds = cfg["dataset"]
    label_col = ds["label_col"]

    if "transaction_path" in ds:  # IEEE-CIS gốc (2 file)
        tx_path = Path(ds["transaction_path"])
        _need(tx_path, "Tải train_transaction.csv từ Kaggle ieee-fraud-detection.")
        print(f"[LOAD] Reading transactions: {tx_path}")
        df = pd.read_csv(tx_path)
        id_p = ds.get("identity_path")
        if id_p and Path(id_p).exists():
            print(f"[LOAD] Reading identity: {id_p}")
            ident = pd.read_csv(id_p)
            key = ds.get("key_col", "TransactionID")
            df = df.merge(ident, how="left", on=key)  # giữ NaN khi thiếu identity
            print(f"[LOAD] Merged on {key}: -> {df.shape}")
        else:
            print("[LOAD] [WARN] Không có identity file -> chỉ dùng transaction features")
    else:
        path = Path(ds["path"])
        _need(path, "Đặt CSV (PaySim / CC2023 / IEEE-CIS đã merge) vào đường dẫn trong config.")
        print(f"[LOAD] Reading: {path}")
        df = pd.read_csv(path)

    if label_col not in df.columns:
        raise ValueError(f"Không có cột nhãn '{label_col}'. Cột hiện có: {list(df.columns)[:30]}...")

    frac = ds.get("sample_frac")
    if frac is not None and float(frac) < 1.0:
        n0 = len(df)
        df = df.sample(frac=float(frac), random_state=int(ds.get("random_state", 42)))
        print(f"[LOAD] [WARN] sample_frac={frac} (KHÔNG phải paper): {n0:,} -> {len(df):,}")

    time_col = ds.get("time_col")
    if time_col and time_col not in df.columns:
        raise ValueError(f"time_col='{time_col}' không có trong dữ liệu. Đặt time_col: null nếu dataset không có thời gian.")

    print(f"[LOAD] shape={df.shape} labels={df[label_col].value_counts().sort_index().to_dict()}")
    if time_col:
        print(f"[LOAD] time_col '{time_col}': min={df[time_col].min()}, max={df[time_col].max()}")
    else:
        print("[LOAD] Dataset không có time_col -> graph chỉ dùng cosine similarity")
    return df


__all__ = ["load_dataset"]
