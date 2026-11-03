# ============================================================
# src/data/load_data.py
#
# Theo paper Section V-A-1:
#   - Load 3 dataset: PaySim, Credit Card 2023, IEEE-CIS
#   - IEEE-CIS: merge transaction + identity bằng TransactionID,
#     xử lý trường hợp thiếu identity info (left join, giữ NaN)
#
# [WARN] [INFERRED] Paper nói "for PaySim, we use an 80-20 random split"
# nhưng KHÔNG nói có sample_frac. Nhóm KHÔNG stratified-sample như bản
# cũ (đó là giả lập paper). Chỉ load toàn bộ CSV gốc. Nếu tràn RAM thì
# user tự giảm sample_frac trong config — KHÔNG mặc định sample.
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def load_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load dataset theo cfg["dataset"].

    Hỗ trợ 2 mode:
      A. cfg["dataset"]["transaction_path"] (+ identity_path) → IEEE-CIS
      B. cfg["dataset"]["path"] → file CSV đơn (PaySim, CC2023)

    Sau khi load:
      - Tạo cột time_col nếu config yêu cầu (vd. time_idx từ TransactionDT)
      - sample_frac (nếu < 1.0) — chỉ dùng để test nhanh, KHÔNG phải paper
    """
    ds = cfg["dataset"]

    # ============================================================
    # A. IEEE-CIS: transaction + identity
    # ============================================================
    if "transaction_path" in ds:
        tx_path = Path(ds["transaction_path"])
        id_path_str = ds.get("identity_path", "")
        id_path = Path(id_path_str) if id_path_str else None
        key_col = ds.get("key_col", "TransactionID")

        if not tx_path.exists():
            raise FileNotFoundError(f"Transaction file not found: {tx_path}")

        print(f"[LOAD] Reading transactions: {tx_path}")
        tx = pd.read_csv(tx_path)

        if id_path is not None and id_path.exists():
            print(f"[LOAD] Reading identity: {id_path}")
            ident = pd.read_csv(id_path)
            # Paper: "merge using TransactionID, handling cases where identity
            # information is missing" → LEFT JOIN, giữ NaN cho identity thiếu.
            df = tx.merge(ident, how="left", on=key_col)
            print(f"[LOAD] Merged: {tx.shape} + {ident.shape} → {df.shape}")
        else:
            print("[LOAD] No identity file → chỉ dùng transaction features")
            df = tx

    # ============================================================
    # B. File CSV đơn (PaySim, CC2023)
    # ============================================================
    else:
        path = Path(ds["path"])
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        print(f"[LOAD] Reading: {path}")
        df = pd.read_csv(path)

    label_col = ds["label_col"]
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. "
            f"Có sẵn: {list(df.columns)[:30]}..."
        )

    # ============================================================
    # Tạo cột time_col nếu config yêu cầu & cột gốc khác
    # ============================================================
    time_col = ds.get("time_col")
    if time_col and time_col not in df.columns:
        # [INFERRED] Paper không nói cách ánh xạ, nhóm chọn heuristic
        if time_col == "time_idx":
            if "TransactionDT" in df.columns:
                df["time_idx"] = df["TransactionDT"]
                print("[LOAD] Tạo time_idx = TransactionDT (giây)")
            elif "Time" in df.columns:
                df["time_idx"] = df["Time"]
                print("[LOAD] Tạo time_idx = Time (giây, CC2023)")
            elif "time" in df.columns:
                df["time_idx"] = df["time"]
                print("[LOAD] Tạo time_idx = time")
            elif "step" in df.columns:
                df["time_idx"] = df["step"]
                print("[LOAD] Tạo time_idx = step (giờ, PaySim)")
            else:
                df["time_idx"] = range(len(df))
                print("[LOAD] [WARN] Không có cột thời gian gốc — dùng row index [INFERRED]")

    # ============================================================
    # sample_frac (OPTIONAL — không phải paper, chỉ để test nhanh)
    # ============================================================
    sample_frac = ds.get("sample_frac", 1.0)
    if sample_frac is not None and float(sample_frac) < 1.0:
        original_len = len(df)
        df = df.sample(frac=float(sample_frac), random_state=int(ds.get("random_state", 42)))
        print(f"[LOAD] [WARN] sample_frac={sample_frac} (KHÔNG phải paper): "
              f"{original_len:,} → {len(df):,}")

    # ============================================================
    # LOG
    # ============================================================
    print("\n[LOAD] Loaded dataset:")
    print(f"  shape : {df.shape}")
    print(f"  labels: {df[label_col].value_counts().sort_index().to_dict()}")
    if time_col and time_col in df.columns:
        print(f"  time_col '{time_col}': min={df[time_col].min()}, max={df[time_col].max()}")
    else:
        print(f"  [WARN] time_col '{time_col}' not found!")

    return df


__all__ = ["load_dataset"]