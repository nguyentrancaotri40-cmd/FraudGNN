# ============================================================
# src/data/preprocess.py
# Preprocessing theo Section V-A-1 & V-A-4 của paper.
#
# Paper (nguyên văn, V-A-1-d):
#   "For all datasets, we perform the following preprocessing steps:
#    - Missing value imputation using mean values for numerical features
#      and mode for categorical features
#    - Feature scaling using min-max normalization for numerical features
#    - One-hot encoding for categorical variables
#    - Temporal feature extraction including hour of day, day of week,
#      and time differences between consecutive transactions
#    For the IEEE-CIS dataset, we additionally merge the transaction and
#    identity information using the TransactionID as the key, handling
#    cases where identity information is missing."
#
# → Imputer numerical = MEAN (paper nói rõ)
# → Imputer categorical = MODE = 'most_frequent' (paper nói rõ, b8 xác nhận)
# → Scaler = MinMax (paper nói rõ)
# → OneHot cho categorical (paper nói rõ)
# → Temporal features: hour, day-of-week, delta_t (paper nói rõ)
# ============================================================

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class FraudPreprocessor:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        ds_cfg = self.cfg.get("dataset", {})
        pre_cfg = self.cfg.get("preprocess", {})

        self.label_col = ds_cfg.get("label_col", "isFraud")
        self.time_col = ds_cfg.get("time_col", None)

        # Paper nói rõ: mean cho numerical, mode cho categorical
        self.numeric_strategy = pre_cfg.get("numeric_strategy", "mean")
        self.categorical_strategy = pre_cfg.get("categorical_strategy", "most_frequent")

        self.drop_cols = list(pre_cfg.get("drop_cols", []))
        self.categorical_cols_cfg = list(pre_cfg.get("categorical_cols", []))

        # Sẽ được điền trong _identify_columns()
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.preprocessor: ColumnTransformer | None = None

        # Lưu tên cột temporal đã thêm để transform lại
        self._temporal_cols: List[str] = []

    # ------------------------------------------------------------------
    # Bước 1: Xác định cột numeric / categorical
    # ------------------------------------------------------------------
    def _identify_columns(self, df: pd.DataFrame) -> None:
        all_cols = set(df.columns)

        cols_to_drop = set()
        if self.label_col in all_cols:
            cols_to_drop.add(self.label_col)
        if self.time_col and self.time_col in all_cols:
            cols_to_drop.add(self.time_col)
        for c in self.drop_cols:
            if c in all_cols:
                cols_to_drop.add(c)
        # Các cột temporal sẽ được thêm sau — không đưa vào feature gốc
        for c in self._temporal_cols:
            cols_to_drop.add(c)

        feature_cols = all_cols - cols_to_drop

        explicit_cat = set(self.categorical_cols_cfg) - cols_to_drop

        numeric_cols: List[str] = []
        categorical_cols: List[str] = []

        for c in feature_cols:
            if c in explicit_cat:
                categorical_cols.append(c)
            elif pd.api.types.is_numeric_dtype(df[c]):
                numeric_cols.append(c)
            else:
                categorical_cols.append(c)

        self.numeric_cols = sorted(numeric_cols)
        self.categorical_cols = sorted(categorical_cols)

        total = len(self.numeric_cols) + len(self.categorical_cols)
        if total > 1000:
            warnings.warn(
                f"[MEMORY WARNING] {total} features — có thể tràn RAM khi OHE.",
                UserWarning,
            )

    # ------------------------------------------------------------------
    # Bước 2: Thêm temporal features (paper nói rõ)
    # ------------------------------------------------------------------
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thêm hour-of-day, day-of-week, delta_t từ time_col.

        [INFERRED — chi tiết] Paper nói "hour of day, day of week, and time
        differences between consecutive transactions" nhưng KHÔNG nói:
          - time_col là giây/giờ/index?
          - delta_t tính trên df đã sort theo thời gian chưa?
        Nhóm chọn:
          - Sort df theo time_col (tăng dần) trước khi tính delta_t.
          - hour = (t // 3600) % 24 nếu đơn vị là giây (IEEE-CIS);
                  = t % 24 nếu đơn vị là giờ (PaySim 'step').
          - day_of_week = (hour_of_day_index // 24) % 7.
          - delta_t = t.diff().fillna(0).
        """
        df = df.copy()
        if not self.time_col or self.time_col not in df.columns:
            return df

        t = pd.to_numeric(df[self.time_col], errors="coerce").fillna(0).astype(float)

        # Sort theo thời gian để delta_t có nghĩa (giữ index gốc để reset sau)
        order = np.argsort(t.values, kind="stable")
        t_sorted = t.values[order]

        # Xác định đơn vị: paper không nói; nhóm suy luận theo dataset
        ds_name = self.cfg.get("experiment", {}).get("dataset", "").lower()
        if ds_name in ("ieee", "ieee-cis", "ieee_cis"):
            unit_sec = True   # TransactionDT tính bằng giây
        elif ds_name in ("paysim",):
            unit_sec = False  # 'step' tính bằng giờ
        else:
            # [INFERRED] fallback: nếu max > 1000 → giây
            unit_sec = bool(t_sorted.max() > 1000)

        if unit_sec:
            hour = ((t_sorted // 3600) % 24).astype(np.float32)
            total_hours = t_sorted / 3600.0
        else:
            hour = (t_sorted % 24).astype(np.float32)
            total_hours = t_sorted

        day_of_week = ((total_hours // 24) % 7).astype(np.float32)
        delta_t = np.diff(t_sorted, prepend=t_sorted[0]).astype(np.float32)

        # Gán lại theo index gốc
        hour_full = np.empty(len(df), dtype=np.float32)
        dow_full = np.empty(len(df), dtype=np.float32)
        dt_full = np.empty(len(df), dtype=np.float32)
        hour_full[order] = hour
        dow_full[order] = day_of_week
        dt_full[order] = delta_t

        df["__hour_of_day"] = hour_full
        df["__day_of_week"] = dow_full
        df["__delta_t"] = dt_full
        self._temporal_cols = ["__hour_of_day", "__day_of_week", "__delta_t"]
        return df

    # ------------------------------------------------------------------
    # Bước 3: Build ColumnTransformer
    # ------------------------------------------------------------------
    def _build_preprocessor(self) -> ColumnTransformer:
        transformers = []

        if self.numeric_cols:
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy=self.numeric_strategy)),  # mean
                ("scaler", MinMaxScaler()),                                  # min-max
            ])
            transformers.append(("num", numeric_pipeline, self.numeric_cols))

        if self.categorical_cols:
            categorical_pipeline = Pipeline([
                # b8: mode = most_frequent (paper: "mode for categorical")
                ("imputer", SimpleImputer(strategy=self.categorical_strategy)),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                )),
            ])
            transformers.append(("cat", categorical_pipeline, self.categorical_cols))

        return ColumnTransformer(
            transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def _densify_if_sparse(self, x):
        if sparse.issparse(x):
            x = x.toarray()
        return np.asarray(x, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "FraudPreprocessor":
        # Categorical phải là str trước khi impute / one-hot
        df = df.copy()
        if self.categorical_cols_cfg:
            for c in self.categorical_cols_cfg:
                if c in df.columns:
                    df[c] = df[c].astype(str)

        # Thêm temporal features trước khi identify
        df = self._add_temporal_features(df)

        self._identify_columns(df)

        for c in self.categorical_cols:
            if c in df.columns:
                df[c] = df[c].astype(str)

        self.preprocessor = self._build_preprocessor()
        self.preprocessor.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.preprocessor is None:
            raise RuntimeError("FraudPreprocessor chưa fit().")

        df = df.copy()
        if self.categorical_cols_cfg:
            for c in self.categorical_cols_cfg:
                if c in df.columns:
                    df[c] = df[c].astype(str)

        df = self._add_temporal_features(df)

        y = df[self.label_col].astype(int).to_numpy()

        if self.time_col and self.time_col in df.columns:
            t = pd.to_numeric(df[self.time_col], errors="coerce").fillna(0).astype(float).to_numpy()
        else:
            t = np.zeros(len(df), dtype=np.float32)

        for c in self.categorical_cols:
            if c in df.columns:
                df[c] = df[c].astype(str)

        X = self.preprocessor.transform(df)
        X = self._densify_if_sparse(X)
        return X, y, t

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.fit(df)
        return self.transform(df)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    def get_feature_names_out(self) -> List[str]:
        if self.preprocessor is not None:
            return list(self.preprocessor.get_feature_names_out())
        return []


def load_preprocessor(path: str) -> FraudPreprocessor:
    return joblib.load(path)


__all__ = ["FraudPreprocessor", "load_preprocessor"]