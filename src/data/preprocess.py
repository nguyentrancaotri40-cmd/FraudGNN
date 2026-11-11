# ============================================================
# src/data/preprocess.py — Paper V-A-1-d (nguyên văn):
#   - Missing value imputation: MEAN (numeric), MODE (categorical)
#   - Feature scaling: MIN-MAX (numeric)
#   - One-hot encoding (categorical)
#   - Temporal features: hour of day, day of week, time differences
#     between consecutive transactions
#   - IEEE-CIS: merge transaction + identity (xử lý ở load_data.py)
#
# fit() CHỈ trên train; transform() cho val/test (không rò rỉ).
#
# [INFERRED] chi tiết paper không nói:
#   - hour/day_of_week/delta_t tính theo đơn vị dataset.time_unit (second|hour),
#     delta_t tính trong từng split sau khi sort theo thời gian.
#   - Temporal features được đưa vào như numeric (min-max) — bản cũ đã loại nhầm.
# ============================================================
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

TEMPORAL_COLS = ["__hour_of_day", "__day_of_week", "__delta_t"]


class FraudPreprocessor:
    def __init__(self, cfg: Dict[str, Any]):
        ds, pre = cfg.get("dataset", {}), cfg.get("preprocess", {})
        self.label_col = ds.get("label_col", "isFraud")
        self.time_col = ds.get("time_col")
        self.time_unit = ds.get("time_unit", "second")
        self.numeric_strategy = pre.get("numeric_strategy", "mean")
        self.categorical_strategy = pre.get("categorical_strategy", "most_frequent")
        self.drop_cols = list(pre.get("drop_cols", []))
        self.categorical_cols_cfg = list(pre.get("categorical_cols", []))
        self.onehot_min_frequency = pre.get("onehot_min_frequency")  # None = đúng paper
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.preprocessor: ColumnTransformer | None = None

    # ------------------------------------------------------------------
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.time_col or self.time_col not in df.columns:
            return df
        df = df.copy()
        t = pd.to_numeric(df[self.time_col], errors="coerce").fillna(0).astype(float).to_numpy()
        order = np.argsort(t, kind="stable")
        ts = t[order]
        hours = ts / 3600.0 if self.time_unit == "second" else ts
        hour = np.floor(hours % 24)
        dow = np.floor(hours / 24) % 7
        dt = np.diff(ts, prepend=ts[:1])
        for name, arr in zip(TEMPORAL_COLS, (hour, dow, dt)):
            full = np.empty(len(df), dtype=np.float32)
            full[order] = arr
            df[name] = full
        return df

    @staticmethod
    def _clean_categorical(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Chuỗi hoá giá trị nhưng GIỮ NaN để SimpleImputer(most_frequent) xử lý đúng (mode)."""
        for c in cols:
            if c in df.columns:
                s = df[c]
                df[c] = s.where(s.isna(), s.astype(str)).astype(object)
        return df

    def _identify_columns(self, df: pd.DataFrame) -> None:
        drop = {self.label_col, *self.drop_cols}
        if self.time_col:
            drop.add(self.time_col)
        feats = [c for c in df.columns if c not in drop]
        explicit_cat = set(self.categorical_cols_cfg)
        num, cat = [], []
        for c in feats:
            if c in explicit_cat or not pd.api.types.is_numeric_dtype(df[c]):
                cat.append(c)
            else:
                num.append(c)
        self.numeric_cols, self.categorical_cols = sorted(num), sorted(cat)
        if len(num) + len(cat) > 1000:
            warnings.warn(f"{len(num)+len(cat)} cột — có thể tràn RAM sau one-hot (xem preprocess.onehot_min_frequency).")

    def _build(self) -> ColumnTransformer:
        tf = []
        if self.numeric_cols:
            tf.append(("num", Pipeline([
                ("imputer", SimpleImputer(strategy=self.numeric_strategy)),
                ("scaler", MinMaxScaler()),
            ]), self.numeric_cols))
        if self.categorical_cols:
            tf.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy=self.categorical_strategy)),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist" if self.onehot_min_frequency else "ignore",
                                         min_frequency=self.onehot_min_frequency,
                                         sparse_output=True, dtype=np.float32)),
            ]), self.categorical_cols))
        return ColumnTransformer(tf, remainder="drop", verbose_feature_names_out=False)

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "FraudPreprocessor":
        df = self._add_temporal_features(df)
        self._identify_columns(df)
        df = self._clean_categorical(df.copy(), self.categorical_cols)
        self.preprocessor = self._build().fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self.preprocessor is None:
            raise RuntimeError("FraudPreprocessor chưa fit().")
        df = self._add_temporal_features(df)
        df = self._clean_categorical(df.copy(), self.categorical_cols)
        y = df[self.label_col].astype(int).to_numpy()
        t = (pd.to_numeric(df[self.time_col], errors="coerce").fillna(0).astype(float).to_numpy()
             if self.time_col and self.time_col in df.columns else None)
        x = self.preprocessor.transform(df)
        if sparse.issparse(x):
            x = x.toarray()
        return np.asarray(x, dtype=np.float32), y, t

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    def get_feature_names_out(self) -> List[str]:
        return list(self.preprocessor.get_feature_names_out()) if self.preprocessor else []

    def save(self, path: str) -> None:
        joblib.dump(self, path)


def load_preprocessor(path: str) -> FraudPreprocessor:
    return joblib.load(path)


__all__ = ["FraudPreprocessor", "load_preprocessor", "TEMPORAL_COLS"]
