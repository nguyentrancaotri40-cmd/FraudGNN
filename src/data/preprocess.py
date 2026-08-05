from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings


@dataclass
class FraudPreprocessor:
    cfg: Dict[str, Any]
    label_col: str = field(init=False)
    time_col: str = field(init=False)
    drop_cols: List[str] = field(init=False)
    categorical_cols: List[str] = field(init=False)
    numeric_cols: List[str] = field(init=False)
    numeric_strategy: str = field(init=False)
    preprocessor: ColumnTransformer = field(init=False, default=None)
    
    def __post_init__(self):
        ds_cfg = self.cfg.get("dataset", {})
        pre_cfg = self.cfg.get("preprocess", {})
        
        self.label_col = ds_cfg.get("label_col", "isFraud")
        self.time_col = ds_cfg.get("time_col", None)
        self.drop_cols = pre_cfg.get("drop_cols", [])
        self.categorical_cols = pre_cfg.get("categorical_cols", [])
        self.numeric_strategy = pre_cfg.get("numeric_strategy", "median")
        self.numeric_cols = []
        self.preprocessor = None
    
    def _identify_columns(self, df: pd.DataFrame) -> None:
        all_cols = set(df.columns)
        
        cols_to_drop = set()
        if self.label_col in all_cols:
            cols_to_drop.add(self.label_col)
        if self.time_col and self.time_col in all_cols:
            cols_to_drop.add(self.time_col)
        for col in self.drop_cols:
            if col in all_cols:
                cols_to_drop.add(col)
        
        feature_cols = all_cols - cols_to_drop
        
        config_categorical = set(self.categorical_cols)
        valid_categorical = config_categorical - cols_to_drop
        
        self.numeric_cols = []
        detected_categorical = []
        
        for col in feature_cols:
            if col in valid_categorical:
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_cols.append(col)
            else:
                detected_categorical.append(col)
        
        self.categorical_cols = list(valid_categorical) + detected_categorical
        
        total_features = len(self.numeric_cols) + len(self.categorical_cols)
        if total_features > 1000:
            warnings.warn(
                f"[MEMORY WARNING] Total features: {total_features} "
                f"(numeric: {len(self.numeric_cols)}, categorical: {len(self.categorical_cols)}). "
                f"Potential memory issue.",
                UserWarning
            )
    
    def _build_preprocessor(self) -> ColumnTransformer:
        transformers = []
        
        if self.numeric_cols:
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=self.numeric_strategy)),
                ('scaler', MinMaxScaler()),
            ])
            transformers.append(('num', numeric_pipeline, self.numeric_cols))
        
        if self.categorical_cols:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=True,
                    max_categories=500,
                    dtype=np.float32,
                    categories='auto',
                )),
            ])
            transformers.append(('cat', categorical_pipeline, self.categorical_cols))
        
        return ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)
    
    def _densify_if_sparse(self, x):
        """✅ FIX: Convert sparse matrix to dense numpy array."""
        if sparse.issparse(x):
            print(f"[PREPROCESS] Converting sparse matrix ({x.shape}) to dense array...")
            x = x.toarray()
        return np.asarray(x, dtype=np.float32)
    
    def fit(self, df: pd.DataFrame) -> FraudPreprocessor:
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        self._identify_columns(df)
        
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        self.preprocessor = self._build_preprocessor()
        self.preprocessor.fit(df)
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        y = df[self.label_col].astype(int).to_numpy()
        
        if self.time_col and self.time_col in df.columns:
            t = pd.to_numeric(df[self.time_col], errors='coerce').fillna(0).astype(float).to_numpy()
        else:
            t = np.zeros(len(df), dtype=np.float32)
        
        # ✅ FIX: Densify sparse output
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
            return self.preprocessor.get_feature_names_out().tolist()
        return []


def load_preprocessor(path: str) -> FraudPreprocessor:
    return joblib.load(path)