import pandas as pd
import numpy as np
from src.data.preprocess import FraudPreprocessor


def test_preprocess_fit_transform():
    cfg = {
        'dataset': {'label_col': 'isFraud', 'time_col': 'step', 'time_unit': 'hour'},
        'preprocess': {'drop_cols': [], 'categorical_cols': ['type'], 'numeric_strategy': 'median'}
    }
    df = pd.DataFrame({
        'step': [1, 2, 3],
        'type': ['A', 'B', 'A'],
        'amount': [10.0, None, 30.0],
        'isFraud': [0, 1, 0]
    })
    pre = FraudPreprocessor(cfg)
    x, y, t = pre.fit_transform(df)
    
    assert x.shape[0] == 3
    assert y.tolist() == [0, 1, 0]
    assert t is not None
    assert len(t) == 3


def test_time_col_removed_without_drop_cols():
    """ FIX #15: Test that time_col is removed even without drop_cols config."""
    cfg = {
        'dataset': {'label_col': 'isFraud', 'time_col': 'step', 'time_unit': 'hour'},
        'preprocess': {'drop_cols': [], 'categorical_cols': ['type'], 'numeric_strategy': 'median'}
    }
    df = pd.DataFrame({
        'step': [1, 2, 3],
        'type': ['A', 'B', 'A'],
        'amount': [10.0, None, 30.0],
        'isFraud': [0, 1, 0]
    })
    pre = FraudPreprocessor(cfg)
    x, y, t = pre.fit_transform(df)
    
    # Kiểm tra t chứa step
    assert t is not None
    assert len(t) == 3
    assert t[0] == 1.0
    assert t[1] == 2.0
    assert t[2] == 3.0
    
    # Kiểm tra x không chứa step (bằng cách kiểm tra số lượng features)
    # Sau preprocessing: amount (1) + type one-hot (2) = 3
    expected_features = 3
    actual_features = x.shape[1]
    print(f"x.shape: {x.shape}")
    assert actual_features <= expected_features, f"x.shape[1]={actual_features} > {expected_features}, possible step leakage"


def test_time_col_preserved_in_t():
    """ FIX #15: Test that time_col is preserved in t (time values)."""
    cfg = {
        'dataset': {'label_col': 'isFraud', 'time_col': 'step', 'time_unit': 'hour'},
        'preprocess': {'drop_cols': [], 'categorical_cols': ['type'], 'numeric_strategy': 'median'}
    }
    df = pd.DataFrame({
        'step': [1, 2, 3],
        'type': ['A', 'B', 'A'],
        'amount': [10.0, None, 30.0],
        'isFraud': [0, 1, 0]
    })
    pre = FraudPreprocessor(cfg)
    x, y, t = pre.fit_transform(df)
    
    assert t is not None
    assert len(t) == 3
    assert t[0] == 1.0
    assert t[1] == 2.0
    assert t[2] == 3.0


def test_drop_cols_config_respected():
    """ FIX #15: Test that drop_cols from config is respected."""
    cfg = {
        'dataset': {'label_col': 'isFraud', 'time_col': 'step', 'time_unit': 'hour'},
        'preprocess': {
            'drop_cols': ['step'],
            'categorical_cols': ['type'],
            'numeric_strategy': 'median'
        }
    }
    df = pd.DataFrame({
        'step': [1, 2, 3],
        'type': ['A', 'B', 'A'],
        'amount': [10.0, None, 30.0],
        'isFraud': [0, 1, 0]
    })
    pre = FraudPreprocessor(cfg)
    x, y, t = pre.fit_transform(df)
    
    assert t is not None
    assert len(t) == 3
    assert t[0] == 1.0
    
    # Kiểm tra x không chứa step
    expected_features = 3  # amount (1) + type one-hot (2)
    actual_features = x.shape[1]
    print(f"x.shape: {x.shape}")
    assert actual_features <= expected_features, f"x.shape[1]={actual_features} > {expected_features}, step not dropped"