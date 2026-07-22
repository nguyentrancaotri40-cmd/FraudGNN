# tests/test_pipeline.py
import pytest
import yaml
import tempfile
from pathlib import Path
import numpy as np
import torch

from src.train.pipeline_fraudgnn import resolve_flags, build_graph_from_flags, run_pipeline


class TestPipeline:
    """Test pipeline functions."""
    
    def test_resolve_flags(self):
        """Test resolve_flags with defaults."""
        cfg = {}
        flags = resolve_flags(cfg)
        
        assert flags['hard_edges'] == True
        assert flags['soft_edges'] == False
        assert flags['hybrid_graph'] == False
        assert flags['weighted_fusion'] == False
        assert flags['federated'] == True
        assert flags['rl'] == True
        assert flags['pruning'] == False
    
    def test_resolve_flags_override(self):
        """Test resolve_flags with custom flags."""
        cfg = {
            'flags': {
                'hard_edges': False,
                'soft_edges': True,
                'hybrid_graph': True,
                'pruning': True,
            }
        }
        flags = resolve_flags(cfg)
        
        assert flags['hard_edges'] == False
        assert flags['soft_edges'] == True
        assert flags['hybrid_graph'] == True
        assert flags['pruning'] == True
        assert flags['federated'] == True  # Default
        assert flags['rl'] == True  # Default
    
    def test_build_graph_from_flags_baseline(self):
        """Test build_graph_from_flags for baseline."""
        cfg = {
            'dataset': {'time_unit': 'hour'},
            'graph': {'time_window_hours': 1.0, 'similarity_threshold': 0.5}
        }
        flags = {'hard_edges': True, 'soft_edges': False, 'hybrid_graph': False}
        
        x = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randint(0, 2, 10)
        t = np.arange(10).astype(np.float32)
        
        graph = build_graph_from_flags(x, y, t, cfg, flags)
        
        assert graph is not None
        assert graph.x.shape[0] == 10
    
    def test_build_graph_from_flags_soft_only(self):
        """Test build_graph_from_flags for soft only."""
        cfg = {
            'dataset': {'time_unit': 'hour'},
            'graph': {'time_window_hours': 1.0},
            'soft_graph': {'enabled': True, 'similarity_threshold': 0.5},
        }
        flags = {'hard_edges': False, 'soft_edges': True, 'hybrid_graph': False}
        
        x = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randint(0, 2, 10)
        t = np.arange(10).astype(np.float32)
        
        graph = build_graph_from_flags(x, y, t, cfg, flags)
        
        assert graph is not None
        assert graph.x.shape[0] == 10
    
    def test_build_graph_from_flags_hybrid(self):
        """Test build_graph_from_flags for hybrid."""
        cfg = {
            'dataset': {'time_unit': 'hour'},
            'graph': {'time_window_hours': 1.0, 'similarity_threshold': 0.5},
            'soft_graph': {'enabled': True, 'similarity_threshold': 0.5},
            'hybrid_graph': {'enabled': True, 'merge_prefer': 'min_delta'},
        }
        flags = {'hard_edges': True, 'soft_edges': True, 'hybrid_graph': True}
        
        x = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randint(0, 2, 10)
        t = np.arange(10).astype(np.float32)
        
        graph = build_graph_from_flags(x, y, t, cfg, flags)
        
        assert graph is not None
        assert graph.x.shape[0] == 10
    
    def test_pipeline_minimal(self):
        """Test pipeline with minimal config (quick)."""
        # ✅ FIX: Tăng sample_frac lên 0.5 để có đủ data
        cfg = {
            'flags': {
                'hard_edges': True,
                'soft_edges': False,
                'hybrid_graph': False,
                'weighted_fusion': False,
                'federated': False,
                'rl': False,
                'pruning': False,
            },
            'experiment': {
                'name': 'test',
                'pipeline': 'fraudgnn_rl',
                'dataset': 'test',
            },
            'dataset': {
                'name': 'Test',
                'path': 'data/raw/test.csv',
                'label_col': 'label',
                'time_col': 'time',
                'time_unit': 'hour',
                'sample_frac': 0.5,  # ✅ Tăng từ 0.001 lên 0.5
                'random_state': 42,
            },
            'split': {
                'strategy': 'temporal',
                'train_ratio': 0.7,
                'val_ratio': 0.15,
            },
            'preprocess': {
                'drop_cols': [],
                'categorical_cols': [],
                'numeric_strategy': 'median',
            },
            'graph': {
                'time_window_hours': 1.0,
                'similarity_threshold': 0.5,
                'max_neighbors_per_node': 2,
                'add_self_loops': True,
            },
            'model': {
                'hidden_dim': 8,
                'num_layers': 2,
                'dropout': 0.2,
                'num_node_types': 1,
            },
            'train': {
                'epochs': 1,
                'patience': 1,
                'batch_size': 8,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
            },
            'rl': {
                'enabled': False,
                'threshold_bins': [0.3, 0.5, 0.7],
                'fpr_penalty': 0.5,
                'epochs': 1,
                'batch_size': 8,
            },
        }
        
        # ✅ Tạo dữ liệu giả với nhiều samples hơn
        import pandas as pd
        import os
        
        # Đảm bảo directory tồn tại
        os.makedirs('data/raw', exist_ok=True)
        
        df = pd.DataFrame({
            'time': np.arange(200),  # ✅ Tăng từ 100 lên 200
            'label': np.random.randint(0, 2, 200),
            'feat1': np.random.randn(200),
            'feat2': np.random.randn(200),
        })
        df.to_csv('data/raw/test.csv', index=False)
        
        try:
            result = run_pipeline(cfg)
            assert result is not None
            assert 'model' in result
            assert 'test_metrics' in result
        finally:
            # Cleanup
            if os.path.exists('data/raw/test.csv'):
                os.remove('data/raw/test.csv')