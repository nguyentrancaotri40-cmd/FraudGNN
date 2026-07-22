# tests/test_pruning.py
import pytest
import torch
import torch.nn as nn

from src.utils.pruning import (
    apply_pruning,
    apply_pruning_inplace,
    update_pruning_mask,
    remove_pruning,
    get_pruning_stats,
    update_pruning_amount,
)


class TestPruning:
    """Test pruning utilities."""
    
    def test_apply_pruning(self):
        """Test apply_pruning returns new model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        
        pruned = apply_pruning(model, amount=0.2)
        
        assert pruned is not model  # Should be new model
        assert hasattr(pruned[0], 'weight_mask')
        assert hasattr(pruned[2], 'weight_mask')
    
    def test_apply_pruning_inplace(self):
        """Test apply_pruning_inplace mutates model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        
        apply_pruning_inplace(model, amount=0.2)
        
        assert hasattr(model[0], 'weight_mask')
        assert hasattr(model[2], 'weight_mask')
    
    def test_update_pruning_mask(self):
        """Test update_pruning_mask updates mask."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        
        # First apply pruning
        apply_pruning_inplace(model, amount=0.1)
        old_mask = model[0].weight_mask.clone()
        
        # Update mask
        update_pruning_mask(model, amount=0.3)
        new_mask = model[0].weight_mask
        
        # Should be different
        assert not torch.allclose(old_mask, new_mask)
    
    def test_remove_pruning(self):
        """Test remove_pruning removes masks."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        
        apply_pruning_inplace(model, amount=0.2)
        assert hasattr(model[0], 'weight_mask')
        
        remove_pruning(model)
        assert not hasattr(model[0], 'weight_mask')
    
    def test_get_pruning_stats(self):
        """Test get_pruning_stats returns correct stats."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        
        stats = get_pruning_stats(model)
        assert stats['total_params'] == 0
        assert stats['pruned_params'] == 0
        assert stats['pruning_ratio'] == 0.0
        
        apply_pruning_inplace(model, amount=0.2)
        stats = get_pruning_stats(model)
        assert stats['total_params'] > 0
        assert stats['pruned_params'] > 0
        assert stats['pruning_ratio'] > 0
    
    def test_update_pruning_amount(self):
        """Test update_pruning_amount gradual pruning."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        
        # First epoch (before pruning starts)
        model = update_pruning_amount(model, current_epoch=0, start_epoch=5)
        stats = get_pruning_stats(model)
        assert stats['pruning_ratio'] == 0.0
        
        # After pruning starts
        model = update_pruning_amount(model, current_epoch=10, start_epoch=5, end_epoch=25)
        stats = get_pruning_stats(model)
        assert stats['pruning_ratio'] > 0
    
    def test_update_pruning_amount_progressive(self):
        """Test pruning amount increases progressively."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        
        # Epoch 5: initial amount 0.1
        model = update_pruning_amount(model, current_epoch=5, start_epoch=5, end_epoch=25, initial_amount=0.1, final_amount=0.3)
        stats1 = get_pruning_stats(model)
        
        # Epoch 15: should have more pruning
        model = update_pruning_amount(model, current_epoch=15, start_epoch=5, end_epoch=25, initial_amount=0.1, final_amount=0.3)
        stats2 = get_pruning_stats(model)
        
        # Should have more pruning
        assert stats2['pruning_ratio'] >= stats1['pruning_ratio']