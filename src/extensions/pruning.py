from __future__ import annotations
import torch
import torch.nn.utils.prune as prune
from typing import Dict, Any
import copy

_PRUNABLE = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)

def apply_pruning(model: torch.nn.Module, amount: float = 0.2, prune_weights: bool = True, prune_biases: bool = False) -> torch.nn.Module:
    model = copy.deepcopy(model)
    for name, module in model.named_modules():
        if isinstance(module, _PRUNABLE):
            if prune_weights:
                prune.l1_unstructured(module, name='weight', amount=amount)
            if prune_biases and hasattr(module, 'bias') and module.bias is not None:
                prune.l1_unstructured(module, name='bias', amount=amount)
    return model

def apply_pruning_inplace(model: torch.nn.Module, amount: float = 0.2, prune_weights: bool = True, prune_biases: bool = False) -> torch.nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, _PRUNABLE):
            if prune_weights and not hasattr(module, 'weight_mask'):
                prune.l1_unstructured(module, name='weight', amount=amount)
            if prune_biases and hasattr(module, 'bias') and module.bias is not None and not hasattr(module, 'bias_mask'):
                prune.l1_unstructured(module, name='bias', amount=amount)
    return model

def update_pruning_mask(model: torch.nn.Module, amount: float) -> torch.nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, _PRUNABLE) and hasattr(module, 'weight_mask'):
            with torch.no_grad():
                w_abs = module.weight_orig.detach().abs()
                n = w_abs.numel()
                k = int(round(amount * n))
                if k <= 0:
                    new_mask = torch.ones_like(module.weight_mask)
                elif k >= n:
                    new_mask = torch.zeros_like(module.weight_mask)
                else:
                    threshold = torch.kthvalue(w_abs.flatten(), k).values
                    new_mask = (w_abs > threshold).to(module.weight_mask.dtype)
                module.weight_mask.data.copy_(new_mask)
    return model

def remove_pruning(model: torch.nn.Module) -> torch.nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, _PRUNABLE):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass
            try:
                prune.remove(module, 'bias')
            except ValueError:
                pass
    return model

def get_pruning_stats(model: torch.nn.Module) -> Dict[str, Any]:
    total_params = 0
    pruned_params = 0
    for name, module in model.named_modules():
        if isinstance(module, _PRUNABLE) and hasattr(module, 'weight_mask'):
            total_params += module.weight.numel()
            pruned_params += (module.weight_mask == 0).sum().item()
    return {
        "total_params": total_params,
        "pruned_params": pruned_params,
        "pruning_ratio": pruned_params / total_params if total_params > 0 else 0.0,
    }

def update_pruning_amount(model: torch.nn.Module, current_epoch: int, start_epoch: int = 5, end_epoch: int = 25, initial_amount: float = 0.1, final_amount: float = 0.3) -> torch.nn.Module:
    if current_epoch < start_epoch:
        return model
    if current_epoch > end_epoch:
        return model
    total_steps = max(1, end_epoch - start_epoch)
    progress = (current_epoch - start_epoch) / total_steps
    amount = initial_amount + (final_amount - initial_amount) * progress
    has_mask = any(hasattr(module, 'weight_mask') for _, module in model.named_modules() if isinstance(module, _PRUNABLE))
    if not has_mask:
        apply_pruning_inplace(model, amount=amount)
    else:
        update_pruning_mask(model, amount=amount)
    return model
