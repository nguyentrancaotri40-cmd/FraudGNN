from __future__ import annotations

import torch
import torch.nn.utils.prune as prune
from typing import Dict, Any
import copy

_PRUNABLE = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)


def apply_pruning(
    model: torch.nn.Module,
    pruning_method: str = "l1_unstructured",
    amount: float = 0.2,
    prune_weights: bool = True,
    prune_biases: bool = False,
) -> torch.nn.Module:
    """Apply pruning to all linear and conv layers. Returns a NEW (deep-copied) model.

    NOTE: because this deep-copies the model, any optimizer already bound to the
    original model's parameters becomes STALE after calling this (the optimizer
    still points at the old, now-orphaned Parameter objects). Recreate the
    optimizer against the returned model if you plan to keep training it.
    For gradual pruning inside an active training loop (existing optimizer),
    use `apply_pruning_inplace` + `update_pruning_mask` instead.
    """
    model = copy.deepcopy(model)
    for name, module in model.named_modules():
        if isinstance(module, _PRUNABLE):
            if prune_weights:
                prune.l1_unstructured(module, name='weight', amount=amount)
            if prune_biases and hasattr(module, 'bias') and module.bias is not None:
                prune.l1_unstructured(module, name='bias', amount=amount)
    return model


def apply_pruning_inplace(
    model: torch.nn.Module,
    amount: float = 0.2,
    prune_weights: bool = True,
    prune_biases: bool = False,
) -> torch.nn.Module:
    """Same effect as `apply_pruning` but mutates `model` IN PLACE (no deepcopy).

    Safe to call on a model whose parameters are already tracked by an existing
    optimizer, PROVIDED this is the first pruning call since the model was last
    in a fully "plain" (unpruned / masks removed) state: torch.nn.utils.prune
    preserves the original Parameter object's identity on a first-time call
    (it is renamed to `<name>_orig`, same underlying tensor), so the optimizer's
    internal references stay valid.

    Do NOT call this a second time on an already-pruned module (it will raise
    since `weight` is no longer a plain Parameter) -- use `update_pruning_mask`
    for subsequent updates instead.
    """
    for name, module in model.named_modules():
        if isinstance(module, _PRUNABLE):
            if prune_weights and not hasattr(module, 'weight_mask'):
                prune.l1_unstructured(module, name='weight', amount=amount)
            if (
                prune_biases
                and hasattr(module, 'bias')
                and module.bias is not None
                and not hasattr(module, 'bias_mask')
            ):
                prune.l1_unstructured(module, name='bias', amount=amount)
    return model


def update_pruning_mask(
    model: torch.nn.Module,
    amount: float,
) -> torch.nn.Module:
    """Update an EXISTING pruning mask to a new target sparsity, in place,
    WITHOUT touching Parameter object identity -- so any optimizer already
    bound to `weight_orig` stays valid across repeated calls.

    Must only be called on modules that already have `weight_mask` (i.e.
    after `apply_pruning_inplace` / a first `prune.l1_unstructured` call).
    Recomputes an L1-magnitude mask directly from the current `weight_orig`
    values and copies it into `weight_mask` in place.
    """
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
    """Remove pruning masks (make pruning permanent). In place.

    WARNING: this replaces `weight_orig` with a brand-new plain `weight`
    Parameter object (different Python identity). If an optimizer is
    tracking the old `weight_orig` object, it becomes stale for that
    parameter -- recreate the optimizer after calling this if you plan to
    keep training the SAME model further (this is safe to ignore when
    pruning is removed only once, at the very end of training, since no
    more optimizer steps happen afterwards).
    """
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
    """Get statistics about pruned weights."""
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


def update_pruning_amount(
    model: torch.nn.Module,
    current_epoch: int,
    start_epoch: int = 5,
    end_epoch: int = 25,
    initial_amount: float = 0.1,
    final_amount: float = 0.3,
) -> torch.nn.Module:
    """Update pruning amount gradually across epochs, in place.

    🔧 FIX: the previous implementation called `prune.remove()` and then
    re-applied `prune.l1_unstructured()` on EVERY call. `prune.remove()`
    creates a brand-new plain `weight` Parameter object each time, which
    silently detaches it from whatever optimizer was created before this
    function first ran. In practice this meant: from the *second* pruning
    update onward, `optimizer.step()` kept updating an orphaned tensor that
    was no longer part of the model's forward graph, while the actual
    active weights (the new `weight_orig`) never received a single gradient
    update from the optimizer for the rest of training. The reported
    "pruned" accuracy for `w/o` and `paysim_pruned.yaml`-style runs using
    epochs after `start_epoch` should be re-validated after this fix.

    This version applies pruning once (which preserves the optimizer's
    Parameter identity) and, on every later call, only overwrites the mask
    values in place -- the optimizer stays valid for the whole run.
    """
    if current_epoch < start_epoch:
        return model
    if current_epoch > end_epoch:
        return model

    total_steps = max(1, end_epoch - start_epoch)
    progress = (current_epoch - start_epoch) / total_steps
    amount = initial_amount + (final_amount - initial_amount) * progress

    has_mask = any(
        hasattr(module, 'weight_mask')
        for _, module in model.named_modules()
        if isinstance(module, _PRUNABLE)
    )

    if not has_mask:
        apply_pruning_inplace(model, amount=amount)
    else:
        update_pruning_mask(model, amount=amount)

    return model