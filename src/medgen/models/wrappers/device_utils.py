"""Device placement utilities for model wrappers.

Provides helper functions for detecting and moving modules to the same device
as a reference model. This centralizes the common pattern of:

    try:
        device = next(model.parameters()).device
        module.to(device)
    except StopIteration:
        pass

Used by mode_embed.py, size_bin_embed.py, and combined_embed.py wrappers.
"""
import torch
from torch import nn


def get_model_device(model: nn.Module) -> torch.device:
    """Get the device of a model's first parameter.

    Args:
        model: PyTorch module.

    Returns:
        Device of first parameter, or CPU if model has no parameters.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def move_module_to_model_device(
    model: nn.Module,
    *modules: nn.Module,
) -> None:
    """Move modules to the same device as model.

    Silently handles models with no parameters (keeps modules on CPU).

    Args:
        model: Reference model for device detection.
        *modules: Modules to move to model's device.

    Example:
        >>> # Before: verbose try-except pattern
        >>> try:
        ...     device = next(model.parameters()).device
        ...     self.size_bin_time_embed = self.size_bin_time_embed.to(device)
        ...     model.time_embed = self.size_bin_time_embed
        ... except StopIteration:
        ...     pass

        >>> # After: one-liner
        >>> move_module_to_model_device(model, self.size_bin_time_embed)
        >>> model.time_embed = self.size_bin_time_embed
    """
    try:
        device = next(model.parameters()).device
        for module in modules:
            module.to(device)
    except StopIteration:
        pass  # Model has no parameters, keep modules on CPU
