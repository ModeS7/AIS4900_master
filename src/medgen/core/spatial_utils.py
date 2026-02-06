"""Dimension-agnostic spatial tensor operations.

Provides utilities that abstract over 2D vs 3D tensor shapes,
eliminating duplicated spatial_dims conditionals across the codebase.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


def broadcast_to_spatial(tensor: Tensor, spatial_dims: int) -> Tensor:
    """Reshape [B] tensor to [B,1,1,1] (2D) or [B,1,1,1,1] (3D) for broadcasting.

    Args:
        tensor: 1D tensor of shape [B].
        spatial_dims: 2 or 3.

    Returns:
        Reshaped tensor for broadcasting with [B, C, H, W] or [B, C, D, H, W].
    """
    # [B] -> [B, 1, 1, 1] for 2D or [B, 1, 1, 1, 1] for 3D
    ones = (1,) * (spatial_dims + 1)  # +1 for channel dim
    return tensor.view(-1, *ones)


def get_spatial_sum_dims(spatial_dims: int) -> tuple[int, ...]:
    """Return dimension indices for spatial reduction.

    Args:
        spatial_dims: 2 or 3.

    Returns:
        (2, 3) for 2D or (2, 3, 4) for 3D. For use with torch.sum(dim=...).
    """
    if spatial_dims == 3:
        return (2, 3, 4)
    return (2, 3)


def extract_center_slice(tensor: Tensor, spatial_dims: int) -> Tensor:
    """Extract center depth slice from 3D volume; pass-through for 2D.

    Args:
        tensor: Input tensor [B, C, D, H, W] for 3D or [B, C, H, W] for 2D.
        spatial_dims: 2 or 3.

    Returns:
        2D tensor [B, C, H, W].
    """
    if spatial_dims == 2:
        return tensor
    center_idx = tensor.shape[2] // 2
    return tensor[:, :, center_idx, :, :]


def get_pooling_fn(spatial_dims: int, pool_type: str = 'max') -> Callable:
    """Return the appropriate 2D/3D pooling function.

    Args:
        spatial_dims: 2 or 3.
        pool_type: 'max' for max pooling, 'avg' for average pooling.

    Returns:
        F.max_pool2d/3d or F.avg_pool2d/3d.
    """
    if pool_type == 'avg':
        return F.avg_pool3d if spatial_dims == 3 else F.avg_pool2d
    return F.max_pool3d if spatial_dims == 3 else F.max_pool2d
