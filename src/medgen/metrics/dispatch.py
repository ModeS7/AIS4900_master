"""Dimension-agnostic metric dispatch.

Routes LPIPS and MS-SSIM computation to the correct 2D or 3D implementation
based on spatial_dims. Eliminates duplicated dispatch conditionals across
6+ files.
"""

from collections.abc import Callable

import torch


def compute_lpips_dispatch(
    pred: torch.Tensor,
    gt: torch.Tensor,
    spatial_dims: int,
    device: torch.device | None = None,
) -> float:
    """Route to compute_lpips (2D) or compute_lpips_3d (3D).

    Args:
        pred: Predicted tensor [B, C, (D), H, W].
        gt: Ground truth tensor [B, C, (D), H, W].
        spatial_dims: 2 or 3.
        device: Device for computation (used by 3D variant).

    Returns:
        LPIPS value (lower is better).
    """
    from .quality import compute_lpips, compute_lpips_3d

    if spatial_dims == 3:
        return compute_lpips_3d(pred, gt, device=device)
    return compute_lpips(pred, gt, device=device)


def compute_msssim_dispatch(
    pred: torch.Tensor,
    gt: torch.Tensor,
    spatial_dims: int,
    mode: str = 'slicewise',
) -> float:
    """Route to appropriate MS-SSIM computation.

    Args:
        pred: Predicted tensor [B, C, (D), H, W].
        gt: Ground truth tensor [B, C, (D), H, W].
        spatial_dims: 2 or 3.
        mode: For 3D data only:
            - 'slicewise': 2D slice-wise MS-SSIM (default, used in validation)
            - 'volumetric': True 3D MS-SSIM

    Returns:
        MS-SSIM value (higher is better).
    """
    from .quality import compute_msssim, compute_msssim_2d_slicewise

    if spatial_dims == 3:
        if mode == 'volumetric':
            return compute_msssim(pred, gt, spatial_dims=3)
        return compute_msssim_2d_slicewise(pred, gt)
    return compute_msssim(pred, gt, spatial_dims=2)


def create_lpips_fn(spatial_dims: int) -> Callable:
    """Return the appropriate LPIPS function reference.

    For callers that store the function (compression_trainer, perceptual_manager).

    Args:
        spatial_dims: 2 or 3.

    Returns:
        compute_lpips for 2D, compute_lpips_3d for 3D.
    """
    from .quality import compute_lpips, compute_lpips_3d

    if spatial_dims == 3:
        return compute_lpips_3d
    return compute_lpips
