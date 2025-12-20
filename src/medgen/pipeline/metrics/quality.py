"""
Quality metrics for image comparison.

Provides PSNR and MS-SSIM metrics used by both DiffusionTrainer and VAETrainer.
Uses MONAI's MultiScaleSSIMMetric for native 2D/3D support.

MS-SSIM replaces both SSIM and LPIPS as a single perceptual quality metric
that works in both 2D and 3D.
"""
import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

# Cached MS-SSIM metric instances (one per spatial_dims + image_size combo)
_msssim_cache: dict = {}


def _get_weights_for_size(min_size: int) -> Tuple[float, ...]:
    """Get MS-SSIM weights based on image size.

    MS-SSIM requires minimum image size for each scale (halved at each level).
    Adjust number of scales based on smallest spatial dimension.

    Args:
        min_size: Minimum spatial dimension of the image.

    Returns:
        Tuple of weights for MS-SSIM scales.
    """
    if min_size >= 160:
        # 5 scales (default) - needs 160+ pixels
        return (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    elif min_size >= 80:
        # 4 scales - needs 80+ pixels
        return (0.0448, 0.2856, 0.3001, 0.3695)
    elif min_size >= 40:
        # 3 scales - needs 40+ pixels
        return (0.0448, 0.2856, 0.6696)
    else:
        # 2 scales - minimum for very small images
        return (0.5, 0.5)


def _get_msssim_metric(
    spatial_dims: int,
    min_size: int,
    device: torch.device,
) -> 'MultiScaleSSIMMetric':
    """Get or create cached MS-SSIM metric instance.

    Args:
        spatial_dims: Number of spatial dimensions (2 or 3).
        min_size: Minimum spatial dimension for weight selection.
        device: Device to place the metric on.

    Returns:
        MONAI MultiScaleSSIMMetric instance.
    """
    from monai.metrics import MultiScaleSSIMMetric

    weights = _get_weights_for_size(min_size)
    cache_key = (spatial_dims, len(weights), str(device))

    if cache_key not in _msssim_cache:
        metric = MultiScaleSSIMMetric(
            spatial_dims=spatial_dims,
            data_range=1.0,
            weights=weights,
        )
        _msssim_cache[cache_key] = metric

    return _msssim_cache[cache_key]


def compute_psnr(
    generated: torch.Tensor,
    reference: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """Compute PSNR between generated and reference images.

    Pure PyTorch implementation for GPU acceleration.
    Works with any tensor shape (2D, 3D, any number of channels).

    Args:
        generated: Generated images in [0, 1] range.
        reference: Reference images in [0, 1] range.
        data_range: Maximum pixel value (default 1.0 for normalized images).

    Returns:
        Average PSNR in dB across batch.
    """
    with torch.no_grad():
        gen = torch.clamp(generated.float(), 0, 1)
        ref = torch.clamp(reference.float(), 0, 1)

        mse = torch.mean((gen - ref) ** 2)

        # Avoid log(0)
        if mse < 1e-10:
            return 100.0

        psnr = 20 * torch.log10(torch.tensor(data_range, device=mse.device) / torch.sqrt(mse))
        return float(psnr.item())


def compute_msssim(
    generated: torch.Tensor,
    reference: torch.Tensor,
    data_range: float = 1.0,
    spatial_dims: int = 2,
) -> float:
    """Compute MS-SSIM between generated and reference images.

    Uses MONAI's MultiScaleSSIMMetric for native 2D/3D support.
    Multi-Scale Structural Similarity measures perceptual quality at multiple
    scales, combining structural similarity with multi-resolution analysis.

    Replaces both SSIM and LPIPS as a single metric that works in 2D and 3D.

    Args:
        generated: Generated images [B, C, H, W] or [B, C, D, H, W] in [0, 1] range.
        reference: Reference images [B, C, H, W] or [B, C, D, H, W] in [0, 1] range.
        data_range: Value range of input images (default: 1.0 for [0, 1]).
        spatial_dims: Number of spatial dimensions (2 or 3).

    Returns:
        Average MS-SSIM across batch (higher is better, 1.0 = identical).
    """
    try:
        with torch.no_grad():
            # Ensure float32 and clamp to valid range
            gen = torch.clamp(generated.float(), 0, data_range)
            ref = torch.clamp(reference.float(), 0, data_range)

            # Normalize to [0, 1] if data_range != 1.0
            if data_range != 1.0:
                gen = gen / data_range
                ref = ref / data_range

            # Get minimum spatial dimension for weight selection
            if spatial_dims == 2:
                min_size = min(gen.shape[2], gen.shape[3])
            else:  # 3D
                min_size = min(gen.shape[2], gen.shape[3], gen.shape[4])

            # Get cached metric
            metric = _get_msssim_metric(spatial_dims, min_size, gen.device)

            # Compute MS-SSIM
            # MONAI returns [B, C] tensor, we want scalar mean
            result = metric(gen, ref)

            # Handle NaN values (can occur with edge cases)
            if torch.isnan(result).any():
                logger.warning("MS-SSIM returned NaN values, replacing with 0")
                result = torch.nan_to_num(result, nan=0.0)

            return float(result.mean().item())

    except Exception as e:
        logger.warning(f"MS-SSIM computation failed: {e}")
        return 0.0
