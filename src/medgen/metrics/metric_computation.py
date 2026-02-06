"""Metric computation utilities for unified metrics system.

This module provides helper functions and classes for computing quality metrics
during training and validation. The actual metric implementations are in quality.py;
this module provides batched computation and accumulation utilities.

Note: Most metric computation is delegated to quality.py. This module provides
a thin compatibility layer and accumulation helpers used by UnifiedMetrics.
"""

import torch


class MetricAccumulator:
    """Accumulates quality metrics over batches.

    Provides efficient accumulation for PSNR, MS-SSIM, LPIPS, and other metrics
    during validation. Handles both 2D and 3D data.

    Example:
        accumulator = MetricAccumulator(spatial_dims=2)
        for batch in val_loader:
            metrics = accumulator.update(pred, target)
        avg_metrics = accumulator.compute()
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        compute_psnr: bool = True,
        compute_msssim: bool = True,
        compute_lpips: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """Initialize accumulator.

        Args:
            spatial_dims: 2 or 3 for 2D or 3D data.
            compute_psnr: Whether to compute PSNR.
            compute_msssim: Whether to compute MS-SSIM.
            compute_lpips: Whether to compute LPIPS.
            device: Device for computation.
        """
        self.spatial_dims = spatial_dims
        self.compute_psnr_flag = compute_psnr
        self.compute_msssim_flag = compute_msssim
        self.compute_lpips_flag = compute_lpips
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.reset()

    def reset(self) -> None:
        """Reset all accumulators."""
        self._psnr_sum = 0.0
        self._psnr_count = 0
        self._msssim_sum = 0.0
        self._msssim_count = 0
        self._lpips_sum = 0.0
        self._lpips_count = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, float]:
        """Update accumulators with a batch.

        Args:
            pred: Predicted tensor [B, C, (D), H, W].
            target: Ground truth tensor [B, C, (D), H, W].

        Returns:
            Dict of computed metrics for this batch.
        """
        from .dispatch import compute_lpips_dispatch, compute_msssim_dispatch
        from .quality import compute_psnr

        batch_metrics = {}

        if self.compute_psnr_flag:
            psnr = compute_psnr(pred, target)
            self._psnr_sum += psnr
            self._psnr_count += 1
            batch_metrics['psnr'] = psnr

        if self.compute_msssim_flag:
            msssim = compute_msssim_dispatch(pred, target, self.spatial_dims)
            self._msssim_sum += msssim
            self._msssim_count += 1
            batch_metrics['msssim'] = msssim

        if self.compute_lpips_flag:
            lpips = compute_lpips_dispatch(pred, target, self.spatial_dims, device=self.device)
            self._lpips_sum += lpips
            self._lpips_count += 1
            batch_metrics['lpips'] = lpips

        return batch_metrics

    def compute(self) -> dict[str, float]:
        """Compute averaged metrics.

        Returns:
            Dict of averaged metric values.
        """
        result = {}

        if self._psnr_count > 0:
            result['psnr'] = self._psnr_sum / self._psnr_count
        if self._msssim_count > 0:
            result['msssim'] = self._msssim_sum / self._msssim_count
        if self._lpips_count > 0:
            result['lpips'] = self._lpips_sum / self._lpips_count

        return result


def compute_timestep_bin(t: float, num_bins: int = 10) -> int:
    """Compute bin index for a normalized timestep.

    Args:
        t: Timestep value in [0, 1].
        num_bins: Number of bins.

    Returns:
        Bin index in [0, num_bins-1].
    """
    return min(int(t * num_bins), num_bins - 1)


def get_timestep_bin_label(bin_idx: int, num_bins: int = 10) -> str:
    """Get human-readable label for a timestep bin.

    Args:
        bin_idx: Bin index.
        num_bins: Total number of bins.

    Returns:
        Label string like '0.0-0.1'.
    """
    bin_start = bin_idx / num_bins
    bin_end = (bin_idx + 1) / num_bins
    return f'{bin_start:.1f}-{bin_end:.1f}'
