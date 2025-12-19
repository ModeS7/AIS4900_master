"""
Unified regional loss tracking for validation data.

This module provides RegionalMetricsTracker for computing regional losses
(tumor vs background) on validation data for both Diffusion and VAE trainers.
"""
import logging
import math
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class RegionalMetricsTracker:
    """Unified regional loss tracking for validation data.

    Tracks reconstruction error separately for tumor and background regions,
    with breakdown by tumor size. Used by both DiffusionTrainer and VAETrainer
    during validation.

    Args:
        image_size: Image size in pixels (e.g., 128, 256).
        fov_mm: Field of view in millimeters. Default: 240.0.
        loss_fn: Loss function type: 'mse' (diffusion) or 'l1' (VAE).
        device: PyTorch device for computation.

    Example:
        tracker = RegionalMetricsTracker(image_size=128, loss_fn='mse')

        for batch in val_loader:
            prediction, target, mask = ...
            tracker.update(prediction, target, mask)

        tracker.log_to_tensorboard(writer, epoch)
    """

    def __init__(
        self,
        image_size: int,
        fov_mm: float = 240.0,
        loss_fn: str = 'mse',
        device: Optional[torch.device] = None,
    ):
        self.image_size = image_size
        self.fov_mm = fov_mm
        self.loss_fn = loss_fn
        self.device = device or torch.device('cuda')

        # Compute tumor size thresholds based on clinical definitions
        self.tumor_size_thresholds = self._compute_thresholds()

        # Initialize accumulators
        self.reset()

    def _compute_thresholds(self) -> Dict[str, Tuple[float, float]]:
        """Compute tumor size thresholds based on image resolution.

        Clinical definitions (diameter):
            - tiny:   <10mm  (often non-measurable per RANO-BM)
            - small:  10-20mm (small metastases, SRS alone)
            - medium: 20-30mm (SRS candidates)
            - large:  >30mm  (often surgical)

        Returns:
            Dictionary mapping size names to (low, high) area percentage ranges.
        """
        mm_per_pixel = self.fov_mm / self.image_size
        total_pixels = self.image_size ** 2

        diameter_thresholds = {
            'tiny': (0, 10),
            'small': (10, 20),
            'medium': (20, 30),
            'large': (30, 150),
        }

        thresholds = {}
        for size_name, (d_low, d_high) in diameter_thresholds.items():
            d_low_px = d_low / mm_per_pixel
            d_high_px = d_high / mm_per_pixel
            area_low = math.pi * (d_low_px / 2) ** 2
            area_high = math.pi * (d_high_px / 2) ** 2
            pct_low = area_low / total_pixels
            pct_high = area_high / total_pixels
            thresholds[size_name] = (pct_low, pct_high)

        return thresholds

    def reset(self) -> None:
        """Reset accumulators for new validation run."""
        self.tumor_loss_sum = 0.0
        self.bg_loss_sum = 0.0
        self.count = 0

        # Per-size accumulators
        self.size_loss_sum = {k: 0.0 for k in self.tumor_size_thresholds}
        self.size_count = {k: 0 for k in self.tumor_size_thresholds}

    def update(
        self,
        prediction: Union[Tensor, Dict[str, Tensor]],
        target: Union[Tensor, Dict[str, Tensor]],
        mask: Tensor,
    ) -> None:
        """Accumulate regional losses for a batch.

        Handles both tensor and dict inputs (for dual mode).
        Uses L1 or MSE based on loss_fn parameter.

        Args:
            prediction: Predicted images [B, C, H, W] or dict of channel tensors.
            target: Ground truth images [B, C, H, W] or dict of channel tensors.
            mask: Binary segmentation mask [B, 1, H, W].
        """
        # Handle dict input (dual mode)
        if isinstance(prediction, dict):
            pred = torch.cat(list(prediction.values()), dim=1)
            tgt = torch.cat(list(target.values()), dim=1)
        else:
            pred, tgt = prediction, target

        # Compute error based on loss function type
        if self.loss_fn == 'l1':
            error = torch.abs(pred - tgt)
        else:  # mse
            error = (pred - tgt) ** 2

        # Create masks
        tumor_mask = (mask > 0.5).float()
        if error.shape[1] > 1:
            tumor_mask_expanded = tumor_mask.expand_as(error)
        else:
            tumor_mask_expanded = tumor_mask
        bg_mask_expanded = 1.0 - tumor_mask_expanded

        # Compute per-sample tumor pixel counts
        tumor_pixels = tumor_mask.sum(dim=(1, 2, 3))  # [B]
        total_pixels = mask.shape[2] * mask.shape[3]

        # Safe divisors
        tumor_pixels_safe = tumor_pixels.clamp(min=1)
        bg_pixels = total_pixels - tumor_pixels
        bg_pixels_safe = bg_pixels.clamp(min=1)

        # Compute per-sample losses
        tumor_loss_per_sample = (error * tumor_mask_expanded).sum(dim=(1, 2, 3)) / tumor_pixels_safe
        bg_loss_per_sample = (error * bg_mask_expanded).sum(dim=(1, 2, 3)) / bg_pixels_safe

        # Only count samples with meaningful tumor pixels (>10 pixels)
        has_tumor = (tumor_pixels > 10).float()
        num_valid = has_tumor.sum().item()

        if num_valid > 0:
            # Accumulate weighted by validity
            self.tumor_loss_sum += (tumor_loss_per_sample * has_tumor).sum().item()
            self.bg_loss_sum += (bg_loss_per_sample * has_tumor).sum().item()
            self.count += int(num_valid)

            # Track by tumor size
            tumor_ratios = tumor_pixels / total_pixels
            for size_name, (low, high) in self.tumor_size_thresholds.items():
                size_mask = has_tumor * ((tumor_ratios >= low) & (tumor_ratios < high)).float()
                size_count = size_mask.sum().item()
                if size_count > 0:
                    self.size_loss_sum[size_name] += (tumor_loss_per_sample * size_mask).sum().item()
                    self.size_count[size_name] += int(size_count)

    def compute(self) -> Dict[str, float]:
        """Compute final metrics after all batches processed.

        Returns:
            Dict with 'tumor', 'background', 'ratio', and per-size metrics.
            Empty dict if no samples were tracked.
        """
        if self.count == 0:
            return {}

        tumor_avg = self.tumor_loss_sum / self.count
        bg_avg = self.bg_loss_sum / self.count

        metrics = {
            'tumor': tumor_avg,
            'background': bg_avg,
            'ratio': tumor_avg / (bg_avg + 1e-8),
        }

        # Add per-size metrics
        for size_name in self.tumor_size_thresholds:
            count = self.size_count[size_name]
            if count > 0:
                metrics[f'tumor_size_{size_name}'] = self.size_loss_sum[size_name] / count
            else:
                metrics[f'tumor_size_{size_name}'] = 0.0

        return metrics

    def log_to_tensorboard(
        self,
        writer: Optional[SummaryWriter],
        epoch: int,
        prefix: str = 'tumor',
    ) -> None:
        """Log all metrics to TensorBoard.

        Args:
            writer: TensorBoard SummaryWriter.
            epoch: Current epoch number.
            prefix: TensorBoard tag prefix. Default: 'tumor'.
        """
        if writer is None:
            return

        metrics = self.compute()
        if not metrics:
            return

        writer.add_scalar(f'{prefix}/region_loss', metrics['tumor'], epoch)
        writer.add_scalar(f'{prefix}/background_loss', metrics['background'], epoch)
        writer.add_scalar(f'{prefix}/region_bg_ratio', metrics['ratio'], epoch)

        for size in ['tiny', 'small', 'medium', 'large']:
            writer.add_scalar(f'{prefix}/size_{size}', metrics[f'tumor_size_{size}'], epoch)
