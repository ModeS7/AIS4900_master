"""
Unified regional loss tracking for validation data.

This module provides RegionalMetricsTracker for computing regional losses
(tumor vs background) on validation data for both Diffusion and VAE trainers.

Uses connected component analysis to identify individual tumors and compute
loss per tumor, categorized by size using RANO-BM clinical thresholds with
Feret diameter (longest edge-to-edge distance).
"""
import logging
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor
from scipy.ndimage import label as scipy_label
from skimage.measure import regionprops

from .base import BaseRegionalMetricsTracker

logger = logging.getLogger(__name__)


class RegionalMetricsTracker(BaseRegionalMetricsTracker):
    """Unified regional loss tracking for 2D validation data.

    Tracks reconstruction error separately for tumor and background regions,
    with breakdown by tumor size. Used by both DiffusionTrainer and VAETrainer
    during validation.

    Inherits shared methods from BaseRegionalMetricsTracker:
    - _classify_tumor_size(): RANO-BM tumor size classification
    - compute(): Pixel-weighted metric computation
    - log_to_tensorboard(): TensorBoard logging

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
        super().__init__(fov_mm=fov_mm, loss_fn=loss_fn, device=device)
        self.image_size = image_size
        self.reset()

    def reset(self) -> None:
        """Reset accumulators for new validation run."""
        self.tumor_error_sum = 0.0  # Total error across all tumor pixels
        self.tumor_pixels_total = 0  # Total tumor pixels (for pixel-weighted avg)
        self.bg_error_sum = 0.0  # Total error across all background pixels
        self.bg_pixels_total = 0  # Total background pixels (for pixel-weighted avg)
        self.count = 0  # Number of samples with tumors

        # Per-size accumulators (pixel-weighted)
        self.size_error_sum = {k: 0.0 for k in self.tumor_size_thresholds}
        self.size_pixels = {k: 0 for k in self.tumor_size_thresholds}

    def update(
        self,
        prediction: Union[Tensor, Dict[str, Tensor]],
        target: Union[Tensor, Dict[str, Tensor]],
        mask: Tensor,
    ) -> None:
        """Accumulate regional losses for a batch.

        Uses connected component analysis to identify individual tumors and
        compute loss per tumor. Each tumor is categorized by its Feret diameter
        (longest edge-to-edge distance) using RANO-BM clinical thresholds.

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

        mm_per_pixel = self.fov_mm / self.image_size
        batch_size = mask.shape[0]

        # Process each sample individually (needed for connected components)
        for i in range(batch_size):
            mask_np = mask[i, 0].cpu().numpy() > 0.5
            error_i = error[i]  # [C, H, W]

            # Find connected components
            labeled, num_tumors = scipy_label(mask_np)

            if num_tumors == 0:
                continue

            # Get region properties including Feret diameter
            regions = regionprops(labeled)

            # Compute background error (defer adding until we confirm valid tumor)
            bg_mask_np = ~mask_np
            bg_pixels_count = int(bg_mask_np.sum())
            bg_error_value = None
            if bg_pixels_count > 0:
                bg_mask_tensor = torch.from_numpy(bg_mask_np).to(error_i.device).float()
                # Average over channels then sum over spatial (pixel-weighted)
                if error_i.dim() == 3 and error_i.shape[0] > 1:
                    # Multi-channel: mean over channels first
                    error_mean = error_i.mean(dim=0)
                else:
                    error_mean = error_i.squeeze(0) if error_i.dim() == 3 else error_i
                # Accumulate raw error sum (not per-sample average) for pixel-weighted avg
                bg_error_value = (error_mean * bg_mask_tensor).sum().item()

            sample_has_valid_tumor = False

            # Process each tumor individually
            for region in regions:
                # Skip tiny fragments (<5 pixels)
                if region.area < 5:
                    continue

                # Get Feret diameter (longest edge-to-edge distance)
                feret_px = region.feret_diameter_max
                feret_mm = feret_px * mm_per_pixel

                # Classify by size using RANO-BM thresholds
                size_cat = self._classify_tumor_size(feret_mm)

                # Create mask for this tumor only
                tumor_mask_np = (labeled == region.label)
                tumor_mask_tensor = torch.from_numpy(tumor_mask_np).to(error_i.device).float()
                tumor_pixels = tumor_mask_tensor.sum()

                # Compute total error on this tumor's pixels
                if error_i.dim() == 3 and error_i.shape[0] > 1:
                    error_mean = error_i.mean(dim=0)
                else:
                    error_mean = error_i.squeeze(0) if error_i.dim() == 3 else error_i
                tumor_error = (error_mean * tumor_mask_tensor).sum().item()
                tumor_px = int(tumor_pixels.item())

                # Accumulate raw error and pixel counts (for pixel-weighted average)
                self.tumor_error_sum += tumor_error
                self.tumor_pixels_total += tumor_px
                self.size_error_sum[size_cat] += tumor_error
                self.size_pixels[size_cat] += tumor_px
                sample_has_valid_tumor = True

            if sample_has_valid_tumor:
                self.count += 1  # Count samples with valid tumors
                # Only add background error for samples with valid tumors (pixel-weighted)
                if bg_error_value is not None:
                    self.bg_error_sum += bg_error_value
                    self.bg_pixels_total += bg_pixels_count
