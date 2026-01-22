"""
Unified regional loss tracking for validation data.

This module provides RegionalMetricsTracker for computing regional losses
(tumor vs background) on validation data for both Diffusion and VAE trainers.

Supports both 2D images and 3D volumes via spatial_dims parameter.

Uses connected component analysis to identify individual tumors and compute
loss per tumor, categorized by size using RANO-BM clinical thresholds with
Feret diameter (longest edge-to-edge distance).

For 3D volumes:
- Uses 3D connected components (26-connectivity)
- Feret diameter computed on slice with maximum tumor cross-section
- Error accumulated for ALL voxels of each 3D tumor
"""
import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from scipy.ndimage import label as scipy_label
from skimage.measure import regionprops

from .base import BaseRegionalMetricsTracker

logger = logging.getLogger(__name__)


class RegionalMetricsTracker(BaseRegionalMetricsTracker):
    """Unified regional loss tracking for 2D and 3D validation data.

    Tracks reconstruction error separately for tumor and background regions,
    with breakdown by tumor size. Used by both DiffusionTrainer and VAETrainer
    during validation.

    Inherits shared methods from BaseRegionalMetricsTracker:
    - _classify_tumor_size(): RANO-BM tumor size classification
    - compute(): Pixel/voxel-weighted metric computation
    - log_to_tensorboard(): TensorBoard logging

    Args:
        spatial_dims: Number of spatial dimensions (2 or 3).
        image_size: Image size in pixels for 2D. Required if spatial_dims=2.
        volume_size: Volume dimensions (H, W, D) for 3D. Required if spatial_dims=3.
        fov_mm: Field of view in millimeters. Default: 240.0.
        loss_fn: Loss function type: 'mse' (diffusion) or 'l1' (VAE).
        device: PyTorch device for computation.

    Example:
        # 2D
        tracker = RegionalMetricsTracker(spatial_dims=2, image_size=128, loss_fn='mse')

        # 3D
        tracker = RegionalMetricsTracker(
            spatial_dims=3,
            volume_size=(256, 256, 160),
            loss_fn='l1'
        )

        for batch in val_loader:
            prediction, target, mask = ...
            tracker.update(prediction, target, mask)

        tracker.log_to_tensorboard(writer, epoch)
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        image_size: Optional[int] = None,
        volume_size: Optional[Tuple[int, int, int]] = None,
        fov_mm: float = 240.0,
        loss_fn: str = 'mse',
        device: Optional[torch.device] = None,
    ):
        super().__init__(fov_mm=fov_mm, loss_fn=loss_fn, device=device)
        self.spatial_dims = spatial_dims

        if spatial_dims == 2:
            if image_size is None:
                raise ValueError("image_size required for spatial_dims=2")
            self.image_size = image_size
            self.mm_per_pixel = fov_mm / image_size
        elif spatial_dims == 3:
            if volume_size is None:
                raise ValueError("volume_size required for spatial_dims=3")
            self.volume_size = volume_size
            self.volume_height = volume_size[0]
            self.volume_width = volume_size[1]
            self.volume_depth = volume_size[2]
            # mm per pixel (assuming square pixels in H/W plane)
            self.mm_per_pixel = fov_mm / max(self.volume_height, self.volume_width)
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

        # Min area/voxel threshold
        self._min_area = 5

        self.reset()

    def reset(self) -> None:
        """Reset accumulators for new validation run."""
        self.tumor_error_sum = 0.0
        self.tumor_pixels_total = 0  # Used for both 2D and 3D
        self.bg_error_sum = 0.0
        self.bg_pixels_total = 0
        self.count = 0

        # Per-size accumulators (pixel/voxel-weighted)
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
        using RANO-BM clinical thresholds.

        Args:
            prediction: Predicted images/volumes.
                2D: [B, C, H, W] or dict of channel tensors
                3D: [B, C, D, H, W]
            target: Ground truth images/volumes (same shape as prediction).
            mask: Binary segmentation mask.
                2D: [B, 1, H, W]
                3D: [B, 1, D, H, W]
        """
        if self.spatial_dims == 2:
            self._update_2d(prediction, target, mask)
        else:
            self._update_3d(prediction, target, mask)

    def _update_2d(
        self,
        prediction: Union[Tensor, Dict[str, Tensor]],
        target: Union[Tensor, Dict[str, Tensor]],
        mask: Tensor,
    ) -> None:
        """2D update implementation."""
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
                    error_mean = error_i.mean(dim=0)
                else:
                    error_mean = error_i.squeeze(0) if error_i.dim() == 3 else error_i
                bg_error_value = (error_mean * bg_mask_tensor).sum().item()

            sample_has_valid_tumor = False

            # Process each tumor individually
            for region in regions:
                if region.area < self._min_area:
                    continue

                # Get Feret diameter (longest edge-to-edge distance)
                feret_px = region.feret_diameter_max
                feret_mm = feret_px * self.mm_per_pixel

                # Classify by size
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

                # Accumulate
                self.tumor_error_sum += tumor_error
                self.tumor_pixels_total += tumor_px
                self.size_error_sum[size_cat] += tumor_error
                self.size_pixels[size_cat] += tumor_px
                sample_has_valid_tumor = True

            if sample_has_valid_tumor:
                self.count += 1
                if bg_error_value is not None:
                    self.bg_error_sum += bg_error_value
                    self.bg_pixels_total += bg_pixels_count

    def _update_3d(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> None:
        """3D update implementation."""
        # Compute error based on loss function type
        if self.loss_fn == 'l1':
            error = torch.abs(prediction - target)
        else:  # mse
            error = (prediction - target) ** 2

        batch_size = mask.shape[0]

        # Process each volume individually
        for i in range(batch_size):
            # Extract 3D mask: [D, H, W]
            mask_3d = mask[i, 0].cpu().numpy() > 0.5

            # Average error over channels: [D, H, W]
            error_i = error[i]
            if error_i.dim() == 4 and error_i.shape[0] > 1:
                error_mean = error_i.mean(dim=0).cpu().numpy()
            else:
                error_mean = error_i.squeeze(0).cpu().numpy() if error_i.dim() == 4 else error_i.cpu().numpy()

            # 3D connected component analysis (26-connectivity)
            labeled_3d, num_tumors = scipy_label(mask_3d)

            if num_tumors == 0:
                continue

            # Compute background error (defer adding until we confirm valid tumor)
            bg_mask_3d = ~mask_3d
            bg_voxels_count = int(bg_mask_3d.sum())
            bg_error_value = None
            if bg_voxels_count > 0:
                bg_error_value = (error_mean * bg_mask_3d).sum()

            sample_has_valid_tumor = False

            # Process each 3D tumor
            for tumor_id in range(1, num_tumors + 1):
                tumor_mask_3d = (labeled_3d == tumor_id)
                tumor_voxels = tumor_mask_3d.sum()

                if tumor_voxels < self._min_area:
                    continue

                # Find slice with maximum cross-sectional area
                slice_areas = tumor_mask_3d.sum(axis=(1, 2))  # [D]
                max_slice_idx = np.argmax(slice_areas)

                # Extract 2D slice for Feret diameter computation
                tumor_slice_2d = tumor_mask_3d[max_slice_idx]

                # Run 2D connected components on the max slice
                labeled_2d, _ = scipy_label(tumor_slice_2d)
                regions_2d = regionprops(labeled_2d)

                if not regions_2d:
                    continue

                # Use the largest region in this slice
                largest_region = max(regions_2d, key=lambda r: r.area)

                # Get Feret diameter
                feret_px = largest_region.feret_diameter_max
                feret_mm = feret_px * self.mm_per_pixel

                # Classify by size
                size_cat = self._classify_tumor_size(feret_mm)

                # Compute total error on ALL voxels of this 3D tumor
                tumor_error = (error_mean * tumor_mask_3d).sum()

                # Accumulate (voxel-weighted)
                self.tumor_error_sum += tumor_error
                self.tumor_pixels_total += int(tumor_voxels)
                self.size_error_sum[size_cat] += tumor_error
                self.size_pixels[size_cat] += int(tumor_voxels)
                sample_has_valid_tumor = True

            if sample_has_valid_tumor:
                self.count += 1
                if bg_error_value is not None:
                    self.bg_error_sum += bg_error_value
                    self.bg_pixels_total += bg_voxels_count


# Backwards compatibility aliases
class RegionalMetricsTracker3D(RegionalMetricsTracker):
    """3D regional metrics tracker (backwards compatibility wrapper).

    Equivalent to RegionalMetricsTracker(spatial_dims=3, ...).
    """

    def __init__(self, volume_size=None, **kwargs):
        kwargs['spatial_dims'] = 3
        kwargs['volume_size'] = volume_size
        super().__init__(**kwargs)
