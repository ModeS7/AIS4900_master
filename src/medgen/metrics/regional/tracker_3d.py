"""
3D Regional loss tracking for volumetric validation data.

Extends BaseRegionalMetricsTracker for 3D volumes where tumors span multiple slices.
Uses 3D connected component analysis to identify individual tumors, then
classifies each tumor by its maximum cross-sectional Feret diameter.

Key differences from 2D:
- Input mask is 3D: [B, 1, D, H, W]
- Uses scipy.ndimage.label with 3D connectivity (26-neighborhood)
- Feret diameter computed on the slice with maximum tumor area
- Error accumulated for ALL voxels of each 3D tumor
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from scipy.ndimage import label as scipy_label
from skimage.measure import regionprops

from .base import BaseRegionalMetricsTracker

logger = logging.getLogger(__name__)


class RegionalMetricsTracker3D(BaseRegionalMetricsTracker):
    """3D Regional loss tracking for volumetric validation data.

    Tracks reconstruction error separately for tumor and background regions
    in 3D volumes, with breakdown by tumor size using RANO-BM thresholds.

    For each 3D tumor (connected component spanning multiple slices):
    1. Find the slice with maximum cross-sectional area
    2. Compute Feret diameter on that 2D slice
    3. Classify tumor size (tiny/small/medium/large)
    4. Accumulate error for ALL voxels of the 3D tumor

    Inherits shared methods from BaseRegionalMetricsTracker:
    - _classify_tumor_size(): RANO-BM tumor size classification
    - compute(): Voxel-weighted metric computation
    - log_to_tensorboard(): TensorBoard logging

    Args:
        volume_size: Volume dimensions (height, width, depth).
        fov_mm: Field of view in millimeters. Default: 240.0.
        loss_fn: Loss function type: 'mse' or 'l1'.
        device: PyTorch device for computation.

    Example:
        tracker = RegionalMetricsTracker3D(
            volume_size=(256, 256, 160),
            fov_mm=240.0,
            loss_fn='l1'
        )

        for batch in val_loader:
            prediction, target, mask = ...
            tracker.update(prediction, target, mask)

        tracker.log_to_tensorboard(writer, epoch)
    """

    def __init__(
        self,
        volume_size: Tuple[int, int, int],
        fov_mm: float = 240.0,
        loss_fn: str = 'l1',
        device: Optional[torch.device] = None,
    ):
        super().__init__(fov_mm=fov_mm, loss_fn=loss_fn, device=device)
        self.volume_height = volume_size[0]
        self.volume_width = volume_size[1]
        self.volume_depth = volume_size[2]

        # mm per pixel (assuming square pixels in H/W plane)
        self.mm_per_pixel = fov_mm / max(self.volume_height, self.volume_width)

        # Initialize accumulators
        self.reset()

    def reset(self) -> None:
        """Reset accumulators for new validation run."""
        self.tumor_error_sum = 0.0
        self.tumor_voxels_total = 0
        self.bg_error_sum = 0.0  # Total error across all background voxels
        self.bg_voxels_total = 0  # Total background voxels (for voxel-weighted avg)
        self.count = 0  # Number of samples with tumors

        # Per-size accumulators (voxel-weighted)
        self.size_error_sum = {k: 0.0 for k in self.tumor_size_thresholds}
        self.size_voxels = {k: 0 for k in self.tumor_size_thresholds}

    def update(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> None:
        """Accumulate regional losses for a batch of 3D volumes.

        Uses 3D connected component analysis to identify individual tumors
        spanning multiple slices. Each tumor is classified by its Feret
        diameter measured on the slice with maximum cross-sectional area.

        Args:
            prediction: Predicted volumes [B, C, D, H, W].
            target: Ground truth volumes [B, C, D, H, W].
            mask: Binary segmentation mask [B, 1, D, H, W].
        """
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
                # Accumulate raw error sum (not per-sample average) for voxel-weighted avg
                bg_error_value = (error_mean * bg_mask_3d).sum()

            sample_has_valid_tumor = False

            # Process each 3D tumor
            for tumor_id in range(1, num_tumors + 1):
                tumor_mask_3d = (labeled_3d == tumor_id)
                tumor_voxels = tumor_mask_3d.sum()

                # Skip tiny fragments (<5 voxels)
                if tumor_voxels < 5:
                    continue

                # Find slice with maximum cross-sectional area
                slice_areas = tumor_mask_3d.sum(axis=(1, 2))  # [D]
                max_slice_idx = np.argmax(slice_areas)

                # Extract 2D slice for Feret diameter computation
                tumor_slice_2d = tumor_mask_3d[max_slice_idx]

                # Run 2D connected components on the max slice
                # (should be single component, but handle edge cases)
                labeled_2d, _ = scipy_label(tumor_slice_2d)
                regions_2d = regionprops(labeled_2d)

                if not regions_2d:
                    continue

                # Use the largest region in this slice
                largest_region = max(regions_2d, key=lambda r: r.area)

                # Get Feret diameter (longest edge-to-edge distance)
                feret_px = largest_region.feret_diameter_max
                feret_mm = feret_px * self.mm_per_pixel

                # Classify by size using RANO-BM thresholds
                size_cat = self._classify_tumor_size(feret_mm)

                # Compute total error on ALL voxels of this 3D tumor
                tumor_error = (error_mean * tumor_mask_3d).sum()

                # Accumulate (voxel-weighted)
                self.tumor_error_sum += tumor_error
                self.tumor_voxels_total += int(tumor_voxels)
                self.size_error_sum[size_cat] += tumor_error
                self.size_voxels[size_cat] += int(tumor_voxels)
                sample_has_valid_tumor = True

            if sample_has_valid_tumor:
                self.count += 1
                # Only add background error for samples with valid tumors (voxel-weighted)
                if bg_error_value is not None:
                    self.bg_error_sum += bg_error_value
                    self.bg_voxels_total += bg_voxels_count
