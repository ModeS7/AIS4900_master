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

import numpy as np
import torch
from scipy.ndimage import label as scipy_label
from skimage.measure import regionprops
from torch import Tensor

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
        image_size: int | None = None,
        volume_size: tuple[int, int, int] | None = None,
        fov_mm: float = 240.0,
        loss_fn: str = 'mse',
        device: torch.device | None = None,
    ):
        # Subclass-specific setup BEFORE super().__init__
        # (super().__init__ calls reset() which needs these attributes)
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

        # super().__init__ calls reset() which initializes all accumulators
        super().__init__(fov_mm=fov_mm, loss_fn=loss_fn, device=device)

    def reset(self) -> None:
        """Reset accumulators for new validation run."""
        super().reset()
        # No additional reset needed - all accumulators are in base class

    def update(
        self,
        prediction: Tensor | dict[str, Tensor],
        target: Tensor | dict[str, Tensor],
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

    def _compute_error_mean(self, error_i: Tensor) -> Tensor:
        """Average error over channels if multi-channel.

        Args:
            error_i: Error tensor for a single sample.
                2D: [C, H, W]
                3D: [C, D, H, W]

        Returns:
            Averaged error tensor [H, W] for 2D or [D, H, W] for 3D.
        """
        if error_i.dim() >= 3 and error_i.shape[0] > 1:
            return error_i.mean(dim=0)
        return error_i.squeeze(0) if error_i.dim() >= 3 else error_i

    def _compute_error_tensor(
        self,
        prediction: Tensor | dict[str, Tensor],
        target: Tensor | dict[str, Tensor],
    ) -> Tensor:
        """Compute error tensor from prediction and target.

        Handles dict input (dual mode) by concatenating channels.

        Args:
            prediction: Predicted tensor or dict of tensors.
            target: Target tensor or dict of tensors.

        Returns:
            Error tensor with same shape as prediction.
        """
        if isinstance(prediction, dict):
            pred = torch.cat(list(prediction.values()), dim=1)
            tgt = torch.cat(list(target.values()), dim=1)
        else:
            pred, tgt = prediction, target

        if self.loss_fn == 'l1':
            return torch.abs(pred - tgt)
        else:  # mse
            return (pred - tgt) ** 2

    def _accumulate_tumor(
        self,
        tumor_error: float,
        tumor_pixels: int,
        size_cat: str,
    ) -> None:
        """Accumulate error for a single tumor.

        Args:
            tumor_error: Total error for this tumor.
            tumor_pixels: Number of pixels/voxels in this tumor.
            size_cat: Size category from _classify_tumor_size().
        """
        self.tumor_error_sum += tumor_error
        self.tumor_pixels_total += tumor_pixels
        self.size_error_sum[size_cat] += tumor_error
        self.size_pixels[size_cat] += tumor_pixels

    def _finalize_sample(
        self,
        bg_error_value: float | None,
        bg_pixels_count: int,
    ) -> None:
        """Finalize accumulation for a sample with valid tumors.

        Args:
            bg_error_value: Background error value (None if no background).
            bg_pixels_count: Number of background pixels/voxels.
        """
        self.count += 1
        if bg_error_value is not None:
            self.bg_error_sum += bg_error_value
            self.bg_pixels_total += bg_pixels_count

    def _get_2d_feret(self, region) -> float:
        """Get Feret diameter for a 2D region.

        Args:
            region: skimage regionprops region object.

        Returns:
            Feret diameter in pixels.
        """
        return region.feret_diameter_max

    def _get_3d_feret(self, tumor_mask_3d: np.ndarray) -> float | None:
        """Get Feret diameter for a 3D tumor by extracting max slice.

        Args:
            tumor_mask_3d: 3D binary mask for this tumor [D, H, W].

        Returns:
            Feret diameter in pixels, or None if no valid region found.
        """
        # Find slice with maximum cross-sectional area
        slice_areas = tumor_mask_3d.sum(axis=(1, 2))  # [D]
        max_slice_idx = np.argmax(slice_areas)

        # Extract 2D slice for Feret diameter computation
        tumor_slice_2d = tumor_mask_3d[max_slice_idx]

        # Run 2D connected components on the max slice
        labeled_2d, _ = scipy_label(tumor_slice_2d)
        regions_2d = regionprops(labeled_2d)

        if not regions_2d:
            return None

        # Use the largest region in this slice
        largest_region = max(regions_2d, key=lambda r: r.area)
        return largest_region.feret_diameter_max

    def _update_2d(
        self,
        prediction: Tensor | dict[str, Tensor],
        target: Tensor | dict[str, Tensor],
        mask: Tensor,
    ) -> None:
        """2D update implementation."""
        error = self._compute_error_tensor(prediction, target)
        batch_size = mask.shape[0]

        for i in range(batch_size):
            mask_np = mask[i, 0].cpu().numpy() > 0.5
            error_i = error[i]  # [C, H, W]

            labeled, num_tumors = scipy_label(mask_np)
            if num_tumors == 0:
                continue

            regions = regionprops(labeled)
            error_mean = self._compute_error_mean(error_i)

            # Compute background error (defer adding until we confirm valid tumor)
            bg_mask_np = ~mask_np
            bg_pixels_count = int(bg_mask_np.sum())
            bg_error_value = None
            if bg_pixels_count > 0:
                bg_mask_tensor = torch.from_numpy(bg_mask_np).to(error_i.device).float()
                bg_error_value = (error_mean * bg_mask_tensor).sum().item()

            sample_has_valid_tumor = False

            for region in regions:
                if region.area < self._min_area:
                    continue

                feret_px = self._get_2d_feret(region)
                size_cat = self._classify_tumor_size(feret_px * self.mm_per_pixel)

                tumor_mask_np = (labeled == region.label)
                tumor_mask_tensor = torch.from_numpy(tumor_mask_np).to(error_i.device).float()
                tumor_error = (error_mean * tumor_mask_tensor).sum().item()
                tumor_px = int(tumor_mask_tensor.sum().item())

                self._accumulate_tumor(tumor_error, tumor_px, size_cat)
                sample_has_valid_tumor = True

            if sample_has_valid_tumor:
                self._finalize_sample(bg_error_value, bg_pixels_count)

    def _update_3d(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> None:
        """3D update implementation."""
        error = self._compute_error_tensor(prediction, target)
        batch_size = mask.shape[0]

        for i in range(batch_size):
            mask_3d = mask[i, 0].cpu().numpy() > 0.5
            error_mean = self._compute_error_mean(error[i]).cpu().numpy()

            labeled_3d, num_tumors = scipy_label(mask_3d)
            if num_tumors == 0:
                continue

            # Compute background error (defer adding until we confirm valid tumor)
            bg_mask_3d = ~mask_3d
            bg_voxels_count = int(bg_mask_3d.sum())
            bg_error_value = None
            if bg_voxels_count > 0:
                bg_error_value = float((error_mean * bg_mask_3d).sum())

            sample_has_valid_tumor = False

            for tumor_id in range(1, num_tumors + 1):
                tumor_mask_3d = (labeled_3d == tumor_id)
                tumor_voxels = int(tumor_mask_3d.sum())

                if tumor_voxels < self._min_area:
                    continue

                feret_px = self._get_3d_feret(tumor_mask_3d)
                if feret_px is None:
                    continue

                size_cat = self._classify_tumor_size(feret_px * self.mm_per_pixel)
                tumor_error = float((error_mean * tumor_mask_3d).sum())

                self._accumulate_tumor(tumor_error, tumor_voxels, size_cat)
                sample_has_valid_tumor = True

            if sample_has_valid_tumor:
                self._finalize_sample(bg_error_value, bg_voxels_count)


# Backwards compatibility aliases
class RegionalMetricsTracker3D(RegionalMetricsTracker):
    """3D regional metrics tracker (backwards compatibility wrapper).

    Equivalent to RegionalMetricsTracker(spatial_dims=3, ...).
    """

    def __init__(self, volume_size=None, **kwargs):
        kwargs['spatial_dims'] = 3
        kwargs['volume_size'] = volume_size
        super().__init__(**kwargs)
