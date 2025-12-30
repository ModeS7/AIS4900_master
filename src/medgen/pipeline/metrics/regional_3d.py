"""
3D Regional loss tracking for volumetric validation data.

Extends RegionalMetricsTracker for 3D volumes where tumors span multiple slices.
Uses 3D connected component analysis to identify individual tumors, then
classifies each tumor by its maximum cross-sectional Feret diameter.

Key differences from 2D:
- Input mask is 3D: [B, 1, D, H, W]
- Uses scipy.ndimage.label with 3D connectivity (26-neighborhood)
- Feret diameter computed on the slice with maximum tumor area
- Error accumulated for ALL voxels of each 3D tumor
"""
import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import label as scipy_label
from skimage.measure import regionprops

logger = logging.getLogger(__name__)


class RegionalMetricsTracker3D:
    """3D Regional loss tracking for volumetric validation data.

    Tracks reconstruction error separately for tumor and background regions
    in 3D volumes, with breakdown by tumor size using RANO-BM thresholds.

    For each 3D tumor (connected component spanning multiple slices):
    1. Find the slice with maximum cross-sectional area
    2. Compute Feret diameter on that 2D slice
    3. Classify tumor size (tiny/small/medium/large)
    4. Accumulate error for ALL voxels of the 3D tumor

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
        self.volume_height = volume_size[0]
        self.volume_width = volume_size[1]
        self.volume_depth = volume_size[2]
        self.fov_mm = fov_mm
        self.loss_fn = loss_fn
        self.device = device or torch.device('cuda')

        # mm per pixel (assuming square pixels in H/W plane)
        self.mm_per_pixel = fov_mm / max(self.volume_height, self.volume_width)

        # RANO-BM clinical diameter thresholds
        self.tumor_size_thresholds = self._compute_thresholds()

        # Initialize accumulators
        self.reset()

    def _compute_thresholds(self) -> Dict[str, Tuple[float, float]]:
        """RANO-BM clinical diameter thresholds in mm.

        Uses Feret diameter (longest edge-to-edge distance) measured on
        the slice with maximum tumor cross-sectional area.

        Clinical definitions (diameter):
            - tiny:   <10mm  (often non-measurable per RANO-BM)
            - small:  10-20mm (small metastases, SRS alone)
            - medium: 20-30mm (SRS candidates)
            - large:  >30mm  (often surgical)

        Returns:
            Dictionary mapping size names to (low, high) diameter in mm.
        """
        return {
            'tiny': (0, 10),
            'small': (10, 20),
            'medium': (20, 30),
            'large': (30, 200),
        }

    def _classify_tumor_size(self, diameter_mm: float) -> str:
        """Classify tumor by Feret diameter using RANO-BM thresholds.

        Args:
            diameter_mm: Feret diameter (longest axis) in millimeters.

        Returns:
            Size category: 'tiny', 'small', 'medium', or 'large'.
        """
        for size_name, (low, high) in self.tumor_size_thresholds.items():
            if low <= diameter_mm < high:
                return size_name
        return 'large'

    def reset(self) -> None:
        """Reset accumulators for new validation run."""
        self.tumor_error_sum = 0.0
        self.tumor_voxels_total = 0
        self.bg_loss_sum = 0.0
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

            # Compute background loss for this volume
            bg_mask_3d = ~mask_3d
            bg_voxels = bg_mask_3d.sum()
            if bg_voxels > 0:
                bg_loss = (error_mean * bg_mask_3d).sum() / bg_voxels
                self.bg_loss_sum += bg_loss

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

    def compute(self) -> Dict[str, float]:
        """Compute final metrics after all batches processed.

        Returns:
            Dict with 'tumor', 'background', 'ratio', and per-size loss metrics.
            All tumor metrics are voxel-weighted averages.
            Empty dict if no samples were tracked.
        """
        if self.count == 0:
            return {}

        # Voxel-weighted average: total error / total voxels
        tumor_avg = self.tumor_error_sum / max(self.tumor_voxels_total, 1)
        bg_avg = self.bg_loss_sum / self.count

        metrics = {
            'tumor': tumor_avg,
            'background': bg_avg,
            'ratio': tumor_avg / (bg_avg + 1e-8),
        }

        # Per-size metrics: voxel-weighted average within each category
        for size_name in self.tumor_size_thresholds:
            voxels = self.size_voxels[size_name]
            if voxels > 0:
                metrics[f'tumor_size_{size_name}'] = self.size_error_sum[size_name] / voxels
            else:
                metrics[f'tumor_size_{size_name}'] = 0.0

        return metrics

    def log_to_tensorboard(
        self,
        writer: Optional[SummaryWriter],
        epoch: int,
        prefix: str = 'regional',
    ) -> None:
        """Log all metrics to TensorBoard.

        Args:
            writer: TensorBoard SummaryWriter.
            epoch: Current epoch number.
            prefix: TensorBoard tag prefix. Default: 'regional'.
        """
        if writer is None:
            return

        metrics = self.compute()
        if not metrics:
            return

        writer.add_scalar(f'{prefix}/tumor_loss', metrics['tumor'], epoch)
        writer.add_scalar(f'{prefix}/background_loss', metrics['background'], epoch)
        writer.add_scalar(f'{prefix}/tumor_bg_ratio', metrics['ratio'], epoch)

        for size in ['tiny', 'small', 'medium', 'large']:
            writer.add_scalar(f'{prefix}/{size}', metrics[f'tumor_size_{size}'], epoch)
