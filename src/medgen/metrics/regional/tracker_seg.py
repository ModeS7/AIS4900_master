"""
Regional Dice/IoU tracking for segmentation mask compression.

Computes per-tumor Dice and IoU scores, categorized by tumor size
using RANO-BM clinical thresholds (Feret diameter).

Unlike RegionalMetricsTracker which tracks reconstruction error,
this tracks segmentation quality metrics per region.

Supports both 2D and 3D data via ``spatial_dims`` parameter.
"""
import logging

import numpy as np
import torch
from scipy.ndimage import label as scipy_label
from skimage.measure import regionprops
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from ..constants import TUMOR_SIZE_CATEGORIES, TUMOR_SIZE_THRESHOLDS_MM

logger = logging.getLogger(__name__)

# 26-connectivity structure for 3D connected components
_STRUCTURE_3D = np.ones((3, 3, 3), dtype=np.int32)


class SegRegionalMetricsTracker:
    """Regional Dice/IoU tracking for segmentation.

    Tracks segmentation quality (Dice, IoU) per tumor size category.
    Each tumor is classified by Feret diameter using RANO-BM thresholds.

    Supports both 2D (``spatial_dims=2``) and 3D (``spatial_dims=3``).
    For 3D, uses 26-connectivity connected components and computes
    Feret diameter from the max-area axial slice.

    Args:
        image_size: Image size in pixels (e.g., 64, 128, 256).
        fov_mm: Field of view in millimeters. Default: 240.0.
        device: PyTorch device for computation.
        spatial_dims: 2 for 2D slices, 3 for 3D volumes.
        voxel_spacing: Voxel size in mm for anisotropic 3D data.
            For 2D, only the first value is used (isotropic).
            For 3D, expects ``(D, H, W)`` ordering.
            Default: ``None`` (computed from ``fov_mm / image_size``).

    Example:
        tracker = SegRegionalMetricsTracker(image_size=256)

        for batch in val_loader:
            pred_logits, target_mask = ...
            tracker.update(pred_logits, target_mask)

        metrics = tracker.compute()
        tracker.log_to_tensorboard(writer, epoch)
    """

    def __init__(
        self,
        image_size: int,
        fov_mm: float = 240.0,
        device: torch.device | None = None,
        spatial_dims: int = 2,
        voxel_spacing: tuple[float, ...] | None = None,
        detection_threshold: float = 0.1,
    ):
        self.image_size = image_size
        self.fov_mm = fov_mm
        self.device = device or torch.device('cuda')
        self.spatial_dims = spatial_dims
        self.tumor_size_thresholds = TUMOR_SIZE_THRESHOLDS_MM
        self.detection_threshold = detection_threshold

        if voxel_spacing is not None:
            self.voxel_spacing = voxel_spacing
            # For Feret diameter, use the in-plane (H, W) spacing
            self.mm_per_pixel = voxel_spacing[-1] if len(voxel_spacing) >= 2 else voxel_spacing[0]
        else:
            self.mm_per_pixel = fov_mm / image_size
            self.voxel_spacing = (self.mm_per_pixel,) * spatial_dims

        self.reset()

    def reset(self) -> None:
        """Reset accumulators for new validation run."""
        self.total_dice = 0.0
        self.total_iou = 0.0
        self.total_tumors = 0

        self.size_dice_sum: dict[str, float] = {k: 0.0 for k in TUMOR_SIZE_CATEGORIES}
        self.size_iou_sum: dict[str, float] = {k: 0.0 for k in TUMOR_SIZE_CATEGORIES}
        self.size_tumor_count: dict[str, int] = {k: 0 for k in TUMOR_SIZE_CATEGORIES}

        self._per_tumor_records: list[dict] = []
        self._false_positives: int = 0

    def _classify_tumor_size(self, diameter_mm: float) -> str:
        """Classify tumor by Feret diameter using RANO-BM thresholds."""
        for size_name, (low, high) in self.tumor_size_thresholds.items():
            if low <= diameter_mm < high:
                return size_name
        return 'large'

    def _compute_tumor_dice_iou(
        self,
        pred_binary: np.ndarray,
        target_binary: np.ndarray,
        tumor_mask: np.ndarray,
        smooth: float = 1.0,
    ) -> tuple[float, float]:
        """Compute Dice and IoU for a single tumor region.

        Works for both 2D and 3D arrays.

        Args:
            pred_binary: Binary prediction mask.
            target_binary: Binary target mask.
            tumor_mask: Binary mask for this specific tumor.
            smooth: Smoothing factor to avoid division by zero.

        Returns:
            Tuple of (dice, iou) scores for this tumor.
        """
        pred_in_region = pred_binary[tumor_mask]
        target_in_region = target_binary[tumor_mask]

        intersection = np.sum(pred_in_region & target_in_region)
        pred_sum = np.sum(pred_in_region)
        target_sum = np.sum(target_in_region)

        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

        union = pred_sum + target_sum - intersection
        iou = (intersection + smooth) / (union + smooth)

        return dice, iou

    def _get_2d_feret(self, region: 'regionprops') -> float:
        """Get Feret diameter for a 2D region with fallback.

        Args:
            region: skimage regionprops region object.

        Returns:
            Feret diameter in pixels.
        """
        try:
            return region.feret_diameter_max
        except (ValueError, Exception):
            # Degenerate region — fall back to bounding box diagonal
            bbox = region.bbox
            if len(bbox) == 4:
                extent = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
            else:
                extent = [bbox[i + len(bbox) // 2] - bbox[i] for i in range(len(bbox) // 2)]
            return max(extent) if extent else 0.0

    def _get_3d_feret(self, tumor_mask_3d: np.ndarray) -> float | None:
        """Get Feret diameter for a 3D tumor by extracting max-area axial slice.

        Reuses the approach from RegionalMetricsTracker: find the axial slice
        with the largest cross-sectional area, then compute 2D Feret on that.

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
        return self._get_2d_feret(largest_region)

    def update(
        self,
        prediction: Tensor,
        target: Tensor,
        apply_sigmoid: bool = True,
        threshold: float = 0.5,
    ) -> None:
        """Accumulate per-tumor Dice/IoU scores.

        Uses connected components on the TARGET mask to identify tumors,
        then computes Dice/IoU for each tumor region.

        Args:
            prediction: Predicted mask (logits or probabilities).
                2D: [B, 1, H, W], 3D: [B, 1, D, H, W].
            target: Binary target mask (same shape as prediction).
            apply_sigmoid: Whether to apply sigmoid to prediction.
            threshold: Binarization threshold.
        """
        pred = prediction.detach()
        tgt = target.detach()

        if apply_sigmoid:
            pred = torch.sigmoid(pred)

        pred_binary = (pred > threshold).cpu().numpy()
        target_binary = (tgt > threshold).cpu().numpy()

        batch_size = target.shape[0]

        for i in range(batch_size):
            pred_i = pred_binary[i, 0]  # [H, W] or [D, H, W]
            tgt_i = target_binary[i, 0]

            if self.spatial_dims == 3:
                self._update_3d_sample(pred_i, tgt_i)
            else:
                self._update_2d_sample(pred_i, tgt_i)

    def _update_2d_sample(self, pred_i: np.ndarray, tgt_i: np.ndarray) -> None:
        """Process a single 2D sample [H, W]."""
        labeled, num_tumors = scipy_label(tgt_i)
        if num_tumors == 0:
            return

        regions = regionprops(labeled)

        for region in regions:
            if region.area < 5:
                continue

            feret_px = self._get_2d_feret(region)
            feret_mm = feret_px * self.mm_per_pixel
            size_cat = self._classify_tumor_size(feret_mm)

            tumor_mask = (labeled == region.label)
            dice, iou = self._compute_tumor_dice_iou(pred_i, tgt_i, tumor_mask)

            self._accumulate(size_cat, dice, iou, feret_mm)

        # Count false positives: predicted blobs with no GT overlap
        self._count_false_positives(pred_i, tgt_i)

    def _update_3d_sample(self, pred_i: np.ndarray, tgt_i: np.ndarray) -> None:
        """Process a single 3D sample [D, H, W]."""
        # 26-connectivity for 3D connected components
        labeled, num_tumors = scipy_label(tgt_i, structure=_STRUCTURE_3D)
        if num_tumors == 0:
            return

        regions = regionprops(labeled)

        for region in regions:
            if region.area < 5:
                continue

            # Get tumor mask for this region
            tumor_mask = (labeled == region.label)

            # Compute Feret diameter from max-area axial slice
            feret_px = self._get_3d_feret(tumor_mask)
            if feret_px is None:
                continue

            feret_mm = feret_px * self.mm_per_pixel
            size_cat = self._classify_tumor_size(feret_mm)

            dice, iou = self._compute_tumor_dice_iou(pred_i, tgt_i, tumor_mask)
            self._accumulate(size_cat, dice, iou, feret_mm)

        # Count false positives: predicted blobs with no GT overlap
        self._count_false_positives(pred_i, tgt_i, structure=_STRUCTURE_3D)

    def _accumulate(self, size_cat: str, dice: float, iou: float, feret_mm: float) -> None:
        """Add dice/iou to overall and per-size accumulators."""
        self.total_dice += dice
        self.total_iou += iou
        self.total_tumors += 1

        self.size_dice_sum[size_cat] += dice
        self.size_iou_sum[size_cat] += iou
        self.size_tumor_count[size_cat] += 1

        self._per_tumor_records.append({
            'feret_mm': round(feret_mm, 2),
            'size_cat': size_cat,
            'dice': round(dice, 4),
            'iou': round(iou, 4),
            'detected': bool(dice > self.detection_threshold),
        })

    def _count_false_positives(
        self,
        pred_i: np.ndarray,
        tgt_i: np.ndarray,
        structure: np.ndarray | None = None,
    ) -> None:
        """Count predicted blobs that have zero GT overlap.

        Args:
            pred_i: Binary prediction mask [H, W] or [D, H, W].
            tgt_i: Binary target mask (same shape).
            structure: Connectivity structure for scipy_label (None=default for 2D).
        """
        pred_labeled, num_pred = scipy_label(pred_i, structure=structure)
        for j in range(1, num_pred + 1):
            pred_blob = pred_labeled == j
            if pred_blob.sum() < 5:
                continue
            if not np.any(tgt_i[pred_blob]):
                self._false_positives += 1

    def get_detection_summary(self) -> dict[str, float]:
        """Return detection rates overall and per-size, plus FP count.

        Returns:
            Dict with keys: detection_rate, detection_rate_{size}, false_positives.
        """
        records = self._per_tumor_records
        if not records:
            return {'detection_rate': 0.0, 'false_positives': float(self._false_positives)}

        total_detected = sum(1 for r in records if r['detected'])
        summary: dict[str, float] = {
            'detection_rate': total_detected / len(records),
            'false_positives': float(self._false_positives),
        }

        for size in TUMOR_SIZE_CATEGORIES:
            size_records = [r for r in records if r['size_cat'] == size]
            if size_records:
                detected = sum(1 for r in size_records if r['detected'])
                summary[f'detection_rate_{size}'] = detected / len(size_records)
            else:
                summary[f'detection_rate_{size}'] = 0.0

        return summary

    def get_per_tumor_records(self) -> list[dict]:
        """Return raw per-tumor records for JSON export."""
        return list(self._per_tumor_records)

    def compute(self) -> dict[str, float]:
        """Compute final metrics after all batches processed.

        Returns:
            Dict with overall and per-size Dice/IoU metrics.
        """
        if self.total_tumors == 0:
            return {}

        metrics = {
            'dice': self.total_dice / self.total_tumors,
            'iou': self.total_iou / self.total_tumors,
            'n_tumors': self.total_tumors,
        }

        for size_name in TUMOR_SIZE_CATEGORIES:
            count = self.size_tumor_count[size_name]
            if count > 0:
                metrics[f'dice_{size_name}'] = self.size_dice_sum[size_name] / count
                metrics[f'iou_{size_name}'] = self.size_iou_sum[size_name] / count
            else:
                metrics[f'dice_{size_name}'] = 0.0
                metrics[f'iou_{size_name}'] = 0.0
            metrics[f'n_tumors_{size_name}'] = count

        return metrics

    def log_to_tensorboard(
        self,
        writer: SummaryWriter | None,
        epoch: int,
        prefix: str = 'regional',
    ) -> None:
        """Log metrics to TensorBoard.

        Args:
            writer: TensorBoard SummaryWriter.
            epoch: Current epoch number.
            prefix: TensorBoard tag prefix.
        """
        if writer is None:
            return

        metrics = self.compute()
        if not metrics:
            return

        # Overall metrics
        writer.add_scalar(f'{prefix}/dice', metrics['dice'], epoch)
        writer.add_scalar(f'{prefix}/iou', metrics['iou'], epoch)

        # Per-size Dice and IoU
        for size in TUMOR_SIZE_CATEGORIES:
            writer.add_scalar(
                f'{prefix}/dice_{size}',
                metrics[f'dice_{size}'],
                epoch,
            )
            writer.add_scalar(
                f'{prefix}/iou_{size}',
                metrics[f'iou_{size}'],
                epoch,
            )

        # Detection metrics
        detection = self.get_detection_summary()
        writer.add_scalar(f'{prefix}/detection_rate', detection['detection_rate'], epoch)
        writer.add_scalar(f'{prefix}/false_positives', detection['false_positives'], epoch)
        for size in TUMOR_SIZE_CATEGORIES:
            key = f'detection_rate_{size}'
            if key in detection:
                writer.add_scalar(f'{prefix}/{key}', detection[key], epoch)
