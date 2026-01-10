"""
Regional Dice/IoU tracking for segmentation mask compression.

Computes per-tumor Dice and IoU scores, categorized by tumor size
using RANO-BM clinical thresholds (Feret diameter).

Unlike RegionalMetricsTracker which tracks reconstruction error,
this tracks segmentation quality metrics per region.
"""
import logging
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import label as scipy_label
from skimage.measure import regionprops

from .constants import TUMOR_SIZE_THRESHOLDS_MM, TUMOR_SIZE_CATEGORIES

logger = logging.getLogger(__name__)


class SegRegionalMetricsTracker:
    """Regional Dice/IoU tracking for segmentation compression.

    Tracks segmentation quality (Dice, IoU) per tumor size category.
    Each tumor is classified by Feret diameter using RANO-BM thresholds.

    Unlike RegionalMetricsTracker which computes L1/MSE error per region,
    this computes Dice/IoU scores per individual tumor.

    Args:
        image_size: Image size in pixels (e.g., 64, 128, 256).
        fov_mm: Field of view in millimeters. Default: 240.0.
        device: PyTorch device for computation.

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
        device: Optional[torch.device] = None,
    ):
        self.image_size = image_size
        self.fov_mm = fov_mm
        self.device = device or torch.device('cuda')
        self.tumor_size_thresholds = TUMOR_SIZE_THRESHOLDS_MM
        self.mm_per_pixel = fov_mm / image_size
        self.reset()

    def reset(self) -> None:
        """Reset accumulators for new validation run."""
        # Overall metrics
        self.total_dice = 0.0
        self.total_iou = 0.0
        self.total_tumors = 0

        # Per-size accumulators
        self.size_dice_sum: Dict[str, float] = {k: 0.0 for k in TUMOR_SIZE_CATEGORIES}
        self.size_iou_sum: Dict[str, float] = {k: 0.0 for k in TUMOR_SIZE_CATEGORIES}
        self.size_tumor_count: Dict[str, int] = {k: 0 for k in TUMOR_SIZE_CATEGORIES}

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
    ) -> tuple:
        """Compute Dice and IoU for a single tumor region.

        Args:
            pred_binary: Binary prediction mask [H, W].
            target_binary: Binary target mask [H, W].
            tumor_mask: Binary mask for this specific tumor [H, W].
            smooth: Smoothing factor to avoid division by zero.

        Returns:
            Tuple of (dice, iou) scores for this tumor.
        """
        # Extract predictions and targets within tumor region
        pred_in_region = pred_binary[tumor_mask]
        target_in_region = target_binary[tumor_mask]

        # Compute intersection and union
        intersection = np.sum(pred_in_region & target_in_region)
        pred_sum = np.sum(pred_in_region)
        target_sum = np.sum(target_in_region)

        # Dice = 2*|A∩B| / (|A| + |B|)
        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

        # IoU = |A∩B| / |A∪B|
        union = pred_sum + target_sum - intersection
        iou = (intersection + smooth) / (union + smooth)

        return dice, iou

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
            prediction: Predicted mask (logits or probabilities) [B, 1, H, W].
            target: Binary target mask [B, 1, H, W].
            apply_sigmoid: Whether to apply sigmoid to prediction.
            threshold: Binarization threshold.
        """
        # Move to CPU for connected component analysis
        pred = prediction.detach()
        tgt = target.detach()

        if apply_sigmoid:
            pred = torch.sigmoid(pred)

        # Binarize
        pred_binary = (pred > threshold).cpu().numpy()
        target_binary = (tgt > threshold).cpu().numpy()

        batch_size = target.shape[0]

        for i in range(batch_size):
            # Get 2D arrays
            pred_i = pred_binary[i, 0]  # [H, W]
            tgt_i = target_binary[i, 0]  # [H, W]

            # Find tumors using connected components on TARGET
            labeled, num_tumors = scipy_label(tgt_i)

            if num_tumors == 0:
                continue

            # Get region properties
            regions = regionprops(labeled)

            for region in regions:
                # Skip tiny fragments (<5 pixels)
                if region.area < 5:
                    continue

                # Get Feret diameter
                feret_px = region.feret_diameter_max
                feret_mm = feret_px * self.mm_per_pixel

                # Classify by size
                size_cat = self._classify_tumor_size(feret_mm)

                # Create mask for this tumor
                tumor_mask = (labeled == region.label)

                # Compute Dice/IoU for this tumor
                dice, iou = self._compute_tumor_dice_iou(
                    pred_i, tgt_i, tumor_mask
                )

                # Accumulate
                self.total_dice += dice
                self.total_iou += iou
                self.total_tumors += 1

                self.size_dice_sum[size_cat] += dice
                self.size_iou_sum[size_cat] += iou
                self.size_tumor_count[size_cat] += 1

    def compute(self) -> Dict[str, float]:
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

        # Per-size metrics
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
        writer: Optional[SummaryWriter],
        epoch: int,
        prefix: str = 'regional_seg',
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

        # Per-size Dice
        for size in TUMOR_SIZE_CATEGORIES:
            writer.add_scalar(
                f'{prefix}/dice_{size}',
                metrics[f'dice_{size}'],
                epoch
            )
