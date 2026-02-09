"""
Global segmentation metrics: precision, recall, HD95.

Whole-image metrics (not per-tumor) accumulated across validation batches.
Complements the per-tumor-size SegRegionalMetricsTracker.
"""
import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class GlobalSegMetrics:
    """Whole-image segmentation metrics: precision, recall, HD95.

    Accumulates TP/FP/FN counts across batches for micro-averaged
    precision and recall. HD95 is averaged per-sample.

    Usage:
        metrics = GlobalSegMetrics(compute_hd95=True)
        metrics.reset()
        for batch in val_loader:
            metrics.update(logits, targets)
        results = metrics.compute()
        # results = {'precision': 0.85, 'recall': 0.90, 'hd95': 3.2}
    """

    def __init__(self, compute_hd95: bool = True, device: torch.device | None = None) -> None:
        self.compute_hd95 = compute_hd95
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators."""
        self._tp: int = 0
        self._fp: int = 0
        self._fn: int = 0
        self._hd95_sum: float = 0.0
        self._hd95_count: int = 0

    @torch.no_grad()
    def update(self, prediction: Tensor, target: Tensor, apply_sigmoid: bool = True) -> None:
        """Accumulate metrics from one batch.

        Args:
            prediction: Raw logits or probabilities [B, 1, ...].
            target: Binary ground truth [B, 1, ...].
            apply_sigmoid: Whether to apply sigmoid to prediction.
        """
        if apply_sigmoid:
            pred_binary = (torch.sigmoid(prediction) > 0.5).bool()
        else:
            pred_binary = (prediction > 0.5).bool()

        target_binary = target.bool()

        # Flatten spatial dims for TP/FP/FN counting
        pred_flat = pred_binary.view(pred_binary.shape[0], -1)
        tgt_flat = target_binary.view(target_binary.shape[0], -1)

        self._tp += (pred_flat & tgt_flat).sum().item()
        self._fp += (pred_flat & ~tgt_flat).sum().item()
        self._fn += (~pred_flat & tgt_flat).sum().item()

        # HD95 (per-sample, skip empty masks)
        if self.compute_hd95:
            self._update_hd95(pred_binary, target_binary)

    def _update_hd95(self, pred_binary: Tensor, target_binary: Tensor) -> None:
        """Compute HD95 per sample in the batch."""
        from monai.metrics import compute_hausdorff_distance

        batch_size = pred_binary.shape[0]
        for i in range(batch_size):
            pred_i = pred_binary[i:i+1]  # Keep batch dim [1, 1, ...]
            tgt_i = target_binary[i:i+1]

            # Skip if either mask is empty
            if pred_i.sum() == 0 or tgt_i.sum() == 0:
                continue

            # compute_hausdorff_distance expects [B, C, ...] one-hot format
            hd95 = compute_hausdorff_distance(
                pred_i.float(), tgt_i.float(), percentile=95,
            )

            if not torch.isnan(hd95) and not torch.isinf(hd95):
                self._hd95_sum += hd95.item()
                self._hd95_count += 1

    def compute(self) -> dict[str, float]:
        """Compute final metrics.

        Returns:
            Dict with 'precision', 'recall', and optionally 'hd95'.
        """
        precision = self._tp / max(self._tp + self._fp, 1)
        recall = self._tp / max(self._tp + self._fn, 1)

        results = {
            'precision': precision,
            'recall': recall,
        }

        if self.compute_hd95 and self._hd95_count > 0:
            results['hd95'] = self._hd95_sum / self._hd95_count

        return results
