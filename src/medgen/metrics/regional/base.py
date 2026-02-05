"""
Base class for regional loss tracking.

Provides shared functionality for 2D and 3D regional metrics trackers:
- RANO-BM clinical thresholds for tumor size classification
- Tumor size classification by Feret diameter
- Metric computation (pixel/voxel-weighted averages)
- TensorBoard logging
"""
import functools
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from ..constants import TUMOR_SIZE_CATEGORIES, TUMOR_SIZE_THRESHOLDS_MM


class BaseRegionalMetricsTracker(ABC):
    """Abstract base class for regional loss tracking.

    Provides shared methods for computing and logging regional metrics.
    Subclasses must implement `update` method with dimension-specific logic.

    The base class handles accumulator initialization via reset().
    Subclasses that override reset() must call super().reset().

    Attributes:
        fov_mm: Field of view in millimeters.
        loss_fn: Loss function type ('mse' or 'l1').
        device: PyTorch device for computation.
        tumor_size_thresholds: RANO-BM clinical thresholds from constants.
    """

    # Required accumulator attributes - initialized in reset()
    count: int
    tumor_error_sum: float
    tumor_pixels_total: int
    bg_error_sum: float
    bg_pixels_total: int
    size_error_sum: dict[str, float]
    size_pixels: dict[str, int]

    def __init_subclass__(cls, **kwargs):
        """Validate that subclass reset() calls super().reset()."""
        super().__init_subclass__(**kwargs)

        if 'reset' in cls.__dict__:
            original_reset = cls.__dict__['reset']

            @functools.wraps(original_reset)
            def validated_reset(self):
                original_reset(self)
                required_attrs = [
                    'count', 'tumor_error_sum', 'tumor_pixels_total',
                    'bg_error_sum', 'bg_pixels_total',
                    'size_error_sum', 'size_pixels'
                ]
                missing = [attr for attr in required_attrs if not hasattr(self, attr)]
                if missing:
                    raise AttributeError(
                        f"{cls.__name__}.reset() must initialize: {missing}. "
                        f"Did you forget to call super().reset()?"
                    )

            cls.reset = validated_reset

    def __init__(
        self,
        fov_mm: float = 240.0,
        loss_fn: str = 'l1',
        device: torch.device | None = None,
    ):
        """Initialize base tracker.

        Args:
            fov_mm: Field of view in millimeters.
            loss_fn: Loss function type: 'mse' or 'l1'.
            device: PyTorch device for computation.
        """
        self.fov_mm = fov_mm
        self.loss_fn = loss_fn
        self.device = device or torch.device('cuda')
        self.tumor_size_thresholds = TUMOR_SIZE_THRESHOLDS_MM

        # Initialize accumulators - guarantees attributes exist
        self.reset()

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
        return 'large'  # Fallback for very large tumors

    def reset(self) -> None:
        """Reset accumulators for new validation run.

        Subclasses that override this method must call super().reset().
        """
        self.count = 0
        self.tumor_error_sum = 0.0
        self.tumor_pixels_total = 0
        self.bg_error_sum = 0.0
        self.bg_pixels_total = 0
        self.size_error_sum = {k: 0.0 for k in self.tumor_size_thresholds}
        self.size_pixels = {k: 0 for k in self.tumor_size_thresholds}

    @abstractmethod
    def update(self, prediction: Tensor, target: Tensor, mask: Tensor) -> None:
        """Accumulate regional losses for a batch.

        Args:
            prediction: Predicted images/volumes.
            target: Ground truth images/volumes.
            mask: Binary segmentation mask.
        """
        pass

    def _get_tumor_count_total(self) -> int:
        """Get total tumor pixel/voxel count.

        Subclasses use different attribute names (tumor_pixels_total vs tumor_voxels_total).
        Override this method to return the appropriate count.
        """
        # Default implementation - subclasses may override
        if hasattr(self, 'tumor_pixels_total'):
            return self.tumor_pixels_total
        elif hasattr(self, 'tumor_voxels_total'):
            return self.tumor_voxels_total
        return 0

    def _get_size_counts(self) -> dict[str, int]:
        """Get per-size pixel/voxel counts.

        Subclasses use different attribute names (size_pixels vs size_voxels).
        Override this method to return the appropriate counts.
        """
        if hasattr(self, 'size_pixels'):
            return self.size_pixels
        elif hasattr(self, 'size_voxels'):
            return self.size_voxels
        return {}

    def _get_bg_count_total(self) -> int:
        """Get total background pixel/voxel count.

        Subclasses use different attribute names (bg_pixels_total vs bg_voxels_total).
        Override this method to return the appropriate count.
        """
        if hasattr(self, 'bg_pixels_total'):
            return self.bg_pixels_total
        elif hasattr(self, 'bg_voxels_total'):
            return self.bg_voxels_total
        return 0

    def _get_bg_error_sum(self) -> float:
        """Get total background error sum.

        Returns accumulated raw error for pixel/voxel-weighted averaging.
        """
        return getattr(self, 'bg_error_sum', 0.0)

    def compute(self) -> dict[str, float]:
        """Compute final metrics after all batches processed.

        Returns:
            Dict with 'tumor', 'background', 'ratio', and per-size loss metrics.
            All metrics are pixel/voxel-weighted averages for fair comparison.
            Empty dict if no samples were tracked.
        """
        if self.count == 0:
            return {}

        tumor_count = self._get_tumor_count_total()
        bg_count = self._get_bg_count_total()
        bg_error = self._get_bg_error_sum()

        # Both tumor and background use pixel/voxel-weighted averaging
        tumor_avg = self.tumor_error_sum / max(tumor_count, 1)
        bg_avg = bg_error / max(bg_count, 1)

        metrics = {
            'tumor': tumor_avg,
            'background': bg_avg,
            'ratio': tumor_avg / (bg_avg + 1e-8),
        }

        # Per-size metrics
        size_counts = self._get_size_counts()
        for size_name in TUMOR_SIZE_CATEGORIES:
            count = size_counts.get(size_name, 0)
            if count > 0:
                metrics[f'tumor_size_{size_name}'] = self.size_error_sum[size_name] / count
            else:
                metrics[f'tumor_size_{size_name}'] = 0.0

        return metrics

    def log_to_tensorboard(
        self,
        writer: SummaryWriter | None,
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

        for size in TUMOR_SIZE_CATEGORIES:
            writer.add_scalar(f'{prefix}/{size}', metrics[f'tumor_size_{size}'], epoch)
