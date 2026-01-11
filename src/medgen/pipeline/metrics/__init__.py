"""
Metrics tracking and computation utilities.

This module provides:
- MetricsTracker: GPU-efficient epoch-level metric aggregation
- Quality metrics: PSNR, MS-SSIM (2D/3D), LPIPS (2D only)
- Regional metrics: RegionalMetricsTracker for masked regions
- Unified metrics: TrainerMetricsConfig, LossAccumulator, MetricsLogger
- Visualization: create_reconstruction_figure for figures
"""

from .tracker import MetricsTracker
from .quality import (
    compute_dice,
    compute_iou,
    compute_msssim,
    compute_msssim_2d_slicewise,
    compute_psnr,
    compute_lpips,
    compute_lpips_3d,
    reset_msssim_nan_warning,
    reset_lpips_nan_warning,
    clear_metric_caches,
)
from .regional_base import BaseRegionalMetricsTracker
from .regional import RegionalMetricsTracker
from .regional_3d import RegionalMetricsTracker3D
from .regional_seg import SegRegionalMetricsTracker
from .figures import create_reconstruction_figure, figure_to_buffer
from .constants import TUMOR_SIZE_THRESHOLDS_MM, TUMOR_SIZE_CATEGORIES
from .unified import (
    TrainerMode,
    TrainerMetricsConfig,
    LossAccumulator,
    MetricsLogger,
    LossKey,
    MetricKey,
    create_metrics_config,
)

__all__ = [
    # Main tracker
    'MetricsTracker',
    # Quality metrics
    'compute_dice',
    'compute_iou',
    'compute_msssim',
    'compute_msssim_2d_slicewise',
    'compute_psnr',
    'compute_lpips',
    'compute_lpips_3d',
    'reset_msssim_nan_warning',
    'reset_lpips_nan_warning',
    'clear_metric_caches',
    # Regional metrics
    'BaseRegionalMetricsTracker',
    'RegionalMetricsTracker',
    'RegionalMetricsTracker3D',
    'SegRegionalMetricsTracker',
    # Unified metrics system
    'TrainerMode',
    'TrainerMetricsConfig',
    'LossAccumulator',
    'MetricsLogger',
    'LossKey',
    'MetricKey',
    'create_metrics_config',
    # Constants
    'TUMOR_SIZE_THRESHOLDS_MM',
    'TUMOR_SIZE_CATEGORIES',
    # Visualization
    'create_reconstruction_figure',
    'figure_to_buffer',
]
