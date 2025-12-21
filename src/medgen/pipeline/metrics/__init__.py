"""
Metrics tracking and computation utilities.

This module provides:
- MetricsTracker: GPU-efficient epoch-level metric aggregation
- Quality metrics: PSNR, MS-SSIM (2D/3D), LPIPS (2D only)
- Regional metrics: RegionalMetricsTracker for masked regions
- Visualization: create_reconstruction_figure for figures
"""

from .tracker import MetricsTracker
from .quality import compute_msssim, compute_psnr, compute_lpips, reset_msssim_nan_warning
from .regional import RegionalMetricsTracker
from .figures import create_reconstruction_figure

__all__ = [
    # Main tracker
    'MetricsTracker',
    # Quality metrics
    'compute_msssim',
    'compute_psnr',
    'compute_lpips',
    'reset_msssim_nan_warning',
    # Regional metrics
    'RegionalMetricsTracker',
    # Visualization
    'create_reconstruction_figure',
]
