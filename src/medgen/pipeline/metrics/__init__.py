"""
Metrics tracking and computation utilities.

This module provides:
- MetricsTracker: GPU-efficient epoch-level metric aggregation
- Quality metrics: PSNR, MS-SSIM (2D/3D), LPIPS (2D only)
- Regional metrics: RegionalMetricsTracker for masked regions
- Visualization: create_reconstruction_figure for figures
"""

from .tracker import MetricsTracker
from .quality import compute_msssim, compute_msssim_2d_slicewise, compute_psnr, compute_lpips, compute_lpips_3d, reset_msssim_nan_warning
from .regional import RegionalMetricsTracker
from .regional_3d import RegionalMetricsTracker3D
from .figures import create_reconstruction_figure

__all__ = [
    # Main tracker
    'MetricsTracker',
    # Quality metrics
    'compute_msssim',
    'compute_msssim_2d_slicewise',
    'compute_psnr',
    'compute_lpips',
    'compute_lpips_3d',
    'reset_msssim_nan_warning',
    # Regional metrics
    'RegionalMetricsTracker',
    'RegionalMetricsTracker3D',
    # Visualization
    'create_reconstruction_figure',
]
