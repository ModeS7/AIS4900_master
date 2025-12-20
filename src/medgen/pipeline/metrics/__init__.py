"""
Metrics tracking and computation utilities.

This module provides:
- MetricsTracker: GPU-efficient epoch-level metric aggregation
- Quality metrics: MS-SSIM, PSNR (MS-SSIM replaces both SSIM and LPIPS)
- Regional metrics: RegionalMetricsTracker for masked regions
- Visualization: create_reconstruction_figure for figures
"""

from .tracker import MetricsTracker
from .quality import compute_msssim, compute_psnr
from .regional import RegionalMetricsTracker
from .figures import create_reconstruction_figure

__all__ = [
    # Main tracker
    'MetricsTracker',
    # Quality metrics
    'compute_msssim',
    'compute_psnr',
    # Regional metrics
    'RegionalMetricsTracker',
    # Visualization
    'create_reconstruction_figure',
]
