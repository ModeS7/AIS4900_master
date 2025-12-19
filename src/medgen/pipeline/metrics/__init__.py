"""
Metrics tracking and computation utilities.

This module provides:
- MetricsTracker: GPU-efficient epoch-level metric aggregation
- Quality metrics: SSIM, PSNR, LPIPS
- Regional metrics: RegionalMetricsTracker for masked regions
- Visualization: create_reconstruction_figure for figures
"""

from .tracker import MetricsTracker
from .quality import compute_ssim, compute_psnr, compute_lpips
from .regional import RegionalMetricsTracker
from .figures import create_reconstruction_figure

__all__ = [
    # Main tracker
    'MetricsTracker',
    # Quality metrics
    'compute_ssim',
    'compute_psnr',
    'compute_lpips',
    # Regional metrics
    'RegionalMetricsTracker',
    # Visualization
    'create_reconstruction_figure',
]
