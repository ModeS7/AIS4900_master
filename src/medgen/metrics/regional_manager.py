"""Regional metrics management for UnifiedMetrics.

This module provides a facade over the RegionalMetricsTracker classes,
used by UnifiedMetrics for tumor-size stratified metric tracking.

Note: The actual regional metrics implementation is in regional.py.
This module provides factory functions and helpers for UnifiedMetrics integration.
"""
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def create_regional_tracker(
    spatial_dims: int,
    image_size: int = 256,
    fov_mm: float = 240.0,
    volume_size: tuple[int, int, int] | None = None,
    device: torch.device | None = None,
) -> Any:
    """Create appropriate regional metrics tracker based on spatial dimensions.

    Args:
        spatial_dims: 2 or 3 for 2D or 3D data.
        image_size: Image size for 2D (default 256).
        fov_mm: Field of view in mm (default 240.0).
        volume_size: (H, W, D) for 3D (default (256, 256, 160)).
        device: Device for computation.

    Returns:
        RegionalMetricsTracker or RegionalMetricsTracker3D instance.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if spatial_dims == 3:
        from .regional import RegionalMetricsTracker3D
        return RegionalMetricsTracker3D(
            volume_size=volume_size or (256, 256, 160),
            fov_mm=fov_mm,
            device=device,
        )
    else:
        from .regional import RegionalMetricsTracker
        return RegionalMetricsTracker(
            image_size=image_size,
            fov_mm=fov_mm,
            device=device,
        )


def get_regional_prefix(mode: str, modality: str | None, prefix: str = '') -> str:
    """Get TensorBoard prefix for regional metrics.

    Args:
        mode: Training mode ('seg', 'bravo', 'dual', 'multi', etc.).
        modality: Modality name (e.g., 'bravo', 't1_pre').
        prefix: Base prefix (e.g., 'val', 'test_best').

    Returns:
        Formatted prefix for TensorBoard logging.
    """
    is_single_modality = mode not in ('multi_modality', 'dual', 'multi')

    if prefix:
        base = f'{prefix}_regional'
    else:
        base = 'regional'

    if is_single_modality and modality:
        return f'{base}_{modality}'
    return base


def format_regional_history_entry(metrics: dict[str, Any]) -> dict[str, Any]:
    """Format regional metrics for JSON history export.

    Args:
        metrics: Raw metrics dict from RegionalMetricsTracker.compute().

    Returns:
        Formatted dict suitable for JSON serialization.
    """
    return {
        'tumor': metrics.get('tumor', 0),
        'background': metrics.get('background', 0),
        'tumor_bg_ratio': metrics.get('ratio', 0),
        'by_size': {
            'tiny': metrics.get('tumor_size_tiny', 0),
            'small': metrics.get('tumor_size_small', 0),
            'medium': metrics.get('tumor_size_medium', 0),
            'large': metrics.get('tumor_size_large', 0),
        }
    }
