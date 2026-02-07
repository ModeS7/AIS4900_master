"""Codebook metrics management for VQ-VAE models.

This module provides a facade over the CodebookTracker class,
used by UnifiedMetrics for VQ-VAE codebook usage tracking.

Note: The actual codebook tracking implementation is in tracking.py.
This module provides factory functions and helpers for UnifiedMetrics integration.
"""
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def create_codebook_tracker(
    codebook_size: int,
    device: torch.device | None = None,
    enabled: bool = True,
) -> Any | None:
    """Create a codebook tracker for VQ-VAE.

    Args:
        codebook_size: Size of VQ codebook.
        device: Device for computation.
        enabled: Whether codebook tracking is enabled.

    Returns:
        CodebookTracker instance or None if disabled or unavailable.
    """
    if not enabled:
        return None

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        from .tracking import CodebookTracker
        return CodebookTracker(
            num_embeddings=codebook_size,
            device=device,
        )
    except ImportError:
        logger.warning("CodebookTracker not available, codebook metrics disabled")
        return None
    except (TypeError, ValueError, RuntimeError, AttributeError) as e:
        # Catch unexpected errors to prevent training crash
        logger.warning(
            f"CodebookTracker initialization failed ({type(e).__name__}: {e}), "
            f"codebook metrics disabled"
        )
        return None


def format_codebook_metrics(tracker: Any) -> dict[str, float]:
    """Extract metrics from a codebook tracker.

    Args:
        tracker: CodebookTracker instance.

    Returns:
        Dict with perplexity, utilization, active_codes.
    """
    if tracker is None:
        return {}

    return {
        'perplexity': tracker.get_perplexity(),
        'utilization': tracker.get_utilization(),
        'active_codes': tracker.get_active_codes(),
    }
