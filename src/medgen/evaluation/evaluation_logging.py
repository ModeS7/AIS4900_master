"""Test evaluation logging utilities.

Contains functions for logging test results to console and TensorBoard,
including per-modality metric logging.

Moved from evaluation.py during file split.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from medgen.metrics.unified import UnifiedMetrics

logger = logging.getLogger(__name__)


def log_test_header(label: str) -> None:
    """Log test evaluation header."""
    logger.info("=" * 60)
    logger.info(f"EVALUATING ON TEST SET ({label.upper()} MODEL)")
    logger.info("=" * 60)


def log_test_results(
    metrics: dict[str, float],
    label: str,
    n_samples: int,
) -> None:
    """Log test results to console.

    Args:
        metrics: Dictionary of metric name -> value.
        label: Checkpoint label.
        n_samples: Number of test samples evaluated.
    """
    logger.info(f"Test Results - {label} ({n_samples} samples):")
    # Segmentation metrics
    if 'dice' in metrics:
        logger.info(f"  Dice:    {metrics['dice']:.4f}")
    if 'iou' in metrics:
        logger.info(f"  IoU:     {metrics['iou']:.4f}")
    if 'bce' in metrics:
        logger.info(f"  BCE:     {metrics['bce']:.4f}")
    if 'boundary' in metrics:
        logger.info(f"  Boundary:{metrics['boundary']:.4f}")
    if 'gen' in metrics and 'dice' in metrics:  # Only show gen for seg_mode
        logger.info(f"  Gen Loss:{metrics['gen']:.4f}")
    # Image metrics
    if 'l1' in metrics:
        logger.info(f"  L1 Loss: {metrics['l1']:.6f}")
    if 'mse' in metrics:
        logger.info(f"  MSE Loss: {metrics['mse']:.6f}")
    if 'msssim' in metrics:
        logger.info(f"  MS-SSIM: {metrics['msssim']:.4f}")
    if 'msssim_3d' in metrics:
        logger.info(f"  MS-SSIM-3D: {metrics['msssim_3d']:.4f}")
    if 'psnr' in metrics:
        logger.info(f"  PSNR:    {metrics['psnr']:.2f} dB")
    if 'lpips' in metrics:
        logger.info(f"  LPIPS:   {metrics['lpips']:.4f}")


def log_metrics_to_tensorboard(
    writer: SummaryWriter | None,
    metrics: dict[str, float],
    prefix: str,
    step: int = 0,
    unified_metrics: "UnifiedMetrics | None" = None,
) -> None:
    """Log metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter (can be None).
        metrics: Dictionary of metric name -> value.
        prefix: Prefix for metric names (e.g., "test_best").
        step: Global step for TensorBoard.
        unified_metrics: Optional UnifiedMetrics instance for centralized logging.
    """
    # Use UnifiedMetrics if provided (preferred)
    if unified_metrics is not None:
        # Transform metrics to match UnifiedMetrics expectations
        transformed = {}
        name_map = {
            'l1': 'L1',
            'mse': 'MSE',
            'msssim': 'MS-SSIM',
            'msssim_3d': 'MS-SSIM-3D',
            'psnr': 'PSNR',
            'lpips': 'LPIPS',
            'dice': 'Dice',
            'iou': 'IoU',
            'bce': 'BCE',
            'boundary': 'Boundary',
            'gen': 'Generator',
        }
        for key, value in metrics.items():
            if key in name_map:
                transformed[name_map[key]] = value
            elif key != 'n_samples':
                transformed[key] = value
        unified_metrics.log_test(transformed, prefix=prefix)
        return

    # Fallback to direct writer calls for backward compatibility
    if writer is None:
        return

    # Map metric names to TensorBoard display names
    name_map = {
        'l1': 'L1',
        'mse': 'MSE',
        'msssim': 'MS-SSIM',
        'msssim_3d': 'MS-SSIM-3D',
        'psnr': 'PSNR',
        'lpips': 'LPIPS',
        'dice': 'Dice',
        'iou': 'IoU',
        'bce': 'BCE',
        'boundary': 'Boundary',
        'gen': 'Generator',
    }

    for key, value in metrics.items():
        if key in name_map:
            writer.add_scalar(f'{prefix}/{name_map[key]}', value, step)
        elif key != 'n_samples':  # Skip sample count
            writer.add_scalar(f'{prefix}/{key}', value, step)


def log_test_per_modality(
    writer: SummaryWriter | None,
    metrics: dict[str, float],
    prefix: str,
    modality: str,
    step: int = 0,
    unified_metrics: "UnifiedMetrics | None" = None,
) -> None:
    """Log per-modality test metrics to TensorBoard.

    Uses unified tag naming: {prefix}/{MetricName}_{modality}
    e.g., test_best/MS-SSIM_t1_pre

    Args:
        writer: TensorBoard SummaryWriter (can be None).
        metrics: Dictionary of metric key -> value.
        prefix: Prefix for metric names (e.g., "test_best").
        modality: Modality name (e.g., "t1_pre", "bravo").
        step: Global step for TensorBoard.
        unified_metrics: Optional UnifiedMetrics instance for centralized logging.
    """
    # Use UnifiedMetrics if provided (preferred)
    if unified_metrics is not None:
        # Map metric keys to display names and add modality suffix
        name_map = {
            'msssim': 'MS-SSIM',
            'msssim_3d': 'MS-SSIM-3D',
            'psnr': 'PSNR',
            'lpips': 'LPIPS',
            'dice': 'Dice',
            'iou': 'IoU',
        }
        transformed = {}
        for key, value in metrics.items():
            display_name = name_map.get(key)
            if display_name:
                transformed[f'{display_name}_{modality}'] = value
        unified_metrics.log_test(transformed, prefix=prefix)
        return

    # Fallback to direct writer calls for backward compatibility
    if writer is None:
        return

    # Map metric keys to display names
    name_map = {
        'msssim': 'MS-SSIM',
        'msssim_3d': 'MS-SSIM-3D',
        'psnr': 'PSNR',
        'lpips': 'LPIPS',
        'dice': 'Dice',
        'iou': 'IoU',
    }

    for key, value in metrics.items():
        display_name = name_map.get(key)
        if display_name:
            writer.add_scalar(f'{prefix}/{display_name}_{modality}', value, step)
