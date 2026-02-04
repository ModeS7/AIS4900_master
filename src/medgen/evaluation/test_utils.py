"""Test evaluation utility functions.

Provides shared functionality for test set evaluation:
- Checkpoint loading
- Results saving (JSON + TensorBoard)
- Metric logging helpers
- Metrics configuration
- create_compression_test_evaluator: Factory for compression test evaluators
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from medgen.evaluation import Compression3DTestEvaluator, CompressionTestEvaluator
    from medgen.metrics.unified import UnifiedMetrics
    from medgen.pipeline.compression_trainer import BaseCompressionTrainer

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Utilities
# =============================================================================

def load_checkpoint_if_needed(
    checkpoint_name: str | None,
    save_dir: str,
    model: nn.Module,
    device: torch.device,
) -> str:
    """Load checkpoint weights if specified.

    Args:
        checkpoint_name: "best", "latest", or None for current weights.
        save_dir: Directory containing checkpoints.
        model: Model to load weights into.
        device: Device to load weights to.

    Returns:
        Label string for logging ("best", "latest", or "current").
    """
    if checkpoint_name is not None:
        checkpoint_path = os.path.join(save_dir, f"checkpoint_{checkpoint_name}.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded {checkpoint_name} checkpoint for test evaluation")
        else:
            logger.warning(f"Checkpoint {checkpoint_path} not found, using current weights")
            return "current"

    return checkpoint_name or "current"


def save_test_results(
    metrics: dict[str, float],
    label: str,
    save_dir: str,
) -> str:
    """Save test results to JSON file.

    Args:
        metrics: Dictionary of metric name -> value.
        label: Checkpoint label ("best", "latest", "current").
        save_dir: Directory to save results to.

    Returns:
        Path to saved JSON file.
    """
    results_path = os.path.join(save_dir, f'test_results_{label}.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    return results_path


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
    unified_metrics: Optional["UnifiedMetrics"] = None,
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
    unified_metrics: Optional["UnifiedMetrics"] = None,
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


# =============================================================================
# Metrics Configuration
# =============================================================================

@dataclass
class MetricsConfig:
    """Configuration for which metrics to compute during test evaluation."""

    compute_l1: bool = True
    compute_psnr: bool = True
    compute_lpips: bool = True
    compute_msssim: bool = True
    compute_msssim_3d: bool = False  # For 3D trainers
    compute_regional: bool = False
    seg_mode: bool = False  # Compute Dice/IoU instead of image metrics


def create_compression_test_evaluator(
    trainer: 'BaseCompressionTrainer',
) -> Union['CompressionTestEvaluator', 'Compression3DTestEvaluator']:
    """Create test evaluator for compression trainer.

    Factory that creates appropriate evaluator based on trainer.spatial_dims.
    Creates a CompressionTestEvaluator (2D) or Compression3DTestEvaluator (3D)
    with trainer-specific callbacks.

    Args:
        trainer: Compression trainer instance.

    Returns:
        CompressionTestEvaluator for 2D, Compression3DTestEvaluator for 3D.
    """
    from medgen.evaluation import Compression3DTestEvaluator, CompressionTestEvaluator

    # Check for seg_mode (set by subclasses)
    seg_mode = getattr(trainer, 'seg_mode', False)
    seg_loss_fn = getattr(trainer, 'seg_loss_fn', None)

    # Get modality name for single-modality suffix
    # Use empty string for seg_conditioned modes (no suffix needed)
    mode_name = trainer.cfg.mode.get('name', 'bravo')
    if mode_name.startswith('seg_conditioned'):
        mode_name = ''

    # Get image keys for per-channel metrics
    n_channels = trainer.cfg.mode.get('in_channels', 1)
    image_keys = None
    if n_channels > 1:
        image_keys = trainer.cfg.mode.get('image_keys', None)

    # Regional tracker factory (use seg-specific tracker for seg_mode)
    regional_factory = None
    if trainer.log_regional_losses:
        if seg_mode and hasattr(trainer, '_create_seg_regional_tracker'):
            regional_factory = trainer._create_seg_regional_tracker
        else:
            regional_factory = trainer._create_regional_tracker

    # 3D evaluator
    if trainer.spatial_dims == 3:
        metrics_config = MetricsConfig(
            compute_l1=not seg_mode,
            compute_psnr=not seg_mode,
            compute_lpips=not seg_mode,
            compute_msssim=trainer.log_msssim and not seg_mode,  # 2D slicewise
            compute_msssim_3d=trainer.log_msssim and not seg_mode,  # Volumetric
            compute_regional=trainer.log_regional_losses,
            seg_mode=seg_mode,
        )

        # Worst batch figure callback (3D version)
        worst_batch_fig_fn = trainer._create_worst_batch_figure

        return Compression3DTestEvaluator(
            model=trainer.model_raw,
            device=trainer.device,
            save_dir=trainer.save_dir,
            forward_fn=trainer._test_forward,
            weight_dtype=trainer.weight_dtype,
            writer=trainer.writer,
            metrics_config=metrics_config,
            is_cluster=trainer.is_cluster,
            regional_tracker_factory=regional_factory,
            worst_batch_figure_fn=worst_batch_fig_fn,
            image_keys=image_keys,
            seg_loss_fn=seg_loss_fn if seg_mode else None,
            modality_name=mode_name,
        )

    # 2D evaluator
    metrics_config = MetricsConfig(
        compute_l1=not seg_mode,
        compute_psnr=not seg_mode,
        compute_lpips=not seg_mode,
        compute_msssim=trainer.log_msssim and not seg_mode,
        compute_msssim_3d=False,  # Volume 3D MS-SSIM added via callback
        compute_regional=trainer.log_regional_losses,
        seg_mode=seg_mode,
    )

    # Volume 3D MS-SSIM callback (for 2D trainers reconstructing full volumes)
    def volume_3d_msssim() -> float | None:
        if seg_mode:
            return None
        return trainer._compute_volume_3d_msssim(epoch=0, data_split='test_new')

    # Worst batch figure callback
    worst_batch_fig_fn = trainer._create_worst_batch_figure

    return CompressionTestEvaluator(
        model=trainer.model_raw,
        device=trainer.device,
        save_dir=trainer.save_dir,
        forward_fn=trainer._test_forward,
        weight_dtype=trainer.weight_dtype,
        writer=trainer.writer,
        metrics_config=metrics_config,
        is_cluster=trainer.is_cluster,
        regional_tracker_factory=regional_factory,
        volume_3d_msssim_fn=volume_3d_msssim,
        worst_batch_figure_fn=worst_batch_fig_fn,
        image_keys=image_keys,
        seg_loss_fn=seg_loss_fn if seg_mode else None,
        modality_name=mode_name,
    )
