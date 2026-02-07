"""History recording, JSON export, and console logging for UnifiedMetrics.

All functions take `metrics` (UnifiedMetrics instance) as first argument.
These are called via thin delegation wrappers in UnifiedMetrics.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medgen.pipeline.utils import EpochTimeEstimator

    from .unified import UnifiedMetrics

logger = logging.getLogger(__name__)


def record_epoch_history(metrics: UnifiedMetrics, epoch: int) -> None:
    """Record current epoch data to history for JSON export.

    Call this BEFORE reset_validation() to capture the epoch's data.
    """
    # Regional history
    if metrics._regional_tracker is not None:
        computed = metrics._regional_tracker.compute()
        if computed:
            metrics._regional_history[str(epoch)] = {
                'tumor': computed.get('tumor', 0),
                'background': computed.get('background', 0),
                'tumor_bg_ratio': computed.get('ratio', 0),
                'by_size': {
                    'tiny': computed.get('tumor_size_tiny', 0),
                    'small': computed.get('tumor_size_small', 0),
                    'medium': computed.get('tumor_size_medium', 0),
                    'large': computed.get('tumor_size_large', 0),
                }
            }

    # Timestep history
    epoch_timesteps: dict[str, Any] = {}
    for i in range(metrics.num_timestep_bins):
        if metrics._val_timesteps['counts'][i] > 0:
            bin_start = i / metrics.num_timestep_bins
            bin_end = (i + 1) / metrics.num_timestep_bins
            bin_name = f'{bin_start:.1f}-{bin_end:.1f}'
            epoch_timesteps[bin_name] = metrics._val_timesteps['sums'][i] / metrics._val_timesteps['counts'][i]
    if epoch_timesteps:
        metrics._timestep_history[str(epoch)] = epoch_timesteps

    # Timestep-region history
    epoch_tr: dict[str, Any] = {}
    for i in range(metrics.num_timestep_bins):
        bin_start = i / metrics.num_timestep_bins
        bin_label = f'{bin_start:.1f}'
        tumor_avg = metrics._tr_tumor_sum[i] / max(metrics._tr_tumor_count[i], 1) if metrics._tr_tumor_count[i] > 0 else 0.0
        bg_avg = metrics._tr_bg_sum[i] / max(metrics._tr_bg_count[i], 1) if metrics._tr_bg_count[i] > 0 else 0.0
        epoch_tr[bin_label] = {
            'tumor': tumor_avg,
            'background': bg_avg,
        }
    if any(metrics._tr_tumor_count) or any(metrics._tr_bg_count):
        metrics._timestep_region_history[str(epoch)] = epoch_tr


def save_json_histories(metrics: UnifiedMetrics, save_dir: str) -> None:
    """Save all history data to JSON files."""
    import json
    import os

    import numpy as np

    def convert_to_native(obj: Any) -> Any:
        """Recursively convert numpy types to native Python types for JSON."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    if metrics._regional_history:
        filepath = os.path.join(save_dir, 'regional_losses.json')
        with open(filepath, 'w') as f:
            json.dump(convert_to_native(metrics._regional_history), f, indent=2)

    if metrics._timestep_history:
        filepath = os.path.join(save_dir, 'timestep_losses.json')
        with open(filepath, 'w') as f:
            json.dump(convert_to_native(metrics._timestep_history), f, indent=2)

    if metrics._timestep_region_history:
        filepath = os.path.join(save_dir, 'timestep_region_losses.json')
        with open(filepath, 'w') as f:
            json.dump(convert_to_native(metrics._timestep_region_history), f, indent=2)


def log_console_summary(
    metrics: UnifiedMetrics,
    epoch: int,
    total_epochs: int,
    elapsed_time: float,
    time_estimator: EpochTimeEstimator | None = None,
) -> None:
    """Log epoch completion summary to console."""
    import time as time_module

    timestamp = time_module.strftime("%H:%M:%S")
    epoch_pct = ((epoch + 1) / total_epochs) * 100

    train_losses = metrics.get_training_losses()
    val_metrics = metrics.get_validation_metrics()

    # Build loss string
    loss_parts = []
    total_loss = train_losses.get('Total') or train_losses.get('MSE') or 0
    val_total = val_metrics.get('MSE') or val_metrics.get('Total') or 0
    loss_parts.append(f"Loss: {total_loss:.4f}")
    if val_total > 0:
        loss_parts[-1] += f"(v:{val_total:.4f})"

    for key in ['MSE', 'Perceptual', 'KL', 'VQ', 'BCE', 'Dice']:
        if key in train_losses and key != 'Total':
            loss_parts.append(f"{key}: {train_losses[key]:.4f}")

    # Build validation metrics string
    metric_parts = []
    if metrics.uses_image_quality:
        if 'MS-SSIM' in val_metrics:
            metric_parts.append(f"MS-SSIM: {val_metrics['MS-SSIM']:.3f}")
        if 'MS-SSIM-3D' in val_metrics:
            metric_parts.append(f"MS-SSIM-3D: {val_metrics['MS-SSIM-3D']:.3f}")
        if 'PSNR' in val_metrics:
            metric_parts.append(f"PSNR: {val_metrics['PSNR']:.2f}")
        if 'LPIPS' in val_metrics:
            metric_parts.append(f"LPIPS: {val_metrics['LPIPS']:.3f}")
    else:
        if 'Dice' in val_metrics:
            metric_parts.append(f"Dice: {val_metrics['Dice']:.3f}")
        if 'IoU' in val_metrics:
            metric_parts.append(f"IoU: {val_metrics['IoU']:.3f}")

    # Combine all parts
    all_parts = loss_parts + metric_parts
    all_parts.append(f"Time: {elapsed_time:.1f}s")

    # Add ETA if estimator provided
    if time_estimator is not None:
        time_estimator.update(elapsed_time)
        eta_str = time_estimator.get_eta_string()
        if eta_str:
            all_parts.append(eta_str)

    logger.info(
        f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
        + " | ".join(all_parts)
    )
