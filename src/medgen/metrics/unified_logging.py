"""TensorBoard logging functions for UnifiedMetrics.

All functions take `metrics` (UnifiedMetrics instance) as first argument.
These are called via thin delegation wrappers in UnifiedMetrics.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .unified import UnifiedMetrics


def log_training(metrics: UnifiedMetrics, epoch: int) -> None:
    """Log all training metrics to TensorBoard."""
    if metrics.writer is None:
        return

    # Losses
    for key, data in metrics._train_losses.items():
        if data['count'] > 0:
            metrics.writer.add_scalar(
                f'Loss/{key}_train',
                data['sum'] / data['count'],
                epoch,
            )

    # LR
    if metrics._current_lr is not None:
        metrics.writer.add_scalar('LR/Generator', metrics._current_lr, epoch)

    # Grad norm
    if metrics._grad_norm_count > 0:
        metrics.writer.add_scalar(
            'training/grad_norm_avg',
            metrics._grad_norm_sum / metrics._grad_norm_count,
            epoch,
        )
        metrics.writer.add_scalar('training/grad_norm_max', metrics._grad_norm_max, epoch)

    # VRAM
    if metrics._vram_allocated > 0:
        metrics.writer.add_scalar('VRAM/allocated_GB', metrics._vram_allocated, epoch)
        metrics.writer.add_scalar('VRAM/reserved_GB', metrics._vram_reserved, epoch)
        metrics.writer.add_scalar('VRAM/max_allocated_GB', metrics._vram_max, epoch)

    # FLOPs
    if metrics._flops_epoch > 0:
        metrics.writer.add_scalar('FLOPs/TFLOPs_epoch', metrics._flops_epoch, epoch)
        metrics.writer.add_scalar('FLOPs/TFLOPs_total', metrics._flops_total, epoch)

    # Codebook
    if metrics._codebook_tracker is not None:
        metrics._codebook_tracker.log_to_tensorboard(metrics.writer, epoch)


def log_validation(metrics: UnifiedMetrics, epoch: int) -> None:
    """Log all validation metrics to TensorBoard."""
    if metrics.writer is None:
        return

    suffix = f'_{metrics.modality}' if metrics.modality else ''

    # Losses
    for key, data in metrics._val_losses.items():
        if data['count'] > 0:
            metrics.writer.add_scalar(
                f'Loss/{key}_val',
                data['sum'] / data['count'],
                epoch,
            )

    # Quality metrics (image modes)
    if metrics.uses_image_quality:
        if metrics._val_psnr_count > 0:
            metrics.writer.add_scalar(
                f'Validation/PSNR{suffix}',
                metrics._val_psnr_sum / metrics._val_psnr_count,
                epoch,
            )
        if metrics._val_msssim_count > 0:
            metrics.writer.add_scalar(
                f'Validation/MS-SSIM{suffix}',
                metrics._val_msssim_sum / metrics._val_msssim_count,
                epoch,
            )
        if metrics._val_lpips_count > 0:
            metrics.writer.add_scalar(
                f'Validation/LPIPS{suffix}',
                metrics._val_lpips_sum / metrics._val_lpips_count,
                epoch,
            )
        if metrics._val_msssim_3d_count > 0:
            metrics.writer.add_scalar(
                f'Validation/MS-SSIM-3D{suffix}',
                metrics._val_msssim_3d_sum / metrics._val_msssim_3d_count,
                epoch,
            )

    # Seg metrics (always log if computed)
    if metrics._val_dice_count > 0:
        metrics.writer.add_scalar(
            f'Validation/Dice{suffix}',
            metrics._val_dice_sum / metrics._val_dice_count,
            epoch,
        )
    if metrics._val_iou_count > 0:
        metrics.writer.add_scalar(
            f'Validation/IoU{suffix}',
            metrics._val_iou_sum / metrics._val_iou_count,
            epoch,
        )

    # Validation timesteps (format: Timestep/0.0-0.1, Timestep/0.1-0.2, etc.)
    for i in range(metrics.num_timestep_bins):
        if metrics._val_timesteps['counts'][i] > 0:
            bin_start = i / metrics.num_timestep_bins
            bin_end = (i + 1) / metrics.num_timestep_bins
            avg = metrics._val_timesteps['sums'][i] / metrics._val_timesteps['counts'][i]
            metrics.writer.add_scalar(f'Timestep/{bin_start:.1f}-{bin_end:.1f}', avg, epoch)

    # Regional
    if metrics._regional_tracker is not None:
        prefix = f'regional{suffix}' if suffix else 'regional'
        metrics._regional_tracker.log_to_tensorboard(metrics.writer, epoch, prefix=prefix)


def log_generation(metrics: UnifiedMetrics, epoch: int, results: dict[str, float]) -> None:
    """Log generation metrics to TensorBoard."""
    if metrics.writer is None:
        return
    for key, value in results.items():
        if key.startswith('Diversity/'):
            metric_name = key[len('Diversity/'):]
            metrics.writer.add_scalar(f'Generation_Diversity/{metric_name}', value, epoch)
        else:
            metrics.writer.add_scalar(f'Generation/{key}', value, epoch)


def log_test(metrics: UnifiedMetrics, test_metrics: dict[str, float], prefix: str = 'test_best') -> None:
    """Log test evaluation metrics."""
    if metrics.writer is None:
        return
    suffix = f'_{metrics.modality}' if metrics.modality else ''
    for key, value in test_metrics.items():
        metrics.writer.add_scalar(f'{prefix}/{key}{suffix}', value, 0)


def log_test_generation(
    metrics: UnifiedMetrics,
    results: dict[str, float],
    prefix: str = 'test_best',
) -> dict[str, float]:
    """Log test generation metrics (FID, KID, CMMD, diversity)."""
    exported: dict[str, float] = {}
    if metrics.writer is None:
        return exported

    suffix = f'_{metrics.modality}' if metrics.modality else ''

    for key, value in results.items():
        if key.startswith('Diversity/'):
            metric_name = key[len('Diversity/'):]
            metrics.writer.add_scalar(f'{prefix}_diversity/{metric_name}{suffix}', value, 0)
            exported[f'gen_diversity_{metric_name.lower()}'] = value
        else:
            metrics.writer.add_scalar(f'{prefix}_generation/{key}{suffix}', value, 0)
            exported[f'gen_{key.lower()}'] = value

    return exported


def log_test_regional(
    metrics: UnifiedMetrics,
    regional_tracker: Any,
    prefix: str = 'test_best',
) -> None:
    """Log regional metrics with modality suffix."""
    if metrics.writer is None or regional_tracker is None:
        return

    is_single_modality = metrics.mode not in ('multi_modality', 'dual', 'multi')

    if is_single_modality and metrics.modality:
        regional_prefix = f'{prefix}_regional_{metrics.modality}'
    else:
        regional_prefix = f'{prefix}_regional'

    regional_tracker.log_to_tensorboard(metrics.writer, 0, prefix=regional_prefix)


def log_validation_regional(
    metrics: UnifiedMetrics,
    regional_tracker: Any,
    epoch: int,
    modality_override: str | None = None,
) -> None:
    """Log regional metrics for validation (supports per-modality tracking)."""
    if metrics.writer is None or regional_tracker is None:
        return

    modality = modality_override or metrics.modality
    if modality:
        regional_prefix = f'regional_{modality}'
    else:
        regional_prefix = 'regional'

    regional_tracker.log_to_tensorboard(metrics.writer, epoch, prefix=regional_prefix)


def log_test_timesteps(
    metrics: UnifiedMetrics,
    timestep_bins: dict[str, float],
    prefix: str = 'test_best',
) -> None:
    """Log timestep bin losses."""
    if metrics.writer is None or not timestep_bins:
        return

    suffix = f'_{metrics.modality}' if metrics.modality else ''

    for bin_name, loss in timestep_bins.items():
        metrics.writer.add_scalar(f'{prefix}_timestep/{bin_name}{suffix}', loss, 0)


def log_per_channel_validation(
    metrics: UnifiedMetrics,
    channel_metrics: dict[str, dict[str, float]],
    epoch: int,
) -> None:
    """Log per-channel validation (dual/multi modes)."""
    if metrics.writer is None:
        return

    for channel_key, channel_data in channel_metrics.items():
        count = channel_data.get('count', 0)
        if count > 0:
            suffix = f'_{channel_key}'
            if 'psnr' in channel_data:
                avg_psnr = channel_data['psnr'] / count
                metrics.writer.add_scalar(f'Validation/PSNR{suffix}', avg_psnr, epoch)
            if 'msssim' in channel_data:
                avg_msssim = channel_data['msssim'] / count
                metrics.writer.add_scalar(f'Validation/MS-SSIM{suffix}', avg_msssim, epoch)
            if 'lpips' in channel_data and channel_data.get('lpips', 0) > 0:
                avg_lpips = channel_data['lpips'] / count
                metrics.writer.add_scalar(f'Validation/LPIPS{suffix}', avg_lpips, epoch)


def log_per_modality_validation(
    metrics: UnifiedMetrics,
    modality_metrics: dict[str, float],
    modality: str,
    epoch: int,
) -> None:
    """Log per-modality validation."""
    if metrics.writer is None:
        return

    suffix = f'_{modality}' if modality else ''

    # Image quality metrics
    if 'psnr' in modality_metrics and modality_metrics['psnr'] is not None:
        metrics.writer.add_scalar(f'Validation/PSNR{suffix}', modality_metrics['psnr'], epoch)
    if 'msssim' in modality_metrics and modality_metrics['msssim'] is not None:
        metrics.writer.add_scalar(f'Validation/MS-SSIM{suffix}', modality_metrics['msssim'], epoch)
    if 'lpips' in modality_metrics and modality_metrics['lpips'] is not None:
        metrics.writer.add_scalar(f'Validation/LPIPS{suffix}', modality_metrics['lpips'], epoch)
    if 'msssim_3d' in modality_metrics and modality_metrics['msssim_3d'] is not None:
        metrics.writer.add_scalar(f'Validation/MS-SSIM-3D{suffix}', modality_metrics['msssim_3d'], epoch)

    # Segmentation metrics
    if 'dice' in modality_metrics and modality_metrics['dice'] is not None:
        metrics.writer.add_scalar(f'Validation/Dice{suffix}', modality_metrics['dice'], epoch)
    if 'iou' in modality_metrics and modality_metrics['iou'] is not None:
        metrics.writer.add_scalar(f'Validation/IoU{suffix}', modality_metrics['iou'], epoch)


def log_regularization_loss(
    metrics: UnifiedMetrics,
    loss_type: str,
    weighted_loss: float,
    epoch: int,
    unweighted_loss: float | None = None,
) -> None:
    """Log regularization losses (KL for VAE, VQ for VQVAE)."""
    if metrics.writer is None:
        return

    metrics.writer.add_scalar(f'Loss/{loss_type}_val', weighted_loss, epoch)

    if unweighted_loss is not None:
        metrics.writer.add_scalar(f'Loss/{loss_type}_unweighted_val', unweighted_loss, epoch)


def log_codebook_metrics(
    metrics: UnifiedMetrics,
    codebook_tracker: Any,
    epoch: int,
    prefix: str = 'Codebook',
) -> dict[str, float]:
    """Log codebook metrics from external tracker (VQVAE)."""
    if metrics.writer is None or codebook_tracker is None:
        return {}

    return codebook_tracker.log_to_tensorboard(metrics.writer, epoch, prefix=prefix)


def log_timestep_region_heatmap(metrics: UnifiedMetrics, epoch: int) -> None:
    """Log 2D heatmap of loss by timestep bin and region."""
    if metrics.writer is None:
        return

    if not any(metrics._tr_tumor_count) and not any(metrics._tr_bg_count):
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    heatmap_data = np.zeros((metrics.num_timestep_bins, 2))
    labels_timestep = []

    for i in range(metrics.num_timestep_bins):
        bin_start = i / metrics.num_timestep_bins
        bin_end = (i + 1) / metrics.num_timestep_bins
        labels_timestep.append(f'{bin_start:.1f}-{bin_end:.1f}')

        if metrics._tr_tumor_count[i] > 0:
            heatmap_data[i, 0] = metrics._tr_tumor_sum[i] / metrics._tr_tumor_count[i]

        if metrics._tr_bg_count[i] > 0:
            heatmap_data[i, 1] = metrics._tr_bg_sum[i] / metrics._tr_bg_count[i]

    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Tumor', 'Background'])
    ax.set_yticks(range(metrics.num_timestep_bins))
    ax.set_yticklabels(labels_timestep)
    ax.set_xlabel('Region')
    ax.set_ylabel('Timestep Range')
    ax.set_title(f'Loss by Timestep & Region (Epoch {epoch})')
    plt.colorbar(im, ax=ax, label='MSE Loss')

    for i in range(metrics.num_timestep_bins):
        for j in range(2):
            ax.text(j, i, f'{heatmap_data[i, j]:.4f}',
                    ha='center', va='center', color='black', fontsize=8)

    plt.tight_layout()
    metrics.writer.add_figure('loss/timestep_region_heatmap', fig, epoch)
    plt.close(fig)


def log_flops_from_tracker(metrics: UnifiedMetrics, flops_tracker: Any, epoch: int) -> None:
    """Log FLOPs metrics from FLOPsTracker."""
    if metrics.writer is None or flops_tracker is None:
        return
    if not getattr(flops_tracker, '_measured', False):
        return
    if flops_tracker.forward_flops == 0:
        return

    completed_epochs = epoch + 1
    metrics.writer.add_scalar('FLOPs/TFLOPs_epoch', flops_tracker.get_tflops_epoch(), epoch)
    metrics.writer.add_scalar('FLOPs/TFLOPs_total', flops_tracker.get_tflops_total(completed_epochs), epoch)
    metrics.writer.add_scalar('FLOPs/TFLOPs_bs1', flops_tracker.get_tflops_bs1(), epoch)


def log_grad_norm_from_tracker(
    metrics: UnifiedMetrics,
    grad_tracker: Any,
    epoch: int,
    prefix: str = 'training/grad_norm',
) -> None:
    """Log gradient norm stats from GradientNormTracker."""
    if metrics.writer is None or grad_tracker is None:
        return
    if grad_tracker.count == 0:
        return

    metrics.writer.add_scalar(f'{prefix}_avg', grad_tracker.get_avg(), epoch)
    metrics.writer.add_scalar(f'{prefix}_max', grad_tracker.get_max(), epoch)


def log_sample_images(
    metrics: UnifiedMetrics,
    images: Any,
    tag: str,
    epoch: int,
) -> None:
    """Log image grid to TensorBoard using add_images."""
    if metrics.writer is None:
        return
    metrics.writer.add_images(tag, images, epoch)
