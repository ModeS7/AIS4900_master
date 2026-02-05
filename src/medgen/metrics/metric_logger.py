"""TensorBoard logging utilities for unified metrics system.

This module provides helper functions for logging various metrics to TensorBoard,
including visualization helpers for reconstruction figures, heatmaps, and
generated samples. Used by UnifiedMetrics for all TensorBoard operations.
"""
import logging
from typing import TYPE_CHECKING

import torch
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from medgen.pipeline.utils import EpochTimeEstimator

logger = logging.getLogger(__name__)


def log_scalar_dict(
    writer: SummaryWriter | None,
    scalars: dict[str, float],
    epoch: int,
    prefix: str = '',
    suffix: str = '',
) -> None:
    """Log a dictionary of scalar values to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter (may be None).
        scalars: Dict of metric name -> value.
        epoch: Epoch/step number.
        prefix: Optional prefix for tags.
        suffix: Optional suffix for tags.
    """
    if writer is None:
        return

    for key, value in scalars.items():
        tag = key
        if prefix:
            tag = f'{prefix}/{tag}'
        if suffix:
            tag = f'{tag}{suffix}'
        writer.add_scalar(tag, value, epoch)


def log_losses(
    writer: SummaryWriter | None,
    losses: dict[str, dict[str, float]],
    epoch: int,
    phase: str = 'train',
) -> None:
    """Log accumulated losses to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter.
        losses: Dict of loss_name -> {'sum': float, 'count': int}.
        epoch: Epoch number.
        phase: 'train' or 'val'.
    """
    if writer is None:
        return

    for key, data in losses.items():
        if data['count'] > 0:
            avg_loss = data['sum'] / data['count']
            writer.add_scalar(f'Loss/{key}_{phase}', avg_loss, epoch)


def log_quality_metrics(
    writer: SummaryWriter | None,
    metrics: dict[str, float],
    epoch: int,
    suffix: str = '',
) -> None:
    """Log quality metrics (PSNR, MS-SSIM, LPIPS) to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter.
        metrics: Dict with optional keys: 'psnr', 'msssim', 'lpips', 'msssim_3d'.
        epoch: Epoch number.
        suffix: Modality suffix (e.g., '_bravo').
    """
    if writer is None:
        return

    metric_map = {
        'psnr': 'PSNR',
        'msssim': 'MS-SSIM',
        'lpips': 'LPIPS',
        'msssim_3d': 'MS-SSIM-3D',
        'dice': 'Dice',
        'iou': 'IoU',
        # Allow capitalized versions too
        'PSNR': 'PSNR',
        'MS-SSIM': 'MS-SSIM',
        'LPIPS': 'LPIPS',
        'MS-SSIM-3D': 'MS-SSIM-3D',
        'Dice': 'Dice',
        'IoU': 'IoU',
    }

    for key, value in metrics.items():
        if value is not None and key in metric_map:
            writer.add_scalar(f'Validation/{metric_map[key]}{suffix}', value, epoch)


def log_timestep_losses(
    writer: SummaryWriter | None,
    sums: list[float],
    counts: list[int],
    epoch: int,
    num_bins: int = 10,
) -> None:
    """Log timestep bin losses to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter.
        sums: List of loss sums per bin.
        counts: List of counts per bin.
        epoch: Epoch number.
        num_bins: Number of timestep bins.
    """
    if writer is None:
        return

    for i in range(num_bins):
        if counts[i] > 0:
            bin_start = i / num_bins
            bin_end = (i + 1) / num_bins
            avg = sums[i] / counts[i]
            writer.add_scalar(f'Timestep/{bin_start:.1f}-{bin_end:.1f}', avg, epoch)


def log_timestep_region_heatmap(
    writer: SummaryWriter | None,
    tumor_sums: list[float],
    tumor_counts: list[int],
    bg_sums: list[float],
    bg_counts: list[int],
    epoch: int,
    num_bins: int = 10,
) -> None:
    """Log 2D heatmap of loss by timestep bin and region.

    Creates a visualization showing how loss varies across timesteps
    for tumor vs background regions.

    Args:
        writer: TensorBoard SummaryWriter.
        tumor_sums: List of tumor loss sums per bin.
        tumor_counts: List of tumor counts per bin.
        bg_sums: List of background loss sums per bin.
        bg_counts: List of background counts per bin.
        epoch: Epoch number.
        num_bins: Number of timestep bins.
    """
    if writer is None:
        return

    # Check if we have any data
    if not any(tumor_counts) and not any(bg_counts):
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    heatmap_data = np.zeros((num_bins, 2))
    labels_timestep = []

    for i in range(num_bins):
        bin_start = i / num_bins
        bin_end = (i + 1) / num_bins
        labels_timestep.append(f'{bin_start:.1f}-{bin_end:.1f}')

        # Tumor column
        if tumor_counts[i] > 0:
            heatmap_data[i, 0] = tumor_sums[i] / tumor_counts[i]

        # Background column
        if bg_counts[i] > 0:
            heatmap_data[i, 1] = bg_sums[i] / bg_counts[i]

    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Tumor', 'Background'])
    ax.set_yticks(range(num_bins))
    ax.set_yticklabels(labels_timestep)
    ax.set_xlabel('Region')
    ax.set_ylabel('Timestep Range')
    ax.set_title(f'Loss by Timestep & Region (Epoch {epoch})')
    plt.colorbar(im, ax=ax, label='MSE Loss')

    # Add text annotations
    for i in range(num_bins):
        for j in range(2):
            ax.text(j, i, f'{heatmap_data[i, j]:.4f}',
                    ha='center', va='center', color='black', fontsize=8)

    plt.tight_layout()
    writer.add_figure('loss/timestep_region_heatmap', fig, epoch)
    plt.close(fig)


def log_grad_norms(
    writer: SummaryWriter | None,
    avg_norm: float,
    max_norm: float,
    epoch: int,
    prefix: str = 'training/grad_norm',
) -> None:
    """Log gradient norm statistics.

    Args:
        writer: TensorBoard SummaryWriter.
        avg_norm: Average gradient norm.
        max_norm: Maximum gradient norm.
        epoch: Epoch number.
        prefix: TensorBoard tag prefix.
    """
    if writer is None:
        return

    writer.add_scalar(f'{prefix}_avg', avg_norm, epoch)
    writer.add_scalar(f'{prefix}_max', max_norm, epoch)


def log_vram(
    writer: SummaryWriter | None,
    allocated: float,
    reserved: float,
    max_allocated: float,
    epoch: int,
    prefix: str = 'VRAM',
) -> None:
    """Log VRAM usage to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter.
        allocated: Allocated VRAM in GB.
        reserved: Reserved VRAM in GB.
        max_allocated: Maximum allocated VRAM in GB.
        epoch: Epoch number.
        prefix: TensorBoard tag prefix.
    """
    if writer is None:
        return

    writer.add_scalar(f'{prefix}/allocated_GB', allocated, epoch)
    writer.add_scalar(f'{prefix}/reserved_GB', reserved, epoch)
    writer.add_scalar(f'{prefix}/max_allocated_GB', max_allocated, epoch)


def log_flops(
    writer: SummaryWriter | None,
    tflops_epoch: float,
    tflops_total: float,
    tflops_bs1: float | None,
    epoch: int,
) -> None:
    """Log FLOPs metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter.
        tflops_epoch: TFLOPs for this epoch.
        tflops_total: Cumulative TFLOPs.
        tflops_bs1: TFLOPs per sample (optional).
        epoch: Epoch number.
    """
    if writer is None:
        return

    writer.add_scalar('FLOPs/TFLOPs_epoch', tflops_epoch, epoch)
    writer.add_scalar('FLOPs/TFLOPs_total', tflops_total, epoch)
    if tflops_bs1 is not None:
        writer.add_scalar('FLOPs/TFLOPs_bs1', tflops_bs1, epoch)


def log_generation_metrics(
    writer: SummaryWriter | None,
    results: dict[str, float],
    epoch: int,
) -> None:
    """Log generation metrics (KID, CMMD, FID) to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter.
        results: Dict of generation metric results.
            Keys starting with 'Diversity/' go to 'Generation_Diversity/' section.
        epoch: Epoch number.
    """
    if writer is None:
        return

    for key, value in results.items():
        if key.startswith('Diversity/'):
            # Diversity metrics get their own section
            metric_name = key[len('Diversity/'):]
            writer.add_scalar(f'Generation_Diversity/{metric_name}', value, epoch)
        else:
            writer.add_scalar(f'Generation/{key}', value, epoch)


def format_console_summary(
    epoch: int,
    total_epochs: int,
    elapsed_time: float,
    train_losses: dict[str, float],
    val_metrics: dict[str, float],
    uses_image_quality: bool,
    time_estimator: "EpochTimeEstimator | None" = None,
) -> str:
    """Format epoch completion summary for console.

    Args:
        epoch: Current epoch number (0-indexed).
        total_epochs: Total number of epochs.
        elapsed_time: Time taken for epoch in seconds.
        train_losses: Dict of training losses.
        val_metrics: Dict of validation metrics.
        uses_image_quality: Whether image quality metrics are used.
        time_estimator: Optional estimator for ETA calculation.

    Returns:
        Formatted summary string.
    """
    import time as time_module

    timestamp = time_module.strftime("%H:%M:%S")
    epoch_pct = ((epoch + 1) / total_epochs) * 100

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
    if uses_image_quality:
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

    return (
        f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
        + " | ".join(all_parts)
    )


def extract_center_slice(tensor: torch.Tensor) -> torch.Tensor:
    """Extract center slice from 3D volume.

    Args:
        tensor: 5D tensor [B, C, D, H, W].

    Returns:
        4D tensor [B, C, H, W] with center depth slice.
    """
    if tensor.dim() == 5:
        depth = tensor.shape[2]
        center_idx = depth // 2
        return tensor[:, :, center_idx, :, :]
    return tensor


def extract_multiple_slices(
    tensor: torch.Tensor,
    num_slices: int = 8,
) -> torch.Tensor:
    """Extract multiple evenly-spaced slices from 3D volume.

    For 3D worst_batch visualization, extracts N slices from the depth
    dimension and returns them as a batch of 2D images.

    Args:
        tensor: 5D tensor [B, C, D, H, W] (typically B=1 for 3D).
        num_slices: Number of slices to extract.

    Returns:
        4D tensor [N, C, H, W] with N evenly-spaced slices.
    """
    if tensor.dim() != 5:
        return tensor

    _B, _C, D, _H, _W = tensor.shape
    # Calculate evenly spaced indices (avoid edges)
    margin = D // (num_slices + 1)
    indices = [margin + i * (D - 2 * margin) // (num_slices - 1) for i in range(num_slices)]
    indices = [min(max(0, idx), D - 1) for idx in indices]  # Clamp to valid range

    # Extract slices from first volume (3D typically has batch_size=1)
    slices = [tensor[0, :, idx, :, :] for idx in indices]
    return torch.stack(slices, dim=0)  # [num_slices, C, H, W]
