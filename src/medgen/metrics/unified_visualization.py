"""Visualization and figure generation functions for UnifiedMetrics.

All functions take `metrics` (UnifiedMetrics instance) as first argument.
These are called via thin delegation wrappers in UnifiedMetrics.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .unified import UnifiedMetrics


def log_reconstruction_figure(
    metrics: UnifiedMetrics,
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    epoch: int,
    mask: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
    tag: str = 'Figures/reconstruction',
    max_samples: int = 8,
    figure_metrics: dict[str, float] | None = None,
    save_path: str | None = None,
) -> None:
    """Log reconstruction comparison figure to TensorBoard."""
    if metrics.writer is None and save_path is None:
        return

    import matplotlib.pyplot as plt

    from .figures import create_reconstruction_figure

    # Handle 3D volumes - extract multiple slices for visualization
    if metrics.spatial_dims == 3:
        original = _extract_multiple_slices(metrics, original, num_slices=max_samples)
        reconstructed = _extract_multiple_slices(metrics, reconstructed, num_slices=max_samples)
        if mask is not None:
            mask = _extract_multiple_slices(metrics, mask, num_slices=max_samples)
        # For 3D, slices are from same volume - show timestep in metrics instead of per-column
        if timesteps is not None and len(timesteps) > 0:
            t_val = timesteps[0].item() if hasattr(timesteps[0], 'item') else timesteps[0]
            figure_metrics = figure_metrics.copy() if figure_metrics else {}
            figure_metrics['t'] = t_val
        timesteps = None  # Don't show per-column (all slices have same timestep)

    fig = create_reconstruction_figure(
        original=original,
        generated=reconstructed,
        timesteps=timesteps,
        mask=mask,
        max_samples=max_samples,
        metrics=figure_metrics,
    )

    # Log to TensorBoard
    if metrics.writer is not None:
        metrics.writer.add_figure(tag, fig, epoch)

    # Save to file if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close(fig)


def log_worst_batch(
    metrics: UnifiedMetrics,
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    loss: float,
    epoch: int,
    phase: str = 'train',
    mask: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
    tag_prefix: str | None = None,
    save_path: str | None = None,
    display_metrics: dict[str, float] | None = None,
) -> None:
    """Log worst batch visualization to TensorBoard."""
    if metrics.writer is None and save_path is None:
        return

    # Determine tag
    if tag_prefix is not None:
        tag = f'{tag_prefix}/worst_batch'
    else:
        phase_cap = phase.capitalize()
        tag = f'{phase_cap}/worst_batch'

    # Determine metrics to display
    fig_metrics = display_metrics if display_metrics is not None else {'loss': loss}

    log_reconstruction_figure(
        metrics,
        original=original,
        reconstructed=reconstructed,
        epoch=epoch,
        mask=mask,
        timesteps=timesteps,
        tag=tag,
        figure_metrics=fig_metrics,
        save_path=save_path,
    )


def log_denoising_trajectory(
    metrics: UnifiedMetrics,
    trajectory: list,
    epoch: int,
    tag: str = 'denoising_trajectory',
) -> None:
    """Log denoising step visualization to TensorBoard."""
    if metrics.writer is None or not trajectory:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    steps = len(trajectory)
    sample = trajectory[0]

    # Handle 3D - take center slice
    if metrics.spatial_dims == 3 and sample.dim() == 5:
        trajectory = [_extract_center_slice(metrics, t) for t in trajectory]
        sample = trajectory[0]

    fig, axes = plt.subplots(1, steps, figsize=(2.5 * steps, 3))
    if steps == 1:
        axes = [axes]

    for i, step_tensor in enumerate(trajectory):
        if isinstance(step_tensor, torch.Tensor):
            img = step_tensor[0, 0].cpu().float().numpy()
        else:
            img = step_tensor[0, 0]
        axes[i].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Step {i}', fontsize=8)
        axes[i].axis('off')

    fig.tight_layout()
    metrics.writer.add_figure(f'{tag}/progression', fig, epoch)
    plt.close(fig)


def log_generated_samples(
    metrics: UnifiedMetrics,
    samples: torch.Tensor,
    epoch: int,
    tag: str = 'Generated_Samples',
    nrow: int = 4,
    num_slices: int = 8,
) -> None:
    """Log generated samples grid to TensorBoard."""
    if metrics.writer is None:
        return

    from torchvision.utils import make_grid

    # Handle 3D - show multiple slices per sample
    if metrics.spatial_dims == 3 and samples.dim() == 5:
        _log_generated_samples_3d(metrics, samples, epoch, tag, num_slices)
        return

    # 2D: simple grid
    samples = torch.clamp(samples.float(), 0, 1)
    grid = make_grid(samples, nrow=nrow, normalize=False, padding=2)
    metrics.writer.add_image(tag, grid, epoch)


def _log_generated_samples_3d(
    metrics: UnifiedMetrics,
    samples: torch.Tensor,
    epoch: int,
    tag: str,
    num_slices: int = 8,
) -> None:
    """Log 3D generated samples with multiple slices per sample."""
    import matplotlib.pyplot as plt

    B, C, D, H, W = samples.shape
    samples = torch.clamp(samples.float(), 0, 1).cpu()

    margin = max(1, D // (num_slices + 2))
    indices = torch.linspace(margin, D - margin - 1, num_slices).long().tolist()

    fig, axes = plt.subplots(B, num_slices, figsize=(num_slices * 2, B * 2))

    if B == 1:
        axes = axes.reshape(1, -1)

    for b in range(B):
        for s, slice_idx in enumerate(indices):
            ax = axes[b, s]
            slice_img = samples[b, :, slice_idx, :, :]
            if C == 1:
                ax.imshow(slice_img[0].numpy(), cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(slice_img[0].numpy(), cmap='gray', vmin=0, vmax=1)

            ax.set_xticks([])
            ax.set_yticks([])

            if b == 0:
                ax.set_title(f'z={slice_idx}', fontsize=8)
            if s == 0:
                ax.set_ylabel(f'Sample {b+1}', fontsize=8)

    plt.tight_layout()
    metrics.writer.add_figure(tag, fig, epoch)
    plt.close(fig)


def log_latent_samples(
    metrics: UnifiedMetrics,
    samples: torch.Tensor,
    epoch: int,
    tag: str = 'Latent_Samples',
    num_slices: int = 8,
) -> None:
    """Log latent space samples to TensorBoard (before decoding)."""
    if metrics.writer is None:
        return

    if samples.dim() == 5:
        _log_latent_samples_3d(metrics, samples, epoch, tag, num_slices)
        return

    _log_latent_samples_2d(metrics, samples, epoch, tag)


def _log_latent_samples_2d(
    metrics: UnifiedMetrics,
    samples: torch.Tensor,
    epoch: int,
    tag: str,
) -> None:
    """Log 2D latent samples with per-channel visualization."""
    import matplotlib.pyplot as plt

    B, C_lat, H, W = samples.shape
    samples = samples.float().cpu()

    sample = samples[0]

    n_cols = min(C_lat + 1, 5)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.5, 2.5))

    if n_cols == 1:
        axes = [axes]

    vmin, vmax = sample.min().item(), sample.max().item()

    for c in range(min(C_lat, n_cols - 1)):
        ax = axes[c]
        ax.imshow(sample[c].numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'Ch {c}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    if n_cols > 1:
        ax = axes[-1]
        magnitude = torch.sqrt((sample ** 2).mean(dim=0))
        ax.imshow(magnitude.numpy(), cmap='magma')
        ax.set_title('Magnitude', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'Latent Space (range: [{vmin:.2f}, {vmax:.2f}])', fontsize=10)
    plt.tight_layout()
    metrics.writer.add_figure(tag, fig, epoch)
    plt.close(fig)


def _log_latent_samples_3d(
    metrics: UnifiedMetrics,
    samples: torch.Tensor,
    epoch: int,
    tag: str,
    num_slices: int = 8,
) -> None:
    """Log 3D latent samples with per-channel center slices."""
    import matplotlib.pyplot as plt

    B, C_lat, D, H, W = samples.shape
    samples = samples.float().cpu()

    center_idx = D // 2
    sample = samples[0, :, center_idx, :, :]

    n_cols = min(C_lat + 1, 5)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.5, 2.5))

    if n_cols == 1:
        axes = [axes]

    vmin, vmax = sample.min().item(), sample.max().item()

    for c in range(min(C_lat, n_cols - 1)):
        ax = axes[c]
        ax.imshow(sample[c].numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'Ch {c}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    if n_cols > 1:
        ax = axes[-1]
        magnitude = torch.sqrt((sample ** 2).mean(dim=0))
        ax.imshow(magnitude.numpy(), cmap='magma')
        ax.set_title('Magnitude', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'Latent Space z={center_idx} (range: [{vmin:.2f}, {vmax:.2f}])', fontsize=10)
    plt.tight_layout()
    metrics.writer.add_figure(tag, fig, epoch)
    plt.close(fig)


def log_latent_trajectory(
    metrics: UnifiedMetrics,
    trajectory: list,
    epoch: int,
    tag: str = 'denoising_trajectory',
) -> None:
    """Log latent space denoising trajectory to TensorBoard."""
    if metrics.writer is None or not trajectory:
        return

    import matplotlib.pyplot as plt

    n_steps = len(trajectory)
    first = trajectory[0]

    is_3d = first.dim() == 5

    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 2, 2.5))
    if n_steps == 1:
        axes = [axes]

    for i, latent in enumerate(trajectory):
        ax = axes[i]
        latent = latent[0].float().cpu()

        if is_3d:
            center_idx = latent.shape[1] // 2
            slice_data = latent[:, center_idx, :, :]
        else:
            slice_data = latent

        magnitude = torch.sqrt((slice_data ** 2).mean(dim=0))
        ax.imshow(magnitude.numpy(), cmap='magma')
        ax.set_title(f't={i}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('Latent Denoising Trajectory (Magnitude)', fontsize=10)
    plt.tight_layout()
    metrics.writer.add_figure(f'{tag}/progression_latent', fig, epoch)
    plt.close(fig)


def log_test_figure(
    metrics: UnifiedMetrics,
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    prefix: str = 'test_best',
    mask: torch.Tensor | None = None,
    figure_metrics: dict[str, float] | None = None,
) -> None:
    """Log test evaluation figure to TensorBoard."""
    if metrics.writer is None:
        return

    log_reconstruction_figure(
        metrics,
        original=original,
        reconstructed=reconstructed,
        epoch=0,
        mask=mask,
        tag=f'{prefix}/reconstruction',
        figure_metrics=figure_metrics,
    )


def _extract_center_slice(metrics: UnifiedMetrics, tensor: torch.Tensor) -> torch.Tensor:
    """Extract center slice from 3D volume."""
    if tensor.dim() == 5:
        depth = tensor.shape[2]
        center_idx = depth // 2
        return tensor[:, :, center_idx, :, :]
    return tensor


def _extract_multiple_slices(
    metrics: UnifiedMetrics,
    tensor: torch.Tensor,
    num_slices: int = 8,
) -> torch.Tensor:
    """Extract multiple evenly-spaced slices from 3D volume."""
    if tensor.dim() != 5:
        return tensor

    B, C, D, H, W = tensor.shape

    num_slices = min(num_slices, D)
    if num_slices <= 1:
        mid = D // 2
        return tensor[:, :, mid:mid+1, :, :].squeeze(2)

    margin = max(1, D // (num_slices + 1))
    if D - 2 * margin > 0 and num_slices > 1:
        indices = [margin + i * (D - 2 * margin) // (num_slices - 1) for i in range(num_slices)]
    else:
        indices = [i * (D - 1) // (num_slices - 1) for i in range(num_slices)]
    indices = [min(max(0, idx), D - 1) for idx in indices]

    slices = [tensor[0, :, idx, :, :] for idx in indices]
    return torch.stack(slices, dim=0)
