"""
Reconstruction figure visualization utilities.

Shared function for worst_batch, validation, and test visualizations.
Works for both VAE (no timesteps) and Diffusion (with timesteps).
"""
import io
from typing import Dict, Optional, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def create_reconstruction_figure(
    original: Union[torch.Tensor, Dict[str, torch.Tensor]],
    generated: Union[torch.Tensor, Dict[str, torch.Tensor]],
    title: Optional[str] = None,
    timesteps: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    max_samples: int = 8,
    metrics: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """Create reconstruction visualization figure.

    Shared function for worst_batch, validation, and test visualizations.
    Works for both VAE (no timesteps) and Diffusion (with timesteps).

    Layout:
    - Single mode: 3 rows (Original, Generated, Difference)
    - Dual mode: 6 rows (Original_ch1, Generated_ch1, Diff_ch1, Original_ch2, ...)

    Args:
        original: Ground truth images [B, C, H, W] or dict of channel tensors.
        generated: Generated images [B, C, H, W] or dict of channel tensors.
        title: Optional title for the figure.
        timesteps: Optional per-sample timesteps [B] (shown in column titles).
        mask: Optional segmentation mask [B, 1, H, W] for contour overlay.
        max_samples: Maximum number of samples to display.
        metrics: Optional dict of metrics to show in subtitle (e.g., {'MS-SSIM': 0.95}).

    Returns:
        Matplotlib figure.
    """
    is_dual = isinstance(original, dict)

    if is_dual:
        return _create_dual_reconstruction_figure(
            original, generated, title, timesteps, mask, max_samples, metrics
        )
    else:
        return _create_single_reconstruction_figure(
            original, generated, title, timesteps, mask, max_samples, metrics
        )


def _create_single_reconstruction_figure(
    original: torch.Tensor,
    generated: torch.Tensor,
    title: Optional[str],
    timesteps: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    max_samples: int,
    metrics: Optional[Dict[str, float]],
) -> plt.Figure:
    """Create figure for single-channel reconstruction (seg, bravo modes)."""
    # Convert to numpy (float() handles bfloat16 -> float32)
    if isinstance(original, torch.Tensor):
        orig_np = original.cpu().float().numpy()
        gen_np = generated.cpu().float().numpy()
    else:
        orig_np = original
        gen_np = generated

    n_samples = min(max_samples, orig_np.shape[0])

    # Use first channel
    orig_ch = orig_np[:n_samples, 0]
    gen_ch = gen_np[:n_samples, 0]
    diff = np.abs(orig_ch - gen_ch)

    mask_np = None
    if mask is not None:
        mask_np = mask.cpu().float().numpy() if isinstance(mask, torch.Tensor) else mask

    # Create figure with minimal spacing - tight layout
    fig, axes = plt.subplots(
        3, n_samples, figsize=(2.5 * n_samples, 7),
        gridspec_kw={'hspace': 0.02, 'wspace': 0.02}
    )
    if n_samples == 1:
        axes = axes.reshape(3, 1)

    row_labels = ['Original', 'Generated', '|Diff|']

    for i in range(n_samples):
        # Column title with timestep if provided
        if timesteps is not None:
            t_val = timesteps[i].item() if isinstance(timesteps, torch.Tensor) else timesteps[i]
            axes[0, i].set_title(f't={t_val:.0f}', fontsize=8, pad=2)

        # Row 1: Original
        axes[0, i].imshow(np.clip(orig_ch[i], 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')

        # Row 2: Generated (with optional mask contour)
        axes[1, i].imshow(np.clip(gen_ch[i], 0, 1), cmap='gray', vmin=0, vmax=1)
        if mask_np is not None and mask_np.shape[0] > i:
            axes[1, i].contour(mask_np[i, 0], colors='red', linewidths=0.5, alpha=0.7)
        axes[1, i].axis('off')

        # Row 3: Difference heatmap
        im = axes[2, i].imshow(diff[i], cmap='hot', vmin=0, vmax=diff.max())
        axes[2, i].axis('off')

    # Add row labels on the left side
    for row, label in enumerate(row_labels):
        axes[row, 0].text(-0.02, 0.5, label, transform=axes[row, 0].transAxes,
                          fontsize=9, va='center', ha='right', rotation=90)

    # Build title
    full_title = title or ''
    if metrics:
        metrics_str = ' | '.join(f'{k}: {v:.4f}' for k, v in metrics.items())
        if full_title:
            full_title = f'{full_title}\n{metrics_str}'
        else:
            full_title = metrics_str

    if full_title:
        fig.suptitle(full_title, fontsize=10, y=0.98)

    # Tight margins - minimal whitespace
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.02, hspace=0.02, wspace=0.02)
    return fig


def _create_dual_reconstruction_figure(
    original: Dict[str, torch.Tensor],
    generated: Dict[str, torch.Tensor],
    title: Optional[str],
    timesteps: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    max_samples: int,
    metrics: Optional[Dict[str, float]],
) -> plt.Figure:
    """Create figure for dual-channel reconstruction (dual mode)."""
    keys = list(original.keys())
    key1, key2 = keys[0], keys[1]

    # Convert to numpy (float() handles bfloat16 -> float32)
    orig1 = original[key1].cpu().float().numpy() if isinstance(original[key1], torch.Tensor) else original[key1]
    orig2 = original[key2].cpu().float().numpy() if isinstance(original[key2], torch.Tensor) else original[key2]
    gen1 = generated[key1].cpu().float().numpy() if isinstance(generated[key1], torch.Tensor) else generated[key1]
    gen2 = generated[key2].cpu().float().numpy() if isinstance(generated[key2], torch.Tensor) else generated[key2]

    n_samples = min(max_samples, orig1.shape[0])

    # Use first channel of each
    orig1_ch = orig1[:n_samples, 0]
    orig2_ch = orig2[:n_samples, 0]
    gen1_ch = gen1[:n_samples, 0]
    gen2_ch = gen2[:n_samples, 0]
    diff1 = np.abs(orig1_ch - gen1_ch)
    diff2 = np.abs(orig2_ch - gen2_ch)

    mask_np = None
    if mask is not None:
        mask_np = mask.cpu().float().numpy() if isinstance(mask, torch.Tensor) else mask

    # Create figure: 6 rows (3 per channel) with minimal spacing
    fig, axes = plt.subplots(
        6, n_samples, figsize=(2.5 * n_samples, 14),
        gridspec_kw={'hspace': 0.02, 'wspace': 0.02}
    )
    if n_samples == 1:
        axes = axes.reshape(6, 1)

    # Row labels based on key names
    label1 = key1.replace('_', ' ').title()
    label2 = key2.replace('_', ' ').title()
    row_labels = [
        f'GT {label1}', f'Pred {label1}', f'|Diff| {label1}',
        f'GT {label2}', f'Pred {label2}', f'|Diff| {label2}'
    ]

    for i in range(n_samples):
        # Column title with timestep if provided
        if timesteps is not None:
            t_val = timesteps[i].item() if isinstance(timesteps, torch.Tensor) else timesteps[i]
            axes[0, i].set_title(f't={t_val:.0f}', fontsize=8, pad=2)

        # Channel 1
        axes[0, i].imshow(np.clip(orig1_ch[i], 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')

        axes[1, i].imshow(np.clip(gen1_ch[i], 0, 1), cmap='gray', vmin=0, vmax=1)
        if mask_np is not None and mask_np.shape[0] > i:
            axes[1, i].contour(mask_np[i, 0], colors='red', linewidths=0.5, alpha=0.7)
        axes[1, i].axis('off')

        axes[2, i].imshow(diff1[i], cmap='hot', vmin=0, vmax=diff1.max())
        axes[2, i].axis('off')

        # Channel 2
        axes[3, i].imshow(np.clip(orig2_ch[i], 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[3, i].axis('off')

        axes[4, i].imshow(np.clip(gen2_ch[i], 0, 1), cmap='gray', vmin=0, vmax=1)
        if mask_np is not None and mask_np.shape[0] > i:
            axes[4, i].contour(mask_np[i, 0], colors='red', linewidths=0.5, alpha=0.7)
        axes[4, i].axis('off')

        im = axes[5, i].imshow(diff2[i], cmap='hot', vmin=0, vmax=diff2.max())
        axes[5, i].axis('off')

    # Add row labels on the left side
    for row, label in enumerate(row_labels):
        axes[row, 0].text(-0.02, 0.5, label, transform=axes[row, 0].transAxes,
                          fontsize=8, va='center', ha='right', rotation=90)

    # Build title
    full_title = title or ''
    if metrics:
        metrics_str = ' | '.join(f'{k}: {v:.4f}' for k, v in metrics.items())
        if full_title:
            full_title = f'{full_title}\n{metrics_str}'
        else:
            full_title = metrics_str

    if full_title:
        fig.suptitle(full_title, fontsize=10, y=0.98)

    # Tight margins - minimal whitespace
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.02, hspace=0.02, wspace=0.02)
    return fig


def figure_to_buffer(fig: plt.Figure, dpi: int = 150) -> io.BytesIO:
    """Convert matplotlib figure to PNG buffer and close figure.

    Use this to safely convert figures for TensorBoard logging without
    memory leaks. The figure is closed after conversion.

    Args:
        fig: Matplotlib figure to convert.
        dpi: Resolution for saved figure. Default 150 for good quality.

    Returns:
        BytesIO buffer containing PNG image data.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)  # Prevent memory leak
    return buf
