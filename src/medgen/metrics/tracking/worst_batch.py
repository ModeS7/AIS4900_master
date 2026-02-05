"""
Worst batch tracking and visualization.

Shared utility for tracking and visualizing the worst performing batch
during training for both DiffusionTrainer and VAETrainer.
"""

import logging
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from ..figures import create_reconstruction_figure

logger = logging.getLogger(__name__)


class WorstBatchTracker:
    """Tracks the worst performing batch during training.

    Usage:
        tracker = WorstBatchTracker()

        # During training step:
        tracker.update(loss, original=images, generated=output, extra={'timestep': t})

        # At end of epoch:
        tracker.log_and_reset(writer, epoch)
    """

    def __init__(self, enabled: bool = True):
        """Initialize tracker.

        Args:
            enabled: Whether tracking is enabled.
        """
        self.enabled = enabled
        self.worst_loss: float = 0.0
        self.worst_data: dict[str, Any] | None = None

    def update(
        self,
        loss: float,
        original: Tensor,
        generated: Tensor,
        loss_breakdown: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Update tracker if this batch has higher loss.

        Args:
            loss: Total loss value for this batch.
            original: Ground truth images [B, C, H, W].
            generated: Generated/reconstructed images [B, C, H, W].
            loss_breakdown: Optional dict of individual losses (e.g., {'mse': 0.1, 'perc': 0.2}).
            extra: Optional extra info (e.g., {'timestep': 450} for diffusion).

        Returns:
            True if this batch became the new worst, False otherwise.
        """
        if not self.enabled:
            return False

        if loss > self.worst_loss:
            self.worst_loss = loss
            self.worst_data = {
                'original': original.detach().cpu(),  # Move to CPU to avoid GPU memory leak
                'generated': generated.detach().cpu(),  # Move to CPU to avoid GPU memory leak
                'loss': loss,
                'loss_breakdown': loss_breakdown or {},
                'extra': extra or {},
            }
            return True
        return False

    def get_and_reset(self) -> dict[str, Any] | None:
        """Get worst batch data (moved to CPU) and reset tracker.

        Returns:
            Dict with 'original', 'generated', 'loss', 'loss_breakdown', 'extra',
            or None if no data tracked.
        """
        if self.worst_data is None:
            return None

        # Tensors already on CPU (moved in update())
        data = {
            'original': self.worst_data['original'],
            'generated': self.worst_data['generated'],
            'loss': self.worst_data['loss'],
            'loss_breakdown': self.worst_data['loss_breakdown'],
            'extra': self.worst_data['extra'],
        }

        # Reset for next epoch
        self.worst_loss = 0.0
        self.worst_data = None

        return data

    def log_and_reset(
        self,
        writer: SummaryWriter | None,
        epoch: int,
        tag_prefix: str = "training",
        save_path: str | None = None,
    ) -> None:
        """Log worst batch visualization to TensorBoard and reset.

        Args:
            writer: TensorBoard SummaryWriter.
            epoch: Current epoch number.
            tag_prefix: Prefix for TensorBoard tags (e.g., "training", "validation").
            save_path: Optional path to save figure as PNG.
        """
        data = self.get_and_reset()
        if data is None or writer is None:
            return

        fig = create_worst_batch_figure(
            original=data['original'],
            generated=data['generated'],
            loss=data['loss'],
            loss_breakdown=data['loss_breakdown'],
            extra=data['extra'],
        )

        writer.add_figure(f'{tag_prefix}/worst_batch', fig, epoch)
        writer.add_scalar(f'{tag_prefix}/worst_batch_loss', data['loss'], epoch)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.close(fig)


def create_worst_batch_figure_3d(
    original: Tensor,
    generated: Tensor,
    loss: float,
    loss_breakdown: dict[str, float] | None = None,
    num_slices: int = 8,
    channel_names: list[str] | None = None,
) -> "plt.Figure":
    """Create visualization figure for worst 3D volume.

    Shows the N slices with highest reconstruction error from the worst volume.
    Layout: (3 × n_channels) rows × num_slices columns.
    Each channel gets 3 rows: Original, Generated, Difference.

    Args:
        original: Ground truth volume [1, C, D, H, W] (single worst volume).
        generated: Generated volume [1, C, D, H, W].
        loss: Total loss value.
        loss_breakdown: Optional dict of individual losses.
        num_slices: Number of worst slices to show. Default: 8.
        channel_names: Optional list of channel names (e.g., ['t1_pre', 't1_gd']).
            If None, uses 'Ch 0', 'Ch 1', etc. for multi-channel volumes.

    Returns:
        Matplotlib figure.
    """
    n_channels = original.shape[1]

    # Compute per-slice error (averaged over all channels)
    diff = torch.abs(original.float() - generated.float())
    # Mean over batch, channel, height, width -> [D]
    per_slice_error = diff.mean(dim=(0, 1, 3, 4))

    depth = per_slice_error.shape[0]
    num_slices = min(num_slices, depth)

    # Get indices of top-K worst slices
    worst_indices = torch.topk(per_slice_error, k=num_slices).indices
    worst_indices = worst_indices.sort().values  # Sort for visual order

    # Generate channel labels
    if channel_names is None:
        if n_channels == 1:
            labels = ['']  # No label needed for single channel
        else:
            labels = [f'Ch {i}' for i in range(n_channels)]
    else:
        labels = [name.replace('_', ' ').title() for name in channel_names]

    # Create figure: (3 × n_channels) rows × num_slices columns with minimal spacing
    n_rows = 3 * n_channels
    fig_height = 1.2 * n_channels + 0.5  # Compact height
    fig, axes = plt.subplots(
        n_rows, num_slices, figsize=(1.5 * num_slices, fig_height),
        gridspec_kw={'hspace': 0.02, 'wspace': 0.02}
    )
    if num_slices == 1:
        axes = axes.reshape(n_rows, 1)
    if n_rows == 1:
        axes = axes.reshape(1, num_slices)

    # Process each channel
    for ch_idx in range(n_channels):
        # Convert to numpy for this channel
        orig_np = original[0, ch_idx].cpu().float().numpy()  # [D, H, W]
        gen_np = generated[0, ch_idx].cpu().float().numpy()
        diff_np = diff[0, ch_idx].cpu().float().numpy()
        diff_max = diff_np.max() if diff_np.max() > 0 else 1.0

        # Row offsets for this channel
        row_orig = ch_idx * 3
        row_gen = ch_idx * 3 + 1
        row_diff = ch_idx * 3 + 2

        label_suffix = f' {labels[ch_idx]}' if labels[ch_idx] else ''

        for col, slice_idx in enumerate(worst_indices):
            idx = slice_idx.item()

            # Only show slice titles on first channel group
            if ch_idx == 0:
                axes[row_orig, col].set_title(f'Slice {idx}', fontsize=8)

            # Original
            axes[row_orig, col].imshow(
                np.clip(orig_np[idx], 0, 1), cmap='gray', vmin=0, vmax=1
            )
            axes[row_orig, col].axis('off')

            # Generated
            axes[row_gen, col].imshow(
                np.clip(gen_np[idx], 0, 1), cmap='gray', vmin=0, vmax=1
            )
            axes[row_gen, col].axis('off')

            # Difference heatmap
            axes[row_diff, col].imshow(diff_np[idx], cmap='hot', vmin=0, vmax=diff_max)
            axes[row_diff, col].axis('off')

        # Row labels (only on first column)
        axes[row_orig, 0].set_ylabel(f'Original{label_suffix}', fontsize=9)
        axes[row_gen, 0].set_ylabel(f'Generated{label_suffix}', fontsize=9)
        axes[row_diff, 0].set_ylabel(f'|Diff|{label_suffix}', fontsize=9)

    # Build title
    title_parts = [f"Worst Volume - Loss: {loss:.4f}"]
    if loss_breakdown:
        breakdown_str = ", ".join(f"{k}: {v:.4f}" for k, v in loss_breakdown.items())
        title_parts.append(f"({breakdown_str})")

    fig.suptitle(" ".join(title_parts), fontsize=10)
    plt.tight_layout(pad=0.3, h_pad=0.1, w_pad=0.1)
    return fig


def create_worst_batch_figure(
    original: Tensor,
    generated: Tensor,
    loss: float,
    loss_breakdown: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
    max_samples: int = 8,
) -> plt.Figure:
    """Create visualization figure for worst batch.

    Uses shared create_reconstruction_figure for consistent visualization.

    Args:
        original: Ground truth images [B, C, H, W].
        generated: Generated images [B, C, H, W].
        loss: Total loss value.
        loss_breakdown: Optional dict of individual losses.
        extra: Optional extra info (e.g., {'timesteps': tensor}).
        max_samples: Maximum number of samples to show.

    Returns:
        Matplotlib figure.
    """
    # Build title
    title_parts = [f"Worst Batch - Loss: {loss:.4f}"]

    if loss_breakdown:
        breakdown_str = ", ".join(f"{k}: {v:.4f}" for k, v in loss_breakdown.items())
        title_parts.append(f"({breakdown_str})")

    extra = extra or {}
    timesteps = extra.get('timesteps')

    # Add average timestep to title if available
    if timesteps is not None:
        avg_t = timesteps.float().mean().item() if isinstance(timesteps, torch.Tensor) else np.mean(timesteps)
        title_parts.append(f"[Avg t={avg_t:.0f}]")

    return create_reconstruction_figure(
        original=original,
        generated=generated,
        title=" ".join(title_parts),
        timesteps=timesteps,
        mask=extra.get('mask'),
        max_samples=max_samples,
    )
