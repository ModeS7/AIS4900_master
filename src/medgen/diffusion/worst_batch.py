"""
Worst batch tracking and visualization.

Shared utility for tracking and visualizing the worst performing batch
during training for both DiffusionTrainer and VAETrainer.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

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
        self.worst_data: Optional[Dict[str, Any]] = None

    def update(
        self,
        loss: float,
        original: Tensor,
        generated: Tensor,
        loss_breakdown: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
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
                'original': original.detach(),
                'generated': generated.detach(),
                'loss': loss,
                'loss_breakdown': loss_breakdown or {},
                'extra': extra or {},
            }
            return True
        return False

    def get_and_reset(self) -> Optional[Dict[str, Any]]:
        """Get worst batch data (moved to CPU) and reset tracker.

        Returns:
            Dict with 'original', 'generated', 'loss', 'loss_breakdown', 'extra',
            or None if no data tracked.
        """
        if self.worst_data is None:
            return None

        # Move tensors to CPU
        data = {
            'original': self.worst_data['original'].cpu(),
            'generated': self.worst_data['generated'].cpu(),
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
        writer: Optional[SummaryWriter],
        epoch: int,
        tag_prefix: str = "training",
        save_path: Optional[str] = None,
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


def create_worst_batch_figure(
    original: Tensor,
    generated: Tensor,
    loss: float,
    loss_breakdown: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
    max_samples: int = 4,
) -> plt.Figure:
    """Create visualization figure for worst batch.

    Shows:
    - Row 1: Original/ground truth images
    - Row 2: Generated/reconstructed images
    - Row 3: Absolute difference heatmap

    Args:
        original: Ground truth images [B, C, H, W].
        generated: Generated images [B, C, H, W].
        loss: Total loss value.
        loss_breakdown: Optional dict of individual losses.
        extra: Optional extra info (e.g., {'timestep': 450}).
        max_samples: Maximum number of samples to show.

    Returns:
        Matplotlib figure.
    """
    n_samples = min(max_samples, original.shape[0])

    # Convert to numpy, use first channel if multi-channel
    orig_np = original[:n_samples, 0].numpy()
    gen_np = generated[:n_samples, 0].numpy()
    diff_np = np.abs(orig_np - gen_np)

    # Create figure: 3 rows x n_samples columns
    fig, axes = plt.subplots(3, n_samples, figsize=(3 * n_samples, 9))

    # Handle single sample case (axes won't be 2D)
    if n_samples == 1:
        axes = axes.reshape(3, 1)

    for i in range(n_samples):
        # Row 1: Original
        axes[0, i].imshow(np.clip(orig_np[i], 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')

        # Row 2: Generated
        axes[1, i].imshow(np.clip(gen_np[i], 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Generated {i+1}')
        axes[1, i].axis('off')

        # Row 3: Difference heatmap
        im = axes[2, i].imshow(diff_np[i], cmap='hot', vmin=0, vmax=diff_np.max())
        axes[2, i].set_title(f'|Diff| {i+1}')
        axes[2, i].axis('off')

    # Add colorbar for heatmap
    cbar = fig.colorbar(im, ax=axes[2, :].tolist(), shrink=0.6, aspect=20)
    cbar.set_label('Absolute Error')

    # Build title
    title_parts = [f"Worst Batch - Loss: {loss:.4f}"]

    # Add loss breakdown
    if loss_breakdown:
        breakdown_str = ", ".join(f"{k}: {v:.4f}" for k, v in loss_breakdown.items())
        title_parts.append(f"({breakdown_str})")

    # Add extra info (e.g., timestep for diffusion)
    extra = extra or {}
    if 'timestep' in extra:
        title_parts.append(f"[t={extra['timestep']}]")

    fig.suptitle(" ".join(title_parts), fontsize=12)
    plt.tight_layout()

    return fig
