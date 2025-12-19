"""
Worst batch tracking and visualization.

Shared utility for tracking and visualizing the worst performing batch
during training for both DiffusionTrainer and VAETrainer.
"""
import logging
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from .metrics import create_reconstruction_figure

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
