"""
Gradient norm tracking utilities.

Shared utility for tracking gradient norms during training.
"""

from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class GradientNormTracker:
    """Track gradient norm statistics during training.

    Shared utility for tracking gradient norms across DiffusionTrainer and VAETrainer.

    Usage:
        tracker = GradientNormTracker()
        # In training loop:
        tracker.update(grad_norm)
        # At end of epoch:
        tracker.log(writer, epoch, prefix='training')
        tracker.reset()
    """

    def __init__(self) -> None:
        self.sum: float = 0.0
        self.max: float = 0.0
        self.count: int = 0

    def update(self, grad_norm: float) -> None:
        """Update tracker with a new gradient norm value."""
        self.sum += grad_norm
        self.max = max(self.max, grad_norm)
        self.count += 1

    def get_avg(self) -> float:
        """Get average gradient norm."""
        return self.sum / max(1, self.count)

    def get_max(self) -> float:
        """Get maximum gradient norm."""
        return self.max

    def log(
        self,
        writer: Optional[SummaryWriter],
        epoch: int,
        prefix: str = 'training',
    ) -> None:
        """Log gradient norm stats to TensorBoard.

        Args:
            writer: TensorBoard SummaryWriter instance.
            epoch: Current epoch number.
            prefix: Prefix for metric names. Will create {prefix}_avg and {prefix}_max.
                Example: prefix='training/grad_norm_g' creates 'training/grad_norm_g_avg'.
        """
        if writer is None or self.count == 0:
            return
        writer.add_scalar(f'{prefix}_avg', self.get_avg(), epoch)
        writer.add_scalar(f'{prefix}_max', self.get_max(), epoch)

    def reset(self) -> None:
        """Reset tracker for next epoch."""
        self.sum = 0.0
        self.max = 0.0
        self.count = 0
