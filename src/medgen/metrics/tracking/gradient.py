"""
Gradient norm tracking utilities.

Shared utility for tracking gradient norms during training.
"""


class GradientNormTracker:
    """Track gradient norm statistics during training.

    Shared utility for tracking gradient norms across DiffusionTrainer and VAETrainer.

    Usage:
        tracker = GradientNormTracker()
        # In training loop:
        tracker.update(grad_norm)
        # At end of epoch (use UnifiedMetrics.log_grad_norm_from_tracker):
        unified_metrics.log_grad_norm_from_tracker(tracker, epoch)
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

    def reset(self) -> None:
        """Reset tracker for next epoch."""
        self.sum = 0.0
        self.max = 0.0
        self.count = 0
