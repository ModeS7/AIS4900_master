"""
Plateau detection for progressive training.

Monitors training loss and detects when improvement rate falls below
a threshold, indicating the model has converged at current resolution.
"""
from collections import deque


class PlateauDetector:
    """Detect loss plateau for phase transition.

    Monitors training loss and detects when improvement rate falls below
    a threshold, indicating the model has converged at current resolution.

    Args:
        window_size: Number of epochs for rolling average.
        min_improvement: Minimum % improvement required (e.g., 0.5 = 0.5%).
        min_epochs: Minimum epochs before checking for plateau.
        patience: Epochs below threshold before declaring plateau.
    """

    def __init__(
        self,
        window_size: int = 10,
        min_improvement: float = 0.5,
        min_epochs: int = 20,
        patience: int = 5
    ):
        self.window_size = window_size
        self.min_improvement = min_improvement / 100.0  # Convert to fraction
        self.min_epochs = min_epochs
        self.patience = patience

        self.loss_history: deque = deque(maxlen=window_size * 2)
        self.epochs_without_improvement = 0
        self.best_window_avg = float('inf')

    def reset(self) -> None:
        """Reset detector for new phase."""
        self.loss_history.clear()
        self.epochs_without_improvement = 0
        self.best_window_avg = float('inf')

    def update(self, loss: float, epoch: int) -> None:
        """Update detector with new loss value (command only).

        Args:
            loss: Current epoch loss.
            epoch: Current epoch number (0-indexed).
        """
        self.loss_history.append(loss)

        # Not enough history yet
        if epoch < self.min_epochs or len(self.loss_history) < self.window_size:
            return

        # Calculate rolling average
        recent_losses = list(self.loss_history)[-self.window_size:]
        recent_avg = sum(recent_losses) / len(recent_losses)

        # Check improvement rate
        if self.best_window_avg > 0:
            improvement = (self.best_window_avg - recent_avg) / self.best_window_avg
        else:
            improvement = 0

        if improvement > self.min_improvement:
            self.best_window_avg = recent_avg
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    def is_plateau(self) -> bool:
        """Check if plateau has been detected (query only).

        Returns:
            True if no improvement for patience epochs, False otherwise.
        """
        return self.epochs_without_improvement >= self.patience
