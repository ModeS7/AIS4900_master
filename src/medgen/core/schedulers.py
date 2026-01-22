"""Learning rate scheduler utilities.

Provides factory functions for common scheduler configurations used
across different trainers.
"""
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)


def create_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    eta_min: float = 1e-6,
    start_factor: float = 0.1,
) -> SequentialLR:
    """Create a warmup + cosine annealing scheduler.

    The learning rate starts at start_factor * base_lr, linearly increases
    to base_lr over warmup_epochs, then follows cosine annealing to eta_min.

    Args:
        optimizer: The optimizer to schedule.
        warmup_epochs: Number of epochs for linear warmup.
        total_epochs: Total number of training epochs.
        eta_min: Minimum learning rate for cosine annealing.
        start_factor: Starting LR factor for warmup (0.1 = start at 10% of base LR).

    Returns:
        SequentialLR scheduler combining warmup and cosine annealing.

    Raises:
        ValueError: If warmup_epochs >= total_epochs.
    """
    if warmup_epochs >= total_epochs:
        raise ValueError(
            f"warmup_epochs ({warmup_epochs}) must be less than "
            f"total_epochs ({total_epochs})"
        )

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=eta_min
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )


def create_warmup_constant_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    start_factor: float = 0.1,
) -> SequentialLR:
    """Create a warmup + constant LR scheduler.

    The learning rate starts at start_factor * base_lr, linearly increases
    to base_lr over warmup_epochs, then stays constant at base_lr.

    Args:
        optimizer: The optimizer to schedule.
        warmup_epochs: Number of epochs for linear warmup.
        total_epochs: Total number of training epochs.
        start_factor: Starting LR factor for warmup (0.1 = start at 10% of base LR).

    Returns:
        SequentialLR scheduler combining warmup and constant LR.

    Raises:
        ValueError: If warmup_epochs >= total_epochs.
    """
    if warmup_epochs >= total_epochs:
        raise ValueError(
            f"warmup_epochs ({warmup_epochs}) must be less than "
            f"total_epochs ({total_epochs})"
        )

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    constant_scheduler = ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=total_epochs - warmup_epochs
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, constant_scheduler],
        milestones=[warmup_epochs]
    )


def create_plateau_scheduler(
    optimizer: Optimizer,
    mode: str = 'min',
    factor: float = 0.5,
    patience: int = 10,
    min_lr: float = 1e-6,
    threshold: float = 1e-4,
) -> ReduceLROnPlateau:
    """Create a ReduceLROnPlateau scheduler.

    Reduces learning rate when a metric has stopped improving.
    Useful for adaptive training where you want the LR to decrease
    when validation loss plateaus.

    Note: This scheduler requires calling scheduler.step(metric) with
    the validation metric, not just scheduler.step().

    Args:
        optimizer: The optimizer to schedule.
        mode: 'min' to reduce LR when metric stops decreasing,
              'max' to reduce LR when metric stops increasing.
        factor: Factor by which to reduce LR (new_lr = old_lr * factor).
        patience: Number of epochs with no improvement before reducing LR.
        min_lr: Minimum learning rate.
        threshold: Threshold for measuring improvement.

    Returns:
        ReduceLROnPlateau scheduler.
    """
    return ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        threshold=threshold,
    )
