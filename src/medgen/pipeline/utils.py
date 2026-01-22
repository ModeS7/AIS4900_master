"""
Utility functions for diffusion model training.

This module provides utility functions for logging, VRAM monitoring,
and checkpoint saving/loading. Checkpoint functions are shared between
DiffusionTrainer and VAETrainer.

Note: GradientNormTracker and FLOPsTracker have been moved to tracking/.
"""
import itertools
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EpochTimeEstimator:
    """Track epoch times and estimate completion time.

    Uses total elapsed time for stable estimates. This approach:
    - Naturally averages over all epochs including validation
    - Becomes more stable as training progresses
    - Is immune to individual epoch fluctuations

    Excludes first epoch from calculations since it includes warmup overhead.
    """

    def __init__(self, total_epochs: int):
        """Initialize estimator.

        Args:
            total_epochs: Total number of epochs for training.
        """
        self.total_epochs = total_epochs
        self.epoch_count = 0
        self.total_time = 0.0  # Total time excluding first epoch
        self.first_epoch_time: Optional[float] = None

    def update(self, elapsed_time: float) -> None:
        """Record time for completed epoch."""
        self.epoch_count += 1

        if self.epoch_count == 1:
            # Store first epoch separately (warmup overhead)
            self.first_epoch_time = elapsed_time
        else:
            # Accumulate total time (excluding first epoch warmup)
            self.total_time += elapsed_time

    def get_eta_string(self) -> str:
        """Get formatted ETA string.

        Returns:
            String like "ETA: 2h 30m (Jan 20 15:30)" or empty if not enough data.
        """
        if self.epoch_count < 1:
            return ""

        remaining_epochs = self.total_epochs - self.epoch_count

        if remaining_epochs <= 0:
            return ""

        # Calculate average time per epoch (excluding first epoch warmup)
        if self.epoch_count >= 2:
            # Use total accumulated time / epochs (excluding first)
            avg_time = self.total_time / (self.epoch_count - 1)
        elif self.first_epoch_time is not None:
            # Only have first epoch, use it (likely overestimate)
            avg_time = self.first_epoch_time
        else:
            return ""

        eta_seconds = avg_time * remaining_epochs
        completion_time = datetime.now() + timedelta(seconds=eta_seconds)

        # Format duration
        eta_str = self._format_duration(eta_seconds)

        # Format completion date/time
        if eta_seconds < 86400:  # Less than a day
            date_str = completion_time.strftime("%H:%M")
        else:
            date_str = completion_time.strftime("%b %d %H:%M")

        return f"ETA: {eta_str} ({date_str})"

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            return f"{minutes}m"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            if minutes > 0:
                return f"{hours}h {minutes}m"
            return f"{hours}h"
        else:
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            if hours > 0:
                return f"{days}d {hours}h"
            return f"{days}d"


def log_epoch_summary(
    epoch: int,
    total_epochs: int,
    avg_losses: Tuple[float, float, float],
    elapsed_time: float,
    time_estimator: Optional[EpochTimeEstimator] = None,
) -> None:
    """Log epoch completion summary.

    Args:
        epoch: Current epoch number (0-indexed).
        total_epochs: Total number of epochs.
        avg_losses: Tuple of (total_loss, mse_loss, perceptual_loss).
        elapsed_time: Time taken for the epoch in seconds.
        time_estimator: Optional estimator for ETA calculation.
    """
    timestamp = time.strftime("%H:%M:%S")
    epoch_pct = ((epoch + 1) / total_epochs) * 100
    total_loss, mse_loss, perceptual_loss = avg_losses

    # Only show perceptual components when perceptual loss is enabled
    if perceptual_loss > 0:
        loss_str = f"Total: {total_loss:.6f} | MSE: {mse_loss:.6f} | Perceptual: {perceptual_loss:.6f}"
    else:
        loss_str = f"MSE: {mse_loss:.6f}"

    # Update estimator and get ETA
    eta_str = ""
    if time_estimator is not None:
        time_estimator.update(elapsed_time)
        eta_str = time_estimator.get_eta_string()
        if eta_str:
            eta_str = f" | {eta_str}"

    logger.info(
        f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) completed | "
        f"{loss_str} | Time: {elapsed_time:.1f}s{eta_str}"
    )


def get_vram_usage(device: torch.device) -> str:
    """Get current VRAM usage statistics.

    Args:
        device: CUDA device to query.

    Returns:
        Formatted string with VRAM usage information.
    """
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    return f"VRAM: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, max: {max_allocated:.1f}GB"


def get_vram_stats(device: torch.device) -> Dict[str, float]:
    """Get VRAM usage statistics as numeric values for TensorBoard logging.

    Args:
        device: CUDA device to query.

    Returns:
        Dict with VRAM stats in GB:
        - allocated: Currently allocated memory
        - reserved: Total reserved memory (includes cached)
        - max_allocated: Peak allocated memory since last reset
    """
    if not torch.cuda.is_available():
        return {'allocated': 0.0, 'reserved': 0.0, 'max_allocated': 0.0}

    return {
        'allocated': torch.cuda.memory_allocated(device) / 1024 ** 3,
        'reserved': torch.cuda.memory_reserved(device) / 1024 ** 3,
        'max_allocated': torch.cuda.max_memory_allocated(device) / 1024 ** 3,
    }


def save_full_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_dir: str,
    filename: str,
    model_config: Optional[Dict[str, Any]] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ema: Optional[Any] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Save full training checkpoint with model, optimizer, and scheduler state.

    Used for best/latest checkpoints by both DiffusionTrainer and VAETrainer.
    VAE can pass discriminator state via extra_state.

    Args:
        model: Model to save.
        optimizer: Optimizer to save state from.
        epoch: Current epoch number.
        save_dir: Directory to save checkpoint in.
        filename: Name for the checkpoint file (without extension).
        model_config: Optional model architecture config dict.
        scheduler: Optional learning rate scheduler to save state from.
        ema: Optional EMA wrapper (from ema-pytorch) to save state from.
        extra_state: Optional dict of additional state (e.g., discriminator for VAE).

    Returns:
        Full path to the saved checkpoint (.pt).
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    if model_config is not None:
        checkpoint['config'] = model_config
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()
    if extra_state is not None:
        checkpoint.update(extra_state)
    save_path = os.path.join(save_dir, f"{filename}.pt")
    torch.save(checkpoint, save_path)
    return save_path


def create_epoch_iterator(
    data_loader: DataLoader,
    epoch: int,
    is_cluster: bool,
    is_main_process: bool,
    ncols: int = 100,
    limit_batches: Optional[int] = None,
) -> Union[Iterator, tqdm]:
    """Create progress bar or plain iterator for epoch training.

    Args:
        data_loader: DataLoader to iterate over.
        epoch: Current epoch number.
        is_cluster: Whether running on cluster (disable progress bar).
        is_main_process: Whether this is main process (only main shows progress).
        ncols: Width of tqdm progress bar.
        limit_batches: Optional limit on number of batches per epoch (for quick testing).

    Returns:
        tqdm progress bar or plain iterator.
    """
    total = limit_batches if limit_batches else len(data_loader)
    iterator = itertools.islice(data_loader, limit_batches) if limit_batches else data_loader

    if is_main_process and not is_cluster:
        return tqdm(iterator, desc=f"Epoch {epoch}", ncols=ncols, total=total)
    return iter(iterator)
