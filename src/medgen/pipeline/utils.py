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
import tempfile
import time
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EpochTimeEstimator:
    """Track epoch times and estimate completion time with adaptive correction.

    Uses a blend of overall average and exponential moving average (EMA):
    - Overall average: stable, based on all epochs
    - EMA: responsive, tracks recent trends and corrects for drift

    This automatically corrects for systematic under/overestimation by
    giving more weight to recent epoch times when they consistently
    deviate from the historical average.

    Excludes first epoch from calculations since it includes warmup overhead.
    """

    def __init__(self, total_epochs: int, ema_alpha: float = 0.2):
        """Initialize estimator.

        Args:
            total_epochs: Total number of epochs for training.
            ema_alpha: EMA smoothing factor (0.1=slow adapt, 0.3=fast adapt).
        """
        self.total_epochs = total_epochs
        self.epoch_count = 0
        self.total_time = 0.0  # Total time excluding first epoch
        self.first_epoch_time: float | None = None

        # Adaptive correction using EMA
        self.ema_alpha = ema_alpha
        self.ema_epoch_time: float | None = None

    def update(self, elapsed_time: float) -> None:
        """Record time for completed epoch and update EMA."""
        self.epoch_count += 1

        if self.epoch_count == 1:
            # Store first epoch separately (warmup overhead)
            self.first_epoch_time = elapsed_time
            self.ema_epoch_time = elapsed_time
        else:
            # Accumulate total time (excluding first epoch warmup)
            self.total_time += elapsed_time

            # Update EMA: responds to recent trends
            # EMA = alpha * new_value + (1 - alpha) * old_EMA
            if self.ema_epoch_time is not None:
                self.ema_epoch_time = (
                    self.ema_alpha * elapsed_time +
                    (1 - self.ema_alpha) * self.ema_epoch_time
                )
            else:
                self.ema_epoch_time = elapsed_time

    def get_eta_string(self) -> str:
        """Get formatted ETA string with adaptive correction.

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
            avg_time = self.total_time / (self.epoch_count - 1)
        elif self.first_epoch_time is not None:
            avg_time = self.first_epoch_time
        else:
            return ""

        # Adaptive blending: use EMA to correct for systematic drift
        # EMA tracks recent trends, average provides stability
        # Blend ratio increases as we get more data (more confident in EMA)
        if self.ema_epoch_time is not None and self.epoch_count >= 5:
            # Gradually increase EMA weight: 20% at epoch 5, up to 40% at epoch 50+
            ema_weight = min(0.4, 0.2 + (self.epoch_count - 5) * 0.005)
            blended_avg = (1 - ema_weight) * avg_time + ema_weight * self.ema_epoch_time
        else:
            blended_avg = avg_time

        eta_seconds = blended_avg * remaining_epochs
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
    avg_losses: tuple[float, float, float],
    elapsed_time: float,
    time_estimator: EpochTimeEstimator | None = None,
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


def get_vram_stats(device: torch.device) -> dict[str, float]:
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
    model_config: dict[str, Any] | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: Any | None = None,
    extra_state: dict[str, Any] | None = None,
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
    _safe_torch_save(checkpoint, save_path)
    return save_path


def _safe_torch_save(obj: Any, path: str) -> None:
    """Save with memory-buffered write and atomic rename.

    Serializes to an in-memory buffer first to avoid NFS position tracking
    bugs in PyTorch's PytorchStreamWriter (which does many small incremental
    writes that NFS can corrupt). Then writes raw bytes to a temp file and
    atomically renames to the target path.

    Args:
        obj: Object to save (checkpoint dict).
        path: Target file path.
    """
    import io

    buffer = io.BytesIO()
    torch.save(obj, buffer)

    parent = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=parent, suffix='.pt.tmp')
    try:
        os.write(fd, buffer.getvalue())
        os.close(fd)
        os.replace(tmp_path, path)
    except BaseException:
        os.close(fd)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def create_epoch_iterator(
    data_loader: DataLoader,
    epoch: int,
    is_cluster: bool,
    is_main_process: bool,
    ncols: int = 100,
    limit_batches: int | None = None,
) -> Iterator | tqdm:
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
