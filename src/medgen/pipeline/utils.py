"""
Utility functions for diffusion model training.

This module provides utility functions for logging, VRAM monitoring,
and checkpoint saving/loading. Checkpoint functions are shared between
DiffusionTrainer and VAETrainer.

Note: GradientNormTracker and FLOPsTracker have been moved to tracking/.
"""
import contextlib
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


class GradientSkipDetector:
    """Skip optimizer steps when gradient norms are anomalously high.

    Tracks an exponential moving average (EMA) of gradient norms and skips
    the optimizer step when the current norm exceeds a threshold relative
    to the running average. This prevents training collapse from pathological
    batches that produce gradient spikes.

    After a warmup period (to establish a stable baseline), any gradient norm
    exceeding `threshold * ema` triggers a skip — the optimizer step is not
    applied and the gradients are zeroed.
    """

    def __init__(
        self,
        threshold: float = 10.0,
        warmup_steps: int = 100,
        ema_decay: float = 0.99,
    ):
        """Initialize detector.

        Args:
            threshold: Skip if grad_norm > threshold * running_average.
            warmup_steps: Steps before skip detection activates.
            ema_decay: EMA smoothing factor for grad norm tracking.
        """
        self.threshold = threshold
        self.warmup_steps = warmup_steps
        self.ema_decay = ema_decay
        self.ema_grad_norm: float | None = None
        self.step_count = 0
        self.skip_count = 0

    def should_skip(self, grad_norm: float) -> bool:
        """Check if this optimizer step should be skipped.

        Args:
            grad_norm: Pre-clip gradient norm from clip_grad_norm_.

        Returns:
            True if the step should be skipped (anomalous gradient).
        """
        self.step_count += 1

        # During warmup, build the EMA baseline without skipping
        if self.step_count <= self.warmup_steps:
            if self.ema_grad_norm is None:
                self.ema_grad_norm = grad_norm
            else:
                self.ema_grad_norm = (
                    self.ema_decay * self.ema_grad_norm
                    + (1 - self.ema_decay) * grad_norm
                )
            return False

        # Check for anomaly
        if self.ema_grad_norm is not None and self.ema_grad_norm > 0:
            ratio = grad_norm / self.ema_grad_norm
            if ratio > self.threshold:
                self.skip_count += 1
                logger.warning(
                    f"Gradient spike detected: norm={grad_norm:.2f}, "
                    f"EMA={self.ema_grad_norm:.4f}, ratio={ratio:.1f}x. "
                    f"Skipping optimizer step. (total skips: {self.skip_count})"
                )
                return True

        # Update EMA with non-anomalous norm
        self.ema_grad_norm = (
            self.ema_decay * self.ema_grad_norm
            + (1 - self.ema_decay) * grad_norm
        )
        return False


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
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    logger.info(f"Checkpoint saved: {save_path} ({size_mb:.0f} MB, epoch {epoch})")
    return save_path


def _safe_torch_save(obj: Any, path: str) -> None:
    """Save checkpoint with atomic rename to prevent corruption.

    Writes to a temp file first, then atomically renames. This ensures
    the target path always contains either the old or new complete file,
    never a half-written one.

    For small checkpoints (<2GB): buffers in memory then writes (avoids
    NFS position tracking bugs with PyTorch's many small writes).
    For large checkpoints: writes directly to temp file (avoids OOM).

    After saving, validates the file by checking the zip header (PyTorch
    checkpoints are zip archives). This catches silent filesystem corruption.

    Args:
        obj: Object to save (checkpoint dict).
        path: Target file path.

    Raises:
        RuntimeError: If post-save validation fails (corrupted file).
    """
    import io
    import zipfile

    parent = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=parent, suffix='.pt.tmp')
    fd_closed = False
    try:
        # Try memory-buffered write (fast, avoids NFS incremental write bugs)
        try:
            buffer = io.BytesIO()
            torch.save(obj, buffer)
            os.write(fd, buffer.getvalue())
            del buffer
        except MemoryError:
            # Fall back to direct file write for very large checkpoints
            os.lseek(fd, 0, os.SEEK_SET)
            with os.fdopen(os.dup(fd), 'wb') as f:
                torch.save(obj, f)
        os.fsync(fd)  # Flush file data to disk
        os.close(fd)
        fd_closed = True
        os.replace(tmp_path, path)
        # Fsync parent directory to make the rename durable.
        # Without this, a crash could leave the old directory entry pointing
        # to stale data even though the file contents were fsynced.
        dir_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        if not fd_closed:
            os.close(fd)
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise

    # Post-save validation: verify the file is a valid zip archive
    try:
        zipfile.ZipFile(path).close()
    except (zipfile.BadZipFile, OSError) as e:
        file_size = os.path.getsize(path) if os.path.exists(path) else 0
        raise RuntimeError(
            f"Checkpoint verification failed after save: {path} "
            f"(size={file_size} bytes, error={e}). "
            f"Filesystem may not have flushed data to disk."
        ) from e


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
