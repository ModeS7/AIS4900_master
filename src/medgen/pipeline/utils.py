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

    Excludes first epoch from average calculations since it includes
    warmup overhead (compilation, caching, etc.).
    """

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.epoch_times: List[float] = []

    def update(self, elapsed_time: float) -> None:
        """Record time for completed epoch."""
        self.epoch_times.append(elapsed_time)

    def get_eta_string(self) -> str:
        """Get formatted ETA string.

        Returns:
            String like "ETA: 2h 30m (Jan 20 15:30)" or empty if not enough data.
        """
        if len(self.epoch_times) < 1:
            return ""

        current_epoch = len(self.epoch_times)
        remaining_epochs = self.total_epochs - current_epoch

        if remaining_epochs <= 0:
            return ""

        # Use average excluding first epoch (warmup) if we have enough data
        if len(self.epoch_times) >= 2:
            avg_time = sum(self.epoch_times[1:]) / len(self.epoch_times[1:])
        else:
            # Only have first epoch, use it but it's likely overestimate
            avg_time = self.epoch_times[0]

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
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            if minutes > 0:
                return f"{hours:.0f}h {minutes:.0f}m"
            return f"{hours:.0f}h"
        else:
            days = seconds / 86400
            hours = (seconds % 86400) / 3600
            if hours > 0:
                return f"{days:.0f}d {hours:.0f}h"
            return f"{days:.0f}d"


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


def log_compression_epoch_summary(
    epoch: int,
    total_epochs: int,
    avg_losses: Dict[str, float],
    val_metrics: Optional[Dict[str, float]],
    elapsed_time: float,
    regularization_key: Optional[str] = None,
) -> None:
    """Log compression trainer epoch summary.

    Unified logging for VAE, VQ-VAE, DC-AE, and 3D variants.
    Shows both MS-SSIM and PSNR when available.

    Args:
        epoch: Current epoch number (0-indexed).
        total_epochs: Total number of epochs.
        avg_losses: Dict with 'gen', 'recon', 'disc' keys (and optionally 'kl' or 'vq').
        val_metrics: Dict with validation metrics ('gen', 'l1', 'msssim', 'psnr').
        elapsed_time: Time taken for the epoch in seconds.
        regularization_key: Loss key for regularization ('kl', 'vq', or None for DC-AE).
    """
    timestamp = time.strftime("%H:%M:%S")
    epoch_pct = ((epoch + 1) / total_epochs) * 100

    val_gen = f"(v:{val_metrics.get('gen', 0):.4f})" if val_metrics else ""
    val_l1 = f"(v:{val_metrics.get('l1', 0):.4f})" if val_metrics else ""

    # Build regularization string
    reg_str = ""
    if regularization_key and regularization_key in avg_losses:
        reg_str = f"{regularization_key.upper()}: {avg_losses[regularization_key]:.4f} | "

    # Build quality metrics string - show MS-SSIM, MS-SSIM-3D, and PSNR
    metrics_parts = []
    if val_metrics:
        if val_metrics.get('msssim'):
            metrics_parts.append(f"MS-SSIM: {val_metrics['msssim']:.3f}")
        if val_metrics.get('msssim_3d'):
            metrics_parts.append(f"MS-SSIM-3D: {val_metrics['msssim_3d']:.3f}")
        if val_metrics.get('psnr'):
            metrics_parts.append(f"PSNR: {val_metrics['psnr']:.2f}")
    metric_str = " | ".join(metrics_parts) if metrics_parts else ""

    logger.info(
        f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
        f"G: {avg_losses['gen']:.4f}{val_gen} | "
        f"L1: {avg_losses['recon']:.4f}{val_l1} | "
        f"{reg_str}"
        f"D: {avg_losses['disc']:.4f} | "
        f"{metric_str} | "
        f"Time: {elapsed_time:.1f}s"
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


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ema: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load training checkpoint with model, optimizer, scheduler, and EMA state.

    Used by both DiffusionTrainer and VAETrainer. Returns full checkpoint dict
    so caller can handle extra state (e.g., discriminator for VAE).

    Args:
        checkpoint_path: Path to the checkpoint file (.pt).
        model: Model to load state into.
        optimizer: Optional optimizer to load state into.
        scheduler: Optional scheduler to load state into.
        ema: Optional EMA wrapper to load state into.
        device: Device to map tensors to.
        strict: Whether to strictly enforce state dict key matching.

    Returns:
        Full checkpoint dict (includes 'epoch', 'config', and any extra_state).

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load EMA state if provided and saved
    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])

    return checkpoint


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
