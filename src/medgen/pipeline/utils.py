"""
Utility functions for diffusion model training.

This module provides utility functions for logging, VRAM monitoring,
and checkpoint saving/loading. Checkpoint functions are shared between
DiffusionTrainer and VAETrainer.

Note: GradientNormTracker and FLOPsTracker have been moved to tracking/.
"""
import logging
import os
import time
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def log_epoch_summary(
    epoch: int,
    total_epochs: int,
    avg_losses: Tuple[float, float, float],
    elapsed_time: float
) -> None:
    """Log epoch completion summary.

    Args:
        epoch: Current epoch number (0-indexed).
        total_epochs: Total number of epochs.
        avg_losses: Tuple of (total_loss, mse_loss, perceptual_loss).
        elapsed_time: Time taken for the epoch in seconds.
    """
    timestamp = time.strftime("%H:%M:%S")
    epoch_pct = ((epoch + 1) / total_epochs) * 100
    total_loss, mse_loss, perceptual_loss = avg_losses

    print(
        f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) completed | "
        f"Total: {total_loss:.6f} | MSE: {mse_loss:.6f} | Perceptual: {perceptual_loss:.6f} | "
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
) -> Union[Iterator, tqdm]:
    """Create progress bar or plain iterator for epoch training.

    Args:
        data_loader: DataLoader to iterate over.
        epoch: Current epoch number.
        is_cluster: Whether running on cluster (disable progress bar).
        is_main_process: Whether this is main process (only main shows progress).
        ncols: Width of tqdm progress bar.

    Returns:
        tqdm progress bar or plain iterator.
    """
    if is_main_process and not is_cluster:
        return tqdm(data_loader, desc=f"Epoch {epoch}", ncols=ncols)
    return iter(data_loader)
