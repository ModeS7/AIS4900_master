"""
Utility functions for diffusion model training.

This module provides utility functions for logging, VRAM monitoring,
and checkpoint saving/loading.
"""
import os
import time
from typing import Any, Optional, Tuple

import torch
from torch import nn


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


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    save_dir: str,
    filename: str,
    ema: Optional[Any] = None
) -> str:
    """Save full training checkpoint with model, optimizer, and scheduler state.

    Args:
        model: Model to save.
        optimizer: Optimizer to save state from.
        scheduler: Learning rate scheduler to save state from.
        epoch: Current epoch number.
        save_dir: Directory to save checkpoint in.
        filename: Name for the checkpoint file (without extension).
        ema: Optional EMA wrapper (from ema-pytorch) to save state from.

    Returns:
        Full path to the saved checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
    }
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()
    save_path = f"{save_dir}/{filename}.pt"
    torch.save(checkpoint, save_path)
    return save_path


def save_model_only(
    model: nn.Module,
    epoch: int,
    save_dir: str,
    filename: str,
    ema: Optional[Any] = None
) -> str:
    """Save model weights only (lightweight checkpoint without optimizer/scheduler).

    Args:
        model: Model to save.
        epoch: Current epoch number.
        save_dir: Directory to save checkpoint in.
        filename: Name for the checkpoint file (without extension).
        ema: Optional EMA wrapper (from ema-pytorch) to save state from.

    Returns:
        Full path to the saved checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()
    save_path = f"{save_dir}/{filename}.pt"
    torch.save(checkpoint, save_path)
    return save_path
