"""
Utility functions for diffusion model training.

This module provides utility functions for logging, VRAM monitoring,
and checkpoint saving/loading. Checkpoint functions are shared between
DiffusionTrainer and VAETrainer.
"""
import logging
import os
import time
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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


def measure_model_flops(
    model: nn.Module,
    sample_input: torch.Tensor,
    timesteps: Optional[torch.Tensor] = None,
) -> int:
    """Measure model FLOPs using torch.profiler.

    Args:
        model: Model to measure.
        sample_input: Sample input tensor.
        timesteps: Optional timesteps tensor (for diffusion models).

    Returns:
        FLOPs count per forward pass, or 0 if measurement fails.
    """
    try:
        model.eval()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_flops=True,
            record_shapes=True,
        ) as prof:
            with torch.no_grad():
                if timesteps is not None:
                    _ = model(sample_input, timesteps)
                else:
                    _ = model(sample_input)

        # Sum up all FLOPs from the profiler events
        total_flops = sum(
            event.flops
            for event in prof.key_averages()
            if event.flops is not None and event.flops > 0
        )
        model.train()
        return total_flops

    except Exception as e:
        logger.debug(f"FLOPs measurement failed: {e}")
        model.train()
        return 0


class FLOPsTracker:
    """Track FLOPs during training.

    Measures forward pass FLOPs once, then computes TFLOPs per epoch and total.
    Training step FLOPs ≈ 3x forward (forward + backward + optimizer).

    Usage:
        tracker = FLOPsTracker()
        # At start of training:
        tracker.measure(model, sample_input, timesteps, steps_per_epoch)
        # Each epoch:
        tracker.log_epoch(writer, epoch)
    """

    TRAINING_MULTIPLIER = 3  # forward + backward + optimizer ≈ 3x forward

    def __init__(self) -> None:
        self.forward_flops: int = 0
        self.steps_per_epoch: int = 0
        self.total_epochs: int = 0
        self._measured: bool = False

    def measure(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        steps_per_epoch: int,
        timesteps: Optional[torch.Tensor] = None,
        is_main_process: bool = True,
    ) -> None:
        """Measure FLOPs and store for tracking.

        Args:
            model: Model to measure (should be raw model, not compiled/DDP).
            sample_input: Sample input tensor [B, C, H, W].
            steps_per_epoch: Number of training steps per epoch.
            timesteps: Optional timesteps tensor (for diffusion models).
            is_main_process: Whether to log info messages.
        """
        if self._measured:
            return

        self.forward_flops = measure_model_flops(model, sample_input, timesteps)
        self.steps_per_epoch = steps_per_epoch
        self._measured = True

        if is_main_process:
            if self.forward_flops > 0:
                gflops_forward = self.forward_flops / 1e9
                tflops_epoch = self.get_tflops_epoch()
                logger.info(
                    f"FLOPs measured: {gflops_forward:.2f} GFLOPs/forward, "
                    f"{tflops_epoch:.2f} TFLOPs/epoch"
                )
            else:
                logger.warning("FLOPs measurement returned 0 - TensorBoard FLOPs logging disabled")

    def get_tflops_epoch(self) -> float:
        """Get TFLOPs per epoch (forward + backward + optimizer)."""
        if self.forward_flops == 0:
            return 0.0
        flops_per_step = self.forward_flops * self.TRAINING_MULTIPLIER
        return (flops_per_step * self.steps_per_epoch) / 1e12

    def get_tflops_total(self, completed_epochs: int) -> float:
        """Get total TFLOPs for all completed epochs."""
        return self.get_tflops_epoch() * completed_epochs

    def log_epoch(
        self,
        writer: Optional[SummaryWriter],
        epoch: int,
    ) -> None:
        """Log TFLOPs metrics to TensorBoard.

        Args:
            writer: TensorBoard SummaryWriter.
            epoch: Current epoch number (0-indexed, will log as epoch+1 completed).
        """
        if writer is None or not self._measured or self.forward_flops == 0:
            return

        completed_epochs = epoch + 1
        writer.add_scalar('FLOPs/TFLOPs_epoch', self.get_tflops_epoch(), epoch)
        writer.add_scalar('FLOPs/TFLOPs_total', self.get_tflops_total(completed_epochs), epoch)
