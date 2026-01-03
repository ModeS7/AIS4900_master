"""
FLOPs tracking and measurement utilities.

Shared utility for measuring and tracking model FLOPs during training.
"""

import logging
from typing import Optional

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


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

        if total_flops == 0:
            logger.warning(
                "FLOPs measurement returned 0 - torch.profiler may not support "
                "this model type (e.g., compiled models, custom CUDA kernels)"
            )

        model.train()
        return total_flops

    except Exception as e:
        logger.warning(f"FLOPs measurement failed: {e}")
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
