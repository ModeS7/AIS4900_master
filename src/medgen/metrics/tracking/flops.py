"""
FLOPs tracking and measurement utilities.

Shared utility for measuring and tracking model FLOPs during training.
"""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def measure_model_flops(
    model: nn.Module,
    sample_input: torch.Tensor,
    timesteps: torch.Tensor | None = None,
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
        torch.cuda.empty_cache()  # Clear cache before profiling

        # Ensure FP32 for FLOPs measurement - avoids BF16/FP32 dtype mismatches
        # that can occur with models using weight slicing (like DC-AE 1.5)
        sample_input = sample_input.float()
        if timesteps is not None:
            timesteps = timesteps.float()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_flops=True,
            record_shapes=False,  # Not needed for FLOPs, reduces memory overhead
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
                "FLOPs measurement returned 0 - torch.profiler does not support "
                "Conv3d/3D operations. FLOPs logging will be disabled."
            )

        model.train()
        torch.cuda.empty_cache()  # Clean up after profiling
        return total_flops

    except Exception as e:
        logger.warning(f"FLOPs measurement failed: {e}")
        model.train()
        return 0


class FLOPsTracker:
    """Track FLOPs during training.

    Measures forward pass FLOPs once (with batch_size=1 for memory efficiency),
    then computes TFLOPs per epoch and total by scaling with batch_size.
    Training step FLOPs ≈ 3x forward (forward + backward + optimizer).

    Usage:
        tracker = FLOPsTracker()
        # At start of training:
        tracker.measure(model, sample_input, steps_per_epoch, batch_size, timesteps)
        # Each epoch:
        tracker.log_epoch(writer, epoch)
    """

    TRAINING_MULTIPLIER = 3  # forward + backward + optimizer ≈ 3x forward

    def __init__(self) -> None:
        self.forward_flops: int = 0  # FLOPs for batch_size=1
        self.steps_per_epoch: int = 0
        self.batch_size: int = 1
        self.total_epochs: int = 0
        self._measured: bool = False

    def measure(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        steps_per_epoch: int,
        batch_size: int = 1,
        timesteps: torch.Tensor | None = None,
        is_main_process: bool = True,
    ) -> None:
        """Measure FLOPs and store for tracking.

        Args:
            model: Model to measure (should be raw model, not compiled/DDP).
            sample_input: Sample input tensor [1, C, H, W] (batch_size=1 for measurement).
            steps_per_epoch: Number of training steps per epoch.
            batch_size: Actual training batch size (used to scale FLOPs).
            timesteps: Optional timesteps tensor (for diffusion models).
            is_main_process: Whether to log info messages.
        """
        if self._measured:
            return

        self.forward_flops = measure_model_flops(model, sample_input, timesteps)
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self._measured = True

        if is_main_process:
            if self.forward_flops > 0:
                gflops_forward = self.forward_flops / 1e9
                tflops_epoch = self.get_tflops_epoch()
                tflops_bs1 = self.get_tflops_bs1()
                logger.info(
                    f"FLOPs measured: {gflops_forward:.2f} GFLOPs/forward (bs=1), "
                    f"{tflops_epoch:.2f} TFLOPs/epoch (bs={batch_size}), "
                    f"{tflops_bs1:.2f} TFLOPs/epoch (bs=1)"
                )
            else:
                logger.warning("FLOPs measurement returned 0 - TensorBoard FLOPs logging disabled")

    def mark_measured(self) -> None:
        """Mark FLOPs as measured (for manual/estimated FLOPs)."""
        self._measured = True

    def get_tflops_epoch(self) -> float:
        """Get TFLOPs per epoch (forward + backward + optimizer), scaled by batch_size."""
        if self.forward_flops == 0:
            return 0.0
        # Scale by batch_size since we measure with bs=1
        flops_per_step = self.forward_flops * self.batch_size * self.TRAINING_MULTIPLIER
        return (flops_per_step * self.steps_per_epoch) / 1e12

    def get_tflops_bs1(self) -> float:
        """Get TFLOPs per epoch for batch_size=1 (raw measurement, unscaled)."""
        if self.forward_flops == 0:
            return 0.0
        flops_per_step = self.forward_flops * self.TRAINING_MULTIPLIER
        return (flops_per_step * self.steps_per_epoch) / 1e12

    def get_tflops_total(self, completed_epochs: int) -> float:
        """Get total TFLOPs for all completed epochs."""
        return self.get_tflops_epoch() * completed_epochs
