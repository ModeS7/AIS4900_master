"""Model utilities for training setup.

Provides shared functions for DDP wrapping and torch.compile setup
used across different trainers.
"""
import logging
from typing import Tuple

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def wrap_model_for_training(
    model: nn.Module,
    use_multi_gpu: bool,
    local_rank: int = 0,
    use_compile: bool = True,
    compile_mode: str = "default",
    find_unused_parameters: bool = False,
    disable_ddp_optimizer: bool = False,
    is_main_process: bool = True,
) -> Tuple[nn.Module, nn.Module]:
    """Wrap model with DDP and/or torch.compile for training.

    Args:
        model: Raw PyTorch model (already on device).
        use_multi_gpu: Whether to use DDP for multi-GPU training.
        local_rank: Local GPU rank for DDP.
        use_compile: Whether to use torch.compile.
        compile_mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune').
        find_unused_parameters: DDP find_unused_parameters flag.
        disable_ddp_optimizer: Whether to disable DDP optimizer (for large models).
        is_main_process: Whether this is the main process (for logging).

    Returns:
        Tuple of (wrapped_model, raw_model).
        - wrapped_model: Model ready for training (may be DDP + compiled).
        - raw_model: Original unwrapped model (for checkpointing).
    """
    raw_model = model

    if use_multi_gpu:
        # Disable DDP optimizer if requested (helps with compilation issues for large models)
        if disable_ddp_optimizer:
            torch._dynamo.config.optimize_ddp = False
            if is_main_process:
                logger.info("Disabled DDPOptimizer (compilation workaround)")

        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=False,
        )

        if use_compile:
            # Use default mode for DDP (reduce-overhead uses CUDA graphs which conflict with DDP)
            actual_mode = "default"
            wrapped_model = torch.compile(ddp_model, mode=actual_mode)
            if is_main_process:
                if compile_mode != actual_mode:
                    logger.warning(
                        f"compile_mode='{compile_mode}' overridden to '{actual_mode}' for DDP "
                        "(reduce-overhead mode uses CUDA graphs which conflict with DDP)"
                    )
                logger.info(f"Multi-GPU: Compiled DDP wrapper with mode='{actual_mode}'")
        else:
            wrapped_model = ddp_model
            if is_main_process:
                logger.info("Multi-GPU: DDP wrapper (no compilation)")
    else:
        if use_compile:
            wrapped_model = torch.compile(model, mode=compile_mode)
            if is_main_process:
                logger.info(f"Single-GPU: Compiled model with mode='{compile_mode}'")
        else:
            wrapped_model = model
            if is_main_process:
                logger.info("Single-GPU: Raw model (no compilation)")

    return wrapped_model, raw_model
