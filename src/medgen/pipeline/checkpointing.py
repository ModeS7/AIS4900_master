"""Gradient checkpointing utilities for 3D models.

Provides base class for gradient-checkpointed model wrappers
for memory-efficient training of 3D compression models,
and block-level checkpointing for MONAI DiffusionModelUNet.
"""

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

logger = logging.getLogger(__name__)


class BaseCheckpointedModel(nn.Module):
    """Base class for gradient-checkpointed 3D model wrappers.

    Reduces activation memory by ~50% for 3D volumes by using
    torch.utils.checkpoint to trade compute for memory.

    Subclasses must implement forward() with checkpointed operations.

    Args:
        model: The underlying MONAI model to wrap.

    Example:
        >>> class CheckpointedMyModel(BaseCheckpointedModel):
        ...     def forward(self, x):
        ...         def encode_fn(x):
        ...             return self.model.encode(x)
        ...         z = self.checkpoint(encode_fn, x)
        ...         return z
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def checkpoint(self, fn: Callable[..., Any], *args) -> Any:
        """Apply gradient checkpointing to a function.

        Args:
            fn: Function to checkpoint (usually a model stage).
            *args: Arguments to pass to fn.

        Returns:
            Result of fn(*args) with gradient checkpointing applied.
        """
        return grad_checkpoint(fn, *args, use_reentrant=False)

    def encode(self, x: torch.Tensor) -> Any:
        """Encode input to latent space (inference path, no checkpointing).

        Args:
            x: Input tensor [B, C, D, H, W].

        Returns:
            Encoded representation (type depends on model).
        """
        return self.model.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to output (inference path, no checkpointing).

        Args:
            z: Latent tensor.

        Returns:
            Reconstructed output [B, C, D, H, W].
        """
        return self.model.decode(z)


def enable_unet_gradient_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing on a MONAI DiffusionModelUNet.

    Monkey-patches each down_block, middle_block, and up_block to use
    torch.utils.checkpoint, trading ~30% extra compute for ~50% activation
    memory savings. Required for 3D volumetric training to fit in GPU memory.

    MONAI's DiffusionModelUNet has no native checkpointing support,
    so we patch block-level forward methods directly.

    Args:
        model: A MONAI DiffusionModelUNet instance.
    """
    if not hasattr(model, 'down_blocks'):
        raise TypeError(
            f"Expected MONAI DiffusionModelUNet with down_blocks/up_blocks, "
            f"got {type(model).__name__}"
        )

    def _make_checkpointed_forward(original_forward: Callable) -> Callable:
        """Create a checkpointed version of a block's forward method."""
        def checkpointed_forward(*args, **kwargs):
            # use_reentrant=False handles non-tensor args (like lists) correctly
            # and is the recommended mode for PyTorch >= 2.0
            return grad_checkpoint(original_forward, *args, use_reentrant=False, **kwargs)
        return checkpointed_forward

    count = 0
    for block in model.down_blocks:
        block.forward = _make_checkpointed_forward(block.forward)
        count += 1

    model.middle_block.forward = _make_checkpointed_forward(model.middle_block.forward)
    count += 1

    for block in model.up_blocks:
        block.forward = _make_checkpointed_forward(block.forward)
        count += 1

    logger.info(f"UNet gradient checkpointing enabled ({count} blocks patched)")
