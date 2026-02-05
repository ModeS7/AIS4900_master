"""Gradient checkpointing utilities for 3D models.

Provides base class for gradient-checkpointed model wrappers
for memory-efficient training of 3D compression models.
"""

from typing import Any, Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint


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
