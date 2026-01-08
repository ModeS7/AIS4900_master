"""
Shared loss functions for diffusion and VAE training.

This module provides loss function wrappers that handle edge cases
like multi-channel inputs consistently across all trainers.
"""
import logging
import os
from typing import Dict, Optional, Union

import torch
from torch import nn, Tensor

from monai.losses import PerceptualLoss as MonaiPerceptualLoss

logger = logging.getLogger(__name__)


class PerceptualLoss(nn.Module):
    """Perceptual loss wrapper that handles multi-channel inputs.

    MONAI's RadImageNet perceptual loss expects 3-channel RGB input but
    auto-handles 1-channel by repeating to 3. This wrapper extends support
    to any number of channels by computing per-channel loss and averaging.

    This matches the approach used in DiffusionTrainer for dual mode, where
    each image modality is processed separately.

    Args:
        spatial_dims: Number of spatial dimensions (2 for 2D images).
        network_type: Backbone network type (default: radimagenet_resnet50).
        cache_dir: Directory for caching pretrained weights.
        pretrained: Whether to use pretrained weights.
        device: Target device for the loss function.
        use_compile: Whether to use torch.compile for optimization.

    Example:
        >>> loss_fn = PerceptualLoss(cache_dir="/cache", device=device)
        >>> # Works with any channel count
        >>> loss_1ch = loss_fn(recon_1ch, target_1ch)  # [B, 1, H, W]
        >>> loss_2ch = loss_fn(recon_2ch, target_2ch)  # [B, 2, H, W]
        >>> loss_3ch = loss_fn(recon_3ch, target_3ch)  # [B, 3, H, W]
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        network_type: str = "radimagenet_resnet50",
        cache_dir: Optional[str] = None,
        pretrained: bool = True,
        device: Optional[torch.device] = None,
        use_compile: bool = False,
    ) -> None:
        super().__init__()

        # Ensure cache directory exists
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Create underlying MONAI perceptual loss
        self._loss_fn = MonaiPerceptualLoss(
            spatial_dims=spatial_dims,
            network_type=network_type,
            cache_dir=cache_dir,
            pretrained=pretrained,
        )

        if device is not None:
            self._loss_fn = self._loss_fn.to(device)

        if use_compile:
            # Use "default" mode - "reduce-overhead" uses CUDA graphs which can cause
            # "tensor deallocate during graph recording" errors with dynamic shapes
            self._loss_fn = torch.compile(self._loss_fn, mode="default")

        self.network_type = network_type

    def forward(
        self,
        input: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute perceptual loss between input and target.

        Handles any number of input channels by computing per-channel
        perceptual loss and averaging (same approach as DiffusionTrainer).

        Args:
            input: Predicted/reconstructed images [B, C, H, W].
            target: Ground truth images [B, C, H, W].

        Returns:
            Scalar perceptual loss tensor.
        """
        num_channels = input.shape[1]

        if num_channels == 1:
            # 1-channel: MONAI auto-handles by repeating to 3 channels
            return self._loss_fn(input.float(), target.float())

        # Multi-channel: compute per-channel loss and average
        # This matches DiffusionTrainer's dual mode approach
        losses = []
        for ch in range(num_channels):
            ch_input = input[:, ch:ch+1].float()   # [B, 1, H, W]
            ch_target = target[:, ch:ch+1].float()  # [B, 1, H, W]
            losses.append(self._loss_fn(ch_input, ch_target))

        return sum(losses) / len(losses)

    def forward_dict(
        self,
        input: Dict[str, Tensor],
        target: Dict[str, Tensor],
    ) -> Tensor:
        """Compute perceptual loss for dict inputs (like DiffusionTrainer dual mode).

        Args:
            input: Dict of predicted images {key: [B, C, H, W]}.
            target: Dict of ground truth images {key: [B, C, H, W]}.

        Returns:
            Average perceptual loss across all keys.
        """
        losses = []
        for key, pred in input.items():
            losses.append(self(pred, target[key]))
        return sum(losses) / len(losses)

    def __call__(
        self,
        input: Union[Tensor, Dict[str, Tensor]],
        target: Union[Tensor, Dict[str, Tensor]],
    ) -> Tensor:
        """Compute perceptual loss, auto-detecting input type.

        Args:
            input: Predicted images - tensor [B, C, H, W] or dict {key: tensor}.
            target: Ground truth images - same format as input.

        Returns:
            Scalar perceptual loss tensor.
        """
        if isinstance(input, dict):
            return self.forward_dict(input, target)
        return self.forward(input, target)
