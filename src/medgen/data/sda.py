"""Shifted Data Augmentation (SDA) for diffusion models.

Reference: IEEE Access 2025 - "Regularization for Unconditional Image
Diffusion Models via Shifted Data Augmentation" by Kensuke Nakamura

Key insight: Standard data augmentation on clean images causes "leakage" where
augmented content appears in generated samples. SDA solves this with a dual-path
training approach:

1. Standard path: x -> add_noise(t) -> denoise -> x
2. Augmented path: T(x) -> add_noise(t + delta) -> denoise -> T(x)

The SHIFTED noise level (t + delta) ensures augmented samples have different
SNR characteristics, preventing the model from simply learning augmented outputs.

Key difference from ScoreAug:
- ScoreAug: Transforms NOISY data (after noise addition), requires omega conditioning
- SDA: Transforms CLEAN data (before noise addition), uses noise shift instead

Usage:
    sda = SDATransform(rotation=True, flip=True, noise_shift=0.1)

    # In training loop:
    aug_images, transform_info = sda(images)
    if transform_info is not None:
        shifted_t = sda.shift_timesteps(timesteps)
        aug_target = sda.apply_to_target(target, transform_info)
        loss_aug = compute_loss(aug_images, shifted_t, aug_target)
        loss = loss_orig + sda_weight * loss_aug
"""

import random
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn


class SDATransform:
    """Shifted Data Augmentation transform for diffusion models.

    Applies geometric transforms to CLEAN data before noise addition,
    with shifted timesteps to prevent leakage.

    Args:
        rotation: Enable 90/180/270 degree rotations. Default: True.
        flip: Enable horizontal/vertical flips. Default: True.
        noise_shift: Amount to shift timesteps for augmented path.
            Positive values increase noise level. Default: 0.1.
        prob: Probability of using the augmented path. Default: 0.5.

    Example:
        >>> sda = SDATransform(rotation=True, flip=True, noise_shift=0.1)
        >>> aug_images, info = sda(images)
        >>> if info is not None:
        ...     shifted_t = sda.shift_timesteps(timesteps)
        ...     aug_target = sda.apply_to_target(target, info)
    """

    def __init__(
        self,
        rotation: bool = True,
        flip: bool = True,
        noise_shift: float = 0.1,
        prob: float = 0.5,
    ):
        self.rotation = rotation
        self.flip = flip
        self.noise_shift = noise_shift
        self.prob = prob

        # Build list of available transforms
        self._transforms = []
        if rotation:
            self._transforms.extend([
                ('rot90', {'k': 1}),   # 90 degrees
                ('rot90', {'k': 2}),   # 180 degrees
                ('rot90', {'k': 3}),   # 270 degrees
            ])
        if flip:
            self._transforms.extend([
                ('hflip', {}),
                ('vflip', {}),
            ])

    def _sample_transform(self) -> Tuple[str, Dict[str, Any]]:
        """Sample a random transform."""
        if not self._transforms:
            return 'identity', {}
        return random.choice(self._transforms)

    def _apply(
        self,
        x: torch.Tensor,
        transform_type: str,
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """Apply transform to tensor [B, C, H, W].

        Args:
            x: Input tensor.
            transform_type: Type of transform ('rot90', 'hflip', 'vflip', 'identity').
            params: Transform parameters.

        Returns:
            Transformed tensor.
        """
        if transform_type == 'identity':
            return x
        elif transform_type == 'rot90':
            return torch.rot90(x, k=params['k'], dims=(-2, -1))
        elif transform_type == 'hflip':
            return torch.flip(x, dims=[-1])
        elif transform_type == 'vflip':
            return torch.flip(x, dims=[-2])
        return x

    def shift_timesteps(
        self,
        t: torch.Tensor,
        direction: str = 'forward',
    ) -> torch.Tensor:
        """Shift timesteps for the augmented path.

        The shifted timesteps give augmented samples a different SNR,
        which prevents the model from treating them as regular training samples.

        Args:
            t: Original timesteps [B] in [0, 1] range.
            direction: 'forward' increases noise (t + delta), 'backward' decreases.

        Returns:
            Shifted timesteps clamped to [0, 1].
        """
        if direction == 'forward':
            return (t + self.noise_shift).clamp(0.0, 1.0)
        else:
            return (t - self.noise_shift).clamp(0.0, 1.0)

    def apply_to_target(
        self,
        target: torch.Tensor,
        transform_info: Dict[str, Any],
    ) -> torch.Tensor:
        """Apply the same transform to the target (velocity/noise).

        Args:
            target: Target tensor [B, C, H, W] (velocity for RFlow).
            transform_info: Transform info from __call__.

        Returns:
            Transformed target.
        """
        return self._apply(
            target,
            transform_info['type'],
            transform_info['params'],
        )

    def __call__(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Apply SDA transform to clean images.

        With probability `self.prob`, applies a random geometric transform.
        Otherwise returns original images with None info.

        Args:
            images: Clean images [B, C, H, W] (before noise addition).

        Returns:
            Tuple of:
                - Transformed images (or original if not augmenting).
                - Transform info dict (or None if not augmenting).
                  Contains 'type' and 'params' keys.
        """
        # Decide whether to use augmented path
        if random.random() >= self.prob:
            return images, None

        # Sample and apply transform
        transform_type, params = self._sample_transform()

        if transform_type == 'identity':
            return images, None

        transformed = self._apply(images, transform_type, params)

        transform_info = {
            'type': transform_type,
            'params': params,
        }

        return transformed, transform_info

    def extra_repr(self) -> str:
        return (
            f"rotation={self.rotation}, flip={self.flip}, "
            f"noise_shift={self.noise_shift}, prob={self.prob}"
        )


def create_sda_transform(cfg) -> Optional[SDATransform]:
    """Create SDA transform from config.

    Args:
        cfg: Training config with sda section.

    Returns:
        SDATransform if enabled, None otherwise.
    """
    sda_cfg = cfg.training.get('sda', {})

    if not sda_cfg.get('enabled', False):
        return None

    return SDATransform(
        rotation=sda_cfg.get('rotation', True),
        flip=sda_cfg.get('flip', True),
        noise_shift=sda_cfg.get('noise_shift', 0.1),
        prob=sda_cfg.get('prob', 0.5),
    )
