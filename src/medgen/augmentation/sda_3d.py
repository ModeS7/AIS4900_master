"""Shifted Data Augmentation (SDA) for 3D diffusion models.

Reference: IEEE Access 2025 - "Regularization for Unconditional Image
Diffusion Models via Shifted Data Augmentation" by Kensuke Nakamura

Adapts SDA for 3D volumetric data [B, C, D, H, W].

Key insight: Standard data augmentation on clean images causes "leakage" where
augmented content appears in generated samples. SDA solves this with a dual-path
training approach with shifted timesteps.

Key differences from 2D:
- Rotations around all 3 axes (D, H, W)
- Flips along all 3 axes

Usage:
    sda = SDATransform3D(rotation=True, flip=True, noise_shift=0.1)

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


class SDATransform3D:
    """Shifted Data Augmentation transform for 3D diffusion models.

    Applies geometric transforms to CLEAN 3D data before noise addition,
    with shifted timesteps to prevent leakage.

    Args:
        rotation: Enable 90/180/270 degree rotations around D/H/W axes. Default: True.
        flip: Enable flips along D/H/W axes. Default: True.
        noise_shift: Amount to shift timesteps for augmented path.
            Positive values increase noise level. Default: 0.1.
        prob: Probability of using the augmented path. Default: 0.5.

    Example:
        >>> sda = SDATransform3D(rotation=True, flip=True, noise_shift=0.1)
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
            # Rotations around each axis: 90, 180, 270 degrees
            for axis in ['d', 'h', 'w']:
                for k in [1, 2, 3]:
                    self._transforms.append(('rot90_3d', {'axis': axis, 'k': k}))
        if flip:
            self._transforms.extend([
                ('flip_d', {}),
                ('flip_h', {}),
                ('flip_w', {}),
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
        """Apply transform to 3D tensor [B, C, D, H, W].

        Args:
            x: Input tensor.
            transform_type: Type of transform.
            params: Transform parameters.

        Returns:
            Transformed tensor.
        """
        if transform_type == 'identity':
            return x
        elif transform_type == 'rot90_3d':
            return self._rotate_3d(x, params['axis'], params['k'])
        elif transform_type == 'flip_d':
            return torch.flip(x, dims=[2])  # D dimension
        elif transform_type == 'flip_h':
            return torch.flip(x, dims=[3])  # H dimension
        elif transform_type == 'flip_w':
            return torch.flip(x, dims=[4])  # W dimension
        return x

    def _rotate_3d(self, x: torch.Tensor, axis: str, k: int) -> torch.Tensor:
        """Rotate 3D tensor by k*90 degrees around axis.

        Args:
            x: Input tensor [B, C, D, H, W]
            axis: Rotation axis ('d', 'h', or 'w')
            k: Number of 90-degree rotations (1, 2, or 3)

        Returns:
            Rotated tensor
        """
        # Determine which dimensions to rotate
        if axis == 'd':
            # Rotate around D axis = rotate H-W plane
            dims = (3, 4)  # H, W
        elif axis == 'h':
            # Rotate around H axis = rotate D-W plane
            dims = (2, 4)  # D, W
        elif axis == 'w':
            # Rotate around W axis = rotate D-H plane
            dims = (2, 3)  # D, H
        else:
            return x

        return torch.rot90(x, k=k, dims=dims)

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
            target: Target tensor [B, C, D, H, W] (velocity for RFlow).
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
        """Apply SDA transform to clean 3D images.

        With probability `self.prob`, applies a random geometric transform.
        Otherwise returns original images with None info.

        Args:
            images: Clean images [B, C, D, H, W] (before noise addition).

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


def create_sda_transform_3d(cfg) -> Optional[SDATransform3D]:
    """Create 3D SDA transform from config.

    Args:
        cfg: Training config with sda section.

    Returns:
        SDATransform3D if enabled, None otherwise.
    """
    sda_cfg = cfg.training.get('sda', {})

    if not sda_cfg.get('enabled', False):
        return None

    return SDATransform3D(
        rotation=sda_cfg.get('rotation', True),
        flip=sda_cfg.get('flip', True),
        noise_shift=sda_cfg.get('noise_shift', 0.1),
        prob=sda_cfg.get('prob', 0.5),
    )
