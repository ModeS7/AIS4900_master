"""Perceptual loss management for compression training.

This module provides:
- PerceptualLossManager: Manages perceptual loss computation for 2D and 3D training

Supports both LPIPS (VGG backbone) and RadImageNet (ResNet50) perceptual losses,
with 2.5D support for 3D volumes (sampling slices for 2D loss computation).
"""
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class PerceptualLossManager:
    """Manages perceptual loss computation for 2D and 3D compression training.

    Handles:
    - Creation of LPIPS or RadImageNet perceptual loss networks
    - 2D loss computation for images
    - 2.5D loss computation for 3D volumes (sampling slices)
    - Optional torch.compile optimization
    """

    def __init__(
        self,
        spatial_dims: int,
        weight: float,
        loss_type: str,
        device: torch.device,
        cache_dir: str | None = None,
        use_compile: bool = False,
        use_2_5d: bool = True,
        slice_fraction: float = 0.25,
    ) -> None:
        """Initialize perceptual loss manager.

        Args:
            spatial_dims: Spatial dimensions (2 for 2D, 3 for 3D).
            weight: Weight for perceptual loss in total loss.
            loss_type: Loss type ('lpips' or 'radimagenet').
            device: Device to place the loss network on.
            cache_dir: Cache directory for pretrained weights.
            use_compile: Whether to apply torch.compile to loss network.
            use_2_5d: Whether to use 2.5D loss for 3D volumes.
            slice_fraction: Fraction of slices to sample for 2.5D (0.0-1.0).
        """
        self.spatial_dims = spatial_dims
        self.weight = weight
        self.loss_type = loss_type.lower()
        self.device = device
        self.cache_dir = cache_dir
        self.use_compile = use_compile
        self.use_2_5d = use_2_5d
        self.slice_fraction = slice_fraction
        self._loss_fn: nn.Module | None = None

    @property
    def is_enabled(self) -> bool:
        """Check if perceptual loss is enabled (weight > 0)."""
        return self.weight > 0

    @property
    def loss_fn(self) -> nn.Module | None:
        """Get the perceptual loss function."""
        return self._loss_fn

    def create(self) -> nn.Module | None:
        """Initialize perceptual loss network.

        Creates either LPIPS (VGG backbone) or RadImageNet (ResNet50) loss.
        LPIPS only supports 2D; falls back to RadImageNet for 3D.

        Returns:
            Perceptual loss module, or None if weight <= 0.
        """
        if not self.is_enabled:
            logger.info("Perceptual loss disabled (weight=0)")
            return None

        loss_type = self.loss_type

        if loss_type == 'lpips':
            if self.spatial_dims != 2:
                logger.warning(
                    f"LPIPS only supports 2D images. Got spatial_dims={self.spatial_dims}. "
                    "Falling back to RadImageNet perceptual loss."
                )
                loss_type = 'radimagenet'
            else:
                logger.info("Using LPIPS loss with VGG backbone (DC-AE paper setting)")
                from medgen.losses import LPIPSLoss
                self._loss_fn = LPIPSLoss(
                    net='vgg',
                    device=self.device,
                    use_compile=self.use_compile,
                )
                return self._loss_fn

        # Default: RadImageNet perceptual loss
        logger.info("Using RadImageNet ResNet50 perceptual loss")
        from medgen.losses import PerceptualLoss
        self._loss_fn = PerceptualLoss(
            spatial_dims=self.spatial_dims,
            network_type="radimagenet_resnet50",
            cache_dir=self.cache_dir,
            pretrained=True,
            device=self.device,
            use_compile=self.use_compile,
        )
        return self._loss_fn

    def compute(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss (handles 2D and 3D).

        For 3D volumes with use_2_5d=True, computes loss on sampled 2D slices.
        Otherwise computes standard perceptual loss.

        Args:
            reconstruction: Generated images/volumes.
            target: Target images/volumes.

        Returns:
            Perceptual loss value.
        """
        if self._loss_fn is None:
            return torch.tensor(0.0, device=self.device)

        # 3D with 2.5D perceptual loss
        if self.spatial_dims == 3 and self.use_2_5d:
            return self._compute_2_5d(reconstruction, target)

        return self._loss_fn(reconstruction, target)

    def _compute_2_5d(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss on sampled 2D slices from 3D volumes.

        Samples a fraction of slices along the depth dimension and computes
        2D perceptual loss on each, averaging the results.

        Args:
            reconstruction: Reconstructed volume [B, C, D, H, W].
            target: Target volume [B, C, D, H, W].

        Returns:
            Perceptual loss averaged over sampled slices.
        """
        if self._loss_fn is None:
            return torch.tensor(0.0, device=self.device)

        depth = reconstruction.shape[2]
        n_slices = max(1, int(depth * self.slice_fraction))

        # Sample slice indices
        indices = torch.randperm(depth)[:n_slices]

        total_loss = torch.tensor(0.0, device=reconstruction.device, dtype=reconstruction.dtype)
        for idx in indices:
            recon_slice = reconstruction[:, :, idx, :, :]
            target_slice = target[:, :, idx, :, :]
            total_loss = total_loss + self._loss_fn(recon_slice, target_slice)

        return total_loss / n_slices

    @staticmethod
    def create_lpips_fn(spatial_dims: int):
        """Create LPIPS function appropriate for spatial dimensions.

        Args:
            spatial_dims: Spatial dimensions (2 or 3).

        Returns:
            compute_lpips for 2D, compute_lpips_3d for 3D.
        """
        from medgen.metrics.dispatch import create_lpips_fn
        return create_lpips_fn(spatial_dims)
