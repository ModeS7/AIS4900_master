"""
Shared loss functions for diffusion and VAE training.

This module provides loss function wrappers that handle edge cases
like multi-channel inputs consistently across all trainers.
"""
import logging
import os

import torch
import torch.nn.functional as F
from monai.losses import PerceptualLoss as MonaiPerceptualLoss
from torch import Tensor, nn

from medgen.core.spatial_utils import get_pooling_fn, get_spatial_sum_dims

# Import LPIPS library (Zhang et al. 2018)
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

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
        cache_dir: str | None = None,
        pretrained: bool = True,
        device: torch.device | None = None,
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
        input: dict[str, Tensor],
        target: dict[str, Tensor],
    ) -> Tensor:
        """Compute perceptual loss for dict inputs (like DiffusionTrainer dual mode).

        Args:
            input: Dict of predicted images {key: [B, C, H, W]}.
            target: Dict of ground truth images {key: [B, C, H, W]}.

        Returns:
            Average perceptual loss across all keys.

        Raises:
            KeyError: If input and target dicts have mismatched keys.
        """
        if not input:
            # Empty dict: return zero loss on appropriate device
            device = next(iter(target.values())).device if target else torch.device('cpu')
            return torch.tensor(0.0, device=device)

        # Validate keys match
        if set(input.keys()) != set(target.keys()):
            raise KeyError(
                f"PerceptualLoss.forward_dict: input and target keys mismatch. "
                f"Input: {sorted(input.keys())}, Target: {sorted(target.keys())}"
            )

        losses = []
        for key, pred in input.items():
            losses.append(self(pred, target[key]))
        return sum(losses) / len(losses)

    def __call__(
        self,
        input: Tensor | dict[str, Tensor],
        target: Tensor | dict[str, Tensor],
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


class LPIPSLoss(nn.Module):
    """LPIPS (Learned Perceptual Image Patch Similarity) loss wrapper.

    Uses the actual LPIPS library (Zhang et al. 2018) with learned linear weights
    calibrated to human perceptual judgments. This differs from raw perceptual
    loss which uses unweighted deep features.

    The DC-AE paper uses LPIPS loss for training (not raw perceptual loss).

    Reference: https://arxiv.org/abs/1801.03924

    Args:
        net: Backbone network - 'alex' (fastest), 'vgg', or 'squeeze'.
             DC-AE paper uses VGG for training.
        device: Target device for the loss function.
        use_compile: Whether to use torch.compile for optimization.

    Example:
        >>> loss_fn = LPIPSLoss(net='vgg', device=device)
        >>> loss = loss_fn(recon, target)  # [B, C, H, W]
    """

    def __init__(
        self,
        net: str = "vgg",
        device: torch.device | None = None,
        use_compile: bool = False,
    ) -> None:
        super().__init__()

        if not LPIPS_AVAILABLE:
            raise ImportError(
                "LPIPS library not installed. Install with: pip install lpips"
            )

        # Create LPIPS model
        # spatial=False means return scalar loss (not per-pixel map)
        self._loss_fn = lpips.LPIPS(net=net, spatial=False, verbose=False)

        if device is not None:
            self._loss_fn = self._loss_fn.to(device)

        # Set to eval mode - LPIPS uses pretrained weights
        self._loss_fn.eval()

        if use_compile:
            self._loss_fn = torch.compile(self._loss_fn, mode="default")

        self.net = net
        logger.info(f"LPIPSLoss initialized with {net} backbone")

    def forward(
        self,
        input: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute LPIPS loss between input and target.

        Handles any number of input channels by computing per-channel
        LPIPS loss and averaging.

        Args:
            input: Predicted/reconstructed images [B, C, H, W].
            target: Ground truth images [B, C, H, W].

        Returns:
            Scalar LPIPS loss tensor.
        """
        num_channels = input.shape[1]

        if num_channels == 1:
            # 1-channel: repeat to 3 channels for LPIPS
            input_3ch = input.repeat(1, 3, 1, 1)
            target_3ch = target.repeat(1, 3, 1, 1)
            return self._loss_fn(input_3ch.float(), target_3ch.float()).mean()

        if num_channels == 3:
            # 3-channel: use directly
            return self._loss_fn(input.float(), target.float()).mean()

        # Multi-channel (not 1 or 3): compute per-channel loss and average
        losses = []
        for ch in range(num_channels):
            ch_input = input[:, ch:ch+1].repeat(1, 3, 1, 1).float()   # [B, 3, H, W]
            ch_target = target[:, ch:ch+1].repeat(1, 3, 1, 1).float()  # [B, 3, H, W]
            losses.append(self._loss_fn(ch_input, ch_target).mean())

        return sum(losses) / len(losses)

    def forward_dict(
        self,
        input: dict[str, Tensor],
        target: dict[str, Tensor],
    ) -> Tensor:
        """Compute LPIPS loss for dict inputs.

        Args:
            input: Dict of predicted images {key: [B, C, H, W]}.
            target: Dict of ground truth images {key: [B, C, H, W]}.

        Returns:
            Average LPIPS loss across all keys.

        Raises:
            KeyError: If input and target dicts have mismatched keys.
        """
        if not input:
            # Empty dict: return zero loss on appropriate device
            device = next(iter(target.values())).device if target else torch.device('cpu')
            return torch.tensor(0.0, device=device)

        # Validate keys match
        if set(input.keys()) != set(target.keys()):
            raise KeyError(
                f"LPIPSLoss.forward_dict: input and target keys mismatch. "
                f"Input: {sorted(input.keys())}, Target: {sorted(target.keys())}"
            )

        losses = []
        for key, pred in input.items():
            losses.append(self.forward(pred, target[key]))
        return sum(losses) / len(losses)

    def __call__(
        self,
        input: Tensor | dict[str, Tensor],
        target: Tensor | dict[str, Tensor],
    ) -> Tensor:
        """Compute LPIPS loss, auto-detecting input type.

        Args:
            input: Predicted images - tensor [B, C, H, W] or dict {key: tensor}.
            target: Ground truth images - same format as input.

        Returns:
            Scalar LPIPS loss tensor.
        """
        if isinstance(input, dict):
            return self.forward_dict(input, target)
        return self.forward(input, target)


class SegmentationLoss(nn.Module):
    """Combined segmentation loss: BCE + Dice + Boundary.

    For segmentation mask compression training. All losses are computed
    on logits (BCE) or sigmoid(logits) (Dice, Boundary).

    Supports both 2D and 3D inputs via spatial_dims parameter.

    Args:
        bce_weight: Weight for BCE loss (default 1.0).
        dice_weight: Weight for Dice loss (default 1.0).
        boundary_weight: Weight for Boundary loss (default 0.5).
        smooth: Smoothing factor for Dice to avoid division by zero.
        spatial_dims: Number of spatial dimensions (2 or 3).

    Example:
        >>> loss_fn = SegmentationLoss(bce_weight=1.0, dice_weight=1.0)
        >>> total_loss, breakdown = loss_fn(logits, target_mask)
        >>> print(breakdown)  # {'bce': 0.5, 'dice': 0.2, 'boundary': 0.1}
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_weight: float = 0.5,
        smooth: float = 1.0,
        spatial_dims: int = 2,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.smooth = smooth
        self.spatial_dims = spatial_dims

        # Set pooling function and sum dimensions based on spatial_dims
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        self._max_pool = get_pooling_fn(spatial_dims, pool_type='max')
        self._spatial_sum_dims = get_spatial_sum_dims(spatial_dims)

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute combined segmentation loss.

        Args:
            logits: Model output (before sigmoid) [B, 1, H, W].
            target: Binary target mask [B, 1, H, W].

        Returns:
            Tuple of (total_loss, loss_breakdown_dict).
        """
        # BCE loss (numerically stable, applies sigmoid internally)
        bce = F.binary_cross_entropy_with_logits(logits, target)

        # Dice loss (on probabilities)
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=self._spatial_sum_dims)
        union = probs.sum(dim=self._spatial_sum_dims) + target.sum(dim=self._spatial_sum_dims)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        # Boundary loss (weighted BCE on boundary pixels)
        boundary_loss = self._compute_boundary_loss(logits, target)

        # Combined loss
        total = (
            self.bce_weight * bce
            + self.dice_weight * dice_loss
            + self.boundary_weight * boundary_loss
        )

        # Return detached tensors to avoid CUDA sync during training loop
        # Callers should call .item() only when actually logging metrics
        breakdown = {
            'bce': bce.detach(),
            'dice': dice_loss.detach(),
            'boundary': boundary_loss.detach(),
        }

        return total, breakdown

    def _compute_boundary_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute boundary-weighted BCE loss.

        Extracts boundary pixels using morphological gradient (dilation - erosion)
        and applies higher weight to boundary regions. Important for small tumors
        with high boundary-to-area ratio.

        Args:
            logits: Model output (before sigmoid) [B, 1, H, W] or [B, 1, D, H, W].
            target: Binary target mask [B, 1, H, W] or [B, 1, D, H, W].

        Returns:
            Boundary-weighted BCE loss tensor.
        """
        # Extract boundaries using max_pool - min_pool (morphological gradient)
        kernel_size = 3
        padding = kernel_size // 2

        dilated = self._max_pool(target, kernel_size, stride=1, padding=padding)
        eroded = -self._max_pool(-target, kernel_size, stride=1, padding=padding)
        boundary = dilated - eroded  # Boundary mask

        # Compute BCE only on boundary pixels
        if boundary.sum() > 0:
            boundary_bce = F.binary_cross_entropy_with_logits(
                logits, target, weight=boundary, reduction='sum'
            ) / (boundary.sum() + 1e-6)
        else:
            boundary_bce = torch.tensor(0.0, device=logits.device)

        return boundary_bce
