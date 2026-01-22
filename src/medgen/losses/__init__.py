"""
Shared loss functions for medical image generation.

This package provides:
- PerceptualLoss: RadImageNet perceptual loss with multi-channel support
- LPIPSLoss: LPIPS perceptual similarity loss (Zhang et al. 2018)
- SegmentationLoss: Combined BCE + Dice + Boundary loss for segmentation
- RegionalWeightComputer: Per-pixel loss weighting by tumor size (2D)
- RegionalWeightComputer3D: Per-pixel loss weighting by tumor size (3D)

Usage:
    from medgen.losses import PerceptualLoss, LPIPSLoss, SegmentationLoss

    # Perceptual loss for reconstruction
    perc_loss = PerceptualLoss(cache_dir="/cache", device=device)
    loss = perc_loss(recon, target)

    # LPIPS for perceptual similarity (DC-AE style)
    lpips_loss = LPIPSLoss(net='vgg', device=device)
    loss = lpips_loss(recon, target)

    # Segmentation loss for mask compression
    seg_loss = SegmentationLoss(bce_weight=1.0, dice_weight=1.0)
    loss, breakdown = seg_loss(logits, target_mask)

    # Regional weighting for diffusion training
    from medgen.losses import RegionalWeightComputer
    weight_computer = RegionalWeightComputer(image_size=256)
    weight_map = weight_computer(seg_mask)  # [B, 1, H, W]
"""

from .losses import PerceptualLoss, LPIPSLoss, SegmentationLoss, LPIPS_AVAILABLE
from .regional_weighting import (
    RegionalWeightComputer,
    create_regional_weight_computer,
    RegionalWeightComputer3D,  # Backwards compatibility alias
)

__all__ = [
    'PerceptualLoss',
    'LPIPSLoss',
    'SegmentationLoss',
    'LPIPS_AVAILABLE',
    # Regional weighting
    'RegionalWeightComputer',
    'create_regional_weight_computer',
    'RegionalWeightComputer3D',
]
