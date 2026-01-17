"""
Shared loss functions for medical image generation.

This package provides:
- PerceptualLoss: RadImageNet perceptual loss with multi-channel support
- LPIPSLoss: LPIPS perceptual similarity loss (Zhang et al. 2018)
- SegmentationLoss: Combined BCE + Dice + Boundary loss for segmentation

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
"""

from .losses import PerceptualLoss, LPIPSLoss, SegmentationLoss, LPIPS_AVAILABLE

__all__ = [
    'PerceptualLoss',
    'LPIPSLoss',
    'SegmentationLoss',
    'LPIPS_AVAILABLE',
]
