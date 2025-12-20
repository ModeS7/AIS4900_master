"""Data augmentation for medical image training.

Separate augmentation pipelines for diffusion and VAE training:
- Diffusion: Conservative (flip, rotate, translate) - preserves image quality
- VAE: Aggressive (+ noise, blur, scale, mixup, cutmix) - learns robust features

Batch-level augmentations (mixup, cutmix) are applied via collate function.
"""
import random
from typing import Callable, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch


# =============================================================================
# Diffusion Augmentation (Conservative)
# =============================================================================

def build_diffusion_augmentation(enabled: bool = True) -> Optional[A.Compose]:
    """Build conservative augmentation for diffusion training.

    Only spatial transforms that preserve image quality. Any distortion
    would teach the model to generate distorted images.

    Transforms:
        - HorizontalFlip (p=0.5)
        - Rotate ±10° (p=0.5)
        - Translate ±5% (p=0.5)

    Args:
        enabled: Whether to enable augmentation.

    Returns:
        Albumentations Compose object, or None if disabled.
    """
    if not enabled:
        return None

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=0, p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=1.0,
            rotate=0,
            border_mode=0,
            p=0.5,
        ),
    ])


# =============================================================================
# VAE Augmentation (Aggressive)
# =============================================================================

def build_vae_augmentation(enabled: bool = True) -> Optional[A.Compose]:
    """Build aggressive augmentation for VAE training.

    More variety helps learn robust latent representations.
    Includes spatial transforms + intensity/noise augmentations.

    Transforms:
        - HorizontalFlip (p=0.5)
        - Rotate ±15° (p=0.5)
        - Translate ±10%, Scale 0.9-1.1x (p=0.5)
        - GaussianNoise (p=0.3)
        - GaussianBlur (p=0.2)
        - RandomBrightnessContrast (p=0.3)
        - ElasticTransform (p=0.2)

    Args:
        enabled: Whether to enable augmentation.

    Returns:
        Albumentations Compose object, or None if disabled.
    """
    if not enabled:
        return None

    return A.Compose([
        # Spatial transforms
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=0, p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=0,
            border_mode=0,
            p=0.5,
        ),

        # Intensity augmentations (images normalized to 0-1)
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3,
        ),

        # Elastic deformation (subtle for medical)
        A.ElasticTransform(
            alpha=50,
            sigma=10,
            border_mode=0,
            p=0.2,
        ),
    ])


# =============================================================================
# Batch-level Augmentations (Mixup, CutMix)
# =============================================================================

def mixup(
    images: np.ndarray,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply mixup augmentation to a batch of images.

    Blends pairs of images: mixed = lambda * img1 + (1-lambda) * img2
    Lambda sampled from Beta(alpha, alpha).

    Args:
        images: Batch of images [B, C, H, W].
        alpha: Beta distribution parameter (higher = more mixing).

    Returns:
        Tuple of (mixed_images, shuffled_indices, lambda_value).
    """
    batch_size = images.shape[0]
    lam = np.random.beta(alpha, alpha)

    # Random permutation for pairing
    indices = np.random.permutation(batch_size)

    # Mix images
    mixed = lam * images + (1 - lam) * images[indices]

    return mixed.astype(np.float32), indices, lam


def cutmix(
    images: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply CutMix augmentation to a batch of images.

    Cuts a rectangular region from one image and pastes onto another.
    Lambda proportional to the area ratio.

    Args:
        images: Batch of images [B, C, H, W].
        alpha: Beta distribution parameter for lambda.

    Returns:
        Tuple of (mixed_images, shuffled_indices, lambda_value).
    """
    batch_size, c, h, w = images.shape
    lam = np.random.beta(alpha, alpha)

    # Random permutation for pairing
    indices = np.random.permutation(batch_size)

    # Get cut region size (proportional to 1-lambda)
    cut_ratio = np.sqrt(1 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    # Random center for cut
    cy = np.random.randint(h)
    cx = np.random.randint(w)

    # Bounding box (clipped to image bounds)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)

    # Apply cutmix
    mixed = images.copy()
    mixed[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]

    # Adjust lambda based on actual cut area
    lam = 1 - ((y2 - y1) * (x2 - x1)) / (h * w)

    return mixed.astype(np.float32), indices, lam


def create_vae_collate_fn(
    mixup_prob: float = 0.2,
    cutmix_prob: float = 0.2,
) -> Callable[[List[torch.Tensor]], torch.Tensor]:
    """Create collate function with mixup/cutmix for VAE training.

    Args:
        mixup_prob: Probability of applying mixup (default 0.2).
        cutmix_prob: Probability of applying cutmix (default 0.2).

    Returns:
        Collate function for DataLoader.
    """
    def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
        images = torch.stack(batch)

        r = random.random()
        if r < mixup_prob:
            mixed, _, _ = mixup(images.numpy(), alpha=0.4)
            images = torch.from_numpy(mixed).float()
        elif r < mixup_prob + cutmix_prob:
            mixed, _, _ = cutmix(images.numpy(), alpha=1.0)
            images = torch.from_numpy(mixed).float()

        return images

    return collate_fn


# =============================================================================
# Apply Augmentation (shared logic)
# =============================================================================

def apply_augmentation(
    slice_data: np.ndarray,
    aug: Optional[A.Compose],
    has_mask: bool = False,
    mask_channel: int = -1,
) -> np.ndarray:
    """Apply augmentation to a 2D slice with optional mask channel.

    Handles the critical requirement that spatial augmentations must be
    identical for image and mask channels.

    Args:
        slice_data: [C, H, W] numpy array.
        aug: Albumentations Compose object (None = no augmentation).
        has_mask: Whether one channel is a segmentation mask.
        mask_channel: Index of mask channel (-1 = last).

    Returns:
        Augmented [C, H, W] numpy array.
    """
    if aug is None:
        return slice_data

    # Ensure input is contiguous and make a copy to avoid memory issues
    slice_data = np.ascontiguousarray(slice_data)

    # Transpose: [C, H, W] -> [H, W, C] for albumentations
    slice_hwc = np.transpose(slice_data, (1, 2, 0)).copy()

    if has_mask:
        # Resolve mask index
        num_channels = slice_hwc.shape[-1]
        mask_idx = mask_channel if mask_channel >= 0 else num_channels + mask_channel

        # Extract mask and image channels separately
        mask = slice_hwc[..., mask_idx].copy()

        # Get image channels (all except mask)
        if num_channels == 2:
            # Only 1 image channel + 1 mask
            img_idx = 1 - mask_idx  # 0 if mask is 1, 1 if mask is 0
            image = slice_hwc[..., img_idx].copy()
        else:
            # Multiple image channels
            indices = [i for i in range(num_channels) if i != mask_idx]
            image = slice_hwc[..., indices].copy()

        # Apply augmentation to both image and mask together
        augmented = aug(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']

        # Rebuild output array
        if num_channels == 2:
            # Simple 2-channel case
            result = np.zeros_like(slice_hwc)
            result[..., img_idx] = aug_image
            result[..., mask_idx] = aug_mask
        else:
            # Multi-channel case: rebuild with mask at correct position
            result = np.zeros_like(slice_hwc)
            j = 0
            for i in range(num_channels):
                if i == mask_idx:
                    result[..., i] = aug_mask
                else:
                    if aug_image.ndim == 2:
                        result[..., i] = aug_image
                    else:
                        result[..., i] = aug_image[..., j]
                    j += 1
    else:
        # No mask - augment all channels as image
        augmented = aug(image=slice_hwc)
        result = augmented['image']

    # Ensure result is contiguous before transpose
    result = np.ascontiguousarray(result)

    # Transpose back: [H, W, C] -> [C, H, W]
    output = np.transpose(result, (2, 0, 1)).copy()

    return output


# =============================================================================
# Legacy API (backward compatibility)
# =============================================================================

def build_augmentation(enabled: bool = True) -> Optional[A.Compose]:
    """Build augmentation pipeline (defaults to diffusion-style).

    DEPRECATED: Use build_diffusion_augmentation or build_vae_augmentation.

    Args:
        enabled: Whether to enable augmentation.

    Returns:
        Albumentations Compose object, or None if disabled.
    """
    return build_diffusion_augmentation(enabled)
