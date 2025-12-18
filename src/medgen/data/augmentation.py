"""Data augmentation for medical image training.

Conservative augmentation pipeline using albumentations, shared across
diffusion, VAE, and VAE progressive training. No intensity augmentations
since images are normalized to [0, 1].
"""
import albumentations as A
import numpy as np
from typing import Optional


def build_augmentation(enabled: bool = True) -> Optional[A.Compose]:
    """Build conservative augmentation pipeline for medical images.

    Shared across diffusion, VAE, and VAE progressive training.
    NO intensity augmentations (images normalized to 0-1).

    Args:
        enabled: Whether to enable augmentation. False for validation/test.

    Returns:
        Albumentations Compose object, or None if disabled.
    """
    if not enabled:
        return None

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=0, p=1.0),  # cv2.BORDER_CONSTANT
    ])


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
