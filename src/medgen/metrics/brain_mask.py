"""
Brain mask utilities for validating generated segmentation masks.

Detects if tumor segmentation falls outside the brain volume outline.
Uses morphological operations to handle internal low-intensity regions
(ventricles, CSF) and boundary cases correctly.
"""
import logging

import numpy as np
import torch
from scipy import ndimage

logger = logging.getLogger(__name__)


def create_brain_mask(
    image: torch.Tensor | np.ndarray,
    threshold: float = 0.05,
    fill_holes: bool = True,
    dilate_pixels: int = 3,
) -> np.ndarray:
    """Create a brain mask from MRI image using morphological operations.

    Handles internal low-intensity regions (ventricles, CSF) by:
    1. Thresholding to get initial tissue estimate
    2. Keeping only the largest connected component (brain)
    3. Filling internal holes to include ventricles
    4. Dilating to allow tumors at the boundary

    Args:
        image: MRI image, any shape (batch/channel dims are squeezed).
        threshold: Intensity threshold for initial brain detection.
        fill_holes: Whether to fill internal holes (recommended).
        dilate_pixels: Number of pixels to dilate brain mask (handles boundary tumors).

    Returns:
        Binary brain mask as numpy array (same spatial shape as input).
    """
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    # Remove batch/channel dimensions
    image = np.squeeze(image)

    # Initial threshold
    binary = image > threshold

    # Find connected components
    labeled, n_components = ndimage.label(binary)

    if n_components == 0:
        return np.zeros_like(image, dtype=bool)

    # Find largest component (the brain)
    component_sizes = ndimage.sum(binary, labeled, range(1, n_components + 1))
    largest_label = np.argmax(component_sizes) + 1
    brain_mask = labeled == largest_label

    # Fill internal holes (ventricles, CSF cavities)
    if fill_holes:
        brain_mask = ndimage.binary_fill_holes(brain_mask)

    # Dilate to allow tumors at the boundary
    if dilate_pixels > 0:
        brain_mask = ndimage.binary_dilation(brain_mask, iterations=dilate_pixels)

    return brain_mask


def is_seg_inside_brain(
    image: torch.Tensor | np.ndarray,
    seg: torch.Tensor | np.ndarray,
    brain_threshold: float = 0.05,
    tolerance: float = 0.05,
    dilate_pixels: int = 3,
) -> bool:
    """Check if segmentation mask is inside the brain volume.

    Args:
        image: MRI image (any shape with optional batch/channel dims).
        seg: Segmentation mask (same shape as image).
        brain_threshold: Intensity threshold for brain detection.
        tolerance: Maximum allowed ratio of outside pixels (0-1).
        dilate_pixels: Pixels to dilate brain mask (handles boundary tumors).

    Returns:
        True if segmentation is valid (inside brain), False otherwise.
    """
    result = compute_outside_brain_ratio(image, seg, brain_threshold, dilate_pixels)
    return result['outside_ratio'] <= tolerance


def compute_outside_brain_ratio(
    image: torch.Tensor | np.ndarray,
    seg: torch.Tensor | np.ndarray,
    brain_threshold: float = 0.05,
    dilate_pixels: int = 3,
) -> dict[str, float | int]:
    """Compute ratio of segmentation pixels outside the brain.

    Args:
        image: MRI image.
        seg: Segmentation mask.
        brain_threshold: Intensity threshold for brain detection.
        dilate_pixels: Pixels to dilate brain mask (handles boundary tumors).

    Returns:
        Dict with 'outside_ratio', 'n_tumor_pixels', 'n_outside_pixels'.
    """
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(seg, torch.Tensor):
        seg = seg.detach().cpu().numpy()

    # Remove batch/channel dimensions
    image = np.squeeze(image)
    seg = np.squeeze(seg)

    # Create brain mask with filled holes and dilation
    brain_mask = create_brain_mask(
        image,
        threshold=brain_threshold,
        fill_holes=True,
        dilate_pixels=dilate_pixels,
    )

    # Binarize segmentation
    tumor_mask = seg > 0.5

    # Find tumor pixels outside brain
    tumor_outside = tumor_mask & ~brain_mask

    n_tumor = int(tumor_mask.sum())
    n_outside = int(tumor_outside.sum())

    if n_tumor == 0:
        return {
            'outside_ratio': 0.0,
            'n_tumor_pixels': 0,
            'n_outside_pixels': 0,
        }

    return {
        'outside_ratio': float(n_outside / n_tumor),
        'n_tumor_pixels': n_tumor,
        'n_outside_pixels': n_outside,
    }
