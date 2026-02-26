"""
Brain mask utilities for validating generated segmentation masks.

Detects if tumor segmentation falls outside the brain volume outline.
Uses morphological operations to handle internal low-intensity regions
(ventricles, CSF) and boundary cases correctly.
"""
import logging
from pathlib import Path

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


def load_brain_atlas(atlas_path: str | Path, expected_shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Load a pre-computed brain atlas from NIfTI file.

    Args:
        atlas_path: Path to atlas NIfTI file (stored as [H, W, D]).
        expected_shape: If provided, validate atlas matches (D, H, W).

    Returns:
        Bool array [D, H, W].
    """
    import nibabel as nib

    atlas_path = Path(atlas_path)
    if not atlas_path.exists():
        raise FileNotFoundError(f"Brain atlas not found: {atlas_path}")

    vol = nib.load(str(atlas_path)).get_fdata()
    # NIfTI is [H, W, D] → transpose to [D, H, W]
    atlas = np.transpose(vol, (2, 0, 1)) > 0.5

    if expected_shape is not None and atlas.shape != expected_shape:
        raise ValueError(
            f"Brain atlas shape {atlas.shape} does not match expected {expected_shape}"
        )

    return atlas


def is_seg_inside_atlas(
    seg: torch.Tensor | np.ndarray,
    atlas: np.ndarray,
    tolerance: float = 0.0,
    dilate_pixels: int = 0,
) -> bool:
    """Check if segmentation mask is inside a pre-computed brain atlas.

    Like ``is_seg_inside_brain()`` but uses a pre-computed atlas instead
    of deriving the brain mask from an MRI image at runtime.

    Args:
        seg: Segmentation mask (any shape with optional batch/channel dims).
        atlas: Pre-computed brain atlas as bool array [D, H, W].
        tolerance: Maximum allowed ratio of outside voxels (0-1).
        dilate_pixels: Extra dilation applied to atlas at runtime.

    Returns:
        True if segmentation is valid (inside atlas).
    """
    if isinstance(seg, torch.Tensor):
        seg = seg.detach().cpu().numpy()
    seg = np.squeeze(seg)

    brain_mask = atlas
    if dilate_pixels > 0:
        brain_mask = ndimage.binary_dilation(brain_mask, iterations=dilate_pixels)

    tumor_mask = seg > 0.5
    n_tumor = int(tumor_mask.sum())
    if n_tumor == 0:
        return True

    n_outside = int((tumor_mask & ~brain_mask).sum())
    return (n_outside / n_tumor) <= tolerance


def remove_tumors_outside_brain(
    seg: np.ndarray,
    brain_mask: np.ndarray,
    outside_threshold: float = 0.1,
) -> tuple[np.ndarray, int]:
    """Remove individual tumors that fall outside the brain mask.

    Labels connected components in seg, then removes any component where
    more than ``outside_threshold`` of its voxels are outside the brain mask.

    Args:
        seg: Binary segmentation mask [D, H, W] (or [H, W] for 2D).
        brain_mask: Binary brain mask, same spatial shape as seg.
        outside_threshold: Fraction of voxels outside brain to trigger removal
            (default 0.1 = remove if >10% outside).

    Returns:
        Tuple of (cleaned_seg, n_removed) where cleaned_seg is the
        segmentation with outside tumors zeroed out.
    """
    seg_binary = (seg > 0.5).astype(np.uint8)
    brain_binary = brain_mask.astype(bool)

    labeled, n_components = ndimage.label(seg_binary)
    if n_components == 0:
        return seg_binary.astype(np.float32), 0

    n_removed = 0
    for label_id in range(1, n_components + 1):
        component = labeled == label_id
        n_voxels = int(component.sum())
        n_outside = int((component & ~brain_binary).sum())

        if n_voxels > 0 and (n_outside / n_voxels) > outside_threshold:
            seg_binary[component] = 0
            n_removed += 1

    return seg_binary.astype(np.float32), n_removed
