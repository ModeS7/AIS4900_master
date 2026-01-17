"""Data augmentation for medical image training.

Separate augmentation pipelines for diffusion and VAE training:
- Diffusion: Conservative (flip, discrete translate) - preserves image quality
- VAE: Aggressive (+ noise, blur, scale, mixup, cutmix) - learns robust features
- Segmentation: Hardcore spatial (+ mosaic, cutmix, copy-paste) - binary mask compression

Batch-level augmentations (mixup, cutmix) are applied via collate function.
"""
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
import torch

# Module-level imports for segmentation augmentations (avoid import inside functions)
try:
    from skimage.transform import resize as skimage_resize
    from scipy.ndimage import label as scipy_label
    _SEG_AUG_AVAILABLE = True
except ImportError:
    _SEG_AUG_AVAILABLE = False


# =============================================================================
# Custom Discrete Translation (Lossless)
# =============================================================================

class DiscreteTranslate(ImageOnlyTransform):
    """Discrete pixel translation - truly lossless.

    Shifts image by integer pixels using np.roll, then fills the wrapped
    edges based on pad_mode. No interpolation, no quality loss.

    Args:
        max_percent_x: Maximum translation as fraction of width (e.g., 0.2 = 20%).
        max_percent_y: Maximum translation as fraction of height (e.g., 0.1 = 10%).
        pad_mode: How to fill wrapped edges after translation:
            - 'zero': Fill with zeros (default, original behavior).
            - 'reflect': Mirror edge pixels (preserves tissue continuity for medical images).
        p: Probability of applying the transform.

    Raises:
        ValueError: If pad_mode is not 'zero' or 'reflect'.
    """

    def __init__(
        self,
        max_percent_x: float = 0.2,
        max_percent_y: float = 0.1,
        pad_mode: str = 'zero',
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.max_percent_x = max_percent_x
        self.max_percent_y = max_percent_y
        if pad_mode not in ('zero', 'reflect'):
            raise ValueError(f"pad_mode must be 'zero' or 'reflect', got '{pad_mode}'")
        self.pad_mode = pad_mode

    def apply(self, img: np.ndarray, dx: int = 0, dy: int = 0, **params) -> np.ndarray:
        if dx == 0 and dy == 0:
            return img

        # Roll pixels
        result = np.roll(img, shift=(dy, dx), axis=(0, 1))

        if self.pad_mode == 'zero':
            # Zero out wrapped edges (original behavior)
            if dy > 0:
                result[:dy, :] = 0
            elif dy < 0:
                result[dy:, :] = 0

            if dx > 0:
                result[:, :dx] = 0
            elif dx < 0:
                result[:, dx:] = 0
        else:
            # Reflect mode: mirror edge pixels for continuity
            # After roll, fill edges with flipped content from original image
            if dy > 0:
                result[:dy, :] = np.flip(img[:dy, :], axis=0)
            elif dy < 0:
                result[dy:, :] = np.flip(img[dy:, :], axis=0)

            if dx > 0:
                result[:, :dx] = np.flip(img[:, :dx], axis=1)
            elif dx < 0:
                result[:, dx:] = np.flip(img[:, dx:], axis=1)

        return result

    def get_params_dependent_on_data(self, params: Dict, data: Dict) -> Dict[str, int]:
        h, w = params["shape"][:2]
        max_dx = int(w * self.max_percent_x)
        max_dy = int(h * self.max_percent_y)
        return {
            "dx": random.randint(-max_dx, max_dx) if max_dx > 0 else 0,
            "dy": random.randint(-max_dy, max_dy) if max_dy > 0 else 0,
        }

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("max_percent_x", "max_percent_y", "pad_mode")


# =============================================================================
# Diffusion Augmentation (Conservative)
# =============================================================================

def build_diffusion_augmentation(enabled: bool = True) -> Optional[A.Compose]:
    """Build conservative augmentation for diffusion training.

    Only lossless spatial transforms that preserve image quality.
    Rotation is avoided because interpolation causes blurring.

    Transforms:
        - HorizontalFlip (p=0.5) - lossless
        - DiscreteTranslate ±20% X, ±10% Y (p=0.5) - lossless integer pixel shift
          (more horizontal room since brain is oval/vertical)

    Args:
        enabled: Whether to enable augmentation.

    Returns:
        Albumentations Compose object, or None if disabled.
    """
    if not enabled:
        return None

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        DiscreteTranslate(max_percent_x=0.2, max_percent_y=0.1, p=1.0),
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
    images: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Apply mixup augmentation to a batch of images.

    Blends pairs of images: mixed = lambda * img1 + (1-lambda) * img2
    Lambda sampled from Beta(alpha, alpha).

    Operates directly on tensors (no numpy conversion).

    Args:
        images: Batch of images [B, C, H, W] as tensor.
        alpha: Beta distribution parameter (higher = more mixing).

    Returns:
        Tuple of (mixed_images, shuffled_indices, lambda_value).
    """
    batch_size = images.shape[0]
    lam = float(torch.distributions.Beta(alpha, alpha).sample())

    # Random permutation for pairing
    indices = torch.randperm(batch_size, device=images.device)

    # Mix images
    mixed = lam * images + (1 - lam) * images[indices]

    return mixed, indices, lam


def cutmix(
    images: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Apply CutMix augmentation to a batch of images.

    Cuts a rectangular region from one image and pastes onto another.
    Lambda proportional to the area ratio.

    Operates directly on tensors (no numpy conversion).

    Args:
        images: Batch of images [B, C, H, W] as tensor.
        alpha: Beta distribution parameter for lambda.

    Returns:
        Tuple of (mixed_images, shuffled_indices, lambda_value).
    """
    batch_size, c, h, w = images.shape
    lam = float(torch.distributions.Beta(alpha, alpha).sample())

    # Random permutation for pairing
    indices = torch.randperm(batch_size, device=images.device)

    # Get cut region size (proportional to 1-lambda)
    cut_ratio = (1 - lam) ** 0.5
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    # Random center for cut
    cy = random.randint(0, h - 1)
    cx = random.randint(0, w - 1)

    # Bounding box (clipped to image bounds)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, h)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, w)

    # Apply cutmix
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]

    # Adjust lambda based on actual cut area
    lam = 1 - ((y2 - y1) * (x2 - x1)) / (h * w)

    return mixed, indices, lam


def create_vae_collate_fn(
    mixup_prob: float = 0.2,
    cutmix_prob: float = 0.2,
) -> Callable[[List], Any]:
    """Create collate function with mixup/cutmix for VAE training.

    Handles both tensor batches and tuple batches (image, mask) for regional metrics.

    Args:
        mixup_prob: Probability of applying mixup (default 0.2).
        cutmix_prob: Probability of applying cutmix (default 0.2).

    Returns:
        Collate function for DataLoader.
    """
    def collate_fn(batch: List) -> Any:
        # Check if batch contains tuples (image, mask) for regional metrics
        has_masks = isinstance(batch[0], tuple) and len(batch[0]) == 2

        if has_masks:
            # Separate images and masks
            image_tensors = []
            mask_tensors = []
            for item in batch:
                img, mask = item
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).float()
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask).float()
                image_tensors.append(img)
                mask_tensors.append(mask)
            images = torch.stack(image_tensors)
            masks = torch.stack(mask_tensors)
        else:
            # Single tensor batch
            tensors = []
            for item in batch:
                if isinstance(item, np.ndarray):
                    tensors.append(torch.from_numpy(item).float())
                else:
                    tensors.append(item)
            images = torch.stack(tensors)
            masks = None

        # Apply batch augmentation to images only (not masks)
        # mixup/cutmix now operate directly on tensors (no numpy conversion)
        r = random.random()
        if r < mixup_prob:
            images, _, _ = mixup(images, alpha=0.4)
        elif r < mixup_prob + cutmix_prob:
            images, _, _ = cutmix(images, alpha=1.0)

        if has_masks:
            return (images, masks)
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
# Segmentation Diffusion Augmentation (Conservative for Size-Conditioned)
# =============================================================================

def build_seg_diffusion_augmentation(enabled: bool = True) -> Optional[A.Compose]:
    """Build augmentation for segmentation mask diffusion training.

    Conservative spatial transforms suitable for size-conditioned diffusion
    where we condition on tumor size distribution. Augmentations must preserve
    reasonable tumor shapes while adding variety.

    Transforms:
        - HorizontalFlip, VerticalFlip (p=0.5 each) - lossless
        - Rotate ±15° (p=0.5) - mild rotation
        - Scale 0.9-1.1x (p=0.5) - mild zoom in/out
        - Translate ±5% (p=0.5) - small shifts
        - ElasticTransform (mild, p=0.2) - subtle anatomical variation

    IMPORTANT: After using this augmentation, apply binarize_mask() with
    threshold=0.5 to restore binary values (interpolation creates non-binary).

    Args:
        enabled: Whether to enable augmentation.

    Returns:
        Albumentations Compose object, or None if disabled.
    """
    if not enabled:
        return None

    return A.Compose([
        # Flips (lossless, very effective for brain symmetry)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Rotation ±15 degrees (mild, preserves tumor shapes)
        A.Rotate(
            limit=15,
            border_mode=0,  # Zero padding
            p=0.5,
        ),

        # Scale (zoom in/out) and Translation
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # Max ±5%
            scale=(0.9, 1.1),  # ±10% zoom
            rotate=0,  # Rotation handled separately above
            border_mode=0,  # Zero padding
            p=0.5,
        ),

        # Mild elastic deformation (simulates anatomical variation)
        # Lower alpha than seg_augmentation for subtler effect
        A.ElasticTransform(
            alpha=30,  # Mild (seg_augmentation uses 120)
            sigma=8,
            border_mode=0,
            p=0.2,
        ),
    ])


class BinarizeTransform(ImageOnlyTransform):
    """Binarize mask after augmentation to restore 0/1 values.

    CRITICAL: Must be applied after any augmentation using interpolation
    (rotation, scaling, elastic) to restore binary mask values.

    Args:
        threshold: Binarization threshold (default 0.5).
        p: Always 1.0 - this should always be applied.
    """

    def __init__(self, threshold: float = 0.5, always_apply: bool = True, p: float = 1.0):
        super().__init__(p=p)
        self.threshold = threshold
        self.always_apply = always_apply

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return (img > self.threshold).astype(np.float32)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("threshold",)


def build_seg_diffusion_augmentation_with_binarize(enabled: bool = True) -> Optional[A.Compose]:
    """Build seg diffusion augmentation with automatic binarization.

    Same as build_seg_diffusion_augmentation but includes final
    binarization step. Use this for convenience when you don't need
    to manually control binarization.

    Args:
        enabled: Whether to enable augmentation.

    Returns:
        Albumentations Compose object with binarization, or None if disabled.
    """
    if not enabled:
        return None

    return A.Compose([
        # Flips (lossless)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Rotation ±15 degrees
        A.Rotate(
            limit=15,
            border_mode=0,
            p=0.5,
        ),

        # Scale and Translation
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=0,
            border_mode=0,
            p=0.5,
        ),

        # Mild elastic deformation
        A.ElasticTransform(
            alpha=30,
            sigma=8,
            border_mode=0,
            p=0.2,
        ),

        # CRITICAL: Binarize after all spatial transforms
        BinarizeTransform(threshold=0.5, p=1.0),
    ])


# =============================================================================
# Segmentation Mask Augmentation (Hardcore Spatial for Compression)
# =============================================================================

def build_seg_augmentation(enabled: bool = True) -> Optional[A.Compose]:
    """Build hardcore augmentation for segmentation mask compression.

    Aggressive spatial transforms for learning robust mask representations.
    All augmentations preserve binary nature via final thresholding.

    Transforms:
        - HorizontalFlip, VerticalFlip (p=0.5 each)
        - Rotate 90/180/270 (p=0.5)
        - Affine: translate ±20%, scale 0.8-1.2x (p=0.5)
        - ElasticTransform (p=0.3)
        - GridDistortion (p=0.3)
        - CoarseDropout (p=0.3)

    Note: Mosaic, CutMix, CopyPaste are batch-level augmentations
    implemented separately in collate function.

    Args:
        enabled: Whether to enable augmentation.

    Returns:
        Albumentations Compose object, or None if disabled.
    """
    if not enabled:
        return None

    return A.Compose([
        # Flips (lossless)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # 90-degree rotations (lossless for binary masks)
        A.RandomRotate90(p=0.5),

        # Affine transforms (more aggressive than VAE)
        A.Affine(
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            scale=(0.8, 1.2),
            rotate=(-45, 45),
            border_mode=0,  # Zero padding
            p=0.5,
        ),

        # Elastic deformation (more aggressive for seg)
        A.ElasticTransform(
            alpha=120,
            sigma=12,
            border_mode=0,
            p=0.3,
        ),

        # Grid distortion
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            border_mode=0,
            p=0.3,
        ),

        # Coarse dropout (simulates missing regions)
        # Note: albumentations 2.0+ uses different API
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.03, 0.12),  # ~8-32 pixels on 256x256
            hole_width_range=(0.03, 0.12),
            fill=0,
            p=0.3,
        ),
    ])


def binarize_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binarize mask after augmentation.

    CRITICAL: Must be called after any augmentation that uses interpolation
    (rotation, affine, elastic) to restore binary values.

    Args:
        mask: Mask array (may have non-binary values from interpolation).
        threshold: Binarization threshold.

    Returns:
        Binary mask (0.0 or 1.0).
    """
    return (mask > threshold).astype(np.float32)


def mosaic_augmentation(
    masks: List[np.ndarray],
    output_size: int = 256,
) -> np.ndarray:
    """Create mosaic from 4 masks (2x2 grid).

    Combines 4 different masks into a single training sample.

    Args:
        masks: List of exactly 4 masks [H, W] or [1, H, W] or [H, W, C].
        output_size: Output image size.

    Returns:
        Mosaic mask [1, output_size, output_size].
    """
    if not _SEG_AUG_AVAILABLE:
        raise ImportError("skimage required for mosaic_augmentation")

    assert len(masks) == 4, "Mosaic requires exactly 4 masks"

    half = output_size // 2
    mosaic = np.zeros((output_size, output_size), dtype=np.float32)

    # Resize and place each mask in quadrant
    positions = [(0, 0), (0, half), (half, 0), (half, half)]
    for mask, (y, x) in zip(masks, positions):
        # Handle different channel formats
        if mask.ndim == 3:
            if mask.shape[0] == 1:  # [1, H, W] format
                mask = mask[0]
            elif mask.shape[-1] == 1:  # [H, W, 1] format
                mask = mask[:, :, 0]
            else:  # [H, W, C] format
                mask = mask[:, :, 0]

        # Simple resize (nearest neighbor for binary)
        resized = skimage_resize(mask, (half, half), order=0, preserve_range=True)
        mosaic[y:y+half, x:x+half] = resized

    # Binarize and add channel dimension
    mosaic = (mosaic > 0.5).astype(np.float32)
    return mosaic[np.newaxis, :, :]  # [1, H, W]


def copy_paste_augmentation(
    mask: np.ndarray,
    donor_mask: np.ndarray,
    p: float = 0.5,
) -> np.ndarray:
    """Copy-paste augmentation: paste tumor regions from donor to target.

    Args:
        mask: Target mask to modify [H, W] or [1, H, W].
        donor_mask: Donor mask to copy from [H, W] or [1, H, W].
        p: Probability of applying.

    Returns:
        Augmented mask (same shape as input).
    """
    if not _SEG_AUG_AVAILABLE:
        raise ImportError("scipy required for copy_paste_augmentation")

    if random.random() > p:
        return mask

    had_channel = mask.ndim == 3

    # Normalize to [H, W]
    if had_channel:
        if mask.shape[0] == 1:
            mask = mask[0]
        else:
            mask = mask[:, :, 0]

    donor = donor_mask
    if donor.ndim == 3:
        if donor.shape[0] == 1:
            donor = donor[0]
        else:
            donor = donor[:, :, 0]

    # Find tumor regions in donor
    donor_binary = donor > 0.5
    labeled, num_tumors = scipy_label(donor_binary)

    if num_tumors == 0:
        result = (mask > 0.5).astype(np.float32)
        if had_channel:
            result = result[np.newaxis, :, :]
        return result

    # Randomly select one tumor to paste
    tumor_id = random.randint(1, num_tumors)
    tumor_mask = (labeled == tumor_id)

    # Random translation
    h, w = mask.shape[:2]
    dy = random.randint(-h//4, h//4)
    dx = random.randint(-w//4, w//4)

    # Paste (OR operation)
    result = mask.copy()
    tumor_shifted = np.roll(tumor_mask.astype(np.float32), (dy, dx), axis=(0, 1))
    result = np.maximum(result, tumor_shifted)
    result = (result > 0.5).astype(np.float32)

    # Restore original shape
    if had_channel:
        result = result[np.newaxis, :, :]

    return result


def _cutmix_masks(
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply cutmix between two masks.

    Args:
        mask1: First mask [H, W] or [1, H, W].
        mask2: Second mask [H, W] or [1, H, W].

    Returns:
        Tuple of two swapped masks (same shape as input).
    """
    # Track original shape for restoration
    had_channel = mask1.ndim == 3

    # Normalize to [H, W]
    if had_channel:
        if mask1.shape[0] == 1:
            mask1_2d = mask1[0]
        else:
            mask1_2d = mask1[:, :, 0]
        if mask2.shape[0] == 1:
            mask2_2d = mask2[0]
        else:
            mask2_2d = mask2[:, :, 0]
    else:
        mask1_2d = mask1
        mask2_2d = mask2

    h, w = mask1_2d.shape[:2]

    # Random rectangle
    cx, cy = random.randint(0, w), random.randint(0, h)
    rw, rh = random.randint(w//4, w//2), random.randint(h//4, h//2)

    x1, x2 = max(0, cx - rw//2), min(w, cx + rw//2)
    y1, y2 = max(0, cy - rh//2), min(h, cy + rh//2)

    # Swap regions
    result1, result2 = mask1_2d.copy(), mask2_2d.copy()
    result1[y1:y2, x1:x2] = mask2_2d[y1:y2, x1:x2]
    result2[y1:y2, x1:x2] = mask1_2d[y1:y2, x1:x2]

    # Binarize
    result1 = (result1 > 0.5).astype(np.float32)
    result2 = (result2 > 0.5).astype(np.float32)

    # Restore channel dimension if input had it
    if had_channel:
        result1 = result1[np.newaxis, :, :]
        result2 = result2[np.newaxis, :, :]

    return result1, result2


def create_seg_collate_fn(
    mosaic_prob: float = 0.2,
    cutmix_prob: float = 0.2,
    copy_paste_prob: float = 0.3,
) -> Callable[[List], torch.Tensor]:
    """Create collate function with batch-level seg augmentations.

    Args:
        mosaic_prob: Probability of mosaic augmentation.
        cutmix_prob: Probability of cutmix augmentation.
        copy_paste_prob: Probability of copy-paste per sample.

    Returns:
        Collate function for DataLoader.
    """
    def collate_fn(batch: List) -> torch.Tensor:
        # Convert to numpy if needed
        augmented = []
        for item in batch:
            if isinstance(item, torch.Tensor):
                mask = item.numpy()
            else:
                mask = item
            augmented.append(mask)

        # Copy-paste (needs other samples in batch)
        for i in range(len(augmented)):
            if random.random() < copy_paste_prob and len(augmented) > 1:
                donor_idx = random.choice([j for j in range(len(augmented)) if j != i])
                augmented[i] = copy_paste_augmentation(
                    augmented[i], augmented[donor_idx], p=1.0
                )

        # CRITICAL: Final binarization
        for i in range(len(augmented)):
            augmented[i] = binarize_mask(augmented[i])

        # Mosaic (replaces first sample with mosaic of 4 random samples)
        if random.random() < mosaic_prob and len(augmented) >= 4:
            indices = random.sample(range(len(augmented)), 4)
            mosaic = mosaic_augmentation([augmented[i] for i in indices])
            augmented[0] = mosaic

        # CutMix (swap regions between pairs)
        if random.random() < cutmix_prob and len(augmented) >= 2:
            i, j = random.sample(range(len(augmented)), 2)
            augmented[i], augmented[j] = _cutmix_masks(augmented[i], augmented[j])

        # Stack and convert to tensor
        stacked = np.stack(augmented, axis=0)
        if stacked.ndim == 3:
            stacked = stacked[:, np.newaxis, :, :]  # Add channel dim if missing

        return torch.from_numpy(stacked).float()

    return collate_fn


