"""Consolidated Dataset classes for all dataloader modes.

This module contains all Dataset classes that were previously scattered across
multiple files. Consolidating them here:
1. Reduces code duplication
2. Makes it easier to find and understand Dataset implementations
3. Simplifies imports

Dataset Classes:
    - SegConditionedDataset: 2D seg masks with size bin conditioning
    - MultiDiffusionDataset: Multi-modality diffusion with mode_id
    - AugmentedSegDataset: Seg masks with on-the-fly augmentation (DC-AE)
    - VolumeDataset: Full 3D volumes for single modality
    - DualVolumeDataset: Full 3D volumes for dual modality

Utility Functions:
    - compute_size_bins / compute_size_bins_3d: Compute tumor counts per size bin
    - compute_feret_diameter / compute_feret_diameter_3d: Compute longest axis
    - create_size_bin_maps: Create spatial conditioning maps from size bins
    - extract_seg_slices: Extract 2D slices from seg volumes
    - extract_slices_with_seg_and_mode: Extract slices with mode_id
"""
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial.distance import pdist
from torch.utils.data import Dataset as TorchDataset

from medgen.core.constants import BINARY_THRESHOLD_GT

logger = logging.getLogger(__name__)

# Default size bins (mm) - aligned with RANO-BM thresholds (10, 20, 30mm)
DEFAULT_BIN_EDGES = [0, 3, 6, 10, 15, 20, 30]


# =============================================================================
# Size Bin Utility Functions
# =============================================================================

def create_size_bin_maps(
    size_bins: torch.Tensor,
    spatial_shape: tuple[int, ...],
    normalize: bool = True,
    max_count: int = 10,
) -> torch.Tensor:
    """Create spatial maps from size bin counts for input conditioning.

    Each size bin count becomes a constant-valued spatial map that gets
    concatenated with the noise input to the diffusion model.

    Args:
        size_bins: [num_bins] tensor of counts per bin.
        spatial_shape: Target spatial dimensions (H, W) or (D, H, W).
        normalize: If True, divide counts by max_count to get [0, 1] range.
        max_count: Maximum expected count per bin for normalization.

    Returns:
        [num_bins, *spatial_shape] tensor where each channel is filled
        with the (normalized) count for that bin.
    """
    num_bins = len(size_bins)
    bin_maps = torch.zeros(num_bins, *spatial_shape, dtype=torch.float32)

    for i in range(num_bins):
        count = size_bins[i].float()
        if normalize:
            count = count / max_count
        bin_maps[i].fill_(count.item())

    return bin_maps


def compute_feret_diameter(binary_mask: np.ndarray, pixel_spacing_mm: float) -> float:
    """Compute Feret diameter (longest axis) of a binary region.

    Args:
        binary_mask: 2D binary mask of a single connected component.
        pixel_spacing_mm: Size of one pixel in mm.

    Returns:
        Feret diameter in mm.
    """
    coords = np.argwhere(binary_mask)
    if len(coords) < 2:
        return pixel_spacing_mm  # Single pixel

    # Subsample for large regions
    if len(coords) > 1000:
        idx = np.random.choice(len(coords), 1000, replace=False)
        coords = coords[idx]

    distances = pdist(coords)
    max_dist_pixels = distances.max() if len(distances) > 0 else 1

    return max_dist_pixels * pixel_spacing_mm


def compute_size_bins(
    seg_mask: np.ndarray,
    bin_edges: list[float],
    pixel_spacing_mm: float,
    num_bins: int = None,
) -> np.ndarray:
    """Compute tumor count per size bin for a 2D segmentation mask.

    Args:
        seg_mask: 2D binary segmentation mask [H, W].
        bin_edges: List of bin edges in mm (e.g., [0, 3, 6, 10, 15, 20, 30]).
        pixel_spacing_mm: Size of one pixel in mm.
        num_bins: Number of bins. If > len(edges)-1, last bin is overflow (>= last edge).
                  Default: len(edges)-1 (no separate overflow bin).

    Returns:
        Array of shape [num_bins] with tumor counts per bin.
    """
    n_bounded_bins = len(bin_edges) - 1
    if num_bins is None:
        num_bins = n_bounded_bins

    bin_counts = np.zeros(num_bins, dtype=np.int64)

    # Handle tensor input
    if isinstance(seg_mask, torch.Tensor):
        seg_mask = seg_mask.numpy()

    # Remove batch/channel dimensions if present
    seg_mask = np.squeeze(seg_mask)

    # Label connected components
    labeled, num_features = ndimage.label(seg_mask > 0.5)

    if num_features == 0:
        return bin_counts

    # Compute diameter for each tumor and bin it
    for i in range(1, num_features + 1):
        component_mask = labeled == i
        diameter = compute_feret_diameter(component_mask, pixel_spacing_mm)

        # Find bin - check bounded bins first
        binned = False
        for j in range(n_bounded_bins):
            if bin_edges[j] <= diameter < bin_edges[j + 1]:
                bin_counts[j] += 1
                binned = True
                break

        # If not binned, goes to overflow bin (last bin)
        if not binned:
            bin_counts[num_bins - 1] += 1

    return bin_counts


def compute_feret_diameter_3d(
    binary_mask: np.ndarray,
    voxel_spacing: tuple[float, float, float],
) -> float:
    """Compute 3D Feret diameter (longest axis) of a binary region.

    Unlike 2D, this computes the longest distance in 3D space,
    accounting for anisotropic voxel spacing.

    Args:
        binary_mask: 3D binary mask of a single connected component [D, H, W].
        voxel_spacing: Voxel size in mm as (depth_mm, height_mm, width_mm).

    Returns:
        Feret diameter in mm.
    """
    coords = np.argwhere(binary_mask)  # [N, 3] coordinates
    if len(coords) < 2:
        return min(voxel_spacing)  # Single voxel

    # Scale coordinates by voxel spacing for physical distances
    # coords[:, 0] = depth, coords[:, 1] = height, coords[:, 2] = width
    spacing_array = np.array(voxel_spacing)
    scaled_coords = coords * spacing_array

    # Subsample for large regions (3D can have many more voxels)
    if len(scaled_coords) > 2000:
        idx = np.random.choice(len(scaled_coords), 2000, replace=False)
        scaled_coords = scaled_coords[idx]

    distances = pdist(scaled_coords)
    max_dist = distances.max() if len(distances) > 0 else min(voxel_spacing)

    return max_dist


def compute_size_bins_3d(
    seg_volume: np.ndarray,
    bin_edges: list[float],
    voxel_spacing: tuple[float, float, float],
    num_bins: int = None,
) -> np.ndarray:
    """Compute tumor count per size bin for a 3D segmentation volume.

    Uses 3D connected components so tumors touching in ANY direction
    (including depth) count as ONE tumor.

    Args:
        seg_volume: 3D binary segmentation volume [D, H, W].
        bin_edges: List of bin edges in mm (e.g., [0, 3, 6, 10, 15, 20, 30]).
        voxel_spacing: Voxel size in mm as (depth_mm, height_mm, width_mm).
        num_bins: Number of bins. If > len(edges)-1, last bin is overflow (>= last edge).
                  Default: len(edges)-1 (no separate overflow bin).

    Returns:
        Array of shape [num_bins] with tumor counts per bin.
    """
    n_bounded_bins = len(bin_edges) - 1
    if num_bins is None:
        num_bins = n_bounded_bins

    bin_counts = np.zeros(num_bins, dtype=np.int64)

    # Handle tensor input
    if isinstance(seg_volume, torch.Tensor):
        seg_volume = seg_volume.numpy()

    # Remove batch/channel dimensions if present [1, D, H, W] -> [D, H, W]
    seg_volume = np.squeeze(seg_volume)

    # 3D connected components - ndimage.label works for N-dimensional data
    # This ensures tumors touching in ANY direction are counted as ONE tumor
    labeled, num_features = ndimage.label(seg_volume > 0.5)

    if num_features == 0:
        return bin_counts

    # Compute 3D Feret diameter for each tumor and bin it
    for i in range(1, num_features + 1):
        component_mask = labeled == i
        diameter = compute_feret_diameter_3d(component_mask, voxel_spacing)

        # Find bin - check bounded bins first
        binned = False
        for j in range(n_bounded_bins):
            if bin_edges[j] <= diameter < bin_edges[j + 1]:
                bin_counts[j] += 1
                binned = True
                break

        # If not binned, goes to overflow bin (last bin) - for tumors >= last edge
        if not binned:
            bin_counts[num_bins - 1] += 1

    return bin_counts


# =============================================================================
# Slice Extraction Functions
# =============================================================================

def extract_seg_slices(
    seg_dataset: TorchDataset,
    min_tumor_pixels: int = 10,
) -> list[np.ndarray]:
    """Extract 2D segmentation mask slices from 3D volumes.

    Only keeps slices with actual tumor content (non-empty masks).
    No augmentation is applied here - augmentation happens in AugmentedSegDataset.__getitem__.

    Args:
        seg_dataset: NiFTI dataset with seg volumes.
        min_tumor_pixels: Minimum number of tumor pixels to keep slice.

    Returns:
        List of 2D mask slices [1, H, W] (binary, float32).
    """
    from medgen.augmentation import binarize_mask

    all_slices: list[np.ndarray] = []

    for i in range(len(seg_dataset)):
        volume, _ = seg_dataset[i]  # Shape: [1, H, W, D]

        # Convert to numpy if tensor
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()

        # Extract non-empty slices along depth dimension (axis 3)
        for k in range(volume.shape[3]):
            slice_2d = volume[:, :, :, k]  # [1, H, W]

            # Skip empty slices (no tumor)
            if slice_2d.sum() < min_tumor_pixels:
                continue

            # Ensure float32 and binary
            slice_2d = slice_2d.astype(np.float32)
            slice_2d = binarize_mask(slice_2d, threshold=BINARY_THRESHOLD_GT)

            all_slices.append(slice_2d)

    return all_slices


def extract_slices_with_seg_and_mode(
    image_dataset: TorchDataset,
    seg_dataset: TorchDataset,
    mode_id: int,
    augmentation: Callable | None = None,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Extract 2D slices with paired seg masks and mode_id.

    Each slice is returned as a tuple (image, seg, mode_id) where:
    - image: [1, H, W] single-channel image
    - seg: [1, H, W] binary segmentation mask
    - mode_id: int (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd)

    Args:
        image_dataset: Dataset of 3D image volumes with shape [1, H, W, D].
        seg_dataset: Dataset of 3D seg volumes with shape [1, H, W, D].
        mode_id: Integer mode ID for this modality.
        augmentation: Optional albumentations Compose for data augmentation.

    Returns:
        List of tuples (image_slice, seg_slice, mode_id).
    """
    all_slices: list[tuple[np.ndarray, np.ndarray, int]] = []

    if len(image_dataset) != len(seg_dataset):
        raise ValueError(
            f"Image dataset ({len(image_dataset)}) and seg dataset ({len(seg_dataset)}) "
            "must have same number of patients"
        )

    for i in range(len(image_dataset)):
        image_volume, image_name = image_dataset[i]  # Shape: [1, H, W, D]
        seg_volume, seg_name = seg_dataset[i]  # Shape: [1, H, W, D]

        # Convert to numpy if tensor
        if isinstance(image_volume, torch.Tensor):
            image_volume = image_volume.numpy()
        if isinstance(seg_volume, torch.Tensor):
            seg_volume = seg_volume.numpy()

        # Verify same patient
        if image_name != seg_name:
            raise ValueError(f"Patient mismatch: {image_name} vs {seg_name}")

        # Extract non-empty slices along depth dimension (axis 3)
        for k in range(image_volume.shape[3]):
            image_slice = image_volume[:, :, :, k].copy()
            seg_slice = seg_volume[:, :, :, k].copy()

            if np.sum(image_slice) > 1.0:
                if augmentation is not None:
                    # Transpose to [H, W, C] for albumentations
                    img_hwc = np.transpose(image_slice, (1, 2, 0))
                    seg_hwc = np.transpose(seg_slice, (1, 2, 0))

                    transformed = augmentation(image=img_hwc, mask=seg_hwc)
                    img_aug = transformed['image']
                    seg_aug = transformed['mask']

                    # Transpose back to [C, H, W]
                    image_slice = np.transpose(img_aug, (2, 0, 1))
                    seg_slice = np.transpose(seg_aug, (2, 0, 1))

                # Binarize seg mask
                seg_slice = (seg_slice > BINARY_THRESHOLD_GT).astype(np.float32)

                all_slices.append((image_slice, seg_slice, mode_id))

    return all_slices


# =============================================================================
# 2D Diffusion Datasets
# =============================================================================

class SegConditionedDataset(TorchDataset):
    """Dataset wrapper that adds size bin vectors to seg slices.

    Wraps an existing slice dataset and computes size bins on-the-fly.
    Only includes positive slices (slices with at least one tumor).
    Supports classifier-free guidance dropout.

    IMPORTANT: Augmentation is applied LAZILY in __getitem__ to ensure:
    1. Random augmentation per access (different each epoch)
    2. Size bins are computed AFTER augmentation (bins match augmented mask)
    """

    def __init__(
        self,
        slice_dataset: TorchDataset,
        bin_edges: list[float] = None,
        num_bins: int = None,
        fov_mm: float = 240.0,
        image_size: int = 256,
        positive_only: bool = True,
        cfg_dropout_prob: float = 0.0,
        augmentation: Any | None = None,
        return_bin_maps: bool = False,
        max_count: int = 10,
    ):
        """Initialize the dataset wrapper.

        Args:
            slice_dataset: Base dataset returning [1, H, W] seg tensors (NO augmentation applied).
            bin_edges: Size bin edges in mm. Default: RANO-BM aligned bins.
            num_bins: Number of bins. If > len(edges)-1, last bin is overflow.
            fov_mm: Field of view in mm.
            image_size: Image size in pixels.
            positive_only: If True, only include slices with tumors (default: True).
            cfg_dropout_prob: Probability of dropping conditioning for CFG (default: 0.15).
            augmentation: Albumentations Compose for lazy augmentation (applied in __getitem__).
            return_bin_maps: If True, return spatial bin maps for input conditioning.
            max_count: Max count per bin for normalization (default: 10).
        """
        self.slice_dataset = slice_dataset
        self.bin_edges = bin_edges or DEFAULT_BIN_EDGES
        self.pixel_spacing_mm = fov_mm / image_size
        self.image_size = image_size
        self.num_bins = num_bins if num_bins is not None else len(self.bin_edges) - 1
        self.cfg_dropout_prob = cfg_dropout_prob
        self.augmentation = augmentation
        self.return_bin_maps = return_bin_maps
        self.max_count = max_count

        # Filter to positive slices only (on raw data, before augmentation)
        if positive_only:
            self.positive_indices = self._find_positive_indices()
            logger.info(f"Filtered to {len(self.positive_indices)}/{len(slice_dataset)} positive slices")
        else:
            self.positive_indices = list(range(len(slice_dataset)))

    def _find_positive_indices(self) -> list[int]:
        """Find indices of slices with at least one tumor."""
        positive_indices = []
        for idx in range(len(self.slice_dataset)):
            seg = self.slice_dataset[idx]
            if isinstance(seg, torch.Tensor):
                has_tumor = seg.sum() > 0
            else:
                has_tumor = seg.sum() > 0
            if has_tumor:
                positive_indices.append(idx)
        return positive_indices

    def __len__(self) -> int:
        return len(self.positive_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get seg slice and size bins.

        Augmentation is applied HERE (lazily) to ensure:
        1. Fresh random augmentation each access
        2. Size bins computed on augmented mask (conditioning matches output)

        Returns:
            Dict with keys:
                - 'image': [1, H, W] segmentation mask tensor (augmented, binarized)
                - 'size_bins': [num_bins] integer tensor of tumor counts
                - 'bin_maps': [num_bins, H, W] spatial maps (only if return_bin_maps=True)
        """
        # Map to actual dataset index
        actual_idx = self.positive_indices[idx]
        seg = self.slice_dataset[actual_idx]

        # Convert to numpy for augmentation
        if isinstance(seg, torch.Tensor):
            seg_np = seg.numpy()
        else:
            seg_np = seg

        # Apply augmentation BEFORE computing size bins
        # This ensures bins match the augmented mask
        if self.augmentation is not None:
            # seg_np is [1, H, W], need [H, W] for albumentations
            seg_2d = seg_np[0] if seg_np.ndim == 3 else seg_np
            augmented = self.augmentation(image=seg_2d)
            seg_2d = augmented['image']
            # Restore channel dimension [1, H, W]
            seg_np = seg_2d[np.newaxis, :, :] if seg_2d.ndim == 2 else seg_2d

        # Compute size bins on AUGMENTED mask
        size_bins = compute_size_bins(
            seg_np,
            self.bin_edges,
            self.pixel_spacing_mm,
            num_bins=self.num_bins,
        )
        size_bins = torch.from_numpy(size_bins).long()

        # Convert back to tensor
        if not isinstance(seg_np, torch.Tensor):
            seg = torch.from_numpy(seg_np).float()
        else:
            seg = seg_np

        # Ensure [1, H, W] shape
        if seg.dim() == 2:
            seg = seg.unsqueeze(0)

        # Classifier-free guidance dropout: randomly zero out conditioning
        is_dropout = self.cfg_dropout_prob > 0 and torch.rand(1).item() < self.cfg_dropout_prob
        if is_dropout:
            size_bins = torch.zeros_like(size_bins)

        result = {
            'image': seg,
            'size_bins': size_bins,
        }

        if self.return_bin_maps:
            # Create spatial bin maps for input conditioning
            spatial_shape = (self.image_size, self.image_size)
            if is_dropout:
                # Dropout: zero maps
                bin_maps = torch.zeros(self.num_bins, *spatial_shape, dtype=torch.float32)
            else:
                bin_maps = create_size_bin_maps(
                    size_bins, spatial_shape, normalize=True, max_count=self.max_count
                )
            result['bin_maps'] = bin_maps

        return result


class MultiDiffusionDataset(TorchDataset):
    """Dataset that returns dict with image, seg, mode_id.

    Wraps a list of pre-extracted slices for efficient training.
    Used for multi-modality diffusion with mode embedding.
    """

    def __init__(self, samples: list[tuple[np.ndarray, np.ndarray, int]]):
        """Initialize dataset.

        Args:
            samples: List of (image, seg, mode_id) tuples.
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> dict[str, Any]:
        image, seg, mode_id = self.samples[idx]
        return {
            'image': torch.from_numpy(image).float(),
            'seg': torch.from_numpy(seg).float(),
            'mode_id': torch.tensor(mode_id, dtype=torch.long),
        }


class AugmentedSegDataset(TorchDataset):
    """Dataset wrapper that applies augmentation on-the-fly during __getitem__.

    This ensures augmentation happens during training iteration, not dataloader
    creation, providing both fast startup and training variety.

    Used for DC-AE seg mask compression training.
    """

    def __init__(
        self,
        slices: list[np.ndarray],
        augmentation: Callable | None = None,
    ):
        """Initialize dataset with pre-extracted slices.

        Args:
            slices: List of 2D mask slices [1, H, W].
            augmentation: Optional albumentations transform.
        """
        self.slices = slices
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        from medgen.augmentation import apply_augmentation, binarize_mask

        slice_2d = self.slices[idx].copy()  # Avoid modifying original

        # Apply augmentation during iteration (different each epoch)
        if self.augmentation is not None:
            slice_2d = apply_augmentation(slice_2d, self.augmentation, has_mask=False)

        # CRITICAL: Binarize after augmentation
        slice_2d = binarize_mask(slice_2d, threshold=BINARY_THRESHOLD_GT)

        return torch.from_numpy(slice_2d).float()


# =============================================================================
# 3D Volume Datasets (for VAE volume-level metrics)
# =============================================================================

class VolumeDataset(TorchDataset):
    """Dataset wrapper for returning full 3D volumes.

    Used for volume-level metrics like 3D MS-SSIM. Returns volumes
    without slice extraction, optionally with segmentation masks.
    """

    def __init__(
        self,
        image_dataset: TorchDataset,
        seg_dataset: TorchDataset | None = None,
    ) -> None:
        """Initialize volume dataset.

        Args:
            image_dataset: Dataset of image volumes.
            seg_dataset: Optional dataset of segmentation masks.
        """
        self.image_dataset = image_dataset
        self.seg_dataset = seg_dataset

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, idx: int) -> dict:
        """Get volume and optional segmentation.

        Returns:
            Dict with 'image' key and optional 'seg' key.
            image: [C, H, W, D] tensor
            seg: [1, H, W, D] tensor (if available)
        """
        image, patient = self.image_dataset[idx]
        result = {'image': image, 'patient': patient}

        if self.seg_dataset is not None:
            seg, _ = self.seg_dataset[idx]
            result['seg'] = seg

        return result


class DualVolumeDataset(TorchDataset):
    """Dataset wrapper for dual-modality 3D volumes.

    Stacks t1_pre and t1_gd into 2-channel volumes.
    Used for volume-level metrics like 3D MS-SSIM.
    """

    def __init__(
        self,
        t1_pre_dataset: TorchDataset,
        t1_gd_dataset: TorchDataset,
        seg_dataset: TorchDataset | None = None,
    ) -> None:
        """Initialize dual volume dataset.

        Args:
            t1_pre_dataset: Dataset of t1_pre volumes.
            t1_gd_dataset: Dataset of t1_gd volumes.
            seg_dataset: Optional dataset of segmentation masks.
        """
        self.t1_pre_dataset = t1_pre_dataset
        self.t1_gd_dataset = t1_gd_dataset
        self.seg_dataset = seg_dataset

    def __len__(self) -> int:
        return len(self.t1_pre_dataset)

    def __getitem__(self, idx: int) -> dict:
        """Get stacked dual-modality volume.

        Returns:
            Dict with 'image' key (2 channels) and optional 'seg' key.
            image: [2, H, W, D] tensor (t1_pre, t1_gd stacked)
            seg: [1, H, W, D] tensor (if available)
        """
        t1_pre, patient = self.t1_pre_dataset[idx]
        t1_gd, _ = self.t1_gd_dataset[idx]

        # Stack: [1, H, W, D] + [1, H, W, D] -> [2, H, W, D]
        image = torch.cat([t1_pre, t1_gd], dim=0)
        result = {'image': image, 'patient': patient}

        if self.seg_dataset is not None:
            seg, _ = self.seg_dataset[idx]
            result['seg'] = seg

        return result
