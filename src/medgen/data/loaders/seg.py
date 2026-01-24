"""
3D Dataloader for segmentation volumes with tumor size conditioning.

Provides dataloaders that return (seg_volume, size_bins) tuples where size_bins
is a N-dimensional vector of tumor counts per size bin.

Key difference from 2D: Uses 3D connected components so tumors touching
in ANY direction (including depth) count as ONE tumor.
"""
import logging
import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader, Dataset as TorchDataset

from .volume_3d import (
    VolumeConfig,
    build_3d_transform,
    build_3d_augmentation,
    _create_loader,
)

logger = logging.getLogger(__name__)

# Default size bins (mm) - aligned with RANO-BM thresholds (10, 20, 30mm)
# Same as 2D for consistency
DEFAULT_BIN_EDGES = [0, 3, 6, 10, 15, 20, 30]


def compute_feret_diameter_3d(
    binary_mask: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
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
    bin_edges: List[float],
    voxel_spacing: Tuple[float, float, float],
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


class SegDataset(TorchDataset):
    """3D Dataset that loads segmentation volumes with size bin conditioning.

    Wraps Base3DVolumeDataset to add size bin computation on full 3D volumes.
    Supports classifier-free guidance dropout.

    Key difference from 2D: Computes bins on full 3D volume so tumors touching
    in depth are correctly counted as single tumors.
    """

    def __init__(
        self,
        data_dir: str,
        bin_edges: List[float],
        num_bins: int,
        voxel_spacing: Tuple[float, float, float],
        height: int = 256,
        width: int = 256,
        pad_depth_to: int = 160,
        pad_mode: str = 'replicate',
        slice_step: int = 1,
        positive_only: bool = True,
        cfg_dropout_prob: float = 0.0,
        augmentation: Optional[Callable] = None,
    ):
        """Initialize the 3D seg conditioned dataset.

        Args:
            data_dir: Directory containing patient subdirectories with seg.nii.gz.
            bin_edges: Size bin edges in mm.
            num_bins: Number of bins (last bin is overflow for >= last edge).
            voxel_spacing: Voxel size in mm as (depth_mm, height_mm, width_mm).
            height: Target height dimension.
            width: Target width dimension.
            pad_depth_to: Target depth after padding.
            pad_mode: Padding mode ('replicate' or 'constant').
            slice_step: Take every nth slice (1=all).
            positive_only: If True, only include volumes with tumors.
            cfg_dropout_prob: Probability of dropping conditioning for CFG.
            augmentation: Optional MONAI augmentation transform.
        """
        self.data_dir = data_dir
        self.bin_edges = bin_edges
        self.num_bins = num_bins
        self.voxel_spacing = voxel_spacing
        self.height = height
        self.width = width
        self.pad_depth_to = pad_depth_to
        self.pad_mode = pad_mode
        self.slice_step = slice_step
        self.cfg_dropout_prob = cfg_dropout_prob
        self.augmentation = augmentation

        # Build transform for loading
        self.transform = build_3d_transform(height, width)

        # Find all patients with seg masks
        self.patients = []
        for p in sorted(os.listdir(data_dir)):
            patient_dir = os.path.join(data_dir, p)
            if os.path.isdir(patient_dir):
                seg_path = os.path.join(patient_dir, "seg.nii.gz")
                if os.path.exists(seg_path):
                    self.patients.append(p)

        if not self.patients:
            raise ValueError(f"No patients with seg.nii.gz found in {data_dir}")

        logger.info(f"Found {len(self.patients)} patients with seg masks")

        # Filter to positive volumes only if requested
        if positive_only:
            self.positive_patients = self._find_positive_patients()
            logger.info(f"Filtered to {len(self.positive_patients)}/{len(self.patients)} positive volumes")
        else:
            self.positive_patients = self.patients

    def _find_positive_patients(self) -> List[str]:
        """Find patients with at least one tumor voxel."""
        positive = []
        for patient in self.patients:
            seg_path = os.path.join(self.data_dir, patient, "seg.nii.gz")
            # Quick check: load and check sum > 0
            seg = self.transform(seg_path)
            if not isinstance(seg, torch.Tensor):
                seg = torch.from_numpy(seg)
            if seg.sum() > 0:
                positive.append(patient)
        return positive

    def _load_volume(self, nifti_path: str) -> torch.Tensor:
        """Load and preprocess a 3D volume.

        Args:
            nifti_path: Path to NIfTI file.

        Returns:
            Tensor of shape [1, D, H, W].
        """
        import torch.nn.functional as F

        volume = self.transform(nifti_path)

        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).float()

        # MONAI loads as [C, H, W, D], we need [C, D, H, W] for 3D conv
        volume = volume.permute(0, 3, 1, 2)

        # Subsample slices if slice_step > 1
        if self.slice_step > 1:
            volume = volume[:, ::self.slice_step, :, :]

        # Pad depth
        current_depth = volume.shape[1]
        if current_depth < self.pad_depth_to:
            pad_total = self.pad_depth_to - current_depth
            if self.pad_mode == 'replicate':
                last_slice = volume[:, -1:, :, :]
                padding = last_slice.repeat(1, pad_total, 1, 1)
                volume = torch.cat([volume, padding], dim=1)
            else:
                volume = F.pad(volume, (0, 0, 0, 0, 0, pad_total), mode='constant', value=0)

        return volume

    def __len__(self) -> int:
        return len(self.positive_patients)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get segmentation volume and size bins.

        Returns:
            Tuple of:
                - seg_volume: [1, D, H, W] binary segmentation tensor
                - size_bins: [num_bins] integer tensor of tumor counts
                  (zeros if CFG dropout is applied)
        """
        patient = self.positive_patients[idx]
        seg_path = os.path.join(self.data_dir, patient, "seg.nii.gz")

        # Load volume
        seg_volume = self._load_volume(seg_path)

        # Binarize
        seg_volume = (seg_volume > 0.5).float()

        # Apply augmentation if configured
        if self.augmentation is not None:
            # MONAI transforms expect dict format
            result = self.augmentation({'image': seg_volume})
            seg_volume = result['image'].contiguous()
            # Re-binarize after augmentation (some transforms may interpolate)
            seg_volume = (seg_volume > 0.5).float()

        # Compute 3D size bins on the full volume
        size_bins = compute_size_bins_3d(
            seg_volume,
            self.bin_edges,
            self.voxel_spacing,
            num_bins=self.num_bins,
        )
        size_bins = torch.from_numpy(size_bins).long()

        # Classifier-free guidance dropout: randomly zero out conditioning
        if self.cfg_dropout_prob > 0 and torch.rand(1).item() < self.cfg_dropout_prob:
            size_bins = torch.zeros_like(size_bins)

        return seg_volume, size_bins


def create_seg_dataloader(
    cfg,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, TorchDataset]:
    """Create 3D dataloader for size-conditioned segmentation training.

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        use_distributed: Whether to use distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes.

    Returns:
        Tuple of (DataLoader, train_dataset).
    """
    vcfg = VolumeConfig.from_cfg(cfg)
    data_dir = os.path.join(cfg.paths.data_dir, 'train')

    # Get size bin config from mode config
    size_bin_cfg = cfg.mode.get('size_bins', {})
    bin_edges = list(size_bin_cfg.get('edges', DEFAULT_BIN_EDGES))
    num_bins = int(size_bin_cfg.get('num_bins', len(bin_edges) - 1))

    # Voxel spacing for physical measurements
    # Default: 1mm depth, pixel_spacing for H/W
    default_pixel_spacing = 240.0 / vcfg.height  # fov_mm / image_size
    voxel_spacing_cfg = size_bin_cfg.get('voxel_spacing', [1.0, default_pixel_spacing, default_pixel_spacing])
    voxel_spacing = tuple(float(v) for v in voxel_spacing_cfg)

    # CFG dropout
    cfg_dropout_prob = float(cfg.mode.get('cfg_dropout_prob', 0.0))

    # Check if augmentation is enabled
    augment = getattr(cfg.training, 'augment', False)
    aug = build_3d_augmentation(seg_mode=True) if augment else None

    train_dataset = SegDataset(
        data_dir=data_dir,
        bin_edges=bin_edges,
        num_bins=num_bins,
        voxel_spacing=voxel_spacing,
        height=vcfg.train_height,
        width=vcfg.train_width,
        pad_depth_to=vcfg.pad_depth_to,
        pad_mode=vcfg.pad_mode,
        slice_step=vcfg.slice_step,
        positive_only=True,
        cfg_dropout_prob=cfg_dropout_prob,
        augmentation=aug,
    )

    logger.info(
        f"Created 3D seg dataset: {len(train_dataset)} volumes, "
        f"{num_bins} bins, voxel_spacing={voxel_spacing}"
    )

    loader = _create_loader(
        train_dataset, vcfg, shuffle=True, drop_last=True,
        use_distributed=use_distributed, rank=rank, world_size=world_size
    )

    return loader, train_dataset


def create_seg_validation_dataloader(
    cfg,
) -> Optional[Tuple[DataLoader, TorchDataset]]:
    """Create 3D validation dataloader for size-conditioned segmentation.

    Args:
        cfg: Hydra configuration.

    Returns:
        Tuple of (DataLoader, val_dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, 'val')
    if not os.path.exists(val_dir):
        return None

    vcfg = VolumeConfig.from_cfg(cfg)

    # Get size bin config
    size_bin_cfg = cfg.mode.get('size_bins', {})
    bin_edges = list(size_bin_cfg.get('edges', DEFAULT_BIN_EDGES))
    num_bins = int(size_bin_cfg.get('num_bins', len(bin_edges) - 1))

    default_pixel_spacing = 240.0 / vcfg.height
    voxel_spacing_cfg = size_bin_cfg.get('voxel_spacing', [1.0, default_pixel_spacing, default_pixel_spacing])
    voxel_spacing = tuple(float(v) for v in voxel_spacing_cfg)

    val_dataset = SegDataset(
        data_dir=val_dir,
        bin_edges=bin_edges,
        num_bins=num_bins,
        voxel_spacing=voxel_spacing,
        height=vcfg.height,
        width=vcfg.width,
        pad_depth_to=vcfg.pad_depth_to,
        pad_mode=vcfg.pad_mode,
        slice_step=vcfg.slice_step,
        positive_only=False,  # Evaluate on all volumes
        cfg_dropout_prob=0.0,  # No dropout during validation
        augmentation=None,
    )

    loader = _create_loader(val_dataset, vcfg, shuffle=False)
    return loader, val_dataset


def create_seg_test_dataloader(
    cfg,
) -> Optional[Tuple[DataLoader, TorchDataset]]:
    """Create 3D test dataloader for size-conditioned segmentation.

    Args:
        cfg: Hydra configuration.

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, 'test_new')
    if not os.path.exists(test_dir):
        return None

    vcfg = VolumeConfig.from_cfg(cfg)

    # Get size bin config
    size_bin_cfg = cfg.mode.get('size_bins', {})
    bin_edges = list(size_bin_cfg.get('edges', DEFAULT_BIN_EDGES))
    num_bins = int(size_bin_cfg.get('num_bins', len(bin_edges) - 1))

    default_pixel_spacing = 240.0 / vcfg.height
    voxel_spacing_cfg = size_bin_cfg.get('voxel_spacing', [1.0, default_pixel_spacing, default_pixel_spacing])
    voxel_spacing = tuple(float(v) for v in voxel_spacing_cfg)

    test_dataset = SegDataset(
        data_dir=test_dir,
        bin_edges=bin_edges,
        num_bins=num_bins,
        voxel_spacing=voxel_spacing,
        height=vcfg.height,
        width=vcfg.width,
        pad_depth_to=vcfg.pad_depth_to,
        pad_mode=vcfg.pad_mode,
        slice_step=vcfg.slice_step,
        positive_only=False,
        cfg_dropout_prob=0.0,
        augmentation=None,
    )

    loader = _create_loader(test_dataset, vcfg, shuffle=False)
    return loader, test_dataset
