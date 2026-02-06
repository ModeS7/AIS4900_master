"""
3D Dataloader for segmentation volumes with tumor size conditioning.

Provides dataloaders that return (seg_volume, size_bins) tuples where size_bins
is a N-dimensional vector of tumor counts per size bin.

Key difference from 2D: Uses 3D connected components so tumors touching
in ANY direction (including depth) count as ONE tumor.

NOTE: Utility functions have been consolidated into datasets.py.
This file now imports from there for backward compatibility.
"""
import logging
import os
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

# Import consolidated functions from datasets.py
from medgen.data.loaders.datasets import (
    DEFAULT_BIN_EDGES,
    compute_feret_diameter_3d,
    compute_size_bins_3d,
    create_size_bin_maps,
)

from .common import get_validated_split_dir
from .volume_3d import (
    VolumeConfig,
    _create_loader,
    build_3d_augmentation,
    build_3d_transform,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Size Bin Config Extraction
# =============================================================================

class SegSizeBinConfig:
    """Extracted size bin configuration for segmentation dataloaders.

    Eliminates repeated cfg.mode.get('size_bins', ...) extraction across
    train/val/test loader functions.
    """
    __slots__ = ('bin_edges', 'num_bins', 'voxel_spacing', 'return_bin_maps', 'max_count')

    def __init__(
        self,
        bin_edges: list[float],
        num_bins: int,
        voxel_spacing: tuple[float, ...],
        return_bin_maps: bool,
        max_count: int,
    ) -> None:
        self.bin_edges = bin_edges
        self.num_bins = num_bins
        self.voxel_spacing = voxel_spacing
        self.return_bin_maps = return_bin_maps
        self.max_count = max_count

    @classmethod
    def from_cfg(cls, cfg, vcfg: VolumeConfig) -> 'SegSizeBinConfig':
        """Extract size bin config from Hydra DictConfig.

        Args:
            cfg: Hydra config with mode.size_bins section.
            vcfg: Volume configuration (for pixel spacing defaults).
        """
        size_bins_cfg = cfg.mode.get('size_bins', None)
        if size_bins_cfg is None:
            size_bins_cfg = {}
        elif hasattr(size_bins_cfg, 'to_container'):
            from omegaconf import OmegaConf
            size_bins_cfg = OmegaConf.to_container(size_bins_cfg, resolve=True)

        bin_edges = list(size_bins_cfg.get('edges', DEFAULT_BIN_EDGES))
        num_bins = int(size_bins_cfg.get('num_bins', len(bin_edges) - 1))
        default_pixel_spacing = 240.0 / vcfg.height
        voxel_spacing_cfg = size_bins_cfg.get(
            'voxel_spacing', [1.0, default_pixel_spacing, default_pixel_spacing]
        )
        voxel_spacing = tuple(float(v) for v in voxel_spacing_cfg)
        return_bin_maps = bool(size_bins_cfg.get('return_bin_maps', False))
        max_count = int(size_bins_cfg.get('max_count', 10))

        return cls(
            bin_edges=bin_edges,
            num_bins=num_bins,
            voxel_spacing=voxel_spacing,
            return_bin_maps=return_bin_maps,
            max_count=max_count,
        )


# Re-export for backward compatibility
__all__ = [
    'SegDataset',
    'compute_size_bins_3d',
    'compute_feret_diameter_3d',
    'create_size_bin_maps',
    'DEFAULT_BIN_EDGES',
    'create_seg_dataloader',
    'create_seg_validation_dataloader',
    'create_seg_test_dataloader',
]


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
        bin_edges: list[float],
        num_bins: int,
        voxel_spacing: tuple[float, float, float],
        height: int = 256,
        width: int = 256,
        pad_depth_to: int = 160,
        pad_mode: str = 'replicate',
        slice_step: int = 1,
        positive_only: bool = True,
        cfg_dropout_prob: float = 0.0,
        augmentation: Callable | None = None,
        return_bin_maps: bool = False,
        max_count: int = 10,
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
            return_bin_maps: If True, return spatial bin maps for input conditioning.
            max_count: Max count per bin for normalization (default: 10).
        """
        # Validate data directory exists
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"Data directory not found: {data_dir}")

        # Validate parameter ranges
        if num_bins <= 0:
            raise ValueError(f"num_bins must be > 0, got {num_bins}")
        if height <= 0 or width <= 0:
            raise ValueError(f"height and width must be > 0, got height={height}, width={width}")
        if pad_depth_to <= 0:
            raise ValueError(f"pad_depth_to must be > 0, got {pad_depth_to}")
        if slice_step <= 0:
            raise ValueError(f"slice_step must be > 0, got {slice_step}")
        if not (0 <= cfg_dropout_prob <= 1):
            raise ValueError(f"cfg_dropout_prob must be in [0, 1], got {cfg_dropout_prob}")
        if max_count <= 0:
            raise ValueError(f"max_count must be > 0, got {max_count}")
        if not all(v > 0 for v in voxel_spacing):
            raise ValueError(f"All voxel_spacing values must be > 0, got {voxel_spacing}")
        if bin_edges != sorted(bin_edges):
            raise ValueError(f"bin_edges must be sorted ascending, got {bin_edges}")

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
        self.return_bin_maps = return_bin_maps
        self.max_count = max_count

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

    def _find_positive_patients(self) -> list[str]:
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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get segmentation volume and size bins.

        Returns:
            Dict with keys:
                - 'image': [1, D, H, W] binary segmentation tensor
                - 'size_bins': [num_bins] integer tensor of tumor counts
                  (zeros if CFG dropout is applied)
                - 'bin_maps': [num_bins, D, H, W] spatial maps (only if return_bin_maps=True)
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
            aug_result = self.augmentation({'image': seg_volume})
            seg_volume = aug_result['image'].contiguous()
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
        is_dropout = self.cfg_dropout_prob > 0 and torch.rand(1).item() < self.cfg_dropout_prob
        if is_dropout:
            size_bins = torch.zeros_like(size_bins)

        result = {
            'image': seg_volume,
            'size_bins': size_bins,
        }

        if self.return_bin_maps:
            # Create spatial bin maps for input conditioning [num_bins, D, H, W]
            spatial_shape = seg_volume.shape[1:]  # (D, H, W)
            if is_dropout:
                # Dropout: zero maps for unconditional
                bin_maps = torch.zeros(self.num_bins, *spatial_shape, dtype=torch.float32)
            else:
                bin_maps = create_size_bin_maps(
                    size_bins, spatial_shape, normalize=True, max_count=self.max_count
                )
            result['bin_maps'] = bin_maps

        return result


def create_seg_dataloader(
    cfg,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, TorchDataset]:
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

    # Extract size bin config (typed)
    sbc = SegSizeBinConfig.from_cfg(cfg, vcfg)

    # CFG dropout
    cfg_dropout_prob = float(cfg.mode.get('cfg_dropout_prob', 0.0))

    # Check if augmentation is enabled
    augment = getattr(cfg.training, 'augment', False)
    aug = build_3d_augmentation(seg_mode=True) if augment else None

    train_dataset = SegDataset(
        data_dir=data_dir,
        bin_edges=sbc.bin_edges,
        num_bins=sbc.num_bins,
        voxel_spacing=sbc.voxel_spacing,
        height=vcfg.train_height,
        width=vcfg.train_width,
        pad_depth_to=vcfg.pad_depth_to,
        pad_mode=vcfg.pad_mode,
        slice_step=vcfg.slice_step,
        positive_only=True,
        cfg_dropout_prob=cfg_dropout_prob,
        augmentation=aug,
        return_bin_maps=sbc.return_bin_maps,
        max_count=sbc.max_count,
    )

    logger.info(
        f"Created 3D seg dataset: {len(train_dataset)} volumes, "
        f"{sbc.num_bins} bins, voxel_spacing={sbc.voxel_spacing}"
    )

    loader = _create_loader(
        train_dataset, vcfg, shuffle=True, drop_last=True,
        use_distributed=use_distributed, rank=rank, world_size=world_size
    )

    return loader, train_dataset


def create_seg_validation_dataloader(
    cfg,
) -> tuple[DataLoader, TorchDataset] | None:
    """Create 3D validation dataloader for size-conditioned segmentation.

    Args:
        cfg: Hydra configuration.

    Returns:
        Tuple of (DataLoader, val_dataset) or None if val/ doesn't exist.
    """
    val_dir = get_validated_split_dir(cfg.paths.data_dir, 'val', logger)
    if val_dir is None:
        return None

    vcfg = VolumeConfig.from_cfg(cfg)
    sbc = SegSizeBinConfig.from_cfg(cfg, vcfg)

    val_dataset = SegDataset(
        data_dir=val_dir,
        bin_edges=sbc.bin_edges,
        num_bins=sbc.num_bins,
        voxel_spacing=sbc.voxel_spacing,
        height=vcfg.height,
        width=vcfg.width,
        pad_depth_to=vcfg.pad_depth_to,
        pad_mode=vcfg.pad_mode,
        slice_step=vcfg.slice_step,
        positive_only=False,  # Evaluate on all volumes
        cfg_dropout_prob=0.0,  # No dropout during validation
        augmentation=None,
        return_bin_maps=sbc.return_bin_maps,
        max_count=sbc.max_count,
    )

    loader = _create_loader(val_dataset, vcfg, shuffle=False)
    return loader, val_dataset


def create_seg_test_dataloader(
    cfg,
) -> tuple[DataLoader, TorchDataset] | None:
    """Create 3D test dataloader for size-conditioned segmentation.

    Args:
        cfg: Hydra configuration.

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = get_validated_split_dir(cfg.paths.data_dir, 'test_new', logger)
    if test_dir is None:
        return None

    vcfg = VolumeConfig.from_cfg(cfg)
    sbc = SegSizeBinConfig.from_cfg(cfg, vcfg)

    test_dataset = SegDataset(
        data_dir=test_dir,
        bin_edges=sbc.bin_edges,
        num_bins=sbc.num_bins,
        voxel_spacing=sbc.voxel_spacing,
        height=vcfg.height,
        width=vcfg.width,
        pad_depth_to=vcfg.pad_depth_to,
        pad_mode=vcfg.pad_mode,
        slice_step=vcfg.slice_step,
        positive_only=False,
        cfg_dropout_prob=0.0,
        augmentation=None,
        return_bin_maps=sbc.return_bin_maps,
        max_count=sbc.max_count,
    )

    loader = _create_loader(test_dataset, vcfg, shuffle=False)
    return loader, test_dataset
