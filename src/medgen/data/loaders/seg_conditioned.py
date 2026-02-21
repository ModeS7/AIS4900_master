"""
Dataloader for segmentation masks with tumor size conditioning.

Provides dataloaders that return (seg_mask, size_bins) tuples where size_bins
is a 9-dimensional vector of tumor counts per size bin.
"""
import logging
import os
from typing import Any

import numpy as np
import torch
from monai.data import DataLoader
from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset

from medgen.augmentation import build_seg_diffusion_augmentation_with_binarize
from medgen.data.dataset import NiFTIDataset, build_standard_transform, validate_modality_exists
from medgen.data.loaders.common import DataLoaderConfig, setup_distributed_sampler
from medgen.data.loaders.datasets import (
    DEFAULT_BIN_EDGES,
    compute_size_bins,
    create_size_bin_maps,
)
from medgen.data.utils import extract_slices_single

logger = logging.getLogger(__name__)

# Default CFG dropout probability for input conditioning
DEFAULT_CFG_DROPOUT = 0.15


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

    def __getitem__(self, idx: int):
        """Get seg slice and size bins.

        Augmentation is applied HERE (lazily) to ensure:
        1. Fresh random augmentation each access
        2. Size bins computed on augmented mask (conditioning matches output)

        Returns:
            If return_bin_maps=False:
                Tuple of (seg, size_bins)
            If return_bin_maps=True:
                Tuple of (seg, size_bins, bin_maps)

            Where:
                - seg: [1, H, W] segmentation mask tensor (augmented, binarized)
                - size_bins: [num_bins] integer tensor of tumor counts
                - bin_maps: [num_bins, H, W] spatial maps for input conditioning
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

        if not self.return_bin_maps:
            return seg, size_bins

        # Create spatial bin maps for input conditioning
        spatial_shape = (self.image_size, self.image_size)
        if is_dropout:
            # Dropout: zero maps
            bin_maps = torch.zeros(self.num_bins, *spatial_shape, dtype=torch.float32)
        else:
            bin_maps = create_size_bin_maps(
                size_bins, spatial_shape, normalize=True, max_count=self.max_count
            )

        return seg, size_bins, bin_maps


def create_seg_conditioned_dataloader(
    cfg: DictConfig,
    size_bin_config: dict | None = None,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
) -> tuple[DataLoader, TorchDataset]:
    """Create dataloader for size-conditioned segmentation training.

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        size_bin_config: Optional override for size bin settings. Keys:
            - edges: List of bin edges in mm
            - num_bins: Number of bins
            - fov_mm: Field of view in mm
            - return_bin_maps: If True, return spatial bin maps (for input conditioning)
            - cfg_dropout_prob: CFG dropout probability
            - max_count: Max count per bin for normalization
        use_distributed: Whether to use distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.
        augment: Whether to apply data augmentation.

    Returns:
        Tuple of (DataLoader, train_dataset).
    """
    data_dir = os.path.join(cfg.paths.data_dir, "train")
    image_size = cfg.model.image_size
    batch_size = cfg.training.batch_size

    # Get size bin config from mode config, with optional override
    # Convert OmegaConf to Python types to avoid recursion in MONAI's set_rnd
    size_bin_cfg = cfg.mode.get('size_bins', {})
    if size_bin_config:
        # Merge override config
        size_bin_cfg = {**dict(size_bin_cfg), **size_bin_config}

    bin_edges = list(size_bin_cfg.get('edges', DEFAULT_BIN_EDGES))
    num_bins = int(size_bin_cfg.get('num_bins', len(bin_edges) - 1))
    fov_mm = float(size_bin_cfg.get('fov_mm', 240.0))

    # Input conditioning options
    return_bin_maps = bool(size_bin_cfg.get('return_bin_maps', False))
    max_count = int(size_bin_cfg.get('max_count', 10))

    # Classifier-free guidance dropout (only for training)
    # Default: 0.0 for FiLM mode, 0.15 for input conditioning mode
    cfg_dropout_prob = float(size_bin_cfg.get('cfg_dropout_prob', cfg.mode.get('cfg_dropout_prob', 0.0)))

    validate_modality_exists(data_dir, 'seg')

    transform = build_standard_transform(image_size)

    # Load seg dataset and extract slices (NO augmentation here - applied lazily)
    seg_dataset = NiFTIDataset(
        data_dir=data_dir, mr_sequence="seg", transform=transform
    )
    slice_dataset = extract_slices_single(seg_dataset, augmentation=None)

    # Build augmentation for lazy application in __getitem__
    # This ensures: 1) random aug per access, 2) size bins computed on augmented mask
    # Transforms: H/V flip, ±15° rotation, ±5% translation, 0.9-1.1x scale, mild elastic
    # Includes 0.5 threshold binarization to keep mask binary
    aug = build_seg_diffusion_augmentation_with_binarize(enabled=augment)

    # Wrap with size bin computation
    # positive_only=True: train only on slices with tumors
    # cfg_dropout_prob: randomly drop conditioning for classifier-free guidance
    # augmentation: applied lazily so bins are computed on augmented mask
    # return_bin_maps: if True, return spatial bin maps for input conditioning
    train_dataset = SegConditionedDataset(
        slice_dataset,
        bin_edges=bin_edges,
        num_bins=num_bins,
        fov_mm=fov_mm,
        image_size=image_size,
        positive_only=True,
        cfg_dropout_prob=cfg_dropout_prob,
        augmentation=aug,
        return_bin_maps=return_bin_maps,
        max_count=max_count,
    )

    # Setup distributed sampler
    sampler, batch_size_per_gpu, shuffle = setup_distributed_sampler(
        train_dataset, use_distributed, rank, world_size, batch_size, shuffle=True
    )

    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=shuffle,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    return dataloader, train_dataset


def create_seg_conditioned_validation_dataloader(
    cfg: DictConfig,
    size_bin_config: dict | None = None,
    batch_size: int | None = None,
    world_size: int = 1,
) -> tuple[DataLoader, TorchDataset] | None:
    """Create validation dataloader for size-conditioned segmentation.

    Args:
        cfg: Hydra configuration.
        size_bin_config: Optional override for size bin settings.
        batch_size: Optional batch size override.
        world_size: Number of GPUs for DDP.

    Returns:
        Tuple of (DataLoader, val_dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, "val")

    if not os.path.exists(val_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    if world_size > 1:
        batch_size = max(1, batch_size // world_size)

    # Get size bin config (convert OmegaConf to Python types)
    size_bin_cfg = cfg.mode.get('size_bins', {})
    if size_bin_config:
        size_bin_cfg = {**dict(size_bin_cfg), **size_bin_config}

    bin_edges = list(size_bin_cfg.get('edges', DEFAULT_BIN_EDGES))
    num_bins = int(size_bin_cfg.get('num_bins', len(bin_edges) - 1))
    fov_mm = float(size_bin_cfg.get('fov_mm', 240.0))
    return_bin_maps = bool(size_bin_cfg.get('return_bin_maps', False))
    max_count = int(size_bin_cfg.get('max_count', 10))

    try:
        validate_modality_exists(val_dir, 'seg')
    except ValueError as e:
        logger.warning(f"Validation directory misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    seg_dataset = NiFTIDataset(
        data_dir=val_dir, mr_sequence="seg", transform=transform
    )
    slice_dataset = extract_slices_single(seg_dataset)

    # positive_only=False: evaluate on all slices (including empty ones)
    # cfg_dropout_prob=0: no dropout during evaluation
    val_dataset = SegConditionedDataset(
        slice_dataset,
        bin_edges=bin_edges,
        num_bins=num_bins,
        fov_mm=fov_mm,
        image_size=image_size,
        positive_only=False,
        cfg_dropout_prob=0.0,
        return_bin_maps=return_bin_maps,
        max_count=max_count,
    )

    dl_cfg = DataLoaderConfig.from_cfg(cfg)
    val_generator = torch.Generator().manual_seed(42)

    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=val_generator,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    return dataloader, val_dataset


def create_seg_conditioned_test_dataloader(
    cfg: DictConfig,
    batch_size: int | None = None,
) -> tuple[DataLoader, TorchDataset] | None:
    """Create test dataloader for size-conditioned segmentation.

    Args:
        cfg: Hydra configuration.
        batch_size: Optional batch size override.

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, "test_new")

    if not os.path.exists(test_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Get size bin config (convert OmegaConf to Python types)
    size_bin_cfg = cfg.mode.get('size_bins', {})
    bin_edges = list(size_bin_cfg.get('edges', DEFAULT_BIN_EDGES))
    num_bins = int(size_bin_cfg.get('num_bins', len(bin_edges) - 1))
    fov_mm = float(size_bin_cfg.get('fov_mm', 240.0))

    try:
        validate_modality_exists(test_dir, 'seg')
    except ValueError as e:
        logger.warning(f"Test directory misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    seg_dataset = NiFTIDataset(
        data_dir=test_dir, mr_sequence="seg", transform=transform
    )
    slice_dataset = extract_slices_single(seg_dataset)

    # positive_only=False: evaluate on all slices (including empty ones)
    # cfg_dropout_prob=0: no dropout during evaluation
    test_dataset = SegConditionedDataset(
        slice_dataset,
        bin_edges=bin_edges,
        num_bins=num_bins,
        fov_mm=fov_mm,
        image_size=image_size,
        positive_only=False,
        cfg_dropout_prob=0.0,
    )

    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    return dataloader, test_dataset
