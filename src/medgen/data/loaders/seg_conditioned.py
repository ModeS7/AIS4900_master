"""
Dataloader for segmentation masks with tumor size conditioning.

Provides dataloaders that return (seg_mask, size_bins) tuples where size_bins
is a 9-dimensional vector of tumor counts per size bin.

NOTE: Dataset classes and utility functions have been consolidated into datasets.py.
This file now imports from there for backward compatibility.
"""
import logging
import os
from typing import Optional, Tuple

import torch
from monai.data import DataLoader
from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset

from medgen.augmentation import build_seg_diffusion_augmentation_with_binarize
from medgen.data.loaders.common import (
    DistributedArgs,
    create_dataloader,
    validate_mode_requirements,
)
from medgen.data.dataset import NiFTIDataset, build_standard_transform, validate_modality_exists
from medgen.data.utils import extract_slices_single

# Import consolidated classes and functions from datasets.py
from medgen.data.loaders.datasets import (
    SegConditionedDataset,
    compute_size_bins,
    compute_feret_diameter,
    create_size_bin_maps,
    DEFAULT_BIN_EDGES,
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    'SegConditionedDataset',
    'compute_size_bins',
    'compute_feret_diameter',
    'create_size_bin_maps',
    'DEFAULT_BIN_EDGES',
    'create_seg_conditioned_dataloader',
    'create_seg_conditioned_validation_dataloader',
    'create_seg_conditioned_test_dataloader',
]

# Default CFG dropout probability for input conditioning
DEFAULT_CFG_DROPOUT = 0.15


def create_seg_conditioned_dataloader(
    cfg: DictConfig,
    size_bin_config: Optional[dict] = None,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
) -> Tuple[DataLoader, TorchDataset]:
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

    validate_mode_requirements(data_dir, 'seg_conditioned', validate_modality_exists)

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

    dataloader = create_dataloader(
        train_dataset,
        cfg,
        batch_size=batch_size,
        shuffle=True,
        distributed_args=DistributedArgs(use_distributed, rank, world_size),
    )

    return dataloader, train_dataset


def create_seg_conditioned_validation_dataloader(
    cfg: DictConfig,
    size_bin_config: Optional[dict] = None,
    batch_size: Optional[int] = None,
    world_size: int = 1,
) -> Optional[Tuple[DataLoader, TorchDataset]]:
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
        logger.debug(f"Validation directory not found: {val_dir}")
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
        validate_mode_requirements(val_dir, 'seg_conditioned', validate_modality_exists)
    except ValueError as e:
        logger.warning(f"Seg conditioned validation data not available in {val_dir}: {e}")
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

    val_generator = torch.Generator().manual_seed(42)

    dataloader = create_dataloader(
        val_dataset,
        cfg,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=val_generator,
    )

    return dataloader, val_dataset


def create_seg_conditioned_test_dataloader(
    cfg: DictConfig,
    batch_size: Optional[int] = None,
) -> Optional[Tuple[DataLoader, TorchDataset]]:
    """Create test dataloader for size-conditioned segmentation.

    Args:
        cfg: Hydra configuration.
        batch_size: Optional batch size override.

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, "test_new")

    if not os.path.exists(test_dir):
        logger.debug(f"Test directory not found: {test_dir}")
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Get size bin config (convert OmegaConf to Python types)
    size_bin_cfg = cfg.mode.get('size_bins', {})
    bin_edges = list(size_bin_cfg.get('edges', DEFAULT_BIN_EDGES))
    num_bins = int(size_bin_cfg.get('num_bins', len(bin_edges) - 1))
    fov_mm = float(size_bin_cfg.get('fov_mm', 240.0))

    try:
        validate_mode_requirements(test_dir, 'seg_conditioned', validate_modality_exists)
    except ValueError as e:
        logger.warning(f"Seg conditioned test data not available in {test_dir}: {e}")
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

    dataloader = create_dataloader(
        test_dataset,
        cfg,
        batch_size=batch_size,
        shuffle=False,
    )

    return dataloader, test_dataset
