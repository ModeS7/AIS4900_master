"""Segmentation mask dataloader for DC-AE compression training.

Loads segmentation masks only (no image modalities) for training
autoencoder-based mask compression.

Design note:
    Augmentation is applied during training iteration (in __getitem__),
    NOT during slice extraction. This ensures:
    1. Fast dataloader creation (no per-slice augmentation at startup)
    2. Different augmentations each epoch (training variety)

NOTE: Dataset classes and utility functions have been consolidated into datasets.py.
This file now imports from there for backward compatibility.
"""
import logging
import os
from collections.abc import Callable

from monai.data import DataLoader
from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset

from medgen.augmentation import build_seg_augmentation, create_seg_collate_fn
from medgen.data.dataset import (
    NiFTIDataset,
    build_standard_transform,
    validate_modality_exists,
)
from medgen.data.loaders.common import DistributedArgs, create_dataloader, get_validated_split_dir

# Import consolidated classes and functions from datasets.py
from medgen.data.loaders.datasets import (
    AugmentedSegDataset,
    extract_seg_slices,
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    'AugmentedSegDataset',
    'extract_seg_slices',
    'create_seg_compression_dataloader',
    'create_seg_compression_validation_dataloader',
    'create_seg_compression_test_dataloader',
]


def create_seg_compression_dataloader(
    cfg: DictConfig,
    image_size: int,
    batch_size: int,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
) -> tuple[DataLoader, TorchDataset]:
    """Create dataloader for segmentation mask compression training.

    Augmentation is applied on-the-fly during __getitem__, NOT during slice
    extraction. This ensures fast startup and variety across epochs.

    Args:
        cfg: Hydra configuration with paths.
        image_size: Target image size.
        batch_size: Batch size.
        use_distributed: Whether to use distributed training.
        rank: Process rank.
        world_size: Total processes.
        augment: Whether to apply augmentation.

    Returns:
        Tuple of (DataLoader, train_dataset).
    """
    data_dir = os.path.join(cfg.paths.data_dir, "train")

    # Validate seg modality exists
    validate_modality_exists(data_dir, 'seg')

    transform = build_standard_transform(image_size)

    # Load seg volumes
    seg_dataset = NiFTIDataset(
        data_dir=data_dir,
        mr_sequence='seg',
        transform=transform,
    )

    # Extract 2D slices (no augmentation here - applied in __getitem__)
    slices = extract_seg_slices(seg_dataset)
    logger.info(f"Extracted {len(slices)} seg slices with tumor content")

    # Create augmented dataset (augmentation applied per-access)
    aug = build_seg_augmentation(enabled=augment)
    train_dataset = AugmentedSegDataset(slices, augmentation=aug)

    # Create collate with batch augmentations if enabled
    batch_aug_cfg = cfg.training.get('batch_augment', {})
    collate_fn: Callable | None = None
    if batch_aug_cfg.get('enabled', False):
        collate_fn = create_seg_collate_fn(
            mosaic_prob=batch_aug_cfg.get('mosaic_prob', 0.2),
            cutmix_prob=batch_aug_cfg.get('cutmix_prob', 0.2),
            copy_paste_prob=batch_aug_cfg.get('copy_paste_prob', 0.3),
        )

    dataloader = create_dataloader(
        train_dataset,
        cfg,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        distributed_args=DistributedArgs(use_distributed, rank, world_size),
    )

    return dataloader, train_dataset


def create_seg_compression_validation_dataloader(
    cfg: DictConfig,
    image_size: int,
    batch_size: int,
) -> tuple[DataLoader, TorchDataset] | None:
    """Create validation dataloader for seg compression.

    No augmentation for validation.

    Args:
        cfg: Hydra configuration with paths.
        image_size: Target image size.
        batch_size: Batch size.

    Returns:
        Tuple of (DataLoader, val_dataset) or None if val/ doesn't exist.
    """
    val_dir = get_validated_split_dir(cfg.paths.data_dir, "val", logger)
    if val_dir is None:
        return None

    # Validate seg modality exists
    try:
        validate_modality_exists(val_dir, 'seg')
    except ValueError as e:
        logger.warning(f"Validation seg not found: {e}")
        return None

    transform = build_standard_transform(image_size)

    seg_dataset = NiFTIDataset(
        data_dir=val_dir,
        mr_sequence='seg',
        transform=transform,
    )

    # No augmentation for validation
    slices = extract_seg_slices(seg_dataset)
    logger.info(f"Extracted {len(slices)} validation seg slices")

    # No augmentation wrapper needed for validation
    val_dataset = AugmentedSegDataset(slices, augmentation=None)

    dataloader = create_dataloader(
        val_dataset,
        cfg,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for diverse batch sampling
        drop_last=True,  # Consistent batch sizes for worst_batch visualization
    )

    return dataloader, val_dataset


def create_seg_compression_test_dataloader(
    cfg: DictConfig,
    image_size: int,
    batch_size: int,
) -> tuple[DataLoader, TorchDataset] | None:
    """Create test dataloader for seg compression evaluation.

    No augmentation for testing.

    Args:
        cfg: Hydra configuration with paths.
        image_size: Target image size.
        batch_size: Batch size.

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = get_validated_split_dir(cfg.paths.data_dir, "test_new", logger)
    if test_dir is None:
        return None

    # Validate seg modality exists
    try:
        validate_modality_exists(test_dir, 'seg')
    except ValueError as e:
        logger.warning(f"Test seg not found: {e}")
        return None

    transform = build_standard_transform(image_size)

    seg_dataset = NiFTIDataset(
        data_dir=test_dir,
        mr_sequence='seg',
        transform=transform,
    )

    # No augmentation for testing
    slices = extract_seg_slices(seg_dataset)
    logger.info(f"Extracted {len(slices)} test seg slices")

    # No augmentation wrapper needed for testing
    test_dataset = AugmentedSegDataset(slices, augmentation=None)

    dataloader = create_dataloader(
        test_dataset,
        cfg,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for diverse visualization samples
    )

    return dataloader, test_dataset
