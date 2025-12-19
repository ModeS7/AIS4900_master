"""
Multi-modality dataloaders for progressive VAE training.

Provides dataloaders that combine multiple MR modalities into a single dataset,
enabling training on diverse brain MRI sequences.
"""
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
from monai.data import DataLoader, Dataset
from omegaconf import DictConfig
from torch.utils.data.distributed import DistributedSampler

from medgen.core.constants import DEFAULT_NUM_WORKERS
from medgen.data.augmentation import build_augmentation
from medgen.data.dataset import (
    NiFTIDataset,
    build_standard_transform,
    validate_modality_exists,
)
from medgen.data.utils import extract_slices_single

logger = logging.getLogger(__name__)


def create_multi_modality_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    image_size: int,
    batch_size: int,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True
) -> Tuple[DataLoader, Dataset]:
    """Create dataloader for multi-modality VAE training.

    Loads multiple MR sequences as individual single-channel images and
    combines them into one dataset. Each batch contains mixed slices from
    all modalities, giving 4x more training data than single-modality training.

    Args:
        cfg: Hydra configuration with paths.
        image_keys: List of MR sequences to load (e.g., ['bravo', 'flair', 't1_pre', 't1_gd']).
        image_size: Target image size (passed explicitly for progressive training).
        batch_size: Batch size (passed explicitly for progressive training).
        use_distributed: Whether to use distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.
        augment: Whether to apply data augmentation.

    Returns:
        Tuple of (DataLoader, train_dataset).
        Batches have shape [B, 1, H, W] - single channel images from mixed modalities.
    """
    data_dir = os.path.join(cfg.paths.data_dir, "train")

    # Validate all modalities exist before loading
    for key in image_keys:
        validate_modality_exists(data_dir, key)

    transform = build_standard_transform(image_size)
    aug = build_augmentation(enabled=augment)

    # Collect all slices from all modalities into one list
    all_slices: List[np.ndarray] = []

    for key in image_keys:
        dataset = NiFTIDataset(
            data_dir=data_dir, mr_sequence=key, transform=transform
        )
        # Extract 2D slices from this modality
        slices = extract_slices_single(dataset, augmentation=aug)
        all_slices.extend(list(slices))

    train_dataset = Dataset(all_slices)

    # Setup sampler
    sampler: Optional[DistributedSampler] = None
    shuffle: Optional[bool] = True

    if use_distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = None
        batch_size_per_gpu = batch_size // world_size
    else:
        batch_size_per_gpu = batch_size

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, train_dataset


def create_multi_modality_validation_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    image_size: int,
    batch_size: int
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create validation dataloader for multi-modality VAE training.

    Loads data from the val/ subdirectory if it exists. Returns None if
    val/ directory doesn't exist.

    Args:
        cfg: Hydra configuration with paths.
        image_keys: List of MR sequences to load.
        image_size: Target image size.
        batch_size: Batch size.

    Returns:
        Tuple of (DataLoader, val_dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, "val")

    # Check if validation directory exists
    if not os.path.exists(val_dir):
        return None

    # Validate modalities exist in val directory
    try:
        for key in image_keys:
            validate_modality_exists(val_dir, key)
    except ValueError as e:
        logger.warning(f"Validation directory exists but is misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    # Collect all slices from all modalities into one list
    all_slices: List[np.ndarray] = []

    for key in image_keys:
        dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence=key, transform=transform
        )
        slices = extract_slices_single(dataset)
        all_slices.extend(list(slices))

    val_dataset = Dataset(all_slices)

    # Validation loader: shuffle enabled for diverse batch sampling
    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # Ensure consistent batch sizes for worst_batch visualization
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, val_dataset


def create_multi_modality_test_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    image_size: int,
    batch_size: int
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create test dataloader for multi-modality VAE evaluation.

    Loads data from the test_new/ subdirectory if it exists. Returns None if
    test_new/ directory doesn't exist.

    Args:
        cfg: Hydra configuration with paths.
        image_keys: List of MR sequences to load.
        image_size: Target image size.
        batch_size: Batch size.

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, "test_new")

    # Check if test directory exists
    if not os.path.exists(test_dir):
        return None

    # Validate modalities exist in test directory
    try:
        for key in image_keys:
            validate_modality_exists(test_dir, key)
    except ValueError as e:
        logger.warning(f"Test directory exists but is misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    # Collect all slices from all modalities into one list
    all_slices: List[np.ndarray] = []

    for key in image_keys:
        dataset = NiFTIDataset(
            data_dir=test_dir, mr_sequence=key, transform=transform
        )
        slices = extract_slices_single(dataset)
        all_slices.extend(list(slices))

    test_dataset = Dataset(all_slices)

    # Test loader: shuffle for diverse visualization samples
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, test_dataset
