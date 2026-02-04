"""
Multi-modality dataloaders for progressive VAE training.

Provides dataloaders that combine multiple MR modalities into a single dataset,
enabling training on diverse brain MRI sequences.
"""
import logging
import os
from typing import Callable, List, Optional, Tuple

import numpy as np
from monai.data import DataLoader, Dataset
from omegaconf import DictConfig

from medgen.augmentation import build_vae_augmentation, create_vae_collate_fn
from medgen.data.loaders.common import (
    create_dataloader,
    DistributedArgs,
    validate_mode_requirements,
)
from medgen.data.dataset import (
    NiFTIDataset,
    build_standard_transform,
    validate_modality_exists,
)
from medgen.data.utils import extract_slices_single, extract_slices_single_with_seg

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

    # Validate all modalities exist before loading (no seg required for VAE)
    validate_mode_requirements(
        data_dir, 'multi_modality', validate_modality_exists,
        image_keys=image_keys, require_seg=False
    )

    transform = build_standard_transform(image_size)
    aug = build_vae_augmentation(enabled=augment)

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

    # Get batch augmentation settings (consistent with vae.py)
    batch_aug_cfg = cfg.training.get('batch_augment', {})
    batch_aug_enabled = batch_aug_cfg.get('enabled', False)

    # Create collate function with batch augmentations if enabled
    collate_fn: Optional[Callable] = None
    if batch_aug_enabled:
        mixup_prob = batch_aug_cfg.get('mixup_prob', 0.2)
        cutmix_prob = batch_aug_cfg.get('cutmix_prob', 0.2)
        collate_fn = create_vae_collate_fn(mixup_prob=mixup_prob, cutmix_prob=cutmix_prob)

    # Create dataloader using shared helper
    dataloader = create_dataloader(
        train_dataset,
        cfg,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        distributed_args=DistributedArgs(use_distributed, rank, world_size),
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

    Includes seg masks paired with each modality for regional metrics tracking.
    Batches are tuples of (image [B,1,H,W], seg [B,1,H,W]).

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
        logger.debug(f"Validation directory not found: {val_dir}")
        return None

    # Validate modalities exist in val directory (no seg required for VAE)
    try:
        validate_mode_requirements(
            val_dir, 'multi_modality', validate_modality_exists,
            image_keys=image_keys, require_seg=False
        )
    except ValueError as e:
        logger.warning(f"Validation data for multi-modality mode (keys={image_keys}) not available in {val_dir}: {e}")
        return None

    # Check if seg exists for regional metrics
    has_seg = False
    try:
        validate_modality_exists(val_dir, 'seg')
        has_seg = True
    except ValueError:
        logger.info("No seg masks in val/ - regional metrics will be disabled")

    transform = build_standard_transform(image_size)

    # Collect all slices from all modalities
    all_slices: List = []

    if has_seg:
        # Load seg dataset once (shared across all modalities)
        seg_dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence='seg', transform=transform
        )
        for key in image_keys:
            image_dataset = NiFTIDataset(
                data_dir=val_dir, mr_sequence=key, transform=transform
            )
            # Extract paired (image, seg) tuples
            slices = extract_slices_single_with_seg(image_dataset, seg_dataset)
            all_slices.extend(list(slices))
    else:
        # No seg - just load images
        for key in image_keys:
            dataset = NiFTIDataset(
                data_dir=val_dir, mr_sequence=key, transform=transform
            )
            slices = extract_slices_single(dataset)
            all_slices.extend(list(slices))

    val_dataset = Dataset(all_slices)

    # Validation loader: shuffle enabled for diverse batch sampling
    dataloader = create_dataloader(
        val_dataset,
        cfg,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # Ensure consistent batch sizes for worst_batch visualization
    )

    return dataloader, val_dataset


def create_single_modality_validation_loader(
    cfg: DictConfig,
    modality: str,
    image_size: int,
    batch_size: int
) -> Optional[DataLoader]:
    """Create validation loader for a single modality (for per-modality metrics).

    Includes seg masks paired with each slice for regional metrics tracking.
    Batches are tuples of (image [B,1,H,W], seg [B,1,H,W]).

    Args:
        cfg: Hydra configuration with paths.
        modality: Single modality to load (e.g., 'bravo', 't1_pre', 't1_gd').
        image_size: Target image size.
        batch_size: Batch size.

    Returns:
        DataLoader for single modality or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, "val")

    # Check if validation directory exists
    if not os.path.exists(val_dir):
        logger.debug(f"Validation directory not found for {modality}: {val_dir}")
        return None

    # Validate modality exists in val directory
    try:
        validate_modality_exists(val_dir, modality)
    except ValueError as e:
        logger.warning(f"Validation directory exists but modality {modality} not found: {e}")
        return None

    # Check if seg exists for regional metrics
    has_seg = False
    try:
        validate_modality_exists(val_dir, 'seg')
        has_seg = True
    except ValueError as e:
        logger.debug(f"Seg not available for single modality validation (regional metrics disabled): {e}")

    transform = build_standard_transform(image_size)

    if has_seg:
        # Load with paired seg masks
        image_dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence=modality, transform=transform
        )
        seg_dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence='seg', transform=transform
        )
        slices = extract_slices_single_with_seg(image_dataset, seg_dataset)
        val_dataset = Dataset(list(slices))
    else:
        # No seg - just load images
        dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence=modality, transform=transform
        )
        slices = extract_slices_single(dataset)
        val_dataset = Dataset(list(slices))

    # Create dataloader using shared helper
    dataloader = create_dataloader(
        val_dataset,
        cfg,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return dataloader


def create_multi_modality_test_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    image_size: int,
    batch_size: int
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create test dataloader for multi-modality VAE evaluation.

    Loads data from the test_new/ subdirectory if it exists. Returns None if
    test_new/ directory doesn't exist.

    Includes seg masks paired with each modality for regional metrics tracking.
    Batches are tuples of (image [B,1,H,W], seg [B,1,H,W]).

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
        logger.debug(f"Test directory not found: {test_dir}")
        return None

    # Validate modalities exist in test directory (no seg required for VAE)
    try:
        validate_mode_requirements(
            test_dir, 'multi_modality', validate_modality_exists,
            image_keys=image_keys, require_seg=False
        )
    except ValueError as e:
        logger.warning(f"Test data for multi-modality mode (keys={image_keys}) not available in {test_dir}: {e}")
        return None

    # Check if seg exists for regional metrics
    has_seg = False
    try:
        validate_modality_exists(test_dir, 'seg')
        has_seg = True
    except ValueError:
        logger.info("No seg masks in test_new/ - regional metrics will be disabled")

    transform = build_standard_transform(image_size)

    # Collect all slices from all modalities
    all_slices: List = []

    if has_seg:
        # Load seg dataset once (shared across all modalities)
        seg_dataset = NiFTIDataset(
            data_dir=test_dir, mr_sequence='seg', transform=transform
        )
        for key in image_keys:
            image_dataset = NiFTIDataset(
                data_dir=test_dir, mr_sequence=key, transform=transform
            )
            # Extract paired (image, seg) tuples
            slices = extract_slices_single_with_seg(image_dataset, seg_dataset)
            all_slices.extend(list(slices))
    else:
        # No seg - just load images
        for key in image_keys:
            dataset = NiFTIDataset(
                data_dir=test_dir, mr_sequence=key, transform=transform
            )
            slices = extract_slices_single(dataset)
            all_slices.extend(list(slices))

    test_dataset = Dataset(all_slices)

    # Test loader: shuffle for diverse visualization samples
    dataloader = create_dataloader(
        test_dataset,
        cfg,
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader, test_dataset
