"""Segmentation mask dataloader for DC-AE compression training.

Loads segmentation masks only (no image modalities) for training
autoencoder-based mask compression.
"""
import logging
import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from omegaconf import DictConfig

from medgen.core.constants import BINARY_THRESHOLD_GT
from medgen.data.augmentation import (
    apply_augmentation,
    binarize_mask,
    build_seg_augmentation,
    create_seg_collate_fn,
)
from medgen.data.loaders.common import create_dataloader, DistributedArgs
from medgen.data.dataset import (
    NiFTIDataset,
    build_standard_transform,
    validate_modality_exists,
)

logger = logging.getLogger(__name__)


def extract_seg_slices(
    seg_dataset: Dataset,
    augmentation: Optional[Callable] = None,
    min_tumor_pixels: int = 10,
) -> List[np.ndarray]:
    """Extract 2D segmentation mask slices from 3D volumes.

    Only keeps slices with actual tumor content (non-empty masks).

    Args:
        seg_dataset: NiFTI dataset with seg volumes.
        augmentation: Optional albumentations transform.
        min_tumor_pixels: Minimum number of tumor pixels to keep slice.

    Returns:
        List of 2D mask slices [1, H, W].
    """
    all_slices: List[np.ndarray] = []

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

            # Ensure float32
            slice_2d = slice_2d.astype(np.float32)

            # Apply augmentation if provided
            if augmentation is not None:
                # apply_augmentation expects [C, H, W] and returns [C, H, W]
                slice_2d = apply_augmentation(slice_2d, augmentation, has_mask=False)

            # CRITICAL: Binarize after augmentation to restore binary values
            # (interpolation from rotation/affine creates non-binary values)
            slice_2d = binarize_mask(slice_2d, threshold=BINARY_THRESHOLD_GT)

            all_slices.append(slice_2d)

    return all_slices


def create_seg_compression_dataloader(
    cfg: DictConfig,
    image_size: int,
    batch_size: int,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
) -> Tuple[DataLoader, Dataset]:
    """Create dataloader for segmentation mask compression training.

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
    aug = build_seg_augmentation(enabled=augment)

    # Load seg volumes
    seg_dataset = NiFTIDataset(
        data_dir=data_dir,
        mr_sequence='seg',
        transform=transform,
    )

    # Extract 2D slices
    slices = extract_seg_slices(seg_dataset, augmentation=aug)
    logger.info(f"Extracted {len(slices)} seg slices with tumor content")

    train_dataset = Dataset(slices)

    # Create collate with batch augmentations if enabled
    batch_aug_cfg = cfg.training.get('batch_augment', {})
    collate_fn: Optional[Callable] = None
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
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create validation dataloader for seg compression.

    No augmentation for validation.

    Args:
        cfg: Hydra configuration with paths.
        image_size: Target image size.
        batch_size: Batch size.

    Returns:
        Tuple of (DataLoader, val_dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, "val")

    if not os.path.exists(val_dir):
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
    slices = extract_seg_slices(seg_dataset, augmentation=None)
    logger.info(f"Extracted {len(slices)} validation seg slices")

    val_dataset = Dataset(slices)

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
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create test dataloader for seg compression evaluation.

    No augmentation for testing.

    Args:
        cfg: Hydra configuration with paths.
        image_size: Target image size.
        batch_size: Batch size.

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, "test_new")

    if not os.path.exists(test_dir):
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
    slices = extract_seg_slices(seg_dataset, augmentation=None)
    logger.info(f"Extracted {len(slices)} test seg slices")

    test_dataset = Dataset(slices)

    dataloader = create_dataloader(
        test_dataset,
        cfg,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for diverse visualization samples
    )

    return dataloader, test_dataset
