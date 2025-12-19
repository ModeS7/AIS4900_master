"""
Single-image dataloaders for diffusion training.

Provides dataloaders for segmentation and bravo mode training (single image
or bravo+seg conditioning).
"""
import logging
import os
from typing import Dict, Optional, Tuple

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
from medgen.data.utils import extract_slices_dual, extract_slices_single, merge_sequences

logger = logging.getLogger(__name__)


def create_dataloader(
    cfg: DictConfig,
    image_type: str,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True
) -> Tuple[DataLoader, Dataset]:
    """Create dataloader for single-image training (seg or bravo+seg).

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_type: Image type ('seg' or 'bravo').
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

    # Validate modalities exist before loading
    if image_type == 'seg':
        validate_modality_exists(data_dir, 'seg')
    elif image_type == 'bravo':
        validate_modality_exists(data_dir, 'bravo')
        validate_modality_exists(data_dir, 'seg')

    transform = build_standard_transform(image_size)
    aug = build_augmentation(enabled=augment)

    if image_type == 'seg':
        # Load only segmentation masks
        seg_dataset = NiFTIDataset(
            data_dir=data_dir, mr_sequence="seg", transform=transform
        )
        train_dataset = extract_slices_single(seg_dataset, augmentation=aug)

    elif image_type == 'bravo':
        # Load bravo + seg for conditioning
        bravo_dataset = NiFTIDataset(
            data_dir=data_dir, mr_sequence="bravo", transform=transform
        )
        seg_dataset = NiFTIDataset(
            data_dir=data_dir, mr_sequence="seg", transform=transform
        )

        datasets_dict = {'bravo': bravo_dataset, 'seg': seg_dataset}
        merged = merge_sequences(datasets_dict)
        train_dataset = extract_slices_dual(merged, has_seg=True, augmentation=aug)
    else:
        raise ValueError(f"Unknown image_type: {image_type}")

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


def create_validation_dataloader(
    cfg: DictConfig,
    image_type: str,
    batch_size: Optional[int] = None
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create validation dataloader for single-image diffusion from val/ directory.

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_type: Image type ('seg' or 'bravo').
        batch_size: Optional batch size override. Defaults to training batch size.

    Returns:
        Tuple of (DataLoader, val_dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, "val")

    if not os.path.exists(val_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Validate modalities exist
    try:
        if image_type == 'seg':
            validate_modality_exists(val_dir, 'seg')
        elif image_type == 'bravo':
            validate_modality_exists(val_dir, 'bravo')
            validate_modality_exists(val_dir, 'seg')
        else:
            return None
    except ValueError as e:
        logger.warning(f"Validation directory exists but is misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    if image_type == 'seg':
        seg_dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence="seg", transform=transform
        )
        val_dataset = extract_slices_single(seg_dataset)

    elif image_type == 'bravo':
        bravo_dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence="bravo", transform=transform
        )
        seg_dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence="seg", transform=transform
        )
        datasets_dict = {'bravo': bravo_dataset, 'seg': seg_dataset}
        merged = merge_sequences(datasets_dict)
        val_dataset = extract_slices_dual(merged, has_seg=True)

    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, val_dataset


def create_test_dataloader(
    cfg: DictConfig,
    image_type: str,
    batch_size: Optional[int] = None
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create test dataloader for single-image diffusion from test_new/ directory.

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_type: Image type ('seg' or 'bravo').
        batch_size: Optional batch size override. Defaults to training batch size.

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, "test_new")

    if not os.path.exists(test_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Validate modalities exist
    try:
        if image_type == 'seg':
            validate_modality_exists(test_dir, 'seg')
        elif image_type == 'bravo':
            validate_modality_exists(test_dir, 'bravo')
            validate_modality_exists(test_dir, 'seg')
        else:
            return None
    except ValueError as e:
        logger.warning(f"Test directory exists but is misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    if image_type == 'seg':
        seg_dataset = NiFTIDataset(
            data_dir=test_dir, mr_sequence="seg", transform=transform
        )
        test_dataset = extract_slices_single(seg_dataset)

    elif image_type == 'bravo':
        bravo_dataset = NiFTIDataset(
            data_dir=test_dir, mr_sequence="bravo", transform=transform
        )
        seg_dataset = NiFTIDataset(
            data_dir=test_dir, mr_sequence="seg", transform=transform
        )
        datasets_dict = {'bravo': bravo_dataset, 'seg': seg_dataset}
        merged = merge_sequences(datasets_dict)
        test_dataset = extract_slices_dual(merged, has_seg=True)

    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=DEFAULT_NUM_WORKERS,
        persistent_workers=DEFAULT_NUM_WORKERS > 0
    )

    return dataloader, test_dataset
