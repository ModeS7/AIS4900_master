"""
Single-image dataloaders for diffusion training.

Provides dataloaders for segmentation and bravo mode training (single image
or bravo+seg conditioning).
"""
import logging
import os
from typing import Literal, Optional, Tuple

from monai.data import DataLoader, Dataset
from omegaconf import DictConfig
from torch.utils.data.distributed import DistributedSampler

from medgen.core.constants import DEFAULT_NUM_WORKERS
from medgen.data.augmentation import build_diffusion_augmentation, build_vae_augmentation
from medgen.data.dataset import (
    NiFTIDataset,
    build_standard_transform,
    validate_modality_exists,
)
from medgen.data.utils import extract_slices_dual, extract_slices_single, merge_sequences

logger = logging.getLogger(__name__)

AugmentType = Literal["diffusion", "vae"]


def create_dataloader(
    cfg: DictConfig,
    image_type: str,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
    augment_type: AugmentType = "diffusion",
) -> Tuple[DataLoader, Dataset]:
    """Create dataloader for single-image training (seg or bravo+seg).

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_type: Image type ('seg' or 'bravo').
        use_distributed: Whether to use distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.
        augment: Whether to apply data augmentation.
        augment_type: Type of augmentation ('diffusion' or 'vae').

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

    # Select augmentation based on type
    if augment_type == "vae":
        aug = build_vae_augmentation(enabled=augment)
    else:
        aug = build_diffusion_augmentation(enabled=augment)

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

    # Get DataLoader settings from config
    dl_cfg = cfg.training.get('dataloader', {})
    num_workers = dl_cfg.get('num_workers', DEFAULT_NUM_WORKERS)
    prefetch_factor = dl_cfg.get('prefetch_factor', 4) if num_workers > 0 else None
    pin_memory = dl_cfg.get('pin_memory', True)
    persistent_workers = dl_cfg.get('persistent_workers', True) and num_workers > 0

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    return dataloader, train_dataset


def create_validation_dataloader(
    cfg: DictConfig,
    image_type: str,
    batch_size: Optional[int] = None,
    world_size: int = 1,
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create validation dataloader for single-image diffusion from val/ directory.

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_type: Image type ('seg' or 'bravo').
        batch_size: Optional batch size override. Defaults to training batch size.
        world_size: Number of GPUs for DDP. Validation batch size is reduced
            when world_size > 1 to avoid OOM (validation runs on single GPU).

    Returns:
        Tuple of (DataLoader, val_dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, "val")

    if not os.path.exists(val_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Reduce batch size for DDP (validation runs on single GPU)
    if world_size > 1:
        batch_size = max(1, batch_size // world_size)

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

    # Get DataLoader settings from config
    dl_cfg = cfg.training.get('dataloader', {})
    num_workers = dl_cfg.get('num_workers', DEFAULT_NUM_WORKERS)
    prefetch_factor = dl_cfg.get('prefetch_factor', 4) if num_workers > 0 else None
    pin_memory = dl_cfg.get('pin_memory', True)
    persistent_workers = dl_cfg.get('persistent_workers', True) and num_workers > 0

    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
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

    # Get DataLoader settings from config
    dl_cfg = cfg.training.get('dataloader', {})
    num_workers = dl_cfg.get('num_workers', DEFAULT_NUM_WORKERS)
    prefetch_factor = dl_cfg.get('prefetch_factor', 4) if num_workers > 0 else None
    pin_memory = dl_cfg.get('pin_memory', True)
    persistent_workers = dl_cfg.get('persistent_workers', True) and num_workers > 0

    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    return dataloader, test_dataset
