"""
Single-image dataloaders for diffusion training.

Provides dataloaders for segmentation and bravo mode training (single image
or bravo+seg conditioning).
"""
import logging
import os
from typing import Literal

import torch
from monai.data import DataLoader, Dataset
from omegaconf import DictConfig

from medgen.augmentation import build_diffusion_augmentation, build_vae_augmentation
from medgen.data.dataset import (
    NiFTIDataset,
    build_standard_transform,
    validate_modality_exists,
)
from medgen.data.loaders.common import DataLoaderConfig, setup_distributed_sampler
from medgen.data.utils import (
    CFGDropoutDataset,
    extract_slices_dual,
    extract_slices_single,
    merge_sequences,
)

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
    cfg_dropout_prob: float = 0.15,
) -> tuple[DataLoader, Dataset]:
    """Create dataloader for single-image training (seg or bravo+seg).

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_type: Image type ('seg' or 'bravo').
        use_distributed: Whether to use distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.
        augment: Whether to apply data augmentation.
        augment_type: Type of augmentation ('diffusion' or 'vae').
        cfg_dropout_prob: CFG dropout probability for conditioning (default: 0.15).

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

        # Wrap with CFG dropout for classifier-free guidance training
        if cfg_dropout_prob > 0:
            train_dataset = CFGDropoutDataset(train_dataset, cfg_dropout_prob=cfg_dropout_prob)
            logger.info(f"CFG dropout enabled for bravo mode: {cfg_dropout_prob:.0%} probability")
    else:
        raise ValueError(f"Unknown image_type: {image_type}")

    # Setup distributed sampler and batch size
    sampler, batch_size_per_gpu, shuffle = setup_distributed_sampler(
        train_dataset, use_distributed, rank, world_size, batch_size, shuffle=True
    )

    # Get DataLoader settings from config
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


def create_validation_dataloader(
    cfg: DictConfig,
    image_type: str,
    batch_size: int | None = None,
    world_size: int = 1,
) -> tuple[DataLoader, Dataset] | None:
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
    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    # Validation loader: shuffle with fixed seed for reproducibility + diverse batches
    # Fixed generator ensures same batch composition across epochs/runs
    val_generator = torch.Generator().manual_seed(42)
    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for diverse worst_batch visualization
        drop_last=True,  # Ensure consistent batch sizes
        generator=val_generator,  # Fixed seed for reproducibility
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    return dataloader, val_dataset


def create_test_dataloader(
    cfg: DictConfig,
    image_type: str,
    batch_size: int | None = None
) -> tuple[DataLoader, Dataset] | None:
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
    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Test must be deterministic for reproducible metrics
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    return dataloader, test_dataset
