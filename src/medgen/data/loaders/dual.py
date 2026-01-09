"""
Dual-image dataloaders for diffusion training.

Provides dataloaders for dual mode training (T1 pre + T1 gd with seg conditioning).
"""
import logging
import os
from typing import Dict, List, Literal, Optional, Tuple

from monai.data import DataLoader, Dataset
from omegaconf import DictConfig

from medgen.data.augmentation import build_diffusion_augmentation, build_vae_augmentation
from medgen.data.loaders.common import DataLoaderConfig, setup_distributed_sampler
from medgen.data.dataset import (
    NiFTIDataset,
    build_standard_transform,
    validate_modality_exists,
)
from medgen.data.utils import extract_slices_dual, merge_sequences

logger = logging.getLogger(__name__)

AugmentType = Literal["diffusion", "vae"]


def create_dual_image_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    conditioning: Optional[str],
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
    augment_type: AugmentType = "diffusion",
) -> Tuple[DataLoader, Dataset]:
    """Create dataloader for dual-image training (T1 pre + T1 gd).

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_keys: List of two sequences to train (e.g., ['t1_pre', 't1_gd']).
        conditioning: Conditioning sequence name (e.g., 'seg') or None.
        use_distributed: Whether to use distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.
        augment: Whether to apply data augmentation.
        augment_type: Type of augmentation ('diffusion' or 'vae').

    Returns:
        Tuple of (DataLoader, train_dataset).
        Batches have shape [B, 3, H, W] = [t1_pre, t1_gd, seg].

    Raises:
        ValueError: If image_keys does not contain exactly 2 items.
    """
    if len(image_keys) != 2:
        raise ValueError(f"Dual-image mode requires exactly 2 image types, got {len(image_keys)}: {image_keys}")

    data_dir = os.path.join(cfg.paths.data_dir, "train")
    image_size = cfg.model.image_size
    batch_size = cfg.training.batch_size

    # Validate all modalities exist before loading
    for key in image_keys:
        validate_modality_exists(data_dir, key)
    if conditioning:
        validate_modality_exists(data_dir, conditioning)

    transform = build_standard_transform(image_size)

    # Select augmentation based on type
    if augment_type == "vae":
        aug = build_vae_augmentation(enabled=augment)
    else:
        aug = build_diffusion_augmentation(enabled=augment)

    # Load all required datasets
    datasets_dict: Dict[str, NiFTIDataset] = {}

    for key in image_keys:
        datasets_dict[key] = NiFTIDataset(
            data_dir=data_dir, mr_sequence=key, transform=transform
        )

    # Load conditioning (segmentation)
    if conditioning:
        datasets_dict[conditioning] = NiFTIDataset(
            data_dir=data_dir, mr_sequence=conditioning, transform=transform
        )

    # Merge all sequences
    merged = merge_sequences(datasets_dict)
    train_dataset = extract_slices_dual(merged, has_seg=(conditioning is not None), augmentation=aug)

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


def create_dual_image_validation_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    conditioning: Optional[str] = 'seg',
    batch_size: Optional[int] = None,
    world_size: int = 1,
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create validation dataloader for dual-image diffusion from val/ directory.

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_keys: List of two sequences (e.g., ['t1_pre', 't1_gd']).
        conditioning: Conditioning sequence name (e.g., 'seg') or None.
        batch_size: Optional batch size override. Defaults to training batch size.
        world_size: Number of GPUs for DDP. Validation batch size is reduced
            when world_size > 1 to avoid OOM (validation runs on single GPU).

    Returns:
        Tuple of (DataLoader, val_dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, "val")

    if not os.path.exists(val_dir):
        return None

    if len(image_keys) != 2:
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Reduce batch size for DDP (validation runs on single GPU)
    if world_size > 1:
        batch_size = max(1, batch_size // world_size)

    # Validate modalities exist
    try:
        for key in image_keys:
            validate_modality_exists(val_dir, key)
        if conditioning:
            validate_modality_exists(val_dir, conditioning)
    except ValueError as e:
        logger.warning(f"Validation directory exists but is misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    datasets_dict: Dict[str, NiFTIDataset] = {}
    for key in image_keys:
        datasets_dict[key] = NiFTIDataset(
            data_dir=val_dir, mr_sequence=key, transform=transform
        )

    if conditioning:
        datasets_dict[conditioning] = NiFTIDataset(
            data_dir=val_dir, mr_sequence=conditioning, transform=transform
        )

    merged = merge_sequences(datasets_dict)
    val_dataset = extract_slices_dual(merged, has_seg=(conditioning is not None))

    # Get DataLoader settings from config
    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    return dataloader, val_dataset


def create_dual_image_test_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    conditioning: Optional[str] = 'seg',
    batch_size: Optional[int] = None
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create test dataloader for dual-image diffusion from test_new/ directory.

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        image_keys: List of two sequences (e.g., ['t1_pre', 't1_gd']).
        conditioning: Conditioning sequence name (e.g., 'seg') or None.
        batch_size: Optional batch size override. Defaults to training batch size.

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, "test_new")

    if not os.path.exists(test_dir):
        return None

    if len(image_keys) != 2:
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Validate modalities exist
    try:
        for key in image_keys:
            validate_modality_exists(test_dir, key)
        if conditioning:
            validate_modality_exists(test_dir, conditioning)
    except ValueError as e:
        logger.warning(f"Test directory exists but is misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    datasets_dict: Dict[str, NiFTIDataset] = {}
    for key in image_keys:
        datasets_dict[key] = NiFTIDataset(
            data_dir=test_dir, mr_sequence=key, transform=transform
        )

    if conditioning:
        datasets_dict[conditioning] = NiFTIDataset(
            data_dir=test_dir, mr_sequence=conditioning, transform=transform
        )

    merged = merge_sequences(datasets_dict)
    test_dataset = extract_slices_dual(merged, has_seg=(conditioning is not None))

    # Get DataLoader settings from config
    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    return dataloader, test_dataset
