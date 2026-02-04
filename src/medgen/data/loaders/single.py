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

from medgen.augmentation import build_diffusion_augmentation, build_vae_augmentation
from medgen.data.loaders.common import (
    create_dataloader as create_dataloader_from_dataset,
    DistributedArgs,
    validate_mode_requirements,
)
from medgen.data.dataset import (
    NiFTIDataset,
    build_standard_transform,
    validate_modality_exists,
)
from medgen.data.utils import extract_slices_dual, extract_slices_single, merge_sequences, CFGDropoutDataset

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
        cfg_dropout_prob: CFG dropout probability for conditioning (default: 0.15).

    Returns:
        Tuple of (DataLoader, train_dataset).
    """
    data_dir = os.path.join(cfg.paths.data_dir, "train")
    image_size = cfg.model.image_size

    # Validate modalities exist before loading
    validate_mode_requirements(data_dir, image_type, validate_modality_exists)

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

    # Create DataLoader using unified helper
    dataloader = create_dataloader_from_dataset(
        train_dataset,
        cfg=cfg,
        shuffle=True,
        distributed_args=DistributedArgs(use_distributed, rank, world_size),
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
        logger.debug(f"Validation directory not found: {val_dir}")
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Reduce batch size for DDP (validation runs on single GPU)
    if world_size > 1:
        batch_size = max(1, batch_size // world_size)

    # Validate modalities exist
    try:
        validate_mode_requirements(val_dir, image_type, validate_modality_exists)
    except ValueError as e:
        logger.warning(f"Validation data for {image_type} mode not available in {val_dir}: {e}")
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

    # Create validation DataLoader using unified helper
    # Shuffle with fixed seed for reproducibility + diverse batches
    dataloader = create_dataloader_from_dataset(
        val_dataset,
        cfg=cfg,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for diverse worst_batch visualization
        drop_last=True,  # Ensure consistent batch sizes
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
        logger.debug(f"Test directory not found: {test_dir}")
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Validate modalities exist
    try:
        validate_mode_requirements(test_dir, image_type, validate_modality_exists)
    except ValueError as e:
        logger.warning(f"Test data for {image_type} mode not available in {test_dir}: {e}")
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

    # Create test DataLoader using unified helper
    dataloader = create_dataloader_from_dataset(
        test_dataset,
        cfg=cfg,
        batch_size=batch_size,
        shuffle=False,  # Test must be deterministic for reproducible metrics
    )

    return dataloader, test_dataset
