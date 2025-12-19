"""
VAE dataloaders for training and evaluation.

Provides dataloaders for VAE training with single or dual modality,
with separate handling for training, validation, and test sets.
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


def create_vae_dataloader(
    cfg: DictConfig,
    modality: str,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True
) -> Tuple[DataLoader, Dataset]:
    """Create dataloader for VAE training - correct single/dual modality handling.

    For VAE training, we never concatenate seg with images.
    - Single modality (bravo, seg, t1_pre, t1_gd): 1 channel
    - Dual mode (t1_pre + t1_gd): 2 channels, NO seg

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        modality: Modality to train on ('bravo', 'seg', 't1_pre', 't1_gd', 'dual').
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
    if modality == 'dual':
        for key in ['t1_pre', 't1_gd']:
            validate_modality_exists(data_dir, key)
    else:
        validate_modality_exists(data_dir, modality)

    transform = build_standard_transform(image_size)
    aug = build_augmentation(enabled=augment)

    if modality == 'dual':
        # Dual mode: t1_pre + t1_gd as 2 channels, optionally load seg for metrics
        image_keys = ['t1_pre', 't1_gd']
        datasets_dict: Dict[str, NiFTIDataset] = {}
        for key in image_keys:
            datasets_dict[key] = NiFTIDataset(
                data_dir=data_dir, mr_sequence=key, transform=transform
            )
        # Load seg for regional metrics if available
        has_seg = False
        try:
            validate_modality_exists(data_dir, 'seg')
            datasets_dict['seg'] = NiFTIDataset(
                data_dir=data_dir, mr_sequence='seg', transform=transform
            )
            has_seg = True
        except ValueError:
            pass
        merged = merge_sequences(datasets_dict)
        train_dataset = extract_slices_dual(merged, has_seg=has_seg, augmentation=aug)
    else:
        # Single modality: 1 channel
        nifti_dataset = NiFTIDataset(
            data_dir=data_dir, mr_sequence=modality, transform=transform
        )
        train_dataset = extract_slices_single(nifti_dataset, augmentation=aug)

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


def create_vae_validation_dataloader(
    cfg: DictConfig,
    modality: str,
    batch_size: Optional[int] = None
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create validation dataloader for VAE training from val/ directory.

    Loads data from the val/ subdirectory if it exists. Returns None if
    val/ directory doesn't exist (training will use train dataset sampling
    for validation visualizations).

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        modality: Modality to validate on ('bravo', 'seg', 't1_pre', 't1_gd', 'dual').
        batch_size: Optional batch size override. Defaults to training batch size.

    Returns:
        Tuple of (DataLoader, val_dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, "val")

    # Check if validation directory exists
    if not os.path.exists(val_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Validate modalities exist in val directory
    try:
        if modality == 'dual':
            for key in ['t1_pre', 't1_gd']:
                validate_modality_exists(val_dir, key)
        else:
            validate_modality_exists(val_dir, modality)
    except ValueError as e:
        logger.warning(f"Validation directory exists but is misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    if modality == 'dual':
        # Dual mode: t1_pre + t1_gd as 2 channels, optionally load seg for metrics
        image_keys = ['t1_pre', 't1_gd']
        datasets_dict: Dict[str, NiFTIDataset] = {}
        for key in image_keys:
            datasets_dict[key] = NiFTIDataset(
                data_dir=val_dir, mr_sequence=key, transform=transform
            )
        # Load seg for regional metrics if available
        has_seg = False
        try:
            validate_modality_exists(val_dir, 'seg')
            datasets_dict['seg'] = NiFTIDataset(
                data_dir=val_dir, mr_sequence='seg', transform=transform
            )
            has_seg = True
        except ValueError:
            pass
        merged = merge_sequences(datasets_dict)
        val_dataset = extract_slices_dual(merged, has_seg=has_seg)
    else:
        # Single modality: 1 channel
        nifti_dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence=modality, transform=transform
        )
        val_dataset = extract_slices_single(nifti_dataset)

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


def create_vae_test_dataloader(
    cfg: DictConfig,
    modality: str,
    batch_size: Optional[int] = None
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create test dataloader for VAE evaluation from test_new/ directory.

    Loads data from the test_new/ subdirectory if it exists. Returns None if
    test_new/ directory doesn't exist.

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        modality: Modality to test on ('bravo', 'seg', 't1_pre', 't1_gd', 'dual').
        batch_size: Optional batch size override. Defaults to training batch size.

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, "test_new")

    # Check if test directory exists
    if not os.path.exists(test_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Validate modalities exist in test directory
    try:
        if modality == 'dual':
            for key in ['t1_pre', 't1_gd']:
                validate_modality_exists(test_dir, key)
        else:
            validate_modality_exists(test_dir, modality)
    except ValueError as e:
        logger.warning(f"Test directory exists but is misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    if modality == 'dual':
        # Dual mode: t1_pre + t1_gd as 2 channels, optionally load seg for metrics
        image_keys = ['t1_pre', 't1_gd']
        datasets_dict: Dict[str, NiFTIDataset] = {}
        for key in image_keys:
            datasets_dict[key] = NiFTIDataset(
                data_dir=test_dir, mr_sequence=key, transform=transform
            )
        # Load seg for regional metrics if available
        has_seg = False
        try:
            validate_modality_exists(test_dir, 'seg')
            datasets_dict['seg'] = NiFTIDataset(
                data_dir=test_dir, mr_sequence='seg', transform=transform
            )
            has_seg = True
        except ValueError:
            pass
        merged = merge_sequences(datasets_dict)
        test_dataset = extract_slices_dual(merged, has_seg=has_seg)
    else:
        # Single modality: 1 channel
        nifti_dataset = NiFTIDataset(
            data_dir=test_dir, mr_sequence=modality, transform=transform
        )
        test_dataset = extract_slices_single(nifti_dataset)

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
