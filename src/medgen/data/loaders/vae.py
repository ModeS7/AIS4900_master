"""
VAE dataloaders for training and evaluation.

Provides dataloaders for VAE training with single or dual modality,
with separate handling for training, validation, and test sets.

Uses aggressive VAE-specific augmentation and optional batch-level
augmentations (mixup, cutmix) via custom collate function.
"""
import logging
import os
from collections.abc import Callable

from monai.data import DataLoader, Dataset
from omegaconf import DictConfig

from medgen.augmentation import build_vae_augmentation, create_vae_collate_fn
from medgen.data.dataset import (
    NiFTIDataset,
    build_standard_transform,
    validate_modality_exists,
)
from medgen.data.loaders.common import (
    DistributedArgs,
    create_dataloader,
    get_validated_split_dir,
    validate_mode_requirements,
)
from medgen.data.utils import (
    extract_slices_dual,
    extract_slices_single,
    extract_slices_single_with_seg,
    merge_sequences,
)

logger = logging.getLogger(__name__)


def create_vae_dataloader(
    cfg: DictConfig,
    modality: str,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True
) -> tuple[DataLoader, Dataset]:
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

    # Validate modalities exist before loading (no seg required for VAE)
    if modality == 'dual':
        validate_mode_requirements(
            data_dir, 'dual', validate_modality_exists, require_seg=False
        )
    else:
        validate_modality_exists(data_dir, modality)

    transform = build_standard_transform(image_size)
    aug = build_vae_augmentation(enabled=augment)

    if modality == 'dual':
        # Dual mode: t1_pre + t1_gd as 2 channels, optionally load seg for metrics
        image_keys = ['t1_pre', 't1_gd']
        datasets_dict: dict[str, NiFTIDataset] = {}
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
        except ValueError as e:
            logger.debug(f"Seg not available for dual VAE training (regional metrics disabled): {e}")
        merged = merge_sequences(datasets_dict)
        train_dataset = extract_slices_dual(merged, has_seg=has_seg, augmentation=aug)
    else:
        # Single modality: 1 channel, optionally load seg for regional metrics
        nifti_dataset = NiFTIDataset(
            data_dir=data_dir, mr_sequence=modality, transform=transform
        )
        # Try to load seg for regional metrics
        try:
            validate_modality_exists(data_dir, 'seg')
            seg_dataset = NiFTIDataset(
                data_dir=data_dir, mr_sequence='seg', transform=transform
            )
            train_dataset = extract_slices_single_with_seg(
                nifti_dataset, seg_dataset, augmentation=aug
            )
        except ValueError as e:
            # No seg available, proceed without regional metrics
            logger.debug(f"Seg not available for single VAE training (regional metrics disabled): {e}")
            train_dataset = extract_slices_single(nifti_dataset, augmentation=aug)

    # Get batch augmentation settings
    batch_aug_cfg = cfg.training.get('batch_augment', {})
    batch_aug_enabled = batch_aug_cfg.get('enabled', False)

    # Create collate function with batch augmentations if enabled
    collate_fn: Callable | None = None
    if batch_aug_enabled:
        mixup_prob = batch_aug_cfg.get('mixup_prob', 0.2)
        cutmix_prob = batch_aug_cfg.get('cutmix_prob', 0.2)
        collate_fn = create_vae_collate_fn(mixup_prob=mixup_prob, cutmix_prob=cutmix_prob)

    # Use unified dataloader creation
    dataloader = create_dataloader(
        train_dataset,
        cfg,
        shuffle=True,
        collate_fn=collate_fn,
        distributed_args=DistributedArgs(
            use_distributed=use_distributed,
            rank=rank,
            world_size=world_size,
        ),
    )

    return dataloader, train_dataset


def create_vae_validation_dataloader(
    cfg: DictConfig,
    modality: str,
    batch_size: int | None = None
) -> tuple[DataLoader, Dataset] | None:
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
    val_dir = get_validated_split_dir(cfg.paths.data_dir, "val", logger)
    if val_dir is None:
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Validate modalities exist in val directory (no seg required for VAE)
    try:
        if modality == 'dual':
            validate_mode_requirements(
                val_dir, 'dual', validate_modality_exists, require_seg=False
            )
        else:
            validate_modality_exists(val_dir, modality)
    except ValueError as e:
        logger.warning(f"Validation data for {modality} mode not available in {val_dir}: {e}")
        return None

    transform = build_standard_transform(image_size)

    if modality == 'dual':
        # Dual mode: t1_pre + t1_gd as 2 channels, optionally load seg for metrics
        image_keys = ['t1_pre', 't1_gd']
        datasets_dict: dict[str, NiFTIDataset] = {}
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
        except ValueError as e:
            logger.debug(f"Seg not available for dual VAE validation (regional metrics disabled): {e}")
        merged = merge_sequences(datasets_dict)
        val_dataset = extract_slices_dual(merged, has_seg=has_seg)
    else:
        # Single modality: 1 channel, optionally load seg for regional metrics
        nifti_dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence=modality, transform=transform
        )
        # Try to load seg for regional metrics
        try:
            validate_modality_exists(val_dir, 'seg')
            seg_dataset = NiFTIDataset(
                data_dir=val_dir, mr_sequence='seg', transform=transform
            )
            val_dataset = extract_slices_single_with_seg(nifti_dataset, seg_dataset)
        except ValueError as e:
            logger.debug(f"Seg not available for single VAE validation (regional metrics disabled): {e}")
            val_dataset = extract_slices_single(nifti_dataset)

    # Validation loader: shuffle for diverse sampling, drop_last for consistent batch sizes
    dataloader = create_dataloader(
        val_dataset,
        cfg,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # Ensure consistent batch sizes for worst_batch visualization
    )

    return dataloader, val_dataset


def create_vae_test_dataloader(
    cfg: DictConfig,
    modality: str,
    batch_size: int | None = None
) -> tuple[DataLoader, Dataset] | None:
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
    test_dir = get_validated_split_dir(cfg.paths.data_dir, "test_new", logger)
    if test_dir is None:
        return None

    image_size = cfg.model.image_size
    batch_size = batch_size or cfg.training.batch_size

    # Validate modalities exist in test directory (no seg required for VAE)
    try:
        if modality == 'dual':
            validate_mode_requirements(
                test_dir, 'dual', validate_modality_exists, require_seg=False
            )
        else:
            validate_modality_exists(test_dir, modality)
    except ValueError as e:
        logger.warning(f"Test data for {modality} mode not available in {test_dir}: {e}")
        return None

    transform = build_standard_transform(image_size)

    if modality == 'dual':
        # Dual mode: t1_pre + t1_gd as 2 channels, optionally load seg for metrics
        image_keys = ['t1_pre', 't1_gd']
        datasets_dict: dict[str, NiFTIDataset] = {}
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
        except ValueError as e:
            logger.debug(f"Seg not available for dual VAE test (regional metrics disabled): {e}")
        merged = merge_sequences(datasets_dict)
        test_dataset = extract_slices_dual(merged, has_seg=has_seg)
    else:
        # Single modality: 1 channel, optionally load seg for regional metrics
        nifti_dataset = NiFTIDataset(
            data_dir=test_dir, mr_sequence=modality, transform=transform
        )
        # Try to load seg for regional metrics
        try:
            validate_modality_exists(test_dir, 'seg')
            seg_dataset = NiFTIDataset(
                data_dir=test_dir, mr_sequence='seg', transform=transform
            )
            test_dataset = extract_slices_single_with_seg(nifti_dataset, seg_dataset)
        except ValueError as e:
            logger.debug(f"Seg not available for single VAE test (regional metrics disabled): {e}")
            test_dataset = extract_slices_single(nifti_dataset)

    # Test loader: shuffle for diverse visualization samples
    dataloader = create_dataloader(
        test_dataset,
        cfg,
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader, test_dataset


# NOTE: VolumeDataset and DualVolumeDataset have been consolidated into datasets.py.
# Import from there for backward compatibility.
from medgen.data.loaders.datasets import DualVolumeDataset, VolumeDataset


def create_vae_volume_validation_dataloader(
    cfg: DictConfig,
    modality: str,
    data_split: str = 'val',
) -> tuple[DataLoader, Dataset] | None:
    """Create dataloader that returns full 3D volumes for volume-level metrics.

    Unlike slice-based loaders, this returns [C, H, W, D] volumes without
    slice extraction. Used for computing 3D MS-SSIM on 2D model reconstructions.

    Args:
        cfg: Hydra configuration with paths, model, and training settings.
        modality: Modality ('bravo', 'seg', 't1_pre', 't1_gd', 'dual').
        data_split: Which split to load ('val' or 'test_new').

    Returns:
        Tuple of (DataLoader, volume_dataset) or None if directory doesn't exist.
    """
    data_dir = get_validated_split_dir(cfg.paths.data_dir, data_split, logger)
    if data_dir is None:
        return None

    image_size = cfg.model.image_size

    # Validate modalities exist (no seg required for VAE)
    try:
        if modality == 'dual':
            validate_mode_requirements(
                data_dir, 'dual', validate_modality_exists, require_seg=False
            )
        else:
            validate_modality_exists(data_dir, modality)
    except ValueError as e:
        logger.warning(f"Volume data for {modality} mode not available in {data_dir}: {e}")
        return None

    transform = build_standard_transform(image_size)

    if modality == 'dual':
        # Dual mode: stack t1_pre + t1_gd as 2 channels
        t1_pre_dataset = NiFTIDataset(data_dir, 't1_pre', transform)
        t1_gd_dataset = NiFTIDataset(data_dir, 't1_gd', transform)

        # Try to load seg for regional metrics
        seg_dataset = None
        try:
            validate_modality_exists(data_dir, 'seg')
            seg_dataset = NiFTIDataset(data_dir, 'seg', transform)
        except ValueError as e:
            logger.debug(f"Seg not available for dual volume validation (regional metrics disabled): {e}")

        volume_dataset = DualVolumeDataset(t1_pre_dataset, t1_gd_dataset, seg_dataset)
    else:
        # Single modality
        image_dataset = NiFTIDataset(data_dir, modality, transform)

        # Try to load seg for regional metrics
        seg_dataset = None
        try:
            validate_modality_exists(data_dir, 'seg')
            seg_dataset = NiFTIDataset(data_dir, 'seg', transform)
        except ValueError as e:
            logger.debug(f"Seg not available for single volume validation (regional metrics disabled): {e}")

        volume_dataset = VolumeDataset(image_dataset, seg_dataset)

    # Volume-level loader: batch_size=1, no shuffle (process each volume once)
    dataloader = DataLoader(
        volume_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Simple loader, no need for multi-processing
        pin_memory=True,
    )

    return dataloader, volume_dataset
