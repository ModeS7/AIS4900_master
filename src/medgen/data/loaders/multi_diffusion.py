"""
Multi-modality dataloaders for diffusion training with mode embedding.

Provides dataloaders that pool multiple MR modalities, each paired with seg mask
and mode_id for training a single model on all modalities.
"""
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from omegaconf import DictConfig
from torch.utils.data.distributed import DistributedSampler

from medgen.core.constants import DEFAULT_NUM_WORKERS
from medgen.data.augmentation import build_diffusion_augmentation
from medgen.data.dataset import (
    NiFTIDataset,
    build_standard_transform,
    validate_modality_exists,
)
from medgen.data.mode_embed import MODE_ID_MAP

logger = logging.getLogger(__name__)


def _extract_slices_with_mode_id(
    image_dataset: Dataset,
    seg_dataset: Dataset,
    mode_id: int,
    augmentation=None,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Extract 2D slices with paired seg mask and mode_id.

    Each slice is returned as a tuple (image, seg, mode_id) where:
    - image: [1, H, W] single-channel image
    - seg: [1, H, W] binary segmentation mask
    - mode_id: int (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd)

    Args:
        image_dataset: Dataset of 3D image volumes [1, H, W, D].
        seg_dataset: Dataset of 3D seg volumes [1, H, W, D].
        mode_id: Integer ID for this modality.
        augmentation: Optional albumentations Compose for data augmentation.

    Returns:
        List of tuples (image_slice, seg_slice, mode_id).
    """
    all_slices: List[Tuple[np.ndarray, np.ndarray, int]] = []

    if len(image_dataset) != len(seg_dataset):
        raise ValueError(
            f"Image dataset ({len(image_dataset)}) and seg dataset ({len(seg_dataset)}) "
            "must have same number of patients"
        )

    for i in range(len(image_dataset)):
        image_volume, image_name = image_dataset[i]  # Shape: [1, H, W, D]
        seg_volume, seg_name = seg_dataset[i]  # Shape: [1, H, W, D]

        # Verify same patient
        if image_name != seg_name:
            raise ValueError(f"Patient mismatch: {image_name} vs {seg_name}")

        # Extract non-empty slices along depth dimension (axis 3)
        for k in range(image_volume.shape[3]):
            # Convert MetaTensor to numpy array
            image_slice = np.array(image_volume[:, :, :, k])
            seg_slice = np.array(seg_volume[:, :, :, k])

            if np.sum(image_slice) > 1.0:
                # Apply augmentation if provided
                if augmentation is not None:
                    # Transpose to [H, W, C] for albumentations
                    img_hwc = np.transpose(image_slice, (1, 2, 0))
                    seg_hwc = np.transpose(seg_slice, (1, 2, 0))

                    transformed = augmentation(image=img_hwc, mask=seg_hwc)
                    img_aug = transformed['image']
                    seg_aug = transformed['mask']

                    # Transpose back to [C, H, W]
                    image_slice = np.transpose(img_aug, (2, 0, 1))
                    seg_slice = np.transpose(seg_aug, (2, 0, 1))

                # Binarize seg mask
                seg_slice = (seg_slice > 0.5).astype(np.float32)

                all_slices.append((image_slice, seg_slice, mode_id))

    return all_slices


class MultiDiffusionDataset(Dataset):
    """Dataset that returns (image, seg, mode_id) tuples as tensors."""

    def __init__(self, slices: List[Tuple[np.ndarray, np.ndarray, int]]):
        """Initialize dataset.

        Args:
            slices: List of (image, seg, mode_id) tuples from extraction.
        """
        self.slices = slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        image, seg, mode_id = self.slices[idx]
        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(seg).float(),
            torch.tensor(mode_id, dtype=torch.long),
        )


def create_multi_diffusion_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
) -> Tuple[DataLoader, Dataset]:
    """Create dataloader for multi-modality diffusion training.

    Loads multiple MR sequences, each paired with seg mask and mode_id.
    Each batch contains mixed slices from all modalities with their mode IDs.

    Args:
        cfg: Hydra configuration with paths.
        image_keys: List of MR sequences (e.g., ['bravo', 'flair', 't1_pre', 't1_gd']).
        use_distributed: Whether to use distributed training.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.
        augment: Whether to apply data augmentation.

    Returns:
        Tuple of (DataLoader, train_dataset).
        Batches are tuples of (image [B,1,H,W], seg [B,1,H,W], mode_id [B]).
    """
    data_dir = os.path.join(cfg.paths.data_dir, "train")
    image_size = cfg.model.image_size
    batch_size = cfg.training.batch_size

    # Validate all modalities exist
    for key in image_keys:
        validate_modality_exists(data_dir, key)
    validate_modality_exists(data_dir, 'seg')

    transform = build_standard_transform(image_size)
    aug = build_diffusion_augmentation(enabled=augment)

    # Load seg dataset (shared across all modalities)
    seg_dataset = NiFTIDataset(
        data_dir=data_dir, mr_sequence='seg', transform=transform
    )

    # Collect all slices from all modalities
    all_slices: List[Tuple[np.ndarray, np.ndarray, int]] = []

    for key in image_keys:
        mode_id = MODE_ID_MAP.get(key, 0)
        image_dataset = NiFTIDataset(
            data_dir=data_dir, mr_sequence=key, transform=transform
        )
        slices = _extract_slices_with_mode_id(
            image_dataset, seg_dataset, mode_id, augmentation=aug
        )
        all_slices.extend(slices)
        logger.info(f"Loaded {len(slices)} slices from {key} (mode_id={mode_id})")

    train_dataset = MultiDiffusionDataset(all_slices)
    logger.info(f"Total training slices: {len(train_dataset)}")

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


def create_multi_diffusion_validation_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    world_size: int = 1,
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create validation dataloader for multi-modality diffusion.

    Args:
        cfg: Hydra configuration with paths.
        image_keys: List of MR sequences to load.
        world_size: Number of GPUs for DDP (for batch size adjustment).

    Returns:
        Tuple of (DataLoader, val_dataset) or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, "val")
    if not os.path.exists(val_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = cfg.training.batch_size

    # Reduce batch size for DDP (validation runs on single GPU)
    if world_size > 1:
        batch_size = max(1, batch_size // world_size)

    # Validate modalities exist
    try:
        for key in image_keys:
            validate_modality_exists(val_dir, key)
        validate_modality_exists(val_dir, 'seg')
    except ValueError as e:
        logger.warning(f"Validation directory misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    # Load seg dataset
    seg_dataset = NiFTIDataset(
        data_dir=val_dir, mr_sequence='seg', transform=transform
    )

    # Collect all slices from all modalities
    all_slices: List[Tuple[np.ndarray, np.ndarray, int]] = []

    for key in image_keys:
        mode_id = MODE_ID_MAP.get(key, 0)
        image_dataset = NiFTIDataset(
            data_dir=val_dir, mr_sequence=key, transform=transform
        )
        slices = _extract_slices_with_mode_id(image_dataset, seg_dataset, mode_id)
        all_slices.extend(slices)

    val_dataset = MultiDiffusionDataset(all_slices)

    # Get DataLoader settings
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


def create_multi_diffusion_test_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create test dataloader for multi-modality diffusion.

    Args:
        cfg: Hydra configuration with paths.
        image_keys: List of MR sequences to load.

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, "test_new")
    if not os.path.exists(test_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = cfg.training.batch_size

    # Validate modalities exist
    try:
        for key in image_keys:
            validate_modality_exists(test_dir, key)
        validate_modality_exists(test_dir, 'seg')
    except ValueError as e:
        logger.warning(f"Test directory misconfigured: {e}")
        return None

    transform = build_standard_transform(image_size)

    # Load seg dataset
    seg_dataset = NiFTIDataset(
        data_dir=test_dir, mr_sequence='seg', transform=transform
    )

    # Collect all slices from all modalities
    all_slices: List[Tuple[np.ndarray, np.ndarray, int]] = []

    for key in image_keys:
        mode_id = MODE_ID_MAP.get(key, 0)
        image_dataset = NiFTIDataset(
            data_dir=test_dir, mr_sequence=key, transform=transform
        )
        slices = _extract_slices_with_mode_id(image_dataset, seg_dataset, mode_id)
        all_slices.extend(slices)

    test_dataset = MultiDiffusionDataset(all_slices)

    # Get DataLoader settings
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


def create_single_modality_diffusion_val_loader(
    cfg: DictConfig,
    modality: str,
) -> Optional[DataLoader]:
    """Create validation loader for a single modality (for per-modality metrics).

    Args:
        cfg: Hydra configuration with paths.
        modality: Single modality to load (e.g., 'bravo', 't1_pre').

    Returns:
        DataLoader for single modality or None if val/ doesn't exist.
    """
    val_dir = os.path.join(cfg.paths.data_dir, "val")
    if not os.path.exists(val_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = cfg.training.batch_size

    # Validate modality exists
    try:
        validate_modality_exists(val_dir, modality)
        validate_modality_exists(val_dir, 'seg')
    except ValueError as e:
        logger.warning(f"Modality {modality} not found in val/: {e}")
        return None

    transform = build_standard_transform(image_size)

    # Load datasets
    image_dataset = NiFTIDataset(
        data_dir=val_dir, mr_sequence=modality, transform=transform
    )
    seg_dataset = NiFTIDataset(
        data_dir=val_dir, mr_sequence='seg', transform=transform
    )

    mode_id = MODE_ID_MAP.get(modality, 0)
    slices = _extract_slices_with_mode_id(image_dataset, seg_dataset, mode_id)
    val_dataset = MultiDiffusionDataset(slices)

    # Get DataLoader settings
    dl_cfg = cfg.training.get('dataloader', {})
    num_workers = dl_cfg.get('num_workers', DEFAULT_NUM_WORKERS)
    prefetch_factor = dl_cfg.get('prefetch_factor', 4) if num_workers > 0 else None
    pin_memory = dl_cfg.get('pin_memory', True)
    persistent_workers = dl_cfg.get('persistent_workers', True) and num_workers > 0

    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    return dataloader
