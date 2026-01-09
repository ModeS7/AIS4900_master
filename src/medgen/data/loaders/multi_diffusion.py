"""
Multi-modality dataloaders for diffusion training with mode embedding.

Provides dataloaders that pool multiple MR modalities, each paired with seg mask
and mode_id for training a single model on all modalities.

Uses lazy loading to avoid storing all slices in memory (~125GB for 400 patients).
"""
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from omegaconf import DictConfig

from medgen.data.augmentation import build_diffusion_augmentation
from medgen.data.loaders.common import DataLoaderConfig, setup_distributed_sampler
from medgen.data.dataset import (
    NiFTIDataset,
    build_standard_transform,
    validate_modality_exists,
)
from medgen.data.mode_embed import MODE_ID_MAP
from medgen.core.constants import BINARY_THRESHOLD_GT

logger = logging.getLogger(__name__)


class LazyMultiDiffusionDataset(Dataset):
    """Dataset that lazily loads slices from disk with mode_id.

    Instead of storing all slices in memory (which would require ~125GB for
    400 patients x 160 slices x 4 modalities), this stores only metadata
    and loads data on demand in __getitem__.

    Each sample is (image, seg, mode_id) where:
    - image: [1, H, W] single-channel image
    - seg: [1, H, W] binary segmentation mask
    - mode_id: int (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd)
    """

    def __init__(
        self,
        image_datasets: Dict[str, NiFTIDataset],
        seg_dataset: NiFTIDataset,
        image_keys: List[str],
        augmentation: Optional[Callable] = None,
    ):
        """Initialize lazy dataset.

        Args:
            image_datasets: Dict mapping modality key to NiFTIDataset.
            seg_dataset: NiFTIDataset for segmentation masks.
            image_keys: List of modality keys to include.
            augmentation: Optional albumentations Compose for data augmentation.
        """
        self.image_datasets = image_datasets
        self.seg_dataset = seg_dataset
        self.image_keys = image_keys
        self.augmentation = augmentation

        # Build index: list of (modality_key, volume_idx, slice_idx, mode_id)
        self.index: List[Tuple[str, int, int, int]] = []
        self._build_index()

    def _build_index(self):
        """Build index of all valid slices without loading data."""
        for key in self.image_keys:
            mode_id = MODE_ID_MAP.get(key)
            if mode_id is None:
                raise ValueError(
                    f"Unknown modality key '{key}'. "
                    f"Valid keys: {list(MODE_ID_MAP.keys())}"
                )

            image_dataset = self.image_datasets[key]
            n_volumes = len(image_dataset)

            for vol_idx in range(n_volumes):
                # Get volume to count slices (loads header only for shape)
                image_volume, _ = image_dataset[vol_idx]
                n_slices = image_volume.shape[3]

                for slice_idx in range(n_slices):
                    # Check if slice is non-empty (quick sum check)
                    image_slice = np.array(image_volume[:, :, :, slice_idx])
                    if np.sum(image_slice) > 1.0:
                        self.index.append((key, vol_idx, slice_idx, mode_id))

        logger.info(f"LazyMultiDiffusionDataset: indexed {len(self.index)} slices")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        key, vol_idx, slice_idx, mode_id = self.index[idx]

        # Load data from disk
        image_volume, _ = self.image_datasets[key][vol_idx]
        seg_volume, _ = self.seg_dataset[vol_idx]

        # Extract slice
        image_slice = np.array(image_volume[:, :, :, slice_idx])
        seg_slice = np.array(seg_volume[:, :, :, slice_idx])

        # Apply augmentation if provided
        if self.augmentation is not None:
            # Transpose to [H, W, C] for albumentations
            img_hwc = np.transpose(image_slice, (1, 2, 0))
            seg_hwc = np.transpose(seg_slice, (1, 2, 0))

            transformed = self.augmentation(image=img_hwc, mask=seg_hwc)
            img_aug = transformed['image']
            seg_aug = transformed['mask']

            # Transpose back to [C, H, W]
            image_slice = np.transpose(img_aug, (2, 0, 1))
            seg_slice = np.transpose(seg_aug, (2, 0, 1))

        # Binarize seg mask (using consistent threshold with utils.py)
        seg_slice = (seg_slice > BINARY_THRESHOLD_GT).astype(np.float32)

        return (
            torch.from_numpy(image_slice).float(),
            torch.from_numpy(seg_slice).float(),
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

    Uses lazy loading to avoid storing all slices in memory.

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

    # Create image datasets for each modality
    image_datasets: Dict[str, NiFTIDataset] = {}
    for key in image_keys:
        image_datasets[key] = NiFTIDataset(
            data_dir=data_dir, mr_sequence=key, transform=transform
        )

    # Create lazy dataset
    train_dataset = LazyMultiDiffusionDataset(
        image_datasets=image_datasets,
        seg_dataset=seg_dataset,
        image_keys=image_keys,
        augmentation=aug,
    )
    logger.info(f"Total training slices: {len(train_dataset)}")

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

    # Create image datasets for each modality
    image_datasets: Dict[str, NiFTIDataset] = {}
    for key in image_keys:
        image_datasets[key] = NiFTIDataset(
            data_dir=val_dir, mr_sequence=key, transform=transform
        )

    # Create lazy dataset (no augmentation for validation)
    val_dataset = LazyMultiDiffusionDataset(
        image_datasets=image_datasets,
        seg_dataset=seg_dataset,
        image_keys=image_keys,
        augmentation=None,
    )

    # Get DataLoader settings
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


def create_multi_diffusion_test_dataloader(
    cfg: DictConfig,
    image_keys: List[str],
    world_size: int = 1,
) -> Optional[Tuple[DataLoader, Dataset]]:
    """Create test dataloader for multi-modality diffusion.

    Args:
        cfg: Hydra configuration with paths.
        image_keys: List of MR sequences to load.
        world_size: Number of GPUs for DDP (for batch size adjustment).

    Returns:
        Tuple of (DataLoader, test_dataset) or None if test_new/ doesn't exist.
    """
    test_dir = os.path.join(cfg.paths.data_dir, "test_new")
    if not os.path.exists(test_dir):
        return None

    image_size = cfg.model.image_size
    batch_size = cfg.training.batch_size

    # Reduce batch size for DDP (test runs on single GPU)
    if world_size > 1:
        batch_size = max(1, batch_size // world_size)

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

    # Create image datasets for each modality
    image_datasets: Dict[str, NiFTIDataset] = {}
    for key in image_keys:
        image_datasets[key] = NiFTIDataset(
            data_dir=test_dir, mr_sequence=key, transform=transform
        )

    # Create lazy dataset (no augmentation for test)
    test_dataset = LazyMultiDiffusionDataset(
        image_datasets=image_datasets,
        seg_dataset=seg_dataset,
        image_keys=image_keys,
        augmentation=None,
    )

    # Get DataLoader settings
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

    # Create lazy dataset for single modality
    val_dataset = LazyMultiDiffusionDataset(
        image_datasets={modality: image_dataset},
        seg_dataset=seg_dataset,
        image_keys=[modality],
        augmentation=None,
    )

    # Get DataLoader settings
    dl_cfg = DataLoaderConfig.from_cfg(cfg)

    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )

    return dataloader
