"""Common utilities for data loaders.

Provides shared functions to reduce duplication across loader modules:
- DataLoader configuration extraction
- Distributed sampler setup
- Data directory validation
- Modality validation helpers
"""
import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from omegaconf import DictConfig
from torch.utils.data import Dataset, DistributedSampler

from medgen.core.constants import DEFAULT_NUM_WORKERS

logger = logging.getLogger(__name__)

# Modality key mappings - centralized definition
MODALITY_KEYS = {
    'dual': ['t1_pre', 't1_gd'],
    'multi_modality': ['t1_pre', 't1_gd', 'bravo', 't2_flair'],
}


@dataclass
class DataLoaderConfig:
    """DataLoader configuration extracted from Hydra config."""
    num_workers: int
    prefetch_factor: Optional[int]
    pin_memory: bool
    persistent_workers: bool

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> 'DataLoaderConfig':
        """Extract DataLoader settings from cfg.training.dataloader.

        Args:
            cfg: Hydra configuration object.

        Returns:
            DataLoaderConfig with extracted settings.
        """
        dl_cfg = cfg.training.get('dataloader', {})
        num_workers = dl_cfg.get('num_workers', DEFAULT_NUM_WORKERS)
        prefetch_factor = dl_cfg.get('prefetch_factor', 4) if num_workers > 0 else None
        pin_memory = dl_cfg.get('pin_memory', True)
        persistent_workers = dl_cfg.get('persistent_workers', True) and num_workers > 0

        return cls(
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )


def setup_distributed_sampler(
    dataset: Dataset,
    use_distributed: bool,
    rank: int,
    world_size: int,
    batch_size: int,
    shuffle: bool = True,
) -> Tuple[Optional[DistributedSampler], int, bool]:
    """Setup DistributedSampler and compute per-GPU batch size.

    Args:
        dataset: Dataset to wrap with sampler.
        use_distributed: Whether to use distributed sampling.
        rank: Process rank for distributed training.
        world_size: Total number of processes.
        batch_size: Total batch size (will be divided by world_size).
        shuffle: Whether to shuffle (ignored if distributed, handled by sampler).

    Returns:
        Tuple of (sampler, batch_size_per_gpu, actual_shuffle).
        - sampler: DistributedSampler or None
        - batch_size_per_gpu: Adjusted batch size for this GPU
        - actual_shuffle: Whether DataLoader should shuffle
    """
    if use_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        batch_size_per_gpu = batch_size // world_size
        actual_shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        batch_size_per_gpu = batch_size
        actual_shuffle = shuffle

    return sampler, batch_size_per_gpu, actual_shuffle


def get_data_dir(cfg: DictConfig, split: str) -> Optional[str]:
    """Get data directory for a split, return None if doesn't exist.

    Args:
        cfg: Hydra configuration object.
        split: Data split name ('train', 'val', 'test_new').

    Returns:
        Path to data directory or None if it doesn't exist.
    """
    data_dir = os.path.join(cfg.paths.data_dir, split)
    if not os.path.exists(data_dir):
        return None
    return data_dir


def validate_data_dir(cfg: DictConfig, split: str, required: bool = True) -> Optional[str]:
    """Validate and return data directory for a split.

    Args:
        cfg: Hydra configuration object.
        split: Data split name ('train', 'val', 'test_new').
        required: If True, raise ValueError if directory doesn't exist.

    Returns:
        Path to data directory.

    Raises:
        ValueError: If required=True and directory doesn't exist.
    """
    data_dir = get_data_dir(cfg, split)
    if data_dir is None and required:
        raise ValueError(f"Required data directory '{split}' not found at {cfg.paths.data_dir}")
    return data_dir


@dataclass
class DistributedArgs:
    """Arguments for distributed training setup."""
    use_distributed: bool = False
    rank: int = 0
    world_size: int = 1


def create_dataloader(
    dataset: Dataset,
    cfg: Optional[DictConfig] = None,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None,
    distributed_args: Optional[DistributedArgs] = None,
    loader_config: Optional[DataLoaderConfig] = None,
    scale_batch_for_distributed: bool = True,
) -> 'DataLoader':
    """Create DataLoader with standard configuration.

    This is the unified DataLoader creation function that extracts ~20 lines
    of duplicated code from each loader function. Use this instead of
    manually creating DataLoaders.

    Args:
        dataset: Dataset to wrap.
        cfg: Hydra configuration (for training.batch_size and training.dataloader).
            Required unless loader_config and batch_size are provided.
        batch_size: Override batch size (default: cfg.training.batch_size).
        shuffle: Whether to shuffle (ignored if distributed, handled by sampler).
        drop_last: Whether to drop last incomplete batch.
        collate_fn: Optional custom collate function.
        distributed_args: Optional distributed training configuration.
        loader_config: Pre-extracted DataLoaderConfig (alternative to cfg).
        scale_batch_for_distributed: If True, divide batch_size by world_size
            for distributed training. Set False for 3D volumes where batch_size
            is typically 1-2 and shouldn't be divided.

    Returns:
        Configured DataLoader.

    Example:
        >>> dataset = MyDataset(...)
        >>> loader = create_dataloader(dataset, cfg, shuffle=True)
        >>> for batch in loader:
        ...     train_step(batch)

        >>> # With pre-extracted config (for 3D loaders)
        >>> dl_cfg = DataLoaderConfig.from_cfg(cfg)
        >>> loader = create_dataloader(
        ...     dataset, batch_size=2, loader_config=dl_cfg,
        ...     scale_batch_for_distributed=False
        ... )
    """
    from torch.utils.data import DataLoader

    # Get batch size from config if not provided
    if batch_size is None:
        if cfg is None:
            raise ValueError("Either cfg or batch_size must be provided")
        batch_size = cfg.training.batch_size

    # Setup distributed sampler if needed
    distributed = distributed_args or DistributedArgs()
    sampler, batch_size_per_gpu, actual_shuffle = setup_distributed_sampler(
        dataset,
        distributed.use_distributed,
        distributed.rank,
        distributed.world_size,
        batch_size,
        shuffle=shuffle,
    )

    # Don't scale batch for 3D volumes
    if not scale_batch_for_distributed:
        batch_size_per_gpu = batch_size

    # Get DataLoader settings from config or use provided
    if loader_config is not None:
        dl_cfg = loader_config
    elif cfg is not None:
        dl_cfg = DataLoaderConfig.from_cfg(cfg)
    else:
        raise ValueError("Either cfg or loader_config must be provided")

    return DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=actual_shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor,
        persistent_workers=dl_cfg.persistent_workers,
    )


# =============================================================================
# Modality validation helpers
# =============================================================================

def get_modality_keys(modality: str) -> List[str]:
    """Get image keys for a modality.

    Expands composite modalities like 'dual' into their constituent keys.

    Args:
        modality: Modality name ('dual', 'multi_modality', 't1_pre', 'bravo', etc.)

    Returns:
        List of image keys to load.

    Example:
        >>> get_modality_keys('dual')
        ['t1_pre', 't1_gd']
        >>> get_modality_keys('bravo')
        ['bravo']
    """
    return MODALITY_KEYS.get(modality, [modality])


def validate_modality_keys(
    data_dir: str,
    modality: str,
    validate_fn: Callable,
) -> List[str]:
    """Validate modality files exist and return keys to load.

    Combines get_modality_keys() with validation in a single call.

    Args:
        data_dir: Path to data directory.
        modality: Modality name ('dual', 'multi_modality', 't1_pre', etc.)
        validate_fn: Function to validate existence (e.g., validate_modality_exists).

    Returns:
        List of validated modality keys.

    Raises:
        ValueError: If any required modality file doesn't exist.

    Example:
        >>> from medgen.data import validate_modality_exists
        >>> keys = validate_modality_keys('/data/train', 'dual', validate_modality_exists)
        >>> # keys = ['t1_pre', 't1_gd'], both validated to exist
    """
    keys = get_modality_keys(modality)
    for key in keys:
        validate_fn(data_dir, key)
    return keys


def check_seg_available(data_dir: str, validate_fn: callable) -> bool:
    """Check if seg modality is available without raising exception.

    Use this instead of try/except around validate_modality_exists.

    Args:
        data_dir: Path to data directory.
        validate_fn: Function to validate existence (e.g., validate_modality_exists).

    Returns:
        True if seg exists, False otherwise.

    Example:
        >>> from medgen.data import validate_modality_exists
        >>> has_seg = check_seg_available('/data/train', validate_modality_exists)
        >>> if has_seg:
        ...     seg_dataset = NiFTIDataset(data_dir, 'seg', transform)
    """
    try:
        validate_fn(data_dir, 'seg')
        return True
    except ValueError:
        return False
