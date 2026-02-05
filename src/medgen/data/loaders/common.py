"""Common utilities for data loaders.

Provides shared functions to reduce duplication across loader modules:
- DataLoader configuration extraction
- Distributed sampler setup
- Data directory validation
- Modality validation helpers
"""
import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

if TYPE_CHECKING:
    pass  # DataLoader imported above for both runtime and type checking

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
    prefetch_factor: int | None
    pin_memory: bool
    persistent_workers: bool

    def __post_init__(self):
        """Validate configuration values."""
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError(f"prefetch_factor must be > 0 when set, got {self.prefetch_factor}")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers=True requires num_workers > 0")

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
) -> tuple[DistributedSampler | None, int, bool]:
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

    Raises:
        ValueError: If world_size <= 0, batch_size <= 0, or rank out of range.
    """
    # Validate parameters
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if use_distributed and not (0 <= rank < world_size):
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

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


class GroupedBatchSampler(Sampler[list[int]]):
    """Batch sampler that ensures all samples in a batch belong to the same group.

    Used for multi-modality training where mode embedding requires homogeneous
    batches (all samples must have the same mode_id).

    Groups are shuffled each epoch, and samples within each group are shuffled.
    Batches are formed from consecutive samples within a group.

    Args:
        group_ids: List where group_ids[i] is the group ID for sample i.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle groups and samples within groups each epoch.
        drop_last: Whether to drop the last incomplete batch in each group.
        generator: Optional torch.Generator for reproducible shuffling.

    Example:
        >>> # Dataset with 100 samples, 4 groups (mode_ids 0-3)
        >>> group_ids = [sample[2] for sample in dataset]  # Extract mode_ids
        >>> generator = torch.Generator()
        >>> generator.manual_seed(42)
        >>> sampler = GroupedBatchSampler(group_ids, batch_size=16, generator=generator)
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        group_ids: list[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
    ):
        # Validate parameters
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if not group_ids:
            raise ValueError("group_ids cannot be empty")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

        # Build index groups: {group_id: [sample_indices]}
        self.groups: dict[int, list[int]] = {}
        for idx, gid in enumerate(group_ids):
            if gid not in self.groups:
                self.groups[gid] = []
            self.groups[gid].append(idx)

        self.group_ids = list(self.groups.keys())
        logger.debug(
            f"GroupedBatchSampler: {len(group_ids)} samples, "
            f"{len(self.groups)} groups, batch_size={batch_size}"
        )

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of indices, each batch from a single group."""
        # Shuffle group order using torch for reproducibility
        group_order = self.group_ids.copy()
        if self.shuffle:
            perm = torch.randperm(len(group_order), generator=self.generator)
            group_order = [group_order[i] for i in perm.tolist()]

        for gid in group_order:
            indices = self.groups[gid].copy()
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=self.generator)
                indices = [indices[i] for i in perm.tolist()]

            # Yield full batches
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        """Total number of batches."""
        total = 0
        for indices in self.groups.values():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size != 0:
                n_batches += 1
            total += n_batches
        return total


def get_data_dir(cfg: DictConfig, split: str) -> str | None:
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


def validate_data_dir(cfg: DictConfig, split: str, required: bool = True) -> str | None:
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
    cfg: DictConfig | None = None,
    batch_size: int | None = None,
    shuffle: bool = True,
    drop_last: bool = False,
    collate_fn: Callable | None = None,
    distributed_args: DistributedArgs | None = None,
    loader_config: DataLoaderConfig | None = None,
    scale_batch_for_distributed: bool = True,
    generator: torch.Generator | None = None,
) -> DataLoader:
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
        generator: Optional torch.Generator for reproducible shuffling.
            Useful for validation loaders that need deterministic ordering.

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
        generator=generator,
    )


# =============================================================================
# Modality validation helpers
# =============================================================================

def get_modality_keys(modality: str) -> list[str]:
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
) -> list[str]:
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


def get_validated_split_dir(
    base_dir: str,
    split: str,
    logger: logging.Logger | None = None,
    raise_on_missing: bool = False,
) -> str | None:
    """Get split directory path if it exists.

    This utility centralizes the common pattern of checking for split directories
    (train, val, test_new) and returning None if not found.

    Args:
        base_dir: Base data directory path.
        split: Split name ('train', 'val', 'test_new').
        logger: Optional logger for debug messages.
        raise_on_missing: If True, raise ValueError instead of returning None.

    Returns:
        Full path to split directory, or None if not found.

    Raises:
        ValueError: If raise_on_missing=True and directory doesn't exist.

    Example:
        >>> # Returns None if missing
        >>> val_dir = get_validated_split_dir(data_dir, 'val', logger)
        >>> if val_dir is None:
        ...     return None
        >>> # Or raise if required
        >>> train_dir = get_validated_split_dir(data_dir, 'train', raise_on_missing=True)
    """
    split_dir = os.path.join(base_dir, split)
    if not os.path.exists(split_dir):
        if logger:
            logger.debug(f"{split} directory not found: {split_dir}")
        if raise_on_missing:
            raise ValueError(f"Required directory not found: {split_dir}")
        return None
    return split_dir


def validate_mode_requirements(
    data_dir: str,
    mode: str,
    validate_fn: callable,
    image_keys: list[str] | None = None,
    require_seg: bool = True,
) -> None:
    """Validate all modalities required for a training mode.

    Centralizes validation logic to avoid duplicated if/elif chains across loaders.

    Args:
        data_dir: Path to data directory.
        mode: Training mode ('seg', 'bravo', 'dual', 'multi', 'multi_modality',
              'seg_conditioned', 'seg_conditioned_input', 'dual_vae', 'multi_modality_vae').
        validate_fn: Function to validate existence (e.g., validate_modality_exists).
        image_keys: Optional explicit keys (for dual/multi modes).
        require_seg: If False, skip seg validation even if mode would normally require it.
            Useful for VAE training where seg is optional.

    Raises:
        ValueError: If required modalities are missing or mode is unknown.

    Example:
        >>> from medgen.data import validate_modality_exists
        >>> validate_mode_requirements('/data/train', 'bravo', validate_modality_exists)
        >>> validate_mode_requirements('/data/train', 'dual', validate_modality_exists,
        ...                            image_keys=['t1_pre', 't1_gd'])
        >>> # VAE training (no seg required):
        >>> validate_mode_requirements('/data/train', 'multi_modality', validate_modality_exists,
        ...                            image_keys=['bravo', 'flair'], require_seg=False)
    """
    if mode == 'seg':
        validate_fn(data_dir, 'seg')
    elif mode == 'bravo':
        validate_fn(data_dir, 'bravo')
        if require_seg:
            validate_fn(data_dir, 'seg')
    elif mode in ('dual', 'dual_vae'):
        keys = image_keys or ['t1_pre', 't1_gd']
        for key in keys:
            validate_fn(data_dir, key)
        if require_seg and mode != 'dual_vae':
            validate_fn(data_dir, 'seg')
    elif mode in ('multi', 'multi_modality', 'multi_modality_vae'):
        keys = image_keys or ['bravo', 'flair', 't1_pre', 't1_gd']
        for key in keys:
            validate_fn(data_dir, key)
        if require_seg and mode != 'multi_modality_vae':
            validate_fn(data_dir, 'seg')
    elif mode in ('seg_conditioned', 'seg_conditioned_input'):
        validate_fn(data_dir, 'seg')
    else:
        raise ValueError(f"Unknown mode: {mode}")
