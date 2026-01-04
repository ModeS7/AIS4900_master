"""Common utilities for data loaders.

Provides shared functions to reduce duplication across loader modules:
- DataLoader configuration extraction
- Distributed sampler setup
- Data directory validation
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from omegaconf import DictConfig
from torch.utils.data import Dataset, DistributedSampler

from medgen.core.constants import DEFAULT_NUM_WORKERS


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
