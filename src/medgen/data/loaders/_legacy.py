"""Backward compatibility aliases for legacy loader functions.

All functions here are DEPRECATED. Use DataLoaderFactory.create() instead.

Example migration:
    # Old (deprecated):
    from medgen.data.loaders import create_vae_dataloader
    loader, dataset = create_vae_dataloader(cfg, modality='bravo', ...)

    # New (recommended):
    from medgen.data.loaders import DataLoaderFactory, LoaderConfig, ModelType
    config = LoaderConfig.from_hydra(cfg, ModelType.VAE, "bravo", "train")
    loader, dataset = DataLoaderFactory.create(config)
"""

from __future__ import annotations

import warnings
from functools import wraps
from typing import TYPE_CHECKING, Any

from torch.utils.data import DataLoader, Dataset

from .configs import LoaderConfig, ModelType, SpatialDims
from .factory import DataLoaderFactory

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _deprecated(message: str):
    """Decorator to mark functions as deprecated.

    Emits DeprecationWarning when function is called (not imported).

    Args:
        message: Deprecation message to include in warning.

    Returns:
        Decorator function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated. {message}",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Legacy Single-Image Loader Functions
# =============================================================================

@_deprecated("Use DataLoaderFactory.create() with LoaderConfig instead.")
def create_single_dataloader(
    cfg: "DictConfig",
    image_type: str,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset]:
    """DEPRECATED: Create single-image training dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.DIFFUSION, image_type, "train",
        use_distributed=use_distributed,
        rank=rank,
        world_size=world_size,
        augment=augment,
    )
    return DataLoaderFactory.create(config)


@_deprecated("Use DataLoaderFactory.create() with split='val'.")
def create_single_validation_dataloader(
    cfg: "DictConfig",
    image_type: str,
    batch_size: int | None = None,
    world_size: int = 1,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset] | None:
    """DEPRECATED: Create single-image validation dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.DIFFUSION, image_type, "val",
        batch_size=batch_size or cfg.training.batch_size,
        world_size=world_size,
    )
    try:
        return DataLoaderFactory.create(config)
    except ValueError:
        return None


@_deprecated("Use DataLoaderFactory.create() with split='test'.")
def create_single_test_dataloader(
    cfg: "DictConfig",
    image_type: str,
    batch_size: int | None = None,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset] | None:
    """DEPRECATED: Create single-image test dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.DIFFUSION, image_type, "test",
        batch_size=batch_size or cfg.training.batch_size,
    )
    try:
        return DataLoaderFactory.create(config)
    except ValueError:
        return None


# =============================================================================
# Legacy Dual-Image Loader Functions
# =============================================================================

@_deprecated("Use DataLoaderFactory.create() with mode='dual' instead.")
def legacy_create_dual_image_dataloader(
    cfg: "DictConfig",
    image_keys: list[str],
    conditioning: str | None = "seg",
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset]:
    """DEPRECATED: Create dual-image training dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.DIFFUSION, "dual", "train",
        use_distributed=use_distributed,
        rank=rank,
        world_size=world_size,
        augment=augment,
    )
    return DataLoaderFactory.create(config)


@_deprecated("Use DataLoaderFactory.create() with mode='dual', split='val'.")
def legacy_create_dual_image_validation_dataloader(
    cfg: "DictConfig",
    image_keys: list[str],
    conditioning: str | None = "seg",
    batch_size: int | None = None,
    world_size: int = 1,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset] | None:
    """DEPRECATED: Create dual-image validation dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.DIFFUSION, "dual", "val",
        batch_size=batch_size or cfg.training.batch_size,
        world_size=world_size,
    )
    try:
        return DataLoaderFactory.create(config)
    except ValueError:
        return None


@_deprecated("Use DataLoaderFactory.create() with mode='dual', split='test'.")
def legacy_create_dual_image_test_dataloader(
    cfg: "DictConfig",
    image_keys: list[str],
    conditioning: str | None = "seg",
    batch_size: int | None = None,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset] | None:
    """DEPRECATED: Create dual-image test dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.DIFFUSION, "dual", "test",
        batch_size=batch_size or cfg.training.batch_size,
    )
    try:
        return DataLoaderFactory.create(config)
    except ValueError:
        return None


# =============================================================================
# Legacy VAE Loader Functions
# =============================================================================

@_deprecated("Use DataLoaderFactory.create() with ModelType.VAE instead.")
def legacy_create_vae_dataloader(
    cfg: "DictConfig",
    modality: str,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset]:
    """DEPRECATED: Create VAE training dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.VAE, modality, "train",
        use_distributed=use_distributed,
        rank=rank,
        world_size=world_size,
        augment=augment,
    )
    return DataLoaderFactory.create(config)


@_deprecated("Use DataLoaderFactory.create() with ModelType.VAE, split='val'.")
def legacy_create_vae_validation_dataloader(
    cfg: "DictConfig",
    modality: str,
    batch_size: int | None = None,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset] | None:
    """DEPRECATED: Create VAE validation dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.VAE, modality, "val",
        batch_size=batch_size or cfg.training.batch_size,
    )
    try:
        return DataLoaderFactory.create(config)
    except ValueError:
        return None


@_deprecated("Use DataLoaderFactory.create() with ModelType.VAE, split='test'.")
def legacy_create_vae_test_dataloader(
    cfg: "DictConfig",
    modality: str,
    batch_size: int | None = None,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset] | None:
    """DEPRECATED: Create VAE test dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.VAE, modality, "test",
        batch_size=batch_size or cfg.training.batch_size,
    )
    try:
        return DataLoaderFactory.create(config)
    except ValueError:
        return None


# =============================================================================
# Legacy Multi-Modality Loader Functions
# =============================================================================

@_deprecated("Use DataLoaderFactory.create() with mode='multi_modality' instead.")
def legacy_create_multi_modality_dataloader(
    cfg: "DictConfig",
    image_keys: list[str],
    image_size: int,
    batch_size: int,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset]:
    """DEPRECATED: Create multi-modality training dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.VAE, "multi_modality", "train",
        batch_size=batch_size,
        use_distributed=use_distributed,
        rank=rank,
        world_size=world_size,
        augment=augment,
    )
    return DataLoaderFactory.create(config)


@_deprecated("Use DataLoaderFactory.create() with mode='multi_modality', split='val'.")
def legacy_create_multi_modality_validation_dataloader(
    cfg: "DictConfig",
    image_keys: list[str],
    image_size: int,
    batch_size: int,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset] | None:
    """DEPRECATED: Create multi-modality validation dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.VAE, "multi_modality", "val",
        batch_size=batch_size,
    )
    try:
        return DataLoaderFactory.create(config)
    except ValueError:
        return None


@_deprecated("Use DataLoaderFactory.create() with mode='multi_modality', split='test'.")
def legacy_create_multi_modality_test_dataloader(
    cfg: "DictConfig",
    image_keys: list[str],
    image_size: int,
    batch_size: int,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset] | None:
    """DEPRECATED: Create multi-modality test dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.VAE, "multi_modality", "test",
        batch_size=batch_size,
    )
    try:
        return DataLoaderFactory.create(config)
    except ValueError:
        return None


# =============================================================================
# Legacy Multi-Diffusion Loader Functions
# =============================================================================

@_deprecated("Use DataLoaderFactory.create() with mode='multi' instead.")
def legacy_create_multi_diffusion_dataloader(
    cfg: "DictConfig",
    image_keys: list[str],
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset]:
    """DEPRECATED: Create multi-diffusion training dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.DIFFUSION, "multi", "train",
        use_distributed=use_distributed,
        rank=rank,
        world_size=world_size,
        augment=augment,
    )
    return DataLoaderFactory.create(config)


# =============================================================================
# Legacy Seg-Conditioned Loader Functions
# =============================================================================

@_deprecated("Use DataLoaderFactory.create() with mode='seg_conditioned' instead.")
def legacy_create_seg_conditioned_dataloader(
    cfg: "DictConfig",
    size_bin_config: dict,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset]:
    """DEPRECATED: Create seg_conditioned training dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.DIFFUSION, "seg_conditioned", "train",
        use_distributed=use_distributed,
        rank=rank,
        world_size=world_size,
        augment=augment,
    )
    return DataLoaderFactory.create(config)


@_deprecated("Use DataLoaderFactory.create() with mode='seg_conditioned', split='val'.")
def legacy_create_seg_conditioned_validation_dataloader(
    cfg: "DictConfig",
    size_bin_config: dict,
    batch_size: int | None = None,
    world_size: int = 1,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset] | None:
    """DEPRECATED: Create seg_conditioned validation dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.DIFFUSION, "seg_conditioned", "val",
        batch_size=batch_size or cfg.training.batch_size,
        world_size=world_size,
    )
    try:
        return DataLoaderFactory.create(config)
    except ValueError:
        return None


# =============================================================================
# Legacy Seg-Compression Loader Functions
# =============================================================================

@_deprecated("Use DataLoaderFactory.create() with mode='seg_compression' instead.")
def legacy_create_seg_compression_dataloader(
    cfg: "DictConfig",
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset]:
    """DEPRECATED: Create seg_compression training dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.DCAE, "seg_compression", "train",
        use_distributed=use_distributed,
        rank=rank,
        world_size=world_size,
        augment=augment,
    )
    return DataLoaderFactory.create(config)


@_deprecated("Use DataLoaderFactory.create() with mode='seg_compression', split='val'.")
def legacy_create_seg_compression_validation_dataloader(
    cfg: "DictConfig",
    image_size: int,
    batch_size: int,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset] | None:
    """DEPRECATED: Create seg_compression validation dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.DCAE, "seg_compression", "val",
        batch_size=batch_size,
    )
    try:
        return DataLoaderFactory.create(config)
    except ValueError:
        return None


# =============================================================================
# Legacy 3D VAE Loader Functions
# =============================================================================

@_deprecated("Use DataLoaderFactory.create() with ModelType.VAE, SpatialDims.THREE_D.")
def legacy_create_vae_3d_dataloader(
    cfg: "DictConfig",
    modality: str,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset]:
    """DEPRECATED: Create 3D VAE training dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.VAE, modality, "train",
        use_distributed=use_distributed,
        rank=rank,
        world_size=world_size,
    )
    return DataLoaderFactory.create(config)


@_deprecated("Use DataLoaderFactory.create() with ModelType.VAE, SpatialDims.THREE_D, split='val'.")
def legacy_create_vae_3d_validation_dataloader(
    cfg: "DictConfig",
    modality: str,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset] | None:
    """DEPRECATED: Create 3D VAE validation dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.VAE, modality, "val",
    )
    try:
        return DataLoaderFactory.create(config)
    except ValueError:
        return None


@_deprecated("Use DataLoaderFactory.create() with ModelType.VAE, SpatialDims.THREE_D, split='test'.")
def legacy_create_vae_3d_test_dataloader(
    cfg: "DictConfig",
    modality: str,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset] | None:
    """DEPRECATED: Create 3D VAE test dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.VAE, modality, "test",
    )
    try:
        return DataLoaderFactory.create(config)
    except ValueError:
        return None


@_deprecated("Use DataLoaderFactory.create() with mode='multi_modality', SpatialDims.THREE_D.")
def legacy_create_vae_3d_multi_modality_dataloader(
    cfg: "DictConfig",
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    **kwargs: Any,
) -> tuple[DataLoader, Dataset]:
    """DEPRECATED: Create 3D VAE multi-modality training dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.VAE, "multi_modality", "train",
        use_distributed=use_distributed,
        rank=rank,
        world_size=world_size,
    )
    return DataLoaderFactory.create(config)


@_deprecated("Use DataLoaderFactory.create() with mode='multi_modality', SpatialDims.THREE_D, split='val'.")
def legacy_create_vae_3d_multi_modality_validation_dataloader(
    cfg: "DictConfig",
    **kwargs: Any,
) -> tuple[DataLoader, Dataset] | None:
    """DEPRECATED: Create 3D VAE multi-modality validation dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.VAE, "multi_modality", "val",
    )
    try:
        return DataLoaderFactory.create(config)
    except ValueError:
        return None


@_deprecated("Use DataLoaderFactory.create() with mode='multi_modality', SpatialDims.THREE_D, split='test'.")
def legacy_create_vae_3d_multi_modality_test_dataloader(
    cfg: "DictConfig",
    **kwargs: Any,
) -> tuple[DataLoader, Dataset] | None:
    """DEPRECATED: Create 3D VAE multi-modality test dataloader.

    Use DataLoaderFactory.create() instead.
    """
    config = LoaderConfig.from_hydra(
        cfg, ModelType.VAE, "multi_modality", "test",
    )
    try:
        return DataLoaderFactory.create(config)
    except ValueError:
        return None
