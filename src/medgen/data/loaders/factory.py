"""Unified data loader factory with registry pattern.

This module provides a single entry point for creating all data loaders
via a typed factory pattern, replacing the sprawl of 26+ mode conditionals.

Usage:
    from medgen.data.loaders import DataLoaderFactory, LoaderConfig, ModelType

    # Create typed config
    config = LoaderConfig.from_hydra(cfg, ModelType.DIFFUSION, "bravo", "train")

    # Create loader via factory
    loader, dataset = DataLoaderFactory.create(config)

The factory uses a registry pattern where mode handlers are registered
via decorators, allowing new modes to be added without modifying factory code.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from torch.utils.data import DataLoader, Dataset

from .configs import LoaderConfig, ModelType, SpatialDims

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Type alias for handler functions
HandlerFn = Callable[[LoaderConfig], tuple[DataLoader, Dataset]]


class DataLoaderFactory:
    """Single entry point for all data loaders.

    Uses a registry pattern to map (model_type, spatial_dims, mode) to
    handler functions. New modes are added by decorating functions with
    @DataLoaderFactory.register().

    Example:
        >>> config = LoaderConfig.from_hydra(cfg, ModelType.DIFFUSION, "bravo", "train")
        >>> loader, dataset = DataLoaderFactory.create(config)

    Adding a new mode:
        >>> @DataLoaderFactory.register(ModelType.DIFFUSION, SpatialDims.TWO_D, "new_mode")
        >>> def _create_new_mode_2d(config: LoaderConfig) -> tuple[DataLoader, Dataset]:
        ...     # Implementation
        ...     pass
    """

    # Registry: {(model_type, spatial_dims): {mode: handler_fn}}
    _registry: dict[tuple[ModelType, SpatialDims], dict[str, HandlerFn]] = {}

    @classmethod
    def register(
        cls,
        model_type: ModelType,
        spatial_dims: SpatialDims,
        mode: str,
    ) -> Callable[[HandlerFn], HandlerFn]:
        """Decorator to register a mode handler.

        Args:
            model_type: Type of model (DIFFUSION, VAE, etc.).
            spatial_dims: Spatial dimensions (TWO_D or THREE_D).
            mode: Mode name (e.g., 'bravo', 'dual', 'multi').

        Returns:
            Decorator function.

        Example:
            >>> @DataLoaderFactory.register(ModelType.DIFFUSION, SpatialDims.TWO_D, "bravo")
            >>> def _create_bravo_2d(config: LoaderConfig) -> tuple[DataLoader, Dataset]:
            ...     pass
        """
        def decorator(func: HandlerFn) -> HandlerFn:
            key = (model_type, spatial_dims)
            if key not in cls._registry:
                cls._registry[key] = {}
            cls._registry[key][mode] = func
            logger.debug(
                f"Registered handler for {model_type.name} {spatial_dims.value}D mode={mode}"
            )
            return func
        return decorator

    @classmethod
    def create(cls, config: LoaderConfig) -> tuple[DataLoader, Dataset]:
        """Create dataloader from typed configuration.

        Args:
            config: Validated LoaderConfig instance.

        Returns:
            Tuple of (DataLoader, Dataset).

        Raises:
            ValueError: If mode is not registered for the given model_type/spatial_dims.
        """
        key = (config.model_type, config.spatial_dims)
        registry = cls._registry.get(key, {})

        if config.mode not in registry:
            available = list(registry.keys()) if registry else []
            raise ValueError(
                f"Unknown mode '{config.mode}' for {config.model_type.name} "
                f"{config.spatial_dims.value}D. Available: {available}"
            )

        handler = registry[config.mode]
        logger.debug(
            f"Creating {config.model_type.name} {config.spatial_dims.value}D "
            f"mode={config.mode} split={config.split}"
        )
        return handler(config)

    @classmethod
    def get_available_modes(
        cls,
        model_type: ModelType,
        spatial_dims: SpatialDims,
    ) -> list[str]:
        """Get list of available modes for a model type and spatial dims.

        Args:
            model_type: Type of model.
            spatial_dims: Spatial dimensions.

        Returns:
            List of registered mode names.
        """
        key = (model_type, spatial_dims)
        return list(cls._registry.get(key, {}).keys())

    @classmethod
    def is_registered(
        cls,
        model_type: ModelType,
        spatial_dims: SpatialDims,
        mode: str,
    ) -> bool:
        """Check if a mode is registered.

        Args:
            model_type: Type of model.
            spatial_dims: Spatial dimensions.
            mode: Mode name to check.

        Returns:
            True if mode is registered, False otherwise.
        """
        key = (model_type, spatial_dims)
        return mode in cls._registry.get(key, {})


# =============================================================================
# Diffusion 2D Handlers
# =============================================================================

@DataLoaderFactory.register(ModelType.DIFFUSION, SpatialDims.TWO_D, "seg")
def _create_seg_2d(config: LoaderConfig) -> tuple[DataLoader, Dataset]:
    """Create 2D segmentation mode loader."""
    from .single import create_dataloader, create_test_dataloader, create_validation_dataloader

    if config.split == "train":
        return create_dataloader(
            cfg=_build_cfg_proxy(config),
            image_type="seg",
            use_distributed=config.use_distributed,
            rank=config.rank,
            world_size=config.world_size,
            augment=config.should_augment,
            augment_type="diffusion",
        )
    elif config.split == "val":
        result = create_validation_dataloader(
            cfg=_build_cfg_proxy(config),
            image_type="seg",
            batch_size=config.batch_size,
            world_size=config.world_size,
        )
        if result is None:
            raise ValueError("No validation data found for seg mode")
        return result
    else:  # test
        result = create_test_dataloader(
            cfg=_build_cfg_proxy(config),
            image_type="seg",
            batch_size=config.batch_size,
        )
        if result is None:
            raise ValueError("No test data found for seg mode")
        return result


@DataLoaderFactory.register(ModelType.DIFFUSION, SpatialDims.TWO_D, "bravo")
def _create_bravo_2d(config: LoaderConfig) -> tuple[DataLoader, Dataset]:
    """Create 2D BRAVO mode loader."""
    from .single import create_dataloader, create_test_dataloader, create_validation_dataloader

    if config.split == "train":
        return create_dataloader(
            cfg=_build_cfg_proxy(config),
            image_type="bravo",
            use_distributed=config.use_distributed,
            rank=config.rank,
            world_size=config.world_size,
            augment=config.should_augment,
            augment_type="diffusion",
            cfg_dropout_prob=config.cfg_dropout_prob,
        )
    elif config.split == "val":
        result = create_validation_dataloader(
            cfg=_build_cfg_proxy(config),
            image_type="bravo",
            batch_size=config.batch_size,
            world_size=config.world_size,
        )
        if result is None:
            raise ValueError("No validation data found for bravo mode")
        return result
    else:  # test
        result = create_test_dataloader(
            cfg=_build_cfg_proxy(config),
            image_type="bravo",
            batch_size=config.batch_size,
        )
        if result is None:
            raise ValueError("No test data found for bravo mode")
        return result


@DataLoaderFactory.register(ModelType.DIFFUSION, SpatialDims.TWO_D, "bravo_seg_cond")
def _create_bravo_seg_cond_2d(config: LoaderConfig) -> tuple[DataLoader, Dataset]:
    """Create 2D BRAVO with seg conditioning mode loader.

    Same pixel loader as bravo, latent handling is separate.
    """
    return _create_bravo_2d(config)


@DataLoaderFactory.register(ModelType.DIFFUSION, SpatialDims.TWO_D, "dual")
def _create_dual_2d(config: LoaderConfig) -> tuple[DataLoader, Dataset]:
    """Create 2D dual mode loader (T1_pre + T1_gd with seg conditioning)."""
    from .dual import (
        create_dual_image_dataloader,
        create_dual_image_test_dataloader,
        create_dual_image_validation_dataloader,
    )

    image_keys = list(config.image_keys[:2]) if len(config.image_keys) >= 2 else ["t1_pre", "t1_gd"]
    conditioning = config.conditioning

    if config.split == "train":
        return create_dual_image_dataloader(
            cfg=_build_cfg_proxy(config),
            image_keys=image_keys,
            conditioning=conditioning,
            use_distributed=config.use_distributed,
            rank=config.rank,
            world_size=config.world_size,
            augment=config.should_augment,
            augment_type="diffusion",
            cfg_dropout_prob=config.cfg_dropout_prob,
        )
    elif config.split == "val":
        result = create_dual_image_validation_dataloader(
            cfg=_build_cfg_proxy(config),
            image_keys=image_keys,
            conditioning=conditioning,
            batch_size=config.batch_size,
            world_size=config.world_size,
        )
        if result is None:
            raise ValueError("No validation data found for dual mode")
        return result
    else:  # test
        result = create_dual_image_test_dataloader(
            cfg=_build_cfg_proxy(config),
            image_keys=image_keys,
            conditioning=conditioning,
            batch_size=config.batch_size,
        )
        if result is None:
            raise ValueError("No test data found for dual mode")
        return result


@DataLoaderFactory.register(ModelType.DIFFUSION, SpatialDims.TWO_D, "multi")
def _create_multi_2d(config: LoaderConfig) -> tuple[DataLoader, Dataset]:
    """Create 2D multi-modality diffusion loader."""
    from .multi_diffusion import create_multi_diffusion_dataloader

    if config.split == "train":
        return create_multi_diffusion_dataloader(
            cfg=_build_cfg_proxy(config),
            image_keys=list(config.image_keys),
            use_distributed=config.use_distributed,
            rank=config.rank,
            world_size=config.world_size,
            augment=config.should_augment,
        )
    else:
        raise ValueError(f"Multi-modality {config.split} loader not yet supported")


@DataLoaderFactory.register(ModelType.DIFFUSION, SpatialDims.TWO_D, "seg_conditioned")
def _create_seg_conditioned_2d(config: LoaderConfig) -> tuple[DataLoader, Dataset]:
    """Create 2D seg_conditioned mode loader."""
    from .seg_conditioned import (
        create_seg_conditioned_dataloader,
        create_seg_conditioned_validation_dataloader,
    )

    # Extract size bin config from cfg proxy
    cfg_proxy = _build_cfg_proxy(config)
    size_bin_config = dict(cfg_proxy.mode.get("size_bins", {}))

    if config.split == "train":
        return create_seg_conditioned_dataloader(
            cfg=cfg_proxy,
            size_bin_config=size_bin_config,
            use_distributed=config.use_distributed,
            rank=config.rank,
            world_size=config.world_size,
            augment=config.should_augment,
        )
    elif config.split == "val":
        result = create_seg_conditioned_validation_dataloader(
            cfg=cfg_proxy,
            size_bin_config=size_bin_config,
            batch_size=config.batch_size,
            world_size=config.world_size,
        )
        if result is None:
            raise ValueError("No validation data found for seg_conditioned mode")
        return result
    else:
        raise ValueError("seg_conditioned test loader not yet supported")


@DataLoaderFactory.register(ModelType.DIFFUSION, SpatialDims.TWO_D, "seg_conditioned_input")
def _create_seg_conditioned_input_2d(config: LoaderConfig) -> tuple[DataLoader, Dataset]:
    """Create 2D seg_conditioned_input mode loader.

    Input conditioning mode: returns bin_maps for concatenation with noise.
    """
    from .seg_conditioned import (
        create_seg_conditioned_dataloader,
        create_seg_conditioned_validation_dataloader,
    )

    cfg_proxy = _build_cfg_proxy(config)
    size_bin_config = dict(cfg_proxy.mode.get("size_bins", {}))
    size_bin_config["return_bin_maps"] = True
    size_bin_config["cfg_dropout_prob"] = config.cfg_dropout_prob

    if config.split == "train":
        return create_seg_conditioned_dataloader(
            cfg=cfg_proxy,
            size_bin_config=size_bin_config,
            use_distributed=config.use_distributed,
            rank=config.rank,
            world_size=config.world_size,
            augment=config.should_augment,
        )
    elif config.split == "val":
        result = create_seg_conditioned_validation_dataloader(
            cfg=cfg_proxy,
            size_bin_config=size_bin_config,
            batch_size=config.batch_size,
            world_size=config.world_size,
        )
        if result is None:
            raise ValueError("No validation data found for seg_conditioned_input mode")
        return result
    else:
        raise ValueError("seg_conditioned_input test loader not yet supported")


# =============================================================================
# Diffusion 3D Handlers
# =============================================================================

@DataLoaderFactory.register(ModelType.DIFFUSION, SpatialDims.THREE_D, "seg")
def _create_seg_3d(config: LoaderConfig) -> tuple[DataLoader, Dataset]:
    """Create 3D segmentation mode loader."""
    from .volume_3d import (
        VolumeConfig,
        create_segmentation_dataloader,
        create_segmentation_validation_dataloader,
    )

    cfg_proxy = _build_cfg_proxy(config)
    vol_cfg = VolumeConfig.from_cfg(cfg_proxy)

    if config.split == "train":
        return create_segmentation_dataloader(cfg_proxy, vol_cfg, augment=config.should_augment)
    elif config.split == "val":
        result = create_segmentation_validation_dataloader(cfg_proxy, vol_cfg)
        if result is None:
            raise ValueError("No 3D validation data found for seg mode")
        return result
    else:
        raise ValueError("3D seg test loader not yet supported")


@DataLoaderFactory.register(ModelType.DIFFUSION, SpatialDims.THREE_D, "bravo")
def _create_bravo_3d(config: LoaderConfig) -> tuple[DataLoader, Dataset]:
    """Create 3D BRAVO mode loader (BRAVO conditioned on seg)."""
    from .volume_3d import (
        VolumeConfig,
        create_single_modality_dataloader_with_seg,
        create_single_modality_validation_dataloader_with_seg,
    )

    cfg_proxy = _build_cfg_proxy(config)
    vol_cfg = VolumeConfig.from_cfg(cfg_proxy)

    if config.split == "train":
        return create_single_modality_dataloader_with_seg(
            cfg_proxy, vol_cfg, modality="bravo", augment=config.should_augment
        )
    elif config.split == "val":
        result = create_single_modality_validation_dataloader_with_seg(
            cfg_proxy, vol_cfg, modality="bravo"
        )
        if result is None:
            raise ValueError("No 3D validation data found for bravo mode")
        return result
    else:
        raise ValueError("3D bravo test loader not yet supported")


@DataLoaderFactory.register(ModelType.DIFFUSION, SpatialDims.THREE_D, "bravo_seg_cond")
def _create_bravo_seg_cond_3d(config: LoaderConfig) -> tuple[DataLoader, Dataset]:
    """Create 3D BRAVO with seg conditioning mode loader."""
    return _create_bravo_3d(config)


@DataLoaderFactory.register(ModelType.DIFFUSION, SpatialDims.THREE_D, "seg_conditioned")
def _create_seg_conditioned_3d(config: LoaderConfig) -> tuple[DataLoader, Dataset]:
    """Create 3D seg_conditioned mode loader."""
    from .volume_3d import (
        VolumeConfig,
        create_segmentation_conditioned_dataloader,
        create_segmentation_conditioned_validation_dataloader,
    )

    cfg_proxy = _build_cfg_proxy(config)
    vol_cfg = VolumeConfig.from_cfg(cfg_proxy)
    size_bin_config = dict(cfg_proxy.mode.get("size_bins", {}))

    if config.split == "train":
        return create_segmentation_conditioned_dataloader(
            cfg_proxy, vol_cfg, size_bin_config, augment=config.should_augment
        )
    elif config.split == "val":
        result = create_segmentation_conditioned_validation_dataloader(
            cfg_proxy, vol_cfg, size_bin_config
        )
        if result is None:
            raise ValueError("No 3D validation data found for seg_conditioned mode")
        return result
    else:
        raise ValueError("3D seg_conditioned test loader not yet supported")


@DataLoaderFactory.register(ModelType.DIFFUSION, SpatialDims.THREE_D, "seg_conditioned_input")
def _create_seg_conditioned_input_3d(config: LoaderConfig) -> tuple[DataLoader, Dataset]:
    """Create 3D seg_conditioned_input mode loader."""
    from .seg import create_seg_dataloader, create_seg_test_dataloader, create_seg_validation_dataloader
    from .base import DictDatasetWrapper, dict_collate_fn
    from .common import DataLoaderConfig
    from omegaconf import OmegaConf

    cfg_proxy = _build_cfg_proxy(config)

    # Configure to enable bin_maps output
    size_bin_config = dict(cfg_proxy.mode.get("size_bins", {}))
    size_bin_config["return_bin_maps"] = True
    cfg_dropout_prob = config.cfg_dropout_prob

    # Create modified config
    cfg_modified = OmegaConf.create(OmegaConf.to_container(cfg_proxy, resolve=True))
    cfg_modified.mode.size_bins = size_bin_config
    cfg_modified.mode.cfg_dropout_prob = cfg_dropout_prob

    if config.split == "train":
        loader, raw_dataset = create_seg_dataloader(cfg_modified)
    elif config.split == "val":
        result = create_seg_validation_dataloader(cfg_modified)
        if result is None:
            raise ValueError("No 3D validation data found for seg_conditioned_input mode")
        loader, raw_dataset = result
    else:  # test
        result = create_seg_test_dataloader(cfg_modified)
        if result is None:
            raise ValueError("No 3D test data found for seg_conditioned_input mode")
        loader, raw_dataset = result

    # Wrap dataset for dict format
    wrapped_dataset = DictDatasetWrapper(
        raw_dataset, output_format="seg_conditioned_input", spatial_dims=3
    )

    dl_cfg = DataLoaderConfig.from_cfg(cfg_proxy)
    effective_batch_size = config.batch_size if config.batch_size else loader.batch_size

    sampler = loader.sampler if hasattr(loader, "sampler") else None
    shuffle = False if sampler is not None else (config.split == "train")

    from torch.utils.data import DataLoader as TorchDataLoader
    new_loader = TorchDataLoader(
        wrapped_dataset,
        batch_size=effective_batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=dict_collate_fn,
        pin_memory=dl_cfg.pin_memory,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor if dl_cfg.num_workers > 0 else None,
        persistent_workers=dl_cfg.persistent_workers if dl_cfg.num_workers > 0 else False,
    )

    return new_loader, wrapped_dataset


@DataLoaderFactory.register(ModelType.DIFFUSION, SpatialDims.THREE_D, "dual")
def _create_dual_3d(config: LoaderConfig) -> tuple[DataLoader, Dataset]:
    """Create 3D dual mode loader."""
    raise ValueError("3D dual mode not yet supported")


@DataLoaderFactory.register(ModelType.DIFFUSION, SpatialDims.THREE_D, "multi")
def _create_multi_3d(config: LoaderConfig) -> tuple[DataLoader, Dataset]:
    """Create 3D multi mode loader."""
    raise ValueError("3D multi mode not yet supported")


# =============================================================================
# Compression 2D Handlers (VAE/VQVAE/DCAE)
# =============================================================================

def _register_compression_handler(
    model_type: ModelType,
    spatial_dims: SpatialDims,
    mode: str,
) -> Callable[[HandlerFn], HandlerFn]:
    """Helper to register a handler for multiple compression model types."""
    return DataLoaderFactory.register(model_type, spatial_dims, mode)


# Register compression handlers for all compression model types
for _model_type in [ModelType.VAE, ModelType.VQVAE, ModelType.DCAE]:

    @DataLoaderFactory.register(_model_type, SpatialDims.TWO_D, "seg_compression")
    def _create_seg_compression_2d(
        config: LoaderConfig, _mt=_model_type
    ) -> tuple[DataLoader, Dataset]:
        """Create 2D seg compression loader (DC-AE)."""
        from .seg_compression import (
            create_seg_compression_dataloader,
            create_seg_compression_test_dataloader,
            create_seg_compression_validation_dataloader,
        )

        cfg_proxy = _build_cfg_proxy(config)
        image_size = config.image_size
        effective_batch_size = config.batch_size

        if config.split == "train":
            return create_seg_compression_dataloader(
                cfg=cfg_proxy,
                use_distributed=config.use_distributed,
                rank=config.rank,
                world_size=config.world_size,
                augment=config.should_augment,
            )
        elif config.split == "val":
            result = create_seg_compression_validation_dataloader(
                cfg=cfg_proxy,
                image_size=image_size,
                batch_size=effective_batch_size,
            )
            if result is None:
                raise ValueError("No validation data found for seg_compression mode")
            return result
        else:  # test
            result = create_seg_compression_test_dataloader(
                cfg=cfg_proxy,
                image_size=image_size,
                batch_size=effective_batch_size,
            )
            if result is None:
                raise ValueError("No test data found for seg_compression mode")
            return result


    @DataLoaderFactory.register(_model_type, SpatialDims.TWO_D, "multi_modality")
    def _create_multi_modality_compression_2d(
        config: LoaderConfig, _mt=_model_type
    ) -> tuple[DataLoader, Dataset]:
        """Create 2D multi-modality compression loader."""
        from .multi_modality import (
            create_multi_modality_dataloader,
            create_multi_modality_test_dataloader,
            create_multi_modality_validation_dataloader,
        )

        cfg_proxy = _build_cfg_proxy(config)
        image_keys = list(config.image_keys)
        image_size = config.image_size
        effective_batch_size = config.batch_size

        if config.split == "train":
            return create_multi_modality_dataloader(
                cfg=cfg_proxy,
                image_keys=image_keys,
                image_size=image_size,
                batch_size=effective_batch_size,
                use_distributed=config.use_distributed,
                rank=config.rank,
                world_size=config.world_size,
                augment=config.should_augment,
            )
        elif config.split == "val":
            result = create_multi_modality_validation_dataloader(
                cfg=cfg_proxy,
                image_keys=image_keys,
                image_size=image_size,
                batch_size=effective_batch_size,
            )
            if result is None:
                raise ValueError("No validation data found for multi_modality mode")
            return result
        else:  # test
            result = create_multi_modality_test_dataloader(
                cfg=cfg_proxy,
                image_keys=image_keys,
                image_size=image_size,
                batch_size=effective_batch_size,
            )
            if result is None:
                raise ValueError("No test data found for multi_modality mode")
            return result


    @DataLoaderFactory.register(_model_type, SpatialDims.TWO_D, "dual")
    def _create_dual_compression_2d(
        config: LoaderConfig, _mt=_model_type
    ) -> tuple[DataLoader, Dataset]:
        """Create 2D dual modality compression loader (t1_pre + t1_gd)."""
        from .vae import create_vae_dataloader, create_vae_test_dataloader, create_vae_validation_dataloader

        cfg_proxy = _build_cfg_proxy(config)

        if config.split == "train":
            return create_vae_dataloader(
                cfg=cfg_proxy,
                modality="dual",
                use_distributed=config.use_distributed,
                rank=config.rank,
                world_size=config.world_size,
                augment=config.should_augment,
            )
        elif config.split == "val":
            result = create_vae_validation_dataloader(
                cfg=cfg_proxy,
                modality="dual",
                batch_size=config.batch_size,
            )
            if result is None:
                raise ValueError("No validation data found for dual compression mode")
            return result
        else:  # test
            result = create_vae_test_dataloader(
                cfg=cfg_proxy,
                modality="dual",
                batch_size=config.batch_size,
            )
            if result is None:
                raise ValueError("No test data found for dual compression mode")
            return result


    # Single modality compression (bravo, t1_pre, t1_gd, flair, seg)
    for _modality in ["bravo", "t1_pre", "t1_gd", "flair", "seg"]:
        @DataLoaderFactory.register(_model_type, SpatialDims.TWO_D, _modality)
        def _create_single_compression_2d(
            config: LoaderConfig, _mt=_model_type, _mod=_modality
        ) -> tuple[DataLoader, Dataset]:
            """Create 2D single modality compression loader."""
            from .vae import create_vae_dataloader, create_vae_test_dataloader, create_vae_validation_dataloader

            cfg_proxy = _build_cfg_proxy(config)

            if config.split == "train":
                return create_vae_dataloader(
                    cfg=cfg_proxy,
                    modality=_mod,
                    use_distributed=config.use_distributed,
                    rank=config.rank,
                    world_size=config.world_size,
                    augment=config.should_augment,
                )
            elif config.split == "val":
                result = create_vae_validation_dataloader(
                    cfg=cfg_proxy,
                    modality=_mod,
                    batch_size=config.batch_size,
                )
                if result is None:
                    raise ValueError(f"No validation data found for {_mod} compression mode")
                return result
            else:  # test
                result = create_vae_test_dataloader(
                    cfg=cfg_proxy,
                    modality=_mod,
                    batch_size=config.batch_size,
                )
                if result is None:
                    raise ValueError(f"No test data found for {_mod} compression mode")
                return result


# =============================================================================
# Compression 3D Handlers
# =============================================================================

for _model_type in [ModelType.VAE, ModelType.VQVAE, ModelType.DCAE]:

    @DataLoaderFactory.register(_model_type, SpatialDims.THREE_D, "multi_modality")
    def _create_multi_modality_compression_3d(
        config: LoaderConfig, _mt=_model_type
    ) -> tuple[DataLoader, Dataset]:
        """Create 3D multi-modality compression loader."""
        from .volume_3d import (
            create_vae_3d_multi_modality_dataloader,
            create_vae_3d_multi_modality_test_dataloader,
            create_vae_3d_multi_modality_validation_dataloader,
        )

        cfg_proxy = _build_cfg_proxy(config)

        if config.split == "train":
            return create_vae_3d_multi_modality_dataloader(
                cfg=cfg_proxy,
                use_distributed=config.use_distributed,
                rank=config.rank,
                world_size=config.world_size,
            )
        elif config.split == "val":
            result = create_vae_3d_multi_modality_validation_dataloader(cfg_proxy)
            if result is None:
                raise ValueError("No 3D validation data found for multi_modality mode")
            return result
        else:  # test
            result = create_vae_3d_multi_modality_test_dataloader(cfg_proxy)
            if result is None:
                raise ValueError("No 3D test data found for multi_modality mode")
            return result


    @DataLoaderFactory.register(_model_type, SpatialDims.THREE_D, "dual")
    def _create_dual_compression_3d(
        config: LoaderConfig, _mt=_model_type
    ) -> tuple[DataLoader, Dataset]:
        """Create 3D dual modality compression loader."""
        from .volume_3d import (
            create_vae_3d_dataloader,
            create_vae_3d_test_dataloader,
            create_vae_3d_validation_dataloader,
        )

        cfg_proxy = _build_cfg_proxy(config)

        if config.split == "train":
            return create_vae_3d_dataloader(
                cfg=cfg_proxy,
                modality="dual",
                use_distributed=config.use_distributed,
                rank=config.rank,
                world_size=config.world_size,
            )
        elif config.split == "val":
            result = create_vae_3d_validation_dataloader(cfg_proxy, "dual")
            if result is None:
                raise ValueError("No 3D validation data found for dual compression mode")
            return result
        else:  # test
            result = create_vae_3d_test_dataloader(cfg_proxy, "dual")
            if result is None:
                raise ValueError("No 3D test data found for dual compression mode")
            return result


    # Single modality 3D compression
    for _modality in ["bravo", "t1_pre", "t1_gd", "flair", "seg"]:
        @DataLoaderFactory.register(_model_type, SpatialDims.THREE_D, _modality)
        def _create_single_compression_3d(
            config: LoaderConfig, _mt=_model_type, _mod=_modality
        ) -> tuple[DataLoader, Dataset]:
            """Create 3D single modality compression loader."""
            from .volume_3d import (
                create_vae_3d_dataloader,
                create_vae_3d_test_dataloader,
                create_vae_3d_validation_dataloader,
            )

            cfg_proxy = _build_cfg_proxy(config)

            if config.split == "train":
                return create_vae_3d_dataloader(
                    cfg=cfg_proxy,
                    modality=_mod,
                    use_distributed=config.use_distributed,
                    rank=config.rank,
                    world_size=config.world_size,
                )
            elif config.split == "val":
                result = create_vae_3d_validation_dataloader(cfg_proxy, _mod)
                if result is None:
                    raise ValueError(f"No 3D validation data found for {_mod} compression mode")
                return result
            else:  # test
                result = create_vae_3d_test_dataloader(cfg_proxy, _mod)
                if result is None:
                    raise ValueError(f"No 3D test data found for {_mod} compression mode")
                return result


# =============================================================================
# Utility Functions
# =============================================================================

def _build_cfg_proxy(config: LoaderConfig) -> "DictConfig":
    """Build a Hydra-like config object from LoaderConfig.

    This creates a DictConfig-like object that the underlying loader functions
    can read from, maintaining backward compatibility.

    Args:
        config: LoaderConfig instance.

    Returns:
        DictConfig-like object with expected structure.
    """
    from omegaconf import OmegaConf

    cfg_dict = {
        "paths": {
            "data_dir": config.data_dir,
        },
        "model": {
            "image_size": config.image_size,
            "spatial_dims": config.spatial_dims.value,
            "image_keys": list(config.image_keys),
        },
        "training": {
            "batch_size": config.batch_size,
            "augment": config.should_augment,
            "cfg_dropout_prob": config.cfg_dropout_prob,
            "dataloader": {
                "num_workers": config.num_workers,
                "pin_memory": config.pin_memory,
                "prefetch_factor": config.prefetch_factor,
                "persistent_workers": config.persistent_workers,
            },
        },
        "mode": {
            "image_keys": list(config.image_keys),
            "conditioning": config.conditioning,
            "cfg_dropout_prob": config.cfg_dropout_prob,
            "size_bins": {},  # Will be populated by specific handlers
        },
    }

    # Add volume config for 3D
    if config.is_3d:
        cfg_dict["volume"] = {
            "depth": config.depth,
            "height": config.effective_height,
            "width": config.effective_width,
            "pad_depth_to": config.pad_depth_to or 160,
            "pad_mode": config.pad_mode,
            "slice_step": config.slice_step,
        }

    # Add latent config if present
    if config.compression_checkpoint is not None:
        cfg_dict["latent"] = {
            "enabled": True,
            "compression_checkpoint": config.compression_checkpoint,
            "channels": config.latent_channels,
        }

    return OmegaConf.create(cfg_dict)
