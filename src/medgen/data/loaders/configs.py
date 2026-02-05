"""Typed configuration for data loaders.

This module provides a typed, validated configuration dataclass that replaces
scattered cfg.get() chains throughout the loader codebase.

Usage:
    from medgen.data.loaders.configs import LoaderConfig, ModelType, SpatialDims

    # From Hydra config
    config = LoaderConfig.from_hydra(cfg, ModelType.DIFFUSION, "bravo", "train")

    # Manual creation with validation
    config = LoaderConfig(
        model_type=ModelType.DIFFUSION,
        spatial_dims=SpatialDims.TWO_D,
        split="train",
        mode="bravo",
        batch_size=32,
        image_size=256,
        data_dir="/path/to/data",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from omegaconf import DictConfig


class ModelType(Enum):
    """Type of model being trained."""
    DIFFUSION = auto()
    VAE = auto()
    VQVAE = auto()
    DCAE = auto()

    @classmethod
    def from_string(cls, s: str) -> "ModelType":
        """Convert string to ModelType.

        Args:
            s: String like 'diffusion', 'vae', 'vqvae', 'dcae'.

        Returns:
            Corresponding ModelType enum value.

        Raises:
            ValueError: If string doesn't match a known type.
        """
        mapping = {
            'diffusion': cls.DIFFUSION,
            'vae': cls.VAE,
            'vqvae': cls.VQVAE,
            'dcae': cls.DCAE,
        }
        key = s.lower().replace('-', '_')
        if key not in mapping:
            raise ValueError(f"Unknown model type: {s}. Expected one of: {list(mapping.keys())}")
        return mapping[key]

    def is_compression(self) -> bool:
        """Check if this is a compression model type."""
        return self in (ModelType.VAE, ModelType.VQVAE, ModelType.DCAE)


class SpatialDims(Enum):
    """Spatial dimensions for data loading."""
    TWO_D = 2
    THREE_D = 3

    @classmethod
    def from_int(cls, value: int) -> "SpatialDims":
        """Convert integer to SpatialDims.

        Args:
            value: 2 or 3.

        Returns:
            Corresponding SpatialDims enum value.

        Raises:
            ValueError: If value is not 2 or 3.
        """
        if value == 2:
            return cls.TWO_D
        elif value == 3:
            return cls.THREE_D
        else:
            raise ValueError(f"spatial_dims must be 2 or 3, got {value}")


Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class LoaderConfig:
    """Immutable, validated loader configuration.

    This dataclass captures all parameters needed to create a dataloader,
    with validation at creation time to catch config errors early.

    Attributes:
        model_type: Type of model (DIFFUSION, VAE, VQVAE, DCAE).
        spatial_dims: Spatial dimensions (2D or 3D).
        split: Data split (train, val, test).
        mode: Training mode (bravo, seg, dual, multi, seg_conditioned, etc.).
        batch_size: Batch size for training.
        image_size: Target image size (height/width for 2D).
        data_dir: Base data directory path.
        num_workers: Number of dataloader workers.
        augment: Override augmentation setting (None = auto based on split).
        use_distributed: Enable distributed training sampler.
        rank: Process rank for distributed training.
        world_size: Total number of processes.
        depth: Volume depth for 3D (required for 3D loaders).
        height: Volume height for 3D (optional, defaults to image_size).
        width: Volume width for 3D (optional, defaults to image_size).
        compression_checkpoint: Path to compression model checkpoint (latent diffusion).
        latent_channels: Number of latent channels (latent diffusion).
        cfg_dropout_prob: CFG dropout probability for conditioning.
        image_keys: List of image modality keys for multi-modality modes.
        conditioning: Conditioning modality (e.g., 'seg') for dual mode.
    """
    model_type: ModelType
    spatial_dims: SpatialDims
    split: Split
    mode: str
    batch_size: int
    image_size: int
    data_dir: str
    num_workers: int = 4
    augment: bool | None = None  # None = auto (True for train)

    # Distributed training
    use_distributed: bool = False
    rank: int = 0
    world_size: int = 1

    # 3D-specific
    depth: int | None = None
    height: int | None = None
    width: int | None = None
    pad_depth_to: int | None = None
    pad_mode: str = "replicate"
    slice_step: int = 1

    # Latent diffusion
    compression_checkpoint: str | None = None
    latent_channels: int | None = None

    # Mode-specific
    cfg_dropout_prob: float = 0.15
    image_keys: tuple[str, ...] = field(default_factory=lambda: ("bravo", "flair", "t1_pre", "t1_gd"))
    conditioning: str | None = "seg"

    # DataLoader settings
    pin_memory: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = True

    def __post_init__(self) -> None:
        """Validate configuration at creation time."""
        # Validate 3D requirements
        if self.spatial_dims == SpatialDims.THREE_D:
            # depth is needed for proper 3D volume handling
            # But we allow None and use defaults from VolumeConfig
            pass

        # Validate latent diffusion requirements
        if self.compression_checkpoint is not None:
            if self.latent_channels is None:
                raise ValueError(
                    "latent_channels required when compression_checkpoint is set"
                )

        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")

        # Validate image size
        if self.image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {self.image_size}")

        # Validate distributed settings
        if self.use_distributed:
            if not (0 <= self.rank < self.world_size):
                raise ValueError(
                    f"rank must be in [0, {self.world_size}), got {self.rank}"
                )

    @property
    def should_augment(self) -> bool:
        """Resolve augmentation setting.

        Returns:
            True if augmentation should be applied, False otherwise.
            If augment is None, defaults to True for training split.
        """
        if self.augment is not None:
            return self.augment
        return self.split == "train"

    @property
    def effective_height(self) -> int:
        """Get effective height for 3D volumes."""
        return self.height if self.height is not None else self.image_size

    @property
    def effective_width(self) -> int:
        """Get effective width for 3D volumes."""
        return self.width if self.width is not None else self.image_size

    @property
    def is_3d(self) -> bool:
        """Check if this is a 3D configuration."""
        return self.spatial_dims == SpatialDims.THREE_D

    @property
    def is_compression(self) -> bool:
        """Check if this is for compression model training."""
        return self.model_type.is_compression()

    @classmethod
    def from_hydra(
        cls,
        cfg: "DictConfig",
        model_type: ModelType,
        mode: str,
        split: Split = "train",
        **overrides: Any,
    ) -> "LoaderConfig":
        """Create LoaderConfig from Hydra configuration.

        This method extracts all relevant values from a Hydra config object
        and creates a validated LoaderConfig. Override individual fields
        via keyword arguments.

        Args:
            cfg: Hydra DictConfig object.
            model_type: Type of model being trained.
            mode: Training mode (bravo, seg, dual, multi, etc.).
            split: Data split (train, val, test).
            **overrides: Override any config field (batch_size, augment, etc.).

        Returns:
            Validated LoaderConfig instance.

        Example:
            >>> config = LoaderConfig.from_hydra(
            ...     cfg, ModelType.DIFFUSION, "bravo", "train",
            ...     batch_size=64,  # Override batch size
            ...     augment=False,  # Disable augmentation
            ... )
        """
        # Extract spatial dimensions
        spatial_dims_int = cfg.model.get("spatial_dims", 2)
        spatial_dims = SpatialDims.from_int(spatial_dims_int)

        # Extract batch size (with override support)
        batch_size = overrides.pop("batch_size", cfg.training.batch_size)

        # Extract image size
        image_size = cfg.model.image_size

        # Extract data directory
        data_dir = cfg.paths.data_dir

        # Extract DataLoader settings
        dl_cfg = cfg.training.get("dataloader", {})
        num_workers = dl_cfg.get("num_workers", 4)
        pin_memory = dl_cfg.get("pin_memory", True)
        prefetch_factor = dl_cfg.get("prefetch_factor", 4)
        persistent_workers = dl_cfg.get("persistent_workers", True)

        # Extract 3D volume settings if applicable
        depth = None
        height = None
        width = None
        pad_depth_to = None
        pad_mode = "replicate"
        slice_step = 1

        if spatial_dims == SpatialDims.THREE_D and hasattr(cfg, "volume"):
            vol_cfg = cfg.volume
            depth = vol_cfg.get("depth")
            height = vol_cfg.get("height", image_size)
            width = vol_cfg.get("width", image_size)
            pad_depth_to = vol_cfg.get("pad_depth_to", 160)
            pad_mode = vol_cfg.get("pad_mode", "replicate")
            slice_step = vol_cfg.get("slice_step", 1)

        # Extract latent diffusion settings
        compression_checkpoint = None
        latent_channels = None

        if hasattr(cfg, "latent") and cfg.latent.get("enabled", False):
            compression_checkpoint = cfg.latent.get("compression_checkpoint")
            latent_channels = cfg.latent.get("channels")

        # Extract mode-specific settings
        cfg_dropout_prob = cfg.training.get("cfg_dropout_prob", 0.15)
        if hasattr(cfg, "mode"):
            cfg_dropout_prob = cfg.mode.get("cfg_dropout_prob", cfg_dropout_prob)

        # Extract image keys for multi-modality
        image_keys = ("bravo", "flair", "t1_pre", "t1_gd")
        if hasattr(cfg, "mode") and cfg.mode.get("image_keys"):
            image_keys = tuple(cfg.mode.image_keys)
        elif hasattr(cfg, "model") and cfg.model.get("image_keys"):
            image_keys = tuple(cfg.model.image_keys)

        # Extract conditioning
        conditioning = "seg"
        if hasattr(cfg, "mode"):
            conditioning = cfg.mode.get("conditioning", "seg")

        # Build config with defaults and overrides
        return cls(
            model_type=model_type,
            spatial_dims=spatial_dims,
            split=split,
            mode=mode,
            batch_size=batch_size,
            image_size=image_size,
            data_dir=data_dir,
            num_workers=num_workers,
            augment=overrides.pop("augment", None),
            use_distributed=overrides.pop("use_distributed", False),
            rank=overrides.pop("rank", 0),
            world_size=overrides.pop("world_size", 1),
            depth=depth,
            height=height,
            width=width,
            pad_depth_to=pad_depth_to,
            pad_mode=pad_mode,
            slice_step=slice_step,
            compression_checkpoint=compression_checkpoint,
            latent_channels=latent_channels,
            cfg_dropout_prob=cfg_dropout_prob,
            image_keys=image_keys,
            conditioning=conditioning,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def with_split(self, split: Split) -> "LoaderConfig":
        """Create a new config with a different split.

        Useful for creating validation/test configs from a training config.

        Args:
            split: New data split.

        Returns:
            New LoaderConfig with updated split.
        """
        # Convert frozen dataclass to dict, update split, create new instance
        from dataclasses import asdict
        config_dict = asdict(self)
        config_dict["split"] = split
        # Convert enums back from their values
        config_dict["model_type"] = self.model_type
        config_dict["spatial_dims"] = self.spatial_dims
        return LoaderConfig(**config_dict)

    def with_batch_size(self, batch_size: int) -> "LoaderConfig":
        """Create a new config with a different batch size.

        Args:
            batch_size: New batch size.

        Returns:
            New LoaderConfig with updated batch_size.
        """
        from dataclasses import asdict
        config_dict = asdict(self)
        config_dict["batch_size"] = batch_size
        config_dict["model_type"] = self.model_type
        config_dict["spatial_dims"] = self.spatial_dims
        return LoaderConfig(**config_dict)
