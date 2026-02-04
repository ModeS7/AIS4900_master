"""Mode Factory - Single entry point for dataloader creation.

This module centralizes all mode-specific dataloader creation logic,
replacing scattered if/elif checks throughout train.py with a single
factory pattern.

Usage:
    >>> from medgen.core import ModeFactory, ModeConfig
    >>>
    >>> # Get mode configuration from Hydra config
    >>> mode_config = ModeFactory.get_mode_config(cfg)
    >>>
    >>> # Create dataloaders
    >>> train_loader, train_dataset = ModeFactory.create_train_dataloader(
    ...     cfg, mode_config, use_distributed=True, rank=0, world_size=4
    ... )
    >>> val_result = ModeFactory.create_val_dataloader(cfg, mode_config)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union
from enum import Enum

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from medgen.core.constants import ModeType


class ModeCategory(Enum):
    """Categories for dispatch logic.

    Modes are grouped into categories that share similar dataloader
    creation logic:
    - SINGLE: Single modality generation (seg, bravo, bravo_seg_cond)
    - DUAL: Dual modality generation (t1_pre + t1_gd)
    - MULTI: Multi-modality with mode embedding
    - SEG_CONDITIONED: Segmentation conditioned on tumor sizes
    """
    SINGLE = "single"              # seg, bravo, bravo_seg_cond
    DUAL = "dual"                  # dual (t1_pre + t1_gd)
    MULTI = "multi"                # multi (mode embedding)
    SEG_CONDITIONED = "seg_conditioned"  # seg_conditioned, seg_conditioned_input


@dataclass
class ModeConfig:
    """Mode configuration extracted from Hydra config.

    Attributes:
        mode: The ModeType enum value.
        category: The ModeCategory for dispatch.
        image_keys: List of image keys for multi/dual modes.
        conditioning: Conditioning type ('seg' for dual/multi).
        spatial_dims: 2 for 2D, 3 for 3D.
        use_latent: Whether latent diffusion is enabled.
    """
    mode: ModeType
    category: ModeCategory
    image_keys: List[str]
    conditioning: Optional[str]
    spatial_dims: int
    use_latent: bool


class ModeFactory:
    """Centralized factory for mode-specific dataloader creation.

    This class provides a single entry point for creating dataloaders
    based on the training mode (seg, bravo, dual, multi, etc.). It
    replaces the scattered if/elif checks throughout train.py with
    a clean factory pattern.

    All methods are class methods - no instantiation needed.
    """

    MODE_CATEGORIES = {
        ModeType.SEG: ModeCategory.SINGLE,
        ModeType.BRAVO: ModeCategory.SINGLE,
        ModeType.BRAVO_SEG_COND: ModeCategory.SINGLE,
        ModeType.DUAL: ModeCategory.DUAL,
        ModeType.MULTI: ModeCategory.MULTI,
        ModeType.SEG_CONDITIONED: ModeCategory.SEG_CONDITIONED,
        ModeType.SEG_CONDITIONED_INPUT: ModeCategory.SEG_CONDITIONED,
    }

    DEFAULT_IMAGE_KEYS = {
        ModeCategory.DUAL: ['t1_pre', 't1_gd'],
        ModeCategory.MULTI: ['bravo', 'flair', 't1_pre', 't1_gd'],
    }

    @classmethod
    def normalize_mode(cls, mode: Union[str, ModeType]) -> ModeType:
        """Normalize string/enum to ModeType.

        Args:
            mode: Mode as string or ModeType enum.

        Returns:
            ModeType enum value.

        Raises:
            ValueError: If mode string is unknown.
            TypeError: If mode is neither str nor ModeType.
        """
        if isinstance(mode, ModeType):
            return mode
        if isinstance(mode, str):
            try:
                return ModeType(mode.lower())
            except ValueError:
                raise ValueError(f"Unknown mode: {mode}")
        raise TypeError(f"Mode must be str or ModeType, got {type(mode)}")

    @classmethod
    def get_mode_config(cls, cfg: DictConfig) -> ModeConfig:
        """Extract mode configuration from Hydra config.

        Args:
            cfg: Hydra configuration object.

        Returns:
            ModeConfig dataclass with extracted settings.

        Raises:
            ValueError: If mode has no category mapping.
        """
        mode = cls.normalize_mode(cfg.mode.name)
        category = cls.MODE_CATEGORIES.get(mode)
        if category is None:
            raise ValueError(f"No category for mode: {mode}")

        spatial_dims = cfg.model.get('spatial_dims', 2)
        use_latent = cfg.get('latent', {}).get('enabled', False)

        # Extract image_keys based on category
        default_keys = cls.DEFAULT_IMAGE_KEYS.get(category, [])
        image_keys = list(cfg.mode.get('image_keys', default_keys))

        # Conditioning (DUAL/MULTI use seg)
        conditioning = None
        if category in (ModeCategory.DUAL, ModeCategory.MULTI):
            conditioning = cfg.mode.get('conditioning', 'seg')

        return ModeConfig(
            mode=mode,
            category=category,
            image_keys=image_keys,
            conditioning=conditioning,
            spatial_dims=spatial_dims,
            use_latent=use_latent,
        )

    @classmethod
    def create_train_dataloader(
        cls,
        cfg: DictConfig,
        mode_config: Optional[ModeConfig] = None,
        use_distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        augment: Optional[bool] = None,
    ) -> Tuple[DataLoader, Dataset]:
        """Create training dataloader for any mode.

        Args:
            cfg: Hydra configuration object.
            mode_config: Pre-computed ModeConfig (computed if None).
            use_distributed: Enable distributed training sampler.
            rank: Process rank for distributed training.
            world_size: Total number of processes.
            augment: Override augmentation setting (None = use config).

        Returns:
            Tuple of (DataLoader, Dataset).
        """
        if mode_config is None:
            mode_config = cls.get_mode_config(cfg)

        from medgen.data.loaders.unified import create_dataloader
        return create_dataloader(
            cfg=cfg,
            task='diffusion',
            mode=mode_config.mode.value,
            spatial_dims=mode_config.spatial_dims,
            split='train',
            use_distributed=use_distributed,
            rank=rank,
            world_size=world_size,
            augment=augment,
        )

    @classmethod
    def create_val_dataloader(
        cls,
        cfg: DictConfig,
        mode_config: Optional[ModeConfig] = None,
        world_size: int = 1,
    ) -> Optional[Tuple[DataLoader, Dataset]]:
        """Create validation dataloader for any mode.

        Args:
            cfg: Hydra configuration object.
            mode_config: Pre-computed ModeConfig (computed if None).
            world_size: Total number of processes (for batch size scaling).

        Returns:
            Tuple of (DataLoader, Dataset) or None if no val data.
        """
        if mode_config is None:
            mode_config = cls.get_mode_config(cfg)

        from medgen.data.loaders.unified import create_dataloader
        try:
            return create_dataloader(
                cfg=cfg,
                task='diffusion',
                mode=mode_config.mode.value,
                spatial_dims=mode_config.spatial_dims,
                split='val',
                world_size=world_size,
            )
        except (ValueError, FileNotFoundError):
            return None

    @classmethod
    def create_test_dataloader(
        cls,
        cfg: DictConfig,
        mode_config: Optional[ModeConfig] = None,
    ) -> Optional[Tuple[DataLoader, Dataset]]:
        """Create test dataloader for any mode.

        Args:
            cfg: Hydra configuration object.
            mode_config: Pre-computed ModeConfig (computed if None).

        Returns:
            Tuple of (DataLoader, Dataset) or None if no test data.
        """
        if mode_config is None:
            mode_config = cls.get_mode_config(cfg)

        from medgen.data.loaders.unified import create_dataloader
        try:
            return create_dataloader(
                cfg=cfg,
                task='diffusion',
                mode=mode_config.mode.value,
                spatial_dims=mode_config.spatial_dims,
                split='test',
            )
        except (ValueError, FileNotFoundError):
            return None

    @classmethod
    def create_per_modality_val_loaders(
        cls,
        cfg: DictConfig,
        mode_config: ModeConfig,
    ) -> Dict[str, DataLoader]:
        """Create per-modality validation loaders (MULTI mode only).

        For multi-modality training, this creates separate validation
        loaders for each modality to enable per-modality metrics.

        Args:
            cfg: Hydra configuration object.
            mode_config: ModeConfig with image_keys for multi mode.

        Returns:
            Dict mapping modality names to DataLoaders.
            Empty dict if not MULTI mode.
        """
        if mode_config.category != ModeCategory.MULTI:
            return {}

        from medgen.data.loaders.multi_diffusion import create_single_modality_diffusion_val_loader
        loaders = {}
        for modality in mode_config.image_keys:
            loader = create_single_modality_diffusion_val_loader(cfg, modality)
            if loader:
                loaders[modality] = loader
        return loaders

    @classmethod
    def create_pixel_loader_for_latent_cache(
        cls,
        cfg: DictConfig,
        mode_config: ModeConfig,
        split: str = 'train',
    ) -> Tuple[DataLoader, Dataset]:
        """Create pixel-space loader for latent cache building.

        For latent diffusion, we need to encode pixel-space data
        into latent space before training. This creates the loader
        for that encoding process (no augmentation).

        Args:
            cfg: Hydra configuration object.
            mode_config: ModeConfig for the current mode.
            split: Data split ('train', 'val').

        Returns:
            Tuple of (DataLoader, Dataset) for pixel-space data.
        """
        from medgen.data.loaders.unified import create_dataloader
        return create_dataloader(
            cfg=cfg,
            task='diffusion',
            mode=mode_config.mode.value,
            spatial_dims=mode_config.spatial_dims,
            split=split,
            augment=False,  # No augmentation for cache building
        )

    @classmethod
    def get_image_type_for_mode(cls, mode: ModeType) -> str:
        """Get the image type string for single-modality modes.

        This maps modes to the image type used by legacy dataloaders.

        Args:
            mode: ModeType enum value.

        Returns:
            Image type string ('seg', 'bravo', etc.).
        """
        if mode == ModeType.SEG:
            return 'seg'
        elif mode in (ModeType.SEG_CONDITIONED, ModeType.SEG_CONDITIONED_INPUT):
            return 'seg'
        else:
            # bravo, bravo_seg_cond, etc.
            return 'bravo'
