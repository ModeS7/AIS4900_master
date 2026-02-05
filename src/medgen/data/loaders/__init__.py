"""Dataloader factory functions for diffusion and VAE training.

Primary API (RECOMMENDED):
    from medgen.data.loaders import DataLoaderFactory, LoaderConfig, ModelType, SpatialDims

    config = LoaderConfig.from_hydra(cfg, ModelType.DIFFUSION, "bravo", "train")
    loader, dataset = DataLoaderFactory.create(config)

Legacy API (still works, for backward compatibility):
    from medgen.data.loaders import create_dataloader, create_vae_dataloader
    # These continue to work without deprecation warnings

Dataset Classes (from datasets.py):
    - SegConditionedDataset: 2D seg with size bin conditioning
    - MultiDiffusionDataset: Multi-modality diffusion with mode_id
    - AugmentedSegDataset: DC-AE seg compression
    - VolumeDataset: Full 3D volumes for single modality
    - DualVolumeDataset: Full 3D volumes for dual modality

Utility Functions (from datasets.py):
    - compute_size_bins / compute_size_bins_3d: Tumor size binning
    - compute_feret_diameter / compute_feret_diameter_3d: Longest axis computation
    - create_size_bin_maps: Spatial conditioning maps
"""

# =============================================================================
# NEW Primary API (RECOMMENDED)
# =============================================================================

# Typed configuration and factory
from .configs import LoaderConfig, ModelType, SpatialDims
from .factory import DataLoaderFactory

# =============================================================================
# Primary Exports (unified loader interface)
# =============================================================================

# Base classes for unified 2D/3D loading
from .base import (
    BaseDiffusionDataset,
    BaseDiffusionDataset2D,
    BaseDiffusionDataset3D,
    DictDatasetWrapper,
    dict_collate_fn,
    validate_batch_format,
)

# Unified loader factory (legacy interface, still works)
from .unified import (
    create_dataloader,
    create_diffusion_dataloader,
    get_dataloader_info,
    _get_compression_output_format,
)

# Common utilities
from .common import (
    DataLoaderConfig,
    setup_distributed_sampler,
    get_data_dir,
    validate_data_dir,
    get_modality_keys,
    validate_modality_keys,
    check_seg_available,
    MODALITY_KEYS,
)

# Consolidated Dataset classes and utility functions (NEW)
from .datasets import (
    # Dataset classes
    SegConditionedDataset,
    MultiDiffusionDataset,
    AugmentedSegDataset,
    VolumeDataset,
    DualVolumeDataset,
    # 2D utility functions
    compute_size_bins,
    compute_feret_diameter,
    create_size_bin_maps,
    extract_seg_slices,
    extract_slices_with_seg_and_mode,
    # 3D utility functions
    compute_size_bins_3d,
    compute_feret_diameter_3d,
    # Constants
    DEFAULT_BIN_EDGES,
)

# Latent diffusion (kept separate - complex)
from .latent import (
    LatentDataset,
    LatentCacheBuilder,
    create_latent_dataloader,
    create_latent_validation_dataloader,
    create_latent_test_dataloader,
    load_compression_model,
    detect_compression_type,
)

# 3D volume dataloaders (kept separate - specialized)
from .volume_3d import (
    create_vae_3d_dataloader,
    create_vae_3d_validation_dataloader,
    create_vae_3d_test_dataloader,
    create_vae_3d_multi_modality_dataloader,
    create_vae_3d_multi_modality_validation_dataloader,
    create_vae_3d_multi_modality_test_dataloader,
    create_vae_3d_single_modality_validation_loader,
    Base3DVolumeDataset,
    Volume3DDataset,
    DualVolume3DDataset,
    MultiModality3DDataset,
    SingleModality3DDatasetWithSeg,
)

# 3D Segmentation dataloaders
from .seg import (
    create_seg_dataloader,
    create_seg_validation_dataloader,
    create_seg_test_dataloader,
    SegDataset,
)

# =============================================================================
# Legacy Imports (for backward compatibility - will show deprecation warnings)
# =============================================================================

# Single-image dataloaders
from .single import (
    create_test_dataloader,
    create_validation_dataloader,
)

# Dual-image dataloaders
from .dual import (
    create_dual_image_dataloader,
    create_dual_image_test_dataloader,
    create_dual_image_validation_dataloader,
)

# VAE dataloaders
from .vae import (
    create_vae_dataloader,
    create_vae_test_dataloader,
    create_vae_validation_dataloader,
)

# Multi-modality VAE dataloaders
from .multi_modality import (
    create_multi_modality_dataloader,
    create_multi_modality_test_dataloader,
    create_multi_modality_validation_dataloader,
    create_single_modality_validation_loader,
)

# Multi-modality diffusion dataloaders
from .multi_diffusion import (
    create_multi_diffusion_dataloader,
    create_multi_diffusion_test_dataloader,
    create_multi_diffusion_validation_dataloader,
    create_single_modality_diffusion_val_loader,
)

# Segmentation compression dataloaders
from .seg_compression import (
    create_seg_compression_dataloader,
    create_seg_compression_validation_dataloader,
    create_seg_compression_test_dataloader,
)

# Segmentation conditioned dataloaders
from .seg_conditioned import (
    create_seg_conditioned_dataloader,
    create_seg_conditioned_validation_dataloader,
    create_seg_conditioned_test_dataloader,
)


__all__ = [
    # === NEW Primary API (RECOMMENDED) ===
    'DataLoaderFactory',
    'LoaderConfig',
    'ModelType',
    'SpatialDims',

    # === Primary API (use these) ===
    # Base classes
    'BaseDiffusionDataset',
    'BaseDiffusionDataset2D',
    'BaseDiffusionDataset3D',
    'DictDatasetWrapper',
    'dict_collate_fn',
    'validate_batch_format',
    # Unified factory (legacy interface, still works)
    'create_dataloader',
    'create_diffusion_dataloader',
    'get_dataloader_info',
    # Common utilities
    'DataLoaderConfig',
    'setup_distributed_sampler',
    'MODALITY_KEYS',
    # Dataset classes (from datasets.py)
    'SegConditionedDataset',
    'MultiDiffusionDataset',
    'AugmentedSegDataset',
    'VolumeDataset',
    'DualVolumeDataset',
    # Utility functions (from datasets.py)
    'compute_size_bins',
    'compute_size_bins_3d',
    'compute_feret_diameter',
    'compute_feret_diameter_3d',
    'create_size_bin_maps',
    'extract_seg_slices',
    'extract_slices_with_seg_and_mode',
    'DEFAULT_BIN_EDGES',
    # Latent
    'LatentDataset',
    'LatentCacheBuilder',
    'create_latent_dataloader',
    'create_latent_validation_dataloader',
    'create_latent_test_dataloader',
    # 3D Volume
    'Base3DVolumeDataset',
    'Volume3DDataset',
    'DualVolume3DDataset',
    'MultiModality3DDataset',
    'SingleModality3DDatasetWithSeg',
    'SegDataset',
    # 3D loaders
    'create_vae_3d_dataloader',
    'create_seg_dataloader',

    # === Legacy API (still work, consider migrating) ===
    # Single-image
    'create_validation_dataloader',
    'create_test_dataloader',
    # Dual-image
    'create_dual_image_dataloader',
    'create_dual_image_validation_dataloader',
    'create_dual_image_test_dataloader',
    # VAE
    'create_vae_dataloader',
    'create_vae_validation_dataloader',
    'create_vae_test_dataloader',
    # Multi-modality
    'create_multi_modality_dataloader',
    'create_multi_diffusion_dataloader',
    # Seg compression
    'create_seg_compression_dataloader',
    # Seg conditioned
    'create_seg_conditioned_dataloader',
]
