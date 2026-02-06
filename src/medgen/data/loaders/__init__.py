"""Dataloader factory functions for diffusion and VAE training.

Primary API:
    from medgen.data.loaders import create_dataloader, create_vae_dataloader

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

# Unified loader factory
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

# 2D dataloader builder (consolidated API)
from .builder_2d import (
    LoaderSpec,
    build_2d_loader,
    # Convenience functions (consolidated from thin wrapper files)
    create_single_loader,
    create_dual_loader,
    create_vae_loader,
    create_multi_modality_loader,
    create_single_modality_validation_loader,
    create_multi_diffusion_loader,
    create_seg_compression_loader,
    create_seg_conditioned_loader,
    create_single_modality_diffusion_val_loader,
)

# Consolidated Dataset classes and utility functions
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
    create_vae_volume_validation_dataloader,
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
# Backward-compat aliases (old names -> new convenience functions)
# =============================================================================

# These are used by medgen/data/__init__.py and various callers.
# They wrap the new convenience functions to match old signatures.

def create_validation_dataloader(cfg, image_type, batch_size=None, world_size=1):
    """Backward-compat: single-image validation loader."""
    return create_single_loader(
        cfg, image_type, split='val', batch_size=batch_size, world_size=world_size,
    )

def create_test_dataloader(cfg, image_type, batch_size=None):
    """Backward-compat: single-image test loader."""
    return create_single_loader(cfg, image_type, split='test', batch_size=batch_size)

def create_dual_image_dataloader(
    cfg, image_keys, conditioning, use_distributed=False, rank=0, world_size=1,
    augment=True, augment_type='diffusion', cfg_dropout_prob=0.15,
):
    """Backward-compat: dual-image training loader."""
    return create_dual_loader(
        cfg, image_keys, conditioning, split='train',
        augment_type=augment_type, cfg_dropout_prob=cfg_dropout_prob,
        use_distributed=use_distributed, rank=rank, world_size=world_size,
        augment=augment,
    )

def create_dual_image_validation_dataloader(
    cfg, image_keys, conditioning='seg', batch_size=None, world_size=1,
):
    """Backward-compat: dual-image validation loader."""
    return create_dual_loader(
        cfg, image_keys, conditioning, split='val',
        batch_size=batch_size, world_size=world_size,
    )

def create_dual_image_test_dataloader(
    cfg, image_keys, conditioning='seg', batch_size=None,
):
    """Backward-compat: dual-image test loader."""
    return create_dual_loader(
        cfg, image_keys, conditioning, split='test', batch_size=batch_size,
    )

def create_vae_dataloader(
    cfg, modality, use_distributed=False, rank=0, world_size=1, augment=True,
):
    """Backward-compat: VAE training loader."""
    return create_vae_loader(
        cfg, modality, split='train',
        use_distributed=use_distributed, rank=rank, world_size=world_size,
        augment=augment,
    )

def create_vae_validation_dataloader(cfg, modality, batch_size=None):
    """Backward-compat: VAE validation loader."""
    return create_vae_loader(cfg, modality, split='val', batch_size=batch_size)

def create_vae_test_dataloader(cfg, modality, batch_size=None):
    """Backward-compat: VAE test loader."""
    return create_vae_loader(cfg, modality, split='test', batch_size=batch_size)

def create_multi_modality_dataloader(
    cfg, image_keys, image_size, batch_size,
    use_distributed=False, rank=0, world_size=1, augment=True,
):
    """Backward-compat: multi-modality VAE training loader."""
    return create_multi_modality_loader(
        cfg, image_keys, split='train',
        use_distributed=use_distributed, rank=rank, world_size=world_size,
        augment=augment, batch_size=batch_size, image_size=image_size,
    )

def create_multi_modality_validation_dataloader(
    cfg, image_keys, image_size, batch_size,
):
    """Backward-compat: multi-modality VAE validation loader."""
    return create_multi_modality_loader(
        cfg, image_keys, split='val',
        batch_size=batch_size, image_size=image_size,
    )

def create_multi_modality_test_dataloader(
    cfg, image_keys, image_size, batch_size,
):
    """Backward-compat: multi-modality VAE test loader."""
    return create_multi_modality_loader(
        cfg, image_keys, split='test',
        batch_size=batch_size, image_size=image_size,
    )

def create_multi_diffusion_dataloader(
    cfg, image_keys, use_distributed=False, rank=0, world_size=1, augment=True,
):
    """Backward-compat: multi-modality diffusion training loader."""
    return create_multi_diffusion_loader(
        cfg, image_keys, split='train',
        use_distributed=use_distributed, rank=rank, world_size=world_size,
        augment=augment,
    )

def create_multi_diffusion_validation_dataloader(cfg, image_keys, world_size=1):
    """Backward-compat: multi-modality diffusion validation loader."""
    return create_multi_diffusion_loader(
        cfg, image_keys, split='val', world_size=world_size,
    )

def create_multi_diffusion_test_dataloader(cfg, image_keys, world_size=1):
    """Backward-compat: multi-modality diffusion test loader."""
    return create_multi_diffusion_loader(
        cfg, image_keys, split='test', world_size=world_size,
    )

def create_seg_compression_dataloader(
    cfg, image_size, batch_size, use_distributed=False, rank=0, world_size=1, augment=True,
):
    """Backward-compat: seg compression training loader."""
    return create_seg_compression_loader(
        cfg, split='train',
        image_size=image_size, batch_size=batch_size,
        use_distributed=use_distributed, rank=rank, world_size=world_size,
        augment=augment,
    )

def create_seg_compression_validation_dataloader(cfg, image_size, batch_size):
    """Backward-compat: seg compression validation loader."""
    return create_seg_compression_loader(
        cfg, split='val', image_size=image_size, batch_size=batch_size,
    )

def create_seg_compression_test_dataloader(cfg, image_size, batch_size):
    """Backward-compat: seg compression test loader."""
    return create_seg_compression_loader(
        cfg, split='test', image_size=image_size, batch_size=batch_size,
    )

def create_seg_conditioned_dataloader(
    cfg, size_bin_config=None, use_distributed=False, rank=0, world_size=1, augment=True,
):
    """Backward-compat: seg conditioned training loader."""
    return create_seg_conditioned_loader(
        cfg, split='train', size_bin_config=size_bin_config,
        use_distributed=use_distributed, rank=rank, world_size=world_size,
        augment=augment,
    )

def create_seg_conditioned_validation_dataloader(
    cfg, size_bin_config=None, batch_size=None, world_size=1,
):
    """Backward-compat: seg conditioned validation loader."""
    return create_seg_conditioned_loader(
        cfg, split='val', size_bin_config=size_bin_config,
        batch_size=batch_size, world_size=world_size,
    )

def create_seg_conditioned_test_dataloader(cfg, batch_size=None):
    """Backward-compat: seg conditioned test loader."""
    return create_seg_conditioned_loader(
        cfg, split='test', batch_size=batch_size,
    )


__all__ = [
    # === Primary API ===
    # Base classes
    'BaseDiffusionDataset',
    'BaseDiffusionDataset2D',
    'BaseDiffusionDataset3D',
    'DictDatasetWrapper',
    'dict_collate_fn',
    'validate_batch_format',
    # Unified factory
    'create_dataloader',
    'create_diffusion_dataloader',
    'get_dataloader_info',
    # Common utilities
    'DataLoaderConfig',
    'setup_distributed_sampler',
    'MODALITY_KEYS',
    # 2D builder
    'LoaderSpec',
    'build_2d_loader',
    # Convenience loaders (new names)
    'create_single_loader',
    'create_dual_loader',
    'create_vae_loader',
    'create_multi_modality_loader',
    'create_single_modality_validation_loader',
    'create_multi_diffusion_loader',
    'create_seg_compression_loader',
    'create_seg_conditioned_loader',
    'create_single_modality_diffusion_val_loader',
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
    # Volume validation
    'create_vae_volume_validation_dataloader',

    # === Backward-compat aliases ===
    'create_validation_dataloader',
    'create_test_dataloader',
    'create_dual_image_dataloader',
    'create_dual_image_validation_dataloader',
    'create_dual_image_test_dataloader',
    'create_vae_dataloader',
    'create_vae_validation_dataloader',
    'create_vae_test_dataloader',
    'create_multi_modality_dataloader',
    'create_multi_modality_validation_dataloader',
    'create_multi_modality_test_dataloader',
    'create_multi_diffusion_dataloader',
    'create_multi_diffusion_validation_dataloader',
    'create_multi_diffusion_test_dataloader',
    'create_seg_compression_dataloader',
    'create_seg_compression_validation_dataloader',
    'create_seg_compression_test_dataloader',
    'create_seg_conditioned_dataloader',
    'create_seg_conditioned_validation_dataloader',
    'create_seg_conditioned_test_dataloader',
    'create_single_modality_diffusion_val_loader',
]
