"""Dataloader factory functions for diffusion and VAE training."""

# Base classes for unified 2D/3D loading
from .base import (
    BaseDiffusionDataset,
    BaseDiffusionDataset2D,
    BaseDiffusionDataset3D,
    DictDatasetWrapper,
    dict_collate_fn,
    validate_batch_format,
)

# Unified loader factory (recommended entry point)
from .unified import (
    create_dataloader,
    create_diffusion_dataloader,
    get_dataloader_info,
    _get_compression_output_format,
)

# Common utilities for dataloader configuration
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

# Single-image dataloaders (seg, bravo modes)
from .single import (
    create_test_dataloader,
    create_validation_dataloader,
)

# Dual-image dataloaders (t1_pre + t1_gd with seg conditioning)
from .dual import (
    create_dual_image_dataloader,
    create_dual_image_test_dataloader,
    create_dual_image_validation_dataloader,
)

# VAE dataloaders (single or dual modality, no seg concatenation)
from .vae import (
    create_vae_dataloader,
    create_vae_test_dataloader,
    create_vae_validation_dataloader,
)

# Multi-modality dataloaders (progressive VAE training)
from .multi_modality import (
    create_multi_modality_dataloader,
    create_multi_modality_test_dataloader,
    create_multi_modality_validation_dataloader,
    create_single_modality_validation_loader,
)

# Multi-modality diffusion dataloaders (with mode_id)
from .multi_diffusion import (
    create_multi_diffusion_dataloader,
    create_multi_diffusion_test_dataloader,
    create_multi_diffusion_validation_dataloader,
    create_single_modality_diffusion_val_loader,
)

# 3D volume dataloaders (full volumes, used by both diffusion and compression)
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

# Segmentation mask compression dataloaders (seg-only for DC-AE)
from .seg_compression import (
    create_seg_compression_dataloader,
    create_seg_compression_validation_dataloader,
    create_seg_compression_test_dataloader,
)

# Segmentation conditioned dataloaders (seg + size bins)
from .seg_conditioned import (
    create_seg_conditioned_dataloader,
    create_seg_conditioned_validation_dataloader,
    create_seg_conditioned_test_dataloader,
    SegConditionedDataset,
    compute_size_bins,
)

# 3D Segmentation dataloaders (3D connected components, size-bin conditioned)
from .seg import (
    create_seg_dataloader,
    create_seg_validation_dataloader,
    create_seg_test_dataloader,
    SegDataset,
    compute_size_bins_3d,
    compute_feret_diameter_3d,
)

# Latent diffusion dataloaders (pre-encoded latents)
from .latent import (
    LatentDataset,
    LatentCacheBuilder,
    create_latent_dataloader,
    create_latent_validation_dataloader,
    create_latent_test_dataloader,
    load_compression_model,
    detect_compression_type,
)

__all__ = [
    # Base classes
    'BaseDiffusionDataset',
    'BaseDiffusionDataset2D',
    'BaseDiffusionDataset3D',
    'DictDatasetWrapper',
    'dict_collate_fn',
    'validate_batch_format',
    # Unified factory (recommended)
    'create_dataloader',
    'create_diffusion_dataloader',
    'get_dataloader_info',
    '_get_compression_output_format',
    # Common utilities
    'DataLoaderConfig',
    'setup_distributed_sampler',
    'get_data_dir',
    'validate_data_dir',
    'get_modality_keys',
    'validate_modality_keys',
    'check_seg_available',
    'MODALITY_KEYS',
    # Single-image
    'create_dataloader',
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
    # Multi-modality VAE
    'create_multi_modality_dataloader',
    'create_multi_modality_validation_dataloader',
    'create_multi_modality_test_dataloader',
    'create_single_modality_validation_loader',
    # Multi-modality diffusion
    'create_multi_diffusion_dataloader',
    'create_multi_diffusion_validation_dataloader',
    'create_multi_diffusion_test_dataloader',
    'create_single_modality_diffusion_val_loader',
    # 3D VAE
    'create_vae_3d_dataloader',
    'create_vae_3d_validation_dataloader',
    'create_vae_3d_test_dataloader',
    'create_vae_3d_multi_modality_dataloader',
    'create_vae_3d_multi_modality_validation_dataloader',
    'create_vae_3d_multi_modality_test_dataloader',
    'create_vae_3d_single_modality_validation_loader',
    'Base3DVolumeDataset',
    'Volume3DDataset',
    'DualVolume3DDataset',
    'MultiModality3DDataset',
    'SingleModality3DDatasetWithSeg',
    # Segmentation mask compression
    'create_seg_compression_dataloader',
    'create_seg_compression_validation_dataloader',
    'create_seg_compression_test_dataloader',
    # Segmentation conditioned (2D)
    'create_seg_conditioned_dataloader',
    'create_seg_conditioned_validation_dataloader',
    'create_seg_conditioned_test_dataloader',
    'SegConditionedDataset',
    'compute_size_bins',
    # 3D Segmentation (3D connected components, size-bin conditioned)
    'create_seg_dataloader',
    'create_seg_validation_dataloader',
    'create_seg_test_dataloader',
    'SegDataset',
    'compute_size_bins_3d',
    'compute_feret_diameter_3d',
    # Latent diffusion
    'LatentDataset',
    'LatentCacheBuilder',
    'create_latent_dataloader',
    'create_latent_validation_dataloader',
    'create_latent_test_dataloader',
    'load_compression_model',
    'detect_compression_type',
]
