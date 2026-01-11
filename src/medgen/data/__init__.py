"""Data loading and processing utilities."""

from .augmentation import (
    apply_augmentation,
    build_diffusion_augmentation,
    build_vae_augmentation,
    create_vae_collate_fn,
    cutmix,
    mixup,
)
from .dataset import NiFTIDataset, build_standard_transform, validate_modality_exists
from .base_embed import create_zero_init_mlp
from .score_aug import ScoreAugTransform
from .mode_embed import (
    ModeEmbedModelWrapper,
    ModeEmbedDropoutModelWrapper,
    NoModeModelWrapper,
    LateModeModelWrapper,
    ModeTimeEmbed,
    MODE_ID_MAP,
    encode_mode_id,
)
from .combined_embed import CombinedModelWrapper, CombinedTimeEmbed, create_conditioning_wrapper
from .utils import extract_slices_dual, extract_slices_single, extract_slices_single_with_seg, make_binary, merge_sequences
from .lossless_mask_codec import (
    encode_mask_lossless,
    decode_mask_lossless,
    get_latent_shape,
    encode_f32,
    decode_f32,
    encode_f64,
    decode_f64,
    encode_f128,
    decode_f128,
    FORMATS as LOSSLESS_FORMATS,
)

# Import all loaders from loaders subpackage
from .loaders import (
    # Diffusion single-image
    create_dataloader,
    create_validation_dataloader,
    create_test_dataloader,
    # Diffusion dual-image
    create_dual_image_dataloader,
    create_dual_image_validation_dataloader,
    create_dual_image_test_dataloader,
    # VAE
    create_vae_dataloader,
    create_vae_validation_dataloader,
    create_vae_test_dataloader,
    # Multi-modality VAE
    create_multi_modality_dataloader,
    create_multi_modality_validation_dataloader,
    create_multi_modality_test_dataloader,
    create_single_modality_validation_loader,
    # Multi-modality diffusion
    create_multi_diffusion_dataloader,
    create_multi_diffusion_validation_dataloader,
    create_multi_diffusion_test_dataloader,
    create_single_modality_diffusion_val_loader,
    # 3D VAE loaders
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

__all__ = [
    # Dataset class
    'NiFTIDataset',
    # Transform utilities
    'build_standard_transform',
    'validate_modality_exists',
    # Augmentation
    'apply_augmentation',
    'build_diffusion_augmentation',
    'build_vae_augmentation',
    'create_vae_collate_fn',
    'mixup',
    'cutmix',
    # Score Augmentation
    'ScoreAugTransform',
    # Base embedding helper
    'create_zero_init_mlp',
    # Slice extraction utilities
    'extract_slices_single',
    'extract_slices_single_with_seg',
    'extract_slices_dual',
    'merge_sequences',
    'make_binary',
    # Diffusion single-image loaders
    'create_dataloader',
    'create_validation_dataloader',
    'create_test_dataloader',
    # Diffusion dual-image loaders
    'create_dual_image_dataloader',
    'create_dual_image_validation_dataloader',
    'create_dual_image_test_dataloader',
    # VAE loaders
    'create_vae_dataloader',
    'create_vae_validation_dataloader',
    'create_vae_test_dataloader',
    # Multi-modality VAE loaders
    'create_multi_modality_dataloader',
    'create_multi_modality_validation_dataloader',
    'create_multi_modality_test_dataloader',
    'create_single_modality_validation_loader',
    # Multi-modality diffusion loaders
    'create_multi_diffusion_dataloader',
    'create_multi_diffusion_validation_dataloader',
    'create_multi_diffusion_test_dataloader',
    'create_single_modality_diffusion_val_loader',
    # 3D VAE loaders
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
    # Mode embedding
    'ModeEmbedModelWrapper',
    'ModeEmbedDropoutModelWrapper',
    'NoModeModelWrapper',
    'LateModeModelWrapper',
    'ModeTimeEmbed',
    'MODE_ID_MAP',
    'encode_mode_id',
    'CombinedModelWrapper',
    'CombinedTimeEmbed',
    'create_conditioning_wrapper',
    # Lossless mask codec
    'encode_mask_lossless',
    'decode_mask_lossless',
    'get_latent_shape',
    'encode_f32',
    'decode_f32',
    'encode_f64',
    'decode_f64',
    'encode_f128',
    'decode_f128',
    'LOSSLESS_FORMATS',
]
