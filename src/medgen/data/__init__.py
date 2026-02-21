"""Data loading and processing utilities."""

from medgen.augmentation import (
    apply_augmentation,
    build_diffusion_augmentation,
    build_seg_diffusion_augmentation,
    build_seg_diffusion_augmentation_with_binarize,
    build_vae_augmentation,
    create_vae_collate_fn,
    cutmix,
    mixup,
    binarize_mask,
    BinarizeTransform,
    # Score Augmentation
    ScoreAugTransform,
    ScoreAugTransform3D,
    ScoreAugModelWrapper3D,
    # Shifted Data Augmentation
    SDATransform,
    create_sda_transform,
    SDATransform3D,
    create_sda_transform_3d,
)
from .dataset import NiFTIDataset, build_standard_transform, validate_modality_exists

# Model wrappers (re-exported from models.wrappers for backward compatibility)
from medgen.models.wrappers import (
    create_zero_init_mlp,
    ModeEmbedModelWrapper,
    ModeEmbedDropoutModelWrapper,
    NoModeModelWrapper,
    LateModeModelWrapper,
    ModeTimeEmbed,
    MODE_ID_MAP,
    encode_mode_id,
    CombinedModelWrapper,
    CombinedTimeEmbed,
    create_conditioning_wrapper,
    SizeBinModelWrapper,
    SizeBinTimeEmbed,
    encode_size_bins,
    DEFAULT_BIN_EDGES as SIZE_BIN_EDGES,
    DEFAULT_NUM_BINS as SIZE_NUM_BINS,
    format_size_bins,
)
from .utils import binarize_seg, extract_slices_dual, extract_slices_single, extract_slices_single_with_seg, make_binary, merge_sequences
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
    encode_k8x8,
    decode_k8x8,
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
    # Seg conditioned loaders (2D)
    create_seg_conditioned_dataloader,
    create_seg_conditioned_validation_dataloader,
    create_seg_conditioned_test_dataloader,
    SegConditionedDataset,
    compute_size_bins,
    # 3D Seg loaders (3D connected components, size-bin conditioned)
    create_seg_dataloader,
    create_seg_validation_dataloader,
    create_seg_test_dataloader,
    SegDataset,
    compute_size_bins_3d,
    compute_feret_diameter_3d,
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
    'build_seg_diffusion_augmentation',
    'build_seg_diffusion_augmentation_with_binarize',
    'build_vae_augmentation',
    'create_vae_collate_fn',
    'mixup',
    'cutmix',
    'binarize_mask',
    'BinarizeTransform',
    # Score Augmentation
    'ScoreAugTransform',
    'ScoreAugTransform3D',
    'ScoreAugModelWrapper3D',
    # Shifted Data Augmentation
    'SDATransform',
    'create_sda_transform',
    'SDATransform3D',
    'create_sda_transform_3d',
    # Base embedding helper
    'create_zero_init_mlp',
    # Slice extraction utilities
    'extract_slices_single',
    'extract_slices_single_with_seg',
    'extract_slices_dual',
    'merge_sequences',
    'binarize_seg',
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
    # Seg conditioned loaders (2D)
    'create_seg_conditioned_dataloader',
    'create_seg_conditioned_validation_dataloader',
    'create_seg_conditioned_test_dataloader',
    'SegConditionedDataset',
    'compute_size_bins',
    # 3D Seg loaders (3D connected components, size-bin conditioned)
    'create_seg_dataloader',
    'create_seg_validation_dataloader',
    'create_seg_test_dataloader',
    'SegDataset',
    'compute_size_bins_3d',
    'compute_feret_diameter_3d',
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
    # Size bin embedding
    'SizeBinModelWrapper',
    'SizeBinTimeEmbed',
    'encode_size_bins',
    'SIZE_BIN_EDGES',
    'SIZE_NUM_BINS',
    'format_size_bins',
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
    'encode_k8x8',
    'decode_k8x8',
    'LOSSLESS_FORMATS',
]
