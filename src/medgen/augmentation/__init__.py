"""
Augmentation transforms for medical image generation.

This package provides:
- Standard augmentation: Albumentations-based transforms for VAE/diffusion training
- ScoreAug: Augmentation on noisy data with omega conditioning (2D and 3D)
- SDA: Shifted Data Augmentation with timestep compensation (2D and 3D)
- Batch augmentation: MixUp, CutMix for regularization

Usage:
    from medgen.augmentation import (
        # Standard augmentation
        build_diffusion_augmentation,
        build_vae_augmentation,
        apply_augmentation,
        # ScoreAug
        ScoreAugTransform,
        ScoreAugModelWrapper,
        OmegaTimeEmbed,
        # SDA
        SDATransform,
        create_sda_transform,
        # Batch augmentation
        mixup,
        cutmix,
        create_vae_collate_fn,
    )
"""

# Standard augmentation (Albumentations-based)
from .augmentation import (
    DiscreteTranslate,
    BinarizeTransform,
    build_diffusion_augmentation,
    build_vae_augmentation,
    build_seg_augmentation,
    build_seg_diffusion_augmentation,
    build_seg_diffusion_augmentation_with_binarize,
    apply_augmentation,
    binarize_mask,
    # Batch augmentation
    mixup,
    cutmix,
    create_vae_collate_fn,
    create_seg_collate_fn,
    mosaic_augmentation,
    copy_paste_augmentation,
)

# ScoreAug (2D)
from .score_aug import (
    ScoreAugTransform,
    ScoreAugModelWrapper,
    OmegaTimeEmbed,
    encode_omega,
    generate_pattern_mask,
    apply_mode_intensity_scale,
    inverse_mode_intensity_scale,
    clear_pattern_cache,
    OMEGA_ENCODING_DIM,
)

# ScoreAug (3D) - now unified in score_aug.py
from .score_aug import (
    ScoreAugTransform3D,
    ScoreAugModelWrapper3D,
    OmegaTimeEmbed3D,
    encode_omega_3d,
    generate_pattern_mask_3d,
    apply_mode_intensity_scale_3d,
    inverse_mode_intensity_scale_3d,
    OMEGA_ENCODING_DIM_3D,
)

# SDA - Shifted Data Augmentation (unified 2D/3D)
from .sda import (
    SDATransform,
    SDATransform3D,  # Alias for backwards compatibility
    create_sda_transform,
    create_sda_transform_3d,  # Alias for backwards compatibility
)

__all__ = [
    # Standard augmentation
    'DiscreteTranslate',
    'BinarizeTransform',
    'build_diffusion_augmentation',
    'build_vae_augmentation',
    'build_seg_augmentation',
    'build_seg_diffusion_augmentation',
    'build_seg_diffusion_augmentation_with_binarize',
    'apply_augmentation',
    'binarize_mask',
    # Batch augmentation
    'mixup',
    'cutmix',
    'create_vae_collate_fn',
    'create_seg_collate_fn',
    'mosaic_augmentation',
    'copy_paste_augmentation',
    # ScoreAug 2D
    'ScoreAugTransform',
    'ScoreAugModelWrapper',
    'OmegaTimeEmbed',
    'encode_omega',
    'generate_pattern_mask',
    'apply_mode_intensity_scale',
    'inverse_mode_intensity_scale',
    'clear_pattern_cache',
    'OMEGA_ENCODING_DIM',
    # ScoreAug 3D
    'ScoreAugTransform3D',
    'ScoreAugModelWrapper3D',
    'OmegaTimeEmbed3D',
    'encode_omega_3d',
    'generate_pattern_mask_3d',
    'apply_mode_intensity_scale_3d',
    'inverse_mode_intensity_scale_3d',
    'OMEGA_ENCODING_DIM_3D',
    # SDA (unified 2D/3D)
    'SDATransform',
    'SDATransform3D',  # Alias
    'create_sda_transform',
    'create_sda_transform_3d',  # Alias
]
