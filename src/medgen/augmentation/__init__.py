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
    OMEGA_ENCODING_DIM,
)

# ScoreAug (3D)
from .score_aug_3d import (
    ScoreAugTransform3D,
    ScoreAugModelWrapper3D,
    OmegaTimeEmbed3D,
    encode_omega_3d,
)

# SDA - Shifted Data Augmentation (2D)
from .sda import (
    SDATransform,
    create_sda_transform,
)

# SDA (3D)
from .sda_3d import (
    SDATransform3D,
    create_sda_transform_3d,
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
    'OMEGA_ENCODING_DIM',
    # ScoreAug 3D
    'ScoreAugTransform3D',
    'ScoreAugModelWrapper3D',
    'OmegaTimeEmbed3D',
    'encode_omega_3d',
    # SDA 2D
    'SDATransform',
    'create_sda_transform',
    # SDA 3D
    'SDATransform3D',
    'create_sda_transform_3d',
]
