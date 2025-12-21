"""Data loading and processing utilities."""

from .augmentation import (
    apply_augmentation,
    build_augmentation,
    build_diffusion_augmentation,
    build_vae_augmentation,
    create_vae_collate_fn,
    cutmix,
    mixup,
)
from .dataset import NiFTIDataset, build_standard_transform, validate_modality_exists
from .score_aug import ScoreAugTransform
from .utils import extract_slices_dual, extract_slices_single, make_binary, merge_sequences

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
    # Multi-modality
    create_multi_modality_dataloader,
    create_multi_modality_validation_dataloader,
    create_multi_modality_test_dataloader,
)

__all__ = [
    # Dataset class
    'NiFTIDataset',
    # Transform utilities
    'build_standard_transform',
    'validate_modality_exists',
    # Augmentation
    'apply_augmentation',
    'build_augmentation',
    'build_diffusion_augmentation',
    'build_vae_augmentation',
    'create_vae_collate_fn',
    'mixup',
    'cutmix',
    # Score Augmentation
    'ScoreAugTransform',
    # Slice extraction utilities
    'extract_slices_single',
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
    # Multi-modality loaders
    'create_multi_modality_dataloader',
    'create_multi_modality_validation_dataloader',
    'create_multi_modality_test_dataloader',
]
