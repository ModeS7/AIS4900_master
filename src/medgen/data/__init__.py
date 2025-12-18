"""Data loading and processing utilities."""

from .augmentation import apply_augmentation, build_augmentation
from .nifti_dataset import (
    NiFTIDataset,
    # Diffusion dataloaders
    create_dataloader,
    create_dual_image_dataloader,
    create_validation_dataloader,
    create_dual_image_validation_dataloader,
    create_test_dataloader,
    create_dual_image_test_dataloader,
    # Multi-modality dataloaders
    create_multi_modality_dataloader,
    create_multi_modality_validation_dataloader,
    create_multi_modality_test_dataloader,
    # VAE dataloaders
    create_vae_dataloader,
    create_vae_validation_dataloader,
    create_vae_test_dataloader,
    # Utilities
    extract_slices_single,
    extract_slices_dual,
    extract_slices_multi_modality,
    merge_sequences,
    make_binary,
)
