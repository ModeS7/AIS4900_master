"""Dataloader factory functions for diffusion and VAE training."""

# Single-image dataloaders (seg, bravo modes)
from .single import (
    create_dataloader,
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
)

__all__ = [
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
    # Multi-modality
    'create_multi_modality_dataloader',
    'create_multi_modality_validation_dataloader',
    'create_multi_modality_test_dataloader',
]
