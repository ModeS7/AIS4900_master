"""Centralized default values for training configuration.

These defaults are used when config values are not explicitly specified.
Each trainer type has its own defaults class with dimension-specific values.

Usage:
    from medgen.core.defaults import VAE_DEFAULTS, VQVAE_DEFAULTS, DCAE_DEFAULTS

    # In trainer:
    _DEFAULT_DISC_LR_2D = VAE_DEFAULTS.disc_lr_2d
    _DEFAULT_DISC_LR_3D = VAE_DEFAULTS.disc_lr_3d
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CompressionDefaults:
    """Base default values for compression training.

    These are the defaults used by BaseCompressionTrainer.
    Subclass-specific trainers override these values.
    """

    # Discriminator learning rates
    disc_lr_2d: float = 5e-4
    disc_lr_3d: float = 5e-4

    # Discriminator architecture
    disc_num_layers: int = 3
    disc_num_channels: int = 64

    # Loss weights
    perceptual_weight_2d: float = 0.001
    perceptual_weight_3d: float = 0.001
    adv_weight_2d: float = 0.01
    adv_weight_3d: float = 0.01
    reconstruction_weight: float = 1.0

    # 3D perceptual settings
    use_2_5d_perceptual: bool = True
    perceptual_slice_fraction: float = 0.25

    # EMA
    ema_decay: float = 0.9999


@dataclass(frozen=True)
class VAEDefaults(CompressionDefaults):
    """VAE-specific defaults.

    Lower 3D discriminator LR for training stability.
    """

    disc_lr_3d: float = 1e-4  # Lower for 3D stability


@dataclass(frozen=True)
class VQVAEDefaults(CompressionDefaults):
    """VQ-VAE-specific defaults.

    Slightly higher 3D perceptual weight and lower 3D adversarial weight.
    """

    perceptual_weight_3d: float = 0.002  # Slightly higher for VQ
    adv_weight_3d: float = 0.005  # Lower for 3D stability


@dataclass(frozen=True)
class DCAEDefaults(CompressionDefaults):
    """DC-AE-specific defaults.

    Much higher perceptual weight, no adversarial loss by default.
    """

    disc_lr_3d: float = 1e-4  # Lower for 3D stability
    perceptual_weight_2d: float = 0.1  # Much higher for DC-AE
    perceptual_weight_3d: float = 0.1
    adv_weight_2d: float = 0.0  # No adversarial by default
    adv_weight_3d: float = 0.0


# Singleton instances for easy access
COMPRESSION_DEFAULTS = CompressionDefaults()
VAE_DEFAULTS = VAEDefaults()
VQVAE_DEFAULTS = VQVAEDefaults()
DCAE_DEFAULTS = DCAEDefaults()


# =============================================================================
# Training Constants
# =============================================================================

# Learning rate bounds
DEFAULT_ETA_MIN = 1e-6  # Minimum learning rate for cosine scheduler

# Cosine scheduler
COSINE_SCHEDULER_ETA_MIN = 1e-6

# KL divergence weight
DEFAULT_KL_WEIGHT = 1e-6

# =============================================================================
# Image Constants
# =============================================================================

# Default image dimensions
DEFAULT_IMAGE_SIZE = 256
DEFAULT_STRATEGY_IMAGE_SIZE = 128

# =============================================================================
# Regional Analysis Constants
# =============================================================================

# Field of view
DEFAULT_FOV_MM = 240.0

# Minimum region area thresholds
MIN_TUMOR_AREA_PIXELS_2D = 5
MIN_TUMOR_AREA_PIXELS_3D = 10

# Binary thresholding
MASK_BINARY_THRESHOLD = 0.5

