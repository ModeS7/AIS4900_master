"""Core utilities and shared infrastructure."""

from .defaults import (
    CompressionDefaults,
    VAE_DEFAULTS,
    VQVAE_DEFAULTS,
    DCAE_DEFAULTS,
    COMPRESSION_DEFAULTS,
)
from .dict_utils import get_with_fallbacks, IMAGE_KEYS, MASK_KEYS, PATIENT_KEYS
from .constants import (
    ModeType,
    BINARY_THRESHOLD_GT,
    BINARY_THRESHOLD_GEN,
    MAX_WHITE_PERCENTAGE,
    DEFAULT_CHANNELS,
    DEFAULT_ATTENTION_LEVELS,
    DEFAULT_NUM_RES_BLOCKS,
    DEFAULT_NUM_HEAD_CHANNELS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_DUAL_IMAGE_KEYS,
    get_modality_for_mode,
)
from .cuda_utils import setup_cuda_optimizations
from .distributed import setup_distributed
from .mode_factory import ModeFactory, ModeConfig, ModeCategory
from .schedulers import create_warmup_cosine_scheduler, create_warmup_constant_scheduler, create_plateau_scheduler
from .validation import (
    validate_common_config,
    validate_model_config,
    validate_diffusion_config,
    validate_vae_config,
    validate_vqvae_config,
    validate_training_config,
    validate_strategy_mode_compatibility,
    validate_3d_config,
    validate_latent_config,
    validate_regional_logging,
    validate_strategy_config,
    validate_ema_config,
    validate_optimizer_config,
    validate_augmentation_config,
    run_validation,
)
from .model_utils import wrap_model_for_training
from .spatial_utils import (
    broadcast_to_spatial,
    extract_center_slice,
    get_pooling_fn,
    get_spatial_sum_dims,
)

__all__ = [
    # Defaults
    'CompressionDefaults',
    'VAE_DEFAULTS',
    'VQVAE_DEFAULTS',
    'DCAE_DEFAULTS',
    'COMPRESSION_DEFAULTS',
    # Dict utilities
    'get_with_fallbacks',
    'IMAGE_KEYS',
    'MASK_KEYS',
    'PATIENT_KEYS',
    # Enums
    'ModeType',
    # Constants
    'BINARY_THRESHOLD_GT',
    'BINARY_THRESHOLD_GEN',
    'MAX_WHITE_PERCENTAGE',
    'DEFAULT_CHANNELS',
    'DEFAULT_ATTENTION_LEVELS',
    'DEFAULT_NUM_RES_BLOCKS',
    'DEFAULT_NUM_HEAD_CHANNELS',
    'DEFAULT_NUM_WORKERS',
    'DEFAULT_DUAL_IMAGE_KEYS',
    'get_modality_for_mode',
    # Utilities
    'setup_cuda_optimizations',
    # Distributed training
    'setup_distributed',
    # Mode Factory
    'ModeFactory',
    'ModeConfig',
    'ModeCategory',
    # Schedulers
    'create_warmup_cosine_scheduler',
    'create_warmup_constant_scheduler',
    'create_plateau_scheduler',
    # Validation
    'validate_common_config',
    'validate_model_config',
    'validate_diffusion_config',
    'validate_vae_config',
    'validate_vqvae_config',
    'validate_training_config',
    'validate_strategy_mode_compatibility',
    'validate_3d_config',
    'validate_latent_config',
    'validate_regional_logging',
    'validate_strategy_config',
    'validate_ema_config',
    'validate_optimizer_config',
    'validate_augmentation_config',
    'run_validation',
    # Model utilities
    'wrap_model_for_training',
    # Spatial utilities
    'broadcast_to_spatial',
    'extract_center_slice',
    'get_pooling_fn',
    'get_spatial_sum_dims',
]
