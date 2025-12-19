"""Core utilities and shared infrastructure."""

from .constants import (
    ModeType,
    BINARY_THRESHOLD_GT,
    BINARY_THRESHOLD_GEN,
    BINARY_THRESHOLD,
    MAX_WHITE_PERCENTAGE,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_EMA_UPDATE_AFTER_STEP,
    DEFAULT_EMA_UPDATE_EVERY,
    DEFAULT_MIN_SNR_GAMMA,
    DEFAULT_PERCEPTUAL_WEIGHT,
    DEFAULT_CHANNELS,
    DEFAULT_ATTENTION_LEVELS,
    DEFAULT_NUM_RES_BLOCKS,
    DEFAULT_NUM_HEAD_CHANNELS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_DUAL_IMAGE_KEYS,
)
from .cuda_utils import setup_cuda_optimizations
from .distributed import setup_distributed
from .schedulers import create_warmup_cosine_scheduler
from .validation import (
    validate_common_config,
    validate_model_config,
    validate_diffusion_config,
    validate_vae_config,
    validate_progressive_config,
    run_validation,
)
from .model_utils import wrap_model_for_training

__all__ = [
    # Enums
    'ModeType',
    # Constants
    'BINARY_THRESHOLD_GT',
    'BINARY_THRESHOLD_GEN',
    'BINARY_THRESHOLD',
    'MAX_WHITE_PERCENTAGE',
    'DEFAULT_GRADIENT_CLIP_NORM',
    'DEFAULT_EMA_UPDATE_AFTER_STEP',
    'DEFAULT_EMA_UPDATE_EVERY',
    'DEFAULT_MIN_SNR_GAMMA',
    'DEFAULT_PERCEPTUAL_WEIGHT',
    'DEFAULT_CHANNELS',
    'DEFAULT_ATTENTION_LEVELS',
    'DEFAULT_NUM_RES_BLOCKS',
    'DEFAULT_NUM_HEAD_CHANNELS',
    'DEFAULT_NUM_WORKERS',
    'DEFAULT_DUAL_IMAGE_KEYS',
    # Utilities
    'setup_cuda_optimizations',
    # Distributed training
    'setup_distributed',
    # Schedulers
    'create_warmup_cosine_scheduler',
    # Validation
    'validate_common_config',
    'validate_model_config',
    'validate_diffusion_config',
    'validate_vae_config',
    'validate_progressive_config',
    'run_validation',
    # Model utilities
    'wrap_model_for_training',
]
