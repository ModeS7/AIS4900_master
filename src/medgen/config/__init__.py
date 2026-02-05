"""Typed configuration dataclasses for MedGen.

Provides type-safe configuration access with IDE autocomplete and
validation at startup rather than runtime errors.

This module eliminates deep Hydra config access patterns like:
    cfg.training.get('ema', {}).get('decay', 0.9999)

Instead, use typed access:
    config = Config.from_hydra(cfg)
    decay = config.training.ema.decay  # IDE autocomplete works!

All defaults are defined in ONE place (the dataclass), eliminating
inconsistent defaults scattered across the codebase.

Example:
    from medgen.config import Config

    # Single conversion point from Hydra to typed config
    config = Config.from_hydra(cfg)

    # Now use typed access throughout:
    lr = config.training.learning_rate
    size = config.model.image_size
    use_ema = config.training.ema.enabled
    ema_decay = config.training.ema.decay  # Single source of truth!

    # Validate at startup
    config.validate_and_raise()
"""
from .base import Config
from .latent import LatentConfig
from .mode import ModeConfig, ModeEmbeddingConfig, SizeBinConfig
from .model import ModelConfig
from .paths import PathsConfig
from .strategy import StrategyConfig
from .training import (
    DataLoaderConfig,
    EMAConfig,
    LoggingConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from .validation import validate_and_raise, validate_config
from .volume import VolumeConfig

__all__ = [
    # Main config class
    'Config',
    # Model config
    'ModelConfig',
    # Paths config
    'PathsConfig',
    # Training configs
    'TrainingConfig',
    'EMAConfig',
    'OptimizerConfig',
    'SchedulerConfig',
    'DataLoaderConfig',
    'LoggingConfig',
    # Volume config
    'VolumeConfig',
    # Latent config
    'LatentConfig',
    # Strategy config
    'StrategyConfig',
    # Mode config
    'ModeConfig',
    'ModeEmbeddingConfig',
    'SizeBinConfig',
    # Validation
    'validate_config',
    'validate_and_raise',
]
