"""Base configuration class combining all config sections.

Provides a single Config class that combines all typed configuration
dataclasses into one object for convenient access.
"""
from dataclasses import dataclass

from omegaconf import DictConfig

from .latent import LatentConfig
from .mode import ModeConfig
from .model import ModelConfig
from .paths import PathsConfig
from .strategy import StrategyConfig
from .training import TrainingConfig
from .volume import VolumeConfig


@dataclass
class Config:
    """Complete typed configuration combining all sections.

    This is the main entry point for typed configuration access.
    Create an instance from Hydra config and use typed attributes
    instead of deep dictionary access.

    Attributes:
        model: Model architecture configuration.
        paths: Directory paths configuration.
        training: Training parameters configuration.
        strategy: Diffusion strategy configuration.
        mode: Training mode configuration.
        latent: Latent diffusion configuration.
        volume: 3D volume configuration (None for 2D).

    Example:
        >>> from medgen.config import Config
        >>> config = Config.from_hydra(cfg)
        >>>
        >>> # Now use typed access:
        >>> lr = config.training.learning_rate  # IDE autocomplete works!
        >>> size = config.model.image_size
        >>> use_ema = config.training.ema.enabled
        >>> ema_decay = config.training.ema.decay  # Single source of truth!
    """
    model: ModelConfig
    paths: PathsConfig
    training: TrainingConfig
    strategy: StrategyConfig
    mode: ModeConfig
    latent: LatentConfig
    volume: VolumeConfig | None

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'Config':
        """Create typed config from Hydra DictConfig.

        This is the single conversion point where all config extraction
        happens. After this, all config access should be through the
        typed Config object.

        Args:
            cfg: Hydra configuration object.

        Returns:
            Complete Config instance with all sections typed.
        """
        # Extract mode first to determine is_seg_mode for other configs
        mode = ModeConfig.from_hydra(cfg)
        is_seg_mode = mode.is_seg_mode

        return cls(
            model=ModelConfig.from_hydra(cfg),
            paths=PathsConfig.from_hydra(cfg),
            training=TrainingConfig.from_hydra(cfg, is_seg_mode=is_seg_mode),
            strategy=StrategyConfig.from_hydra(cfg),
            mode=mode,
            latent=LatentConfig.from_hydra(cfg),
            volume=VolumeConfig.from_hydra(cfg),
        )

    def validate(self) -> list[str]:
        """Validate this configuration.

        Returns:
            List of error messages (empty if valid).
        """
        from .validation import validate_config
        return validate_config(self)

    def validate_and_raise(self) -> None:
        """Validate and raise if invalid.

        Raises:
            ValueError: If configuration is invalid.
        """
        from .validation import validate_and_raise
        validate_and_raise(self)
