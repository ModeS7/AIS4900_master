"""Diffusion strategy configuration dataclass.

Provides type-safe access to diffusion strategy settings (DDPM, RFlow).
"""
from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass
class StrategyConfig:
    """Diffusion strategy configuration.

    Attributes:
        name: Strategy name ('ddpm' or 'rflow').
        num_train_timesteps: Number of timesteps for training.
        use_discrete_timesteps: Whether to use discrete (int) timesteps.
        sample_method: Timestep sampling method ('uniform', 'logit-normal').
        use_timestep_transform: Whether to apply timestep transformation.
        prediction_type: What the model predicts ('epsilon', 'v_prediction', 'sample').
    """
    name: str = 'ddpm'
    num_train_timesteps: int = 1000
    use_discrete_timesteps: bool = True
    sample_method: str = 'uniform'
    use_timestep_transform: bool = False
    prediction_type: str = 'epsilon'

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'StrategyConfig':
        """Extract strategy config from Hydra DictConfig.

        Args:
            cfg: Hydra configuration object.

        Returns:
            StrategyConfig instance.
        """
        strategy = cfg.get('strategy', {})
        return cls(
            name=strategy.get('name', 'ddpm'),
            num_train_timesteps=strategy.get('num_train_timesteps', 1000),
            use_discrete_timesteps=strategy.get('use_discrete_timesteps', True),
            sample_method=strategy.get('sample_method', 'uniform'),
            use_timestep_transform=strategy.get('use_timestep_transform', False),
            prediction_type=strategy.get('prediction_type', 'epsilon'),
        )
