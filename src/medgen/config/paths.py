"""Paths configuration dataclass.

Provides type-safe access to directory paths and related settings.
"""
import os
from dataclasses import dataclass, field

from omegaconf import DictConfig


@dataclass
class PathsConfig:
    """Path configuration with auto-derived directories.

    Attributes:
        data_dir: Root directory for datasets.
        model_dir: Directory for model checkpoints.
        log_dir: Directory for logs (defaults to model_dir).
        cache_dir: Directory for caching (derived from model_dir if not set).
        generated_dir: Directory for generated samples.
        fov_mm: Field of view in millimeters (used for size calculations).
        name: Environment name ('local' or 'cluster').
    """
    data_dir: str
    model_dir: str
    log_dir: str = ''
    cache_dir: str | None = None
    generated_dir: str | None = None
    fov_mm: float = 240.0
    name: str = 'local'

    def __post_init__(self) -> None:
        """Set derived paths after initialization."""
        if not self.log_dir:
            self.log_dir = self.model_dir
        if self.cache_dir is None:
            self.cache_dir = os.path.join(self.model_dir, '.cache')

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'PathsConfig':
        """Extract paths config from Hydra DictConfig.

        Args:
            cfg: Hydra configuration object containing paths section.

        Returns:
            PathsConfig instance with extracted values.
        """
        paths = cfg.paths
        return cls(
            data_dir=paths.data_dir,
            model_dir=paths.model_dir,
            log_dir=paths.get('log_dir', paths.model_dir),
            cache_dir=paths.get('cache_dir'),
            generated_dir=paths.get('generated_dir'),
            fov_mm=paths.get('fov_mm', 240.0),
            name=paths.get('name', 'local'),
        )
