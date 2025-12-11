"""
Central path configuration for AIS4005_IP project.

This module provides a centralized configuration system for all paths
used throughout the project, supporting local development, cluster
computing (IDUN), and Windows environments.

Usage:
    from config import PathConfig, get_path_config

    # Explicit environment
    config = PathConfig(compute='local')
    data_dir = config.data_dir

    # Auto-detect environment
    config = get_path_config()
    model_dir = config.model_dir
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union
import os


ComputeEnvironment = Literal['local', 'cluster', 'windows']


@dataclass
class PathConfig:
    """Centralized path configuration for the AIS4005_IP project.

    This class provides consistent path access across different compute
    environments (local Linux, IDUN cluster, Windows).

    Args:
        compute: The compute environment. Options are 'local', 'cluster',
            or 'windows'. Defaults to 'local'.

    Attributes:
        base_prefix: Root directory for user data and projects.
        project_root: Root directory of the AIS4005_IP project.
        data_dir: Directory containing medical datasets.
        model_dir: Directory for storing trained models.
        log_dir: Directory for TensorBoard logs.
        cache_dir: Directory for model caches (RadImageNet, etc.).

    Example:
        >>> config = PathConfig(compute='cluster')
        >>> print(config.data_dir)
        /cluster/work/modestas/MedicalDataSets
    """

    compute: ComputeEnvironment = 'local'
    _base_prefix: Path = field(init=False, repr=False)
    _project_root: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize paths based on compute environment."""
        if self.compute == 'cluster':
            self._base_prefix = Path('/cluster/work/modestas')
        elif self.compute == 'windows':
            self._base_prefix = Path('C:/NTNU')
        else:
            self._base_prefix = Path('/home/mode/NTNU')

        # Set project root based on environment
        if self.compute == 'windows':
            self._project_root = self._base_prefix / 'RepoThesis'
        else:
            self._project_root = self._base_prefix / 'AIS4005_IP'

    @property
    def base_prefix(self) -> Path:
        """Root directory for user data and projects."""
        return self._base_prefix

    @property
    def project_root(self) -> Path:
        """Root directory of the AIS4005_IP project."""
        return self._project_root

    @property
    def data_dir(self) -> Path:
        """Directory containing medical datasets."""
        return self._base_prefix / 'MedicalDataSets'

    @property
    def brainmet_dir(self) -> Path:
        """Directory containing BrainMetShare dataset."""
        return self.data_dir / 'brainmetshare-3'

    @property
    def brainmet_train_dir(self) -> Path:
        """Training data directory for BrainMetShare dataset."""
        return self.brainmet_dir / 'train'

    @property
    def brainmet_test_dir(self) -> Path:
        """Test data directory for BrainMetShare dataset."""
        return self.brainmet_dir / 'test'

    @property
    def model_dir(self) -> Path:
        """Directory for storing training runs (models, logs, metadata)."""
        return self._project_root / 'Generation' / 'runs'

    @property
    def log_dir(self) -> Path:
        """Deprecated: TensorBoard logs now stored in model_dir/run/tensorboard/."""
        return self.model_dir

    @property
    def cache_dir(self) -> Path:
        """Directory for pre-trained model caches."""
        return self._project_root / 'model_cache'

    @property
    def latent_data_dir(self) -> Path:
        """Directory for latent space representations."""
        return self.data_dir / 'latent_data'

    def get_model_path(self, model_name: str) -> Path:
        """Get full path for a specific model checkpoint.

        Args:
            model_name: Name of the model directory or file.

        Returns:
            Full path to the model.
        """
        return self.model_dir / model_name

    def get_log_path(self, experiment_name: str) -> Path:
        """Get full path for experiment logs.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            Full path to the log directory.
        """
        return self.log_dir / experiment_name

    @classmethod
    def from_environment(cls) -> 'PathConfig':
        """Auto-detect compute environment and create PathConfig.

        Detection order:
        1. Check for cluster path existence
        2. Check for Windows path patterns
        3. Default to local Linux

        Returns:
            PathConfig instance for the detected environment.
        """
        if os.path.exists('/cluster/work/modestas'):
            return cls(compute='cluster')
        elif os.name == 'nt' or os.path.exists('C:/NTNU'):
            return cls(compute='windows')
        return cls(compute='local')

    def __str__(self) -> str:
        """Return string representation of configuration."""
        return (
            f"PathConfig(compute='{self.compute}')\n"
            f"  base_prefix: {self.base_prefix}\n"
            f"  project_root: {self.project_root}\n"
            f"  data_dir: {self.data_dir}\n"
            f"  runs_dir: {self.model_dir}"
        )


def get_path_config(compute: Optional[ComputeEnvironment] = None) -> PathConfig:
    """Get a PathConfig instance, optionally specifying environment.

    Args:
        compute: Optional compute environment. If None, auto-detects
            based on available paths.

    Returns:
        PathConfig instance for the specified or detected environment.

    Example:
        >>> config = get_path_config()  # Auto-detect
        >>> config = get_path_config('cluster')  # Explicit
    """
    if compute is None:
        return PathConfig.from_environment()
    return PathConfig(compute=compute)
