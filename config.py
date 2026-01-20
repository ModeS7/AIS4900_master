"""Path configuration for misc utility scripts.

This module provides PathConfig class for misc scripts that need access
to data directories. It mirrors the paths defined in configs/paths/*.yaml.

Usage:
    from config import PathConfig

    path_config = PathConfig()  # Auto-detects local vs cluster
    print(path_config.data_dir)
    print(path_config.brainmet_train_dir)
"""

import os
from pathlib import Path
from typing import Literal, Optional


ComputeEnv = Literal['local', 'cluster']


def _detect_compute_env() -> ComputeEnv:
    """Auto-detect compute environment based on hostname/paths."""
    # Check for cluster indicators
    if os.path.exists('/cluster/work'):
        return 'cluster'
    if 'SLURM_JOB_ID' in os.environ:
        return 'cluster'
    return 'local'


class PathConfig:
    """Path configuration that mirrors Hydra paths configs.

    Provides convenient access to data directories for misc utility scripts.
    Auto-detects local vs cluster environment, or can be explicitly set.

    Attributes:
        base_prefix: Base path prefix (/home/mode/NTNU or /cluster/work/modestas)
        data_dir: Main data directory (brainmetshare-3)
        model_dir: Model checkpoint directory
        brainmet_train_dir: BrainMetShare training data directory

    Example:
        >>> config = PathConfig()  # Auto-detect
        >>> config = PathConfig(compute='cluster')  # Force cluster paths
        >>> print(config.data_dir)
        /home/mode/NTNU/MedicalDataSets/brainmetshare-3
    """

    # Path configurations matching configs/paths/*.yaml
    _CONFIGS = {
        'local': {
            'base_prefix': Path('/home/mode/NTNU'),
            'data_dir': Path('/home/mode/NTNU/MedicalDataSets/brainmetshare-3'),
            'model_dir': Path('/home/mode/NTNU/AIS4900_master/runs'),
        },
        'cluster': {
            'base_prefix': Path('/cluster/work/modestas'),
            'data_dir': Path('/cluster/work/modestas/MedicalDataSets/brainmetshare-3'),
            'model_dir': Path('/cluster/work/modestas/AIS4900_master/runs'),
        },
    }

    def __init__(self, compute: Optional[ComputeEnv] = None):
        """Initialize PathConfig.

        Args:
            compute: Compute environment ('local' or 'cluster').
                If None, auto-detects based on hostname/paths.
        """
        if compute is None:
            compute = _detect_compute_env()

        if compute not in self._CONFIGS:
            raise ValueError(f"Unknown compute environment: {compute}. Use 'local' or 'cluster'.")

        config = self._CONFIGS[compute]
        self._compute = compute
        self.base_prefix: Path = config['base_prefix']
        self.data_dir: Path = config['data_dir']
        self.model_dir: Path = config['model_dir']

    @property
    def compute(self) -> ComputeEnv:
        """Current compute environment."""
        return self._compute

    @property
    def brainmet_train_dir(self) -> Path:
        """BrainMetShare training data directory."""
        return self.data_dir / 'Train'

    @property
    def brainmet_test_dir(self) -> Path:
        """BrainMetShare test data directory."""
        return self.data_dir / 'Test'

    def __repr__(self) -> str:
        return f"PathConfig(compute='{self._compute}', data_dir='{self.data_dir}')"
