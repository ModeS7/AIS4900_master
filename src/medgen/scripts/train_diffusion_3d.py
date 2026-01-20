"""
DEPRECATED: Use train.py with model.spatial_dims=3 instead.

This script is kept for backward compatibility but will be removed in future versions.

Usage (deprecated):
    python -m medgen.scripts.train_diffusion_3d mode=bravo strategy=rflow

Recommended (new):
    python -m medgen.scripts.train mode=bravo strategy=rflow model.spatial_dims=3

Or use the diffusion_3d config which sets spatial_dims=3:
    python -m medgen.scripts.train --config-name diffusion_3d mode=bravo strategy=rflow
"""
import warnings
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="diffusion_3d")
def main(cfg: DictConfig) -> None:
    """Deprecated 3D diffusion training entry point.

    This script forwards to the unified train.py with spatial_dims=3.
    Use 'python -m medgen.scripts.train model.spatial_dims=3' instead.
    """
    warnings.warn(
        "train_diffusion_3d.py is deprecated. "
        "Use 'python -m medgen.scripts.train model.spatial_dims=3' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    logger.warning(
        "=" * 60 + "\n"
        "DEPRECATION WARNING: train_diffusion_3d.py is deprecated.\n"
        "Use 'python -m medgen.scripts.train model.spatial_dims=3' instead.\n"
        "This script will be removed in a future version.\n" +
        "=" * 60
    )

    # Ensure spatial_dims=3 is set (should already be in diffusion_3d config)
    if cfg.model.get('spatial_dims', 2) != 3:
        OmegaConf.update(cfg, 'model.spatial_dims', 3, merge=True)
        logger.info("Set model.spatial_dims=3")

    # Import and run the main training function
    from medgen.scripts.train import main as train_main, validate_config

    # Validate and run
    validate_config(cfg)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # The train.py main function will detect spatial_dims=3 and route to _train_3d
    from medgen.scripts.train import _train_3d
    _train_3d(cfg)


if __name__ == "__main__":
    main()
