"""Training entry point for 3D VAE models.

This module provides the main training script for 3D AutoencoderKL models
for volumetric medical image compression.

Usage:
    # Default config (dual mode)
    python -m medgen.scripts.train_vae_3d

    # Single modality
    python -m medgen.scripts.train_vae_3d mode=bravo

    # Cluster training
    python -m medgen.scripts.train_vae_3d paths=cluster
"""
import hydra
from omegaconf import DictConfig

from .train_compression_3d import train_compression_3d


@hydra.main(version_base=None, config_path="../../../configs", config_name="vae_3d")
def main(cfg: DictConfig) -> None:
    """Main 3D VAE training entry point.

    Args:
        cfg: Hydra configuration object composed from YAML files.
    """
    train_compression_3d(cfg, trainer_type='vae_3d')


if __name__ == "__main__":
    main()
