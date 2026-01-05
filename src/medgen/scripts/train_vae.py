"""Training entry point for VAE models.

This module provides the main training script for AutoencoderKL models
that will be used for latent diffusion training.

Usage:
    # Default config
    python -m medgen.scripts.train_vae

    # Override via CLI
    python -m medgen.scripts.train_vae vae.latent_channels=8 model.image_size=256

    # Cluster training
    python -m medgen.scripts.train_vae paths=cluster
"""
import hydra
from omegaconf import DictConfig

from .train_compression import train_compression


@hydra.main(version_base=None, config_path="../../../configs", config_name="vae")
def main(cfg: DictConfig) -> None:
    """Main VAE training entry point.

    Args:
        cfg: Hydra configuration object composed from YAML files.
    """
    train_compression(cfg, trainer_type='vae')


if __name__ == "__main__":
    main()
