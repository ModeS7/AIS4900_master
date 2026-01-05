"""Training entry point for VQ-VAE models.

This module provides the main training script for VQ-VAE/VQ-GAN models
that will be used for latent diffusion training with discrete codebook.

Usage:
    # Default config
    python -m medgen.scripts.train_vqvae

    # Override via CLI
    python -m medgen.scripts.train_vqvae vqvae.num_embeddings=1024 model.image_size=256

    # Pure VQ-VAE (no GAN)
    python -m medgen.scripts.train_vqvae vqvae.disable_gan=true

    # Cluster training
    python -m medgen.scripts.train_vqvae paths=cluster
"""
import hydra
from omegaconf import DictConfig

from .train_compression import train_compression


@hydra.main(version_base=None, config_path="../../../configs", config_name="vqvae")
def main(cfg: DictConfig) -> None:
    """Main VQ-VAE training entry point.

    Args:
        cfg: Hydra configuration object composed from YAML files.
    """
    train_compression(cfg, trainer_type='vqvae')


if __name__ == "__main__":
    main()
