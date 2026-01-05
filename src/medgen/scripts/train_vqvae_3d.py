"""Training entry point for 3D VQ-VAE models.

This module provides the main training script for 3D VQ-VAE models
for volumetric medical image compression with discrete latent space.

Advantages over 3D KL-VAE:
- Lower memory (no mu/logvar branches)
- Discrete codebook (cleaner latent space)

Usage:
    # Default config (dual mode, 4x compression)
    python -m medgen.scripts.train_vqvae_3d

    # Multi-modality (all sequences)
    python -m medgen.scripts.train_vqvae_3d mode=multi_modality

    # Cluster training
    python -m medgen.scripts.train_vqvae_3d paths=cluster
"""
import hydra
from omegaconf import DictConfig

from .train_compression_3d import train_compression_3d


@hydra.main(version_base=None, config_path="../../../configs", config_name="vqvae_3d")
def main(cfg: DictConfig) -> None:
    """Main 3D VQ-VAE training entry point.

    Args:
        cfg: Hydra configuration object composed from YAML files.
    """
    train_compression_3d(cfg, trainer_type='vqvae_3d')


if __name__ == "__main__":
    main()
