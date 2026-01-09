"""Training entry point for 3D DC-AE (Deep Compression Autoencoder) models.

This module provides the main training script for 3D DC-AE models
for high-compression volumetric medical image encoding.

Usage:
    # Default config (multi_modality mode)
    python -m medgen.scripts.train_dcae_3d

    # Single modality
    python -m medgen.scripts.train_dcae_3d mode=bravo

    # Cluster training
    python -m medgen.scripts.train_dcae_3d paths=cluster

    # Different compression config
    python -m medgen.scripts.train_dcae_3d dcae_3d=f32_d4
"""
import hydra
from omegaconf import DictConfig

from .train_compression_3d import train_compression_3d


@hydra.main(version_base=None, config_path="../../../configs", config_name="dcae_3d")
def main(cfg: DictConfig) -> None:
    """Main 3D DC-AE training entry point.

    Args:
        cfg: Hydra configuration object composed from YAML files.
    """
    train_compression_3d(cfg, trainer_type='dcae_3d')


if __name__ == "__main__":
    main()
