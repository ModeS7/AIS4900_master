"""
Progressive resolution VAE training.

Trains VAE at increasing resolutions (64 -> 128 -> 256) with automatic
plateau detection for phase transitions. Trains on multiple modalities
(bravo, flair, t1_pre, t1_gd) to create a pre-trained model for fine-tuning.

Usage:
    # Full progressive training
    python -m medgen.scripts.train_vae_progressive

    # Resume from checkpoint
    python -m medgen.scripts.train_vae_progressive progressive.resume_from=/path/to/progressive_state.pt

    # Custom settings
    python -m medgen.scripts.train_vae_progressive \
        progressive.final_phase.epochs=100 \
        progressive.plateau.min_improvement=1.0
"""
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from medgen.core import (
    run_validation,
    setup_cuda_optimizations,
    validate_common_config,
    validate_progressive_config,
    validate_vae_config,
)
from medgen.pipeline import ProgressiveVAETrainer

# Enable CUDA optimizations
setup_cuda_optimizations()

log = logging.getLogger(__name__)


def validate_config(cfg: DictConfig) -> None:
    """Validate progressive training configuration.

    Args:
        cfg: Hydra configuration object.

    Raises:
        ValueError: If configuration is invalid.
    """
    run_validation(cfg, [
        validate_common_config,
        validate_vae_config,
        validate_progressive_config,
    ])


@hydra.main(version_base=None, config_path="../../../configs", config_name="vae_progressive")
def main(cfg: DictConfig) -> None:
    """Progressive VAE training entry point.

    Args:
        cfg: Hydra configuration object.
    """
    # Validate configuration
    validate_config(cfg)

    # Log configuration
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    log.info("")
    log.info("=" * 60)
    log.info("PROGRESSIVE VAE TRAINING")
    log.info("=" * 60)
    log.info(f"Resolutions: {cfg.progressive.resolutions}")
    log.info(f"Modalities: {cfg.modalities.image_keys}")
    log.info(f"Plateau detection: window={cfg.progressive.plateau.window_size}, "
             f"min_improvement={cfg.progressive.plateau.min_improvement}%, "
             f"patience={cfg.progressive.plateau.patience}")
    log.info(f"Final phase epochs: {cfg.progressive.final_phase.epochs}")
    log.info("=" * 60)
    log.info("")

    # Create and run trainer
    trainer = ProgressiveVAETrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
