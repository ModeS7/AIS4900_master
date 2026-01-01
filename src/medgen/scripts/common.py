"""
Common utilities for training scripts.

This module provides shared functionality used across multiple training scripts
to reduce code duplication and ensure consistent behavior.
"""
import logging
from typing import Any, Dict, Optional, Tuple

from omegaconf import DictConfig, open_dict

from medgen.core import ModeType

log = logging.getLogger(__name__)


def override_vae_channels(cfg: DictConfig, mode: str) -> int:
    """Override mode channels for VAE/VQ-VAE/DC-AE training.

    VAE variants don't concatenate seg with images, so dual mode = 2 channels (not 3).
    This function modifies cfg in-place and returns the computed channel count.

    Args:
        cfg: Hydra configuration object (modified in-place).
        mode: Mode name (e.g., 'dual', 'bravo', 'multi_modality').

    Returns:
        Number of input/output channels for the autoencoder.

    Example:
        >>> in_channels = override_vae_channels(cfg, cfg.mode.name)
        >>> # cfg.mode.in_channels and cfg.mode.out_channels are now set
    """
    if mode == ModeType.DUAL:
        in_channels = 2  # t1_pre + t1_gd
    else:
        in_channels = 1  # bravo, seg, t1_pre, t1_gd, multi_modality individually

    with open_dict(cfg):
        cfg.mode.in_channels = in_channels
        cfg.mode.out_channels = in_channels

    return in_channels


def run_test_evaluation(
    trainer: Any,
    test_result: Optional[Tuple],
    log: logging.Logger,
    eval_method: str = "evaluate_test_set"
) -> None:
    """Run test evaluation on best and latest checkpoints.

    Args:
        trainer: Trainer instance with evaluate method.
        test_result: Tuple of (test_loader, test_dataset) or None if no test data.
        log: Logger instance for output messages.
        eval_method: Name of evaluation method on trainer ('evaluate_test_set' or 'evaluate_test').

    Example:
        >>> test_result = create_vae_test_dataloader(cfg=cfg, modality=mode)
        >>> run_test_evaluation(trainer, test_result, log)
    """
    if test_result is not None:
        test_loader, test_dataset = test_result
        log.info(f"Test dataset: {len(test_dataset)} samples")

        # Get the evaluation method (different trainers use different names)
        eval_fn = getattr(trainer, eval_method, None)
        if eval_fn is None:
            # Try alternative method name
            eval_fn = getattr(trainer, "evaluate_test", None)

        if eval_fn is not None:
            eval_fn(test_loader, checkpoint_name="best")
            eval_fn(test_loader, checkpoint_name="latest")
        else:
            log.warning(f"Trainer has no {eval_method} or evaluate_test method")
    else:
        log.info("No test_new/ directory found - skipping test evaluation")


def create_per_modality_val_loaders(
    cfg: DictConfig,
    image_keys: list,
    create_loader_fn,
    image_size: int,
    batch_size: int,
    log: logging.Logger,
) -> Dict[str, Any]:
    """Create per-modality validation loaders for multi-modality training.

    Args:
        cfg: Hydra configuration object.
        image_keys: List of modality names (e.g., ['bravo', 'flair', 't1_pre', 't1_gd']).
        create_loader_fn: Function to create a single-modality validation loader.
        image_size: Target image size.
        batch_size: Batch size for validation.
        log: Logger instance for output messages.

    Returns:
        Dictionary mapping modality names to validation loaders.

    Example:
        >>> loaders = create_per_modality_val_loaders(
        ...     cfg, image_keys, create_single_modality_validation_loader,
        ...     cfg.model.image_size, cfg.training.batch_size, log
        ... )
    """
    per_modality_val_loaders = {}

    for modality in image_keys:
        loader = create_loader_fn(
            cfg=cfg,
            modality=modality,
            image_size=image_size,
            batch_size=batch_size
        )
        if loader is not None:
            per_modality_val_loaders[modality] = loader
            log.info(f"  Per-modality validation for {modality}: {len(loader.dataset)} samples")

    return per_modality_val_loaders


def log_training_header(
    trainer_type: str,
    mode: str,
    in_channels: int,
    cfg: DictConfig,
    log: logging.Logger,
    extra_info: Optional[Dict[str, str]] = None
) -> None:
    """Log a consistent training header.

    Args:
        trainer_type: Type of trainer (e.g., 'VAE', 'VQ-VAE', 'DC-AE').
        mode: Mode name.
        in_channels: Number of input channels.
        cfg: Configuration object.
        log: Logger instance.
        extra_info: Additional key-value pairs to log.

    Example:
        >>> log_training_header('VAE', mode, in_channels, cfg, log,
        ...     extra_info={'Latent channels': str(cfg.vae.latent_channels)})
    """
    log.info("")
    log.info("=" * 60)
    log.info(f"Training {trainer_type} for {mode} mode")
    log.info(f"Channels: {in_channels} | Image size: {cfg.model.image_size}")
    log.info(f"Batch size: {cfg.training.batch_size}")
    log.info(f"Epochs: {cfg.training.epochs}")

    if extra_info:
        for key, value in extra_info.items():
            log.info(f"{key}: {value}")

    log.info("=" * 60)
    log.info("")
