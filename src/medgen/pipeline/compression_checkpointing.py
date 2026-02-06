"""Extracted checkpointing and evaluation helpers for BaseCompressionTrainer.

Module-level functions that implement checkpoint save/load, epoch summary
logging, pretrained weight loading, and test evaluation.
Each function takes a `trainer` (BaseCompressionTrainer instance) as its first
argument and accesses trainer attributes like `trainer.device`, `trainer.cfg`, etc.
"""
from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from .compression_trainer import BaseCompressionTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint manager setup
# ---------------------------------------------------------------------------

def setup_checkpoint_manager(trainer: BaseCompressionTrainer) -> None:
    """Setup checkpoint manager with GAN components.

    Overrides BaseTrainer to add discriminator, optimizer_d, scheduler_d.

    Args:
        trainer: BaseCompressionTrainer instance.
    """
    if not trainer.is_main_process:
        return

    from .checkpoint_manager import CheckpointManager

    trainer.checkpoint_manager = CheckpointManager(
        save_dir=trainer.save_dir,
        model=trainer.model_raw,
        optimizer=trainer.optimizer,
        scheduler=trainer.lr_scheduler if not trainer.use_constant_lr else None,
        ema=trainer.ema if trainer.use_ema else None,
        config=trainer._get_model_config(),
        # GAN components
        discriminator=trainer.discriminator_raw if not trainer.disable_gan else None,
        optimizer_d=trainer.optimizer_d if not trainer.disable_gan else None,
        scheduler_d=trainer.lr_scheduler_d if not trainer.disable_gan and not trainer.use_constant_lr else None,
        metric_name=trainer._get_best_metric_name(),
        keep_last_n=trainer._training_config.keep_last_n_checkpoints,
        device=trainer.device,
    )


# ---------------------------------------------------------------------------
# Checkpoint extra state
# ---------------------------------------------------------------------------

def get_checkpoint_extra_state(trainer: BaseCompressionTrainer) -> dict[str, Any] | None:
    """Return extra state for compression trainer checkpoints.

    Includes discriminator config and training flags.

    Args:
        trainer: BaseCompressionTrainer instance.

    Returns:
        Dictionary of extra checkpoint state.
    """
    extra_state = {
        'disable_gan': trainer.disable_gan,
        'use_constant_lr': trainer.use_constant_lr,
    }

    # Add discriminator config if GAN is enabled
    if not trainer.disable_gan and trainer.discriminator_raw is not None:
        extra_state['disc_config'] = {
            'in_channels': trainer.cfg.mode.in_channels,
            'channels': trainer.disc_num_channels,
            'num_layers_d': trainer.disc_num_layers,
        }

    return extra_state


# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    trainer: BaseCompressionTrainer,
    epoch: int,
    name: str,
) -> None:
    """Save checkpoint with standardized format.

    Args:
        trainer: BaseCompressionTrainer instance.
        epoch: Current epoch number.
        name: Checkpoint name ("latest" or "best").
    """
    from .utils import save_full_checkpoint

    if not trainer.is_main_process:
        return

    model_config = trainer._get_model_config()

    # Build extra state
    extra_state = {
        'disable_gan': trainer.disable_gan,
        'use_constant_lr': trainer.use_constant_lr,
    }

    # Add discriminator state if GAN is enabled
    if not trainer.disable_gan and trainer.discriminator_raw is not None:
        extra_state['discriminator_state_dict'] = trainer.discriminator_raw.state_dict()
        extra_state['disc_config'] = {
            'in_channels': trainer.cfg.mode.in_channels,
            'channels': trainer.disc_num_channels,
            'num_layers_d': trainer.disc_num_layers,
        }
        if trainer.optimizer_d is not None:
            extra_state['optimizer_d_state_dict'] = trainer.optimizer_d.state_dict()
        if not trainer.use_constant_lr and trainer.lr_scheduler_d is not None:
            extra_state['scheduler_d_state_dict'] = trainer.lr_scheduler_d.state_dict()

    # Save using standardized format: checkpoint_{name}.pt
    save_full_checkpoint(
        model=trainer.model_raw,
        optimizer=trainer.optimizer,
        epoch=epoch,
        save_dir=trainer.save_dir,
        filename=f"checkpoint_{name}",
        model_config=model_config,
        scheduler=trainer.lr_scheduler if not trainer.use_constant_lr else None,
        ema=trainer.ema,
        extra_state=extra_state,
    )


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(
    trainer: BaseCompressionTrainer,
    path: str,
    load_optimizer: bool = True,
) -> int:
    """Load checkpoint to resume training.

    Uses CheckpointManager if available, otherwise falls back to legacy loading.

    Args:
        trainer: BaseCompressionTrainer instance.
        path: Path to checkpoint file.
        load_optimizer: Whether to load optimizer state.

    Returns:
        Epoch number from checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Use CheckpointManager if available
    if trainer.checkpoint_manager is not None:
        result = trainer.checkpoint_manager.load(
            path,
            strict=True,
            load_optimizer=load_optimizer,
        )
        epoch = result['epoch']
        if trainer.is_main_process:
            logger.info(f"Resuming from epoch {epoch + 1}")
        return epoch

    # Legacy loading (backward compatibility)

    checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)

    # Load model weights
    trainer.model_raw.load_state_dict(checkpoint['model_state_dict'])
    if trainer.is_main_process:
        logger.info(f"Loaded model weights from {path}")

    # Load discriminator weights
    if not trainer.disable_gan and trainer.discriminator_raw is not None:
        if 'discriminator_state_dict' in checkpoint:
            trainer.discriminator_raw.load_state_dict(checkpoint['discriminator_state_dict'])
            if trainer.is_main_process:
                logger.info("Loaded discriminator weights")
        else:
            if trainer.is_main_process:
                logger.warning("Checkpoint has no discriminator weights")

    # Load optimizer states
    if load_optimizer:
        if 'optimizer_state_dict' in checkpoint and trainer.optimizer is not None:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Also try old key name for backwards compat
        elif 'optimizer_g_state_dict' in checkpoint and trainer.optimizer is not None:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_g_state_dict'])

        if not trainer.disable_gan and trainer.optimizer_d is not None:
            if 'optimizer_d_state_dict' in checkpoint:
                trainer.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

        # Load scheduler states
        if not trainer.use_constant_lr:
            if 'scheduler_state_dict' in checkpoint and trainer.lr_scheduler is not None:
                trainer.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            elif 'scheduler_g_state_dict' in checkpoint and trainer.lr_scheduler is not None:
                trainer.lr_scheduler.load_state_dict(checkpoint['scheduler_g_state_dict'])

            if not trainer.disable_gan and trainer.lr_scheduler_d is not None:
                if 'scheduler_d_state_dict' in checkpoint:
                    trainer.lr_scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])

    # Load EMA state
    if trainer.use_ema and trainer.ema is not None:
        if 'ema_state_dict' in checkpoint:
            trainer.ema.load_state_dict(checkpoint['ema_state_dict'])
            if trainer.is_main_process:
                logger.info("Loaded EMA state")

    epoch = checkpoint.get('epoch', 0)
    if trainer.is_main_process:
        logger.info(f"Resuming from epoch {epoch + 1}")

    return epoch


# ---------------------------------------------------------------------------
# Epoch summary logging
# ---------------------------------------------------------------------------

def log_epoch_summary(
    trainer: BaseCompressionTrainer,
    epoch: int,
    total_epochs: int,
    avg_losses: dict[str, float],
    val_metrics: dict[str, float],
    elapsed_time: float,
) -> None:
    """Log epoch completion summary.

    Args:
        trainer: BaseCompressionTrainer instance.
        epoch: Current epoch number.
        total_epochs: Total number of epochs.
        avg_losses: Dictionary of averaged training losses.
        val_metrics: Dictionary of validation metrics.
        elapsed_time: Time taken for epoch in seconds.
    """
    timestamp = time.strftime("%H:%M:%S")
    epoch_pct = ((epoch + 1) / total_epochs) * 100

    # Format validation metrics
    val_gen = f"(v:{val_metrics.get('gen', 0):.4f})" if val_metrics else ""
    val_l1 = f"(v:{val_metrics.get('l1', 0):.4f})" if val_metrics else ""
    msssim_str = f"MS-SSIM: {val_metrics.get('msssim', 0):.3f}" if val_metrics.get('msssim') else ""

    logger.info(
        f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
        f"G: {avg_losses.get('gen', 0):.4f}{val_gen} | "
        f"L1: {avg_losses.get('recon', avg_losses.get('l1', 0)):.4f}{val_l1} | "
        f"D: {avg_losses.get('disc', 0):.4f} | "
        f"{msssim_str} | "
        f"Time: {elapsed_time:.1f}s"
    )


# ---------------------------------------------------------------------------
# Pretrained weight loading
# ---------------------------------------------------------------------------

def load_pretrained_weights(
    trainer: BaseCompressionTrainer,
    raw_model: nn.Module,
    raw_disc: nn.Module | None,
    checkpoint_path: str,
    model_name: str = "model",
) -> None:
    """Load pretrained weights from checkpoint.

    This is the shared implementation for 2D trainers (VAE, VQ-VAE).
    3D trainers use load_pretrained_weights_base() which handles
    prefix stripping for CheckpointedAutoencoder wrapper.

    Args:
        trainer: BaseCompressionTrainer instance.
        raw_model: The raw model to load weights into.
        raw_disc: The raw discriminator (can be None if GAN disabled).
        checkpoint_path: Path to the checkpoint file.
        model_name: Name for logging (e.g., "VAE", "VQ-VAE").
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            raw_model.load_state_dict(checkpoint['model_state_dict'])
            if trainer.is_main_process:
                logger.info(f"Loaded {model_name} weights from {checkpoint_path}")
        if 'discriminator_state_dict' in checkpoint and raw_disc is not None:
            raw_disc.load_state_dict(checkpoint['discriminator_state_dict'])
            if trainer.is_main_process:
                logger.info(f"Loaded discriminator weights from {checkpoint_path}")
    except FileNotFoundError:
        if trainer.is_main_process:
            logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")


def load_pretrained_weights_base(
    trainer: BaseCompressionTrainer,
    base_model: nn.Module,
    checkpoint_path: str,
    model_name: str = "model",
) -> None:
    """Load pretrained weights into base model (before checkpointing wrapper).

    Handles 'model.' prefix stripping for checkpointed model wrappers.
    Used by 3D trainers that wrap models with gradient checkpointing.

    Args:
        trainer: BaseCompressionTrainer instance.
        base_model: The base model (unwrapped) to load weights into.
        checkpoint_path: Path to the checkpoint file.
        model_name: Name for logging (e.g., "3D VAE", "3D VQ-VAE").
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Remove 'model.' prefix if present (from Checkpointed* wrappers)
            keys_with_prefix = [k for k in state_dict.keys() if k.startswith('model.')]
            if keys_with_prefix:
                if len(keys_with_prefix) != len(state_dict) and trainer.is_main_process:
                    logger.warning(
                        f"Mixed prefix state: {len(keys_with_prefix)}/{len(state_dict)} keys "
                        "have 'model.' prefix. Stripping prefix from matching keys."
                    )
                state_dict = {
                    k.replace('model.', '', 1) if k.startswith('model.') else k: v
                    for k, v in state_dict.items()
                }
            base_model.load_state_dict(state_dict)
            if trainer.is_main_process:
                logger.info(f"Loaded {model_name} weights from {checkpoint_path}")
    except FileNotFoundError:
        if trainer.is_main_process:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")


# ---------------------------------------------------------------------------
# Test evaluator creation
# ---------------------------------------------------------------------------

def create_test_evaluator(trainer: BaseCompressionTrainer):
    """Create test evaluator for this trainer (2D or 3D).

    Factory method that creates a CompressionTestEvaluator (2D) or
    Compression3DTestEvaluator (3D) with trainer-specific callbacks.

    Args:
        trainer: BaseCompressionTrainer instance.

    Returns:
        Configured test evaluator instance.
    """
    from medgen.evaluation import (
        Compression3DTestEvaluator,
        CompressionTestEvaluator,
        MetricsConfig,
    )

    # Check for seg_mode (set by subclasses)
    seg_mode = getattr(trainer, 'seg_mode', False)
    seg_loss_fn = getattr(trainer, 'seg_loss_fn', None)

    # Get modality name for single-modality suffix
    # Use empty string for seg_conditioned modes (no suffix needed)
    mode_name = trainer.cfg.mode.name
    if mode_name.startswith('seg_conditioned'):
        mode_name = ''

    # Get image keys for per-channel metrics
    n_channels = trainer.cfg.mode.in_channels
    image_keys = None
    if n_channels > 1:
        image_keys = getattr(trainer.cfg.mode, 'image_keys', None)  # Optional: only used for multi-channel

    # Regional tracker factory (use seg-specific tracker for seg_mode)
    regional_factory = None
    if trainer.log_regional_losses:
        if seg_mode and hasattr(trainer, '_create_seg_regional_tracker'):
            regional_factory = trainer._create_seg_regional_tracker
        else:
            regional_factory = trainer._create_regional_tracker

    # 3D evaluator
    if trainer.spatial_dims == 3:
        metrics_config = MetricsConfig(
            compute_l1=not seg_mode,
            compute_psnr=not seg_mode,
            compute_lpips=not seg_mode,
            compute_msssim=trainer.log_msssim and not seg_mode,  # 2D slicewise
            compute_msssim_3d=trainer.log_msssim and not seg_mode,  # Volumetric
            compute_regional=trainer.log_regional_losses,
            seg_mode=seg_mode,
        )

        # Worst batch figure callback (3D version)
        worst_batch_fig_fn = trainer._create_worst_batch_figure

        return Compression3DTestEvaluator(
            model=trainer.model_raw,
            device=trainer.device,
            save_dir=trainer.save_dir,
            forward_fn=trainer._test_forward,
            weight_dtype=trainer.weight_dtype,
            writer=trainer.writer,
            metrics_config=metrics_config,
            is_cluster=trainer.is_cluster,
            regional_tracker_factory=regional_factory,
            worst_batch_figure_fn=worst_batch_fig_fn,
            image_keys=image_keys,
            seg_loss_fn=seg_loss_fn if seg_mode else None,
            modality_name=mode_name,
        )

    # 2D evaluator
    metrics_config = MetricsConfig(
        compute_l1=not seg_mode,
        compute_psnr=not seg_mode,
        compute_lpips=not seg_mode,
        compute_msssim=trainer.log_msssim and not seg_mode,
        compute_msssim_3d=False,  # Volume 3D MS-SSIM added via callback
        compute_regional=trainer.log_regional_losses,
        seg_mode=seg_mode,
    )

    # Volume 3D MS-SSIM callback (for 2D trainers reconstructing full volumes)
    def volume_3d_msssim() -> float | None:
        if seg_mode:
            return None
        return trainer._compute_volume_3d_msssim(epoch=0, data_split='test_new')

    # Worst batch figure callback
    worst_batch_fig_fn = trainer._create_worst_batch_figure

    return CompressionTestEvaluator(
        model=trainer.model_raw,
        device=trainer.device,
        save_dir=trainer.save_dir,
        forward_fn=trainer._test_forward,
        weight_dtype=trainer.weight_dtype,
        writer=trainer.writer,
        metrics_config=metrics_config,
        is_cluster=trainer.is_cluster,
        regional_tracker_factory=regional_factory,
        volume_3d_msssim_fn=volume_3d_msssim,
        worst_batch_figure_fn=worst_batch_fig_fn,
        image_keys=image_keys,
        seg_loss_fn=seg_loss_fn if seg_mode else None,
        modality_name=mode_name,
    )


# ---------------------------------------------------------------------------
# Test set evaluation
# ---------------------------------------------------------------------------

def evaluate_test_set(
    trainer: BaseCompressionTrainer,
    loader: DataLoader,
    checkpoint_name: str | None = None,
) -> dict[str, float]:
    """Evaluate compression model on test set.

    Uses CompressionTestEvaluator for unified test evaluation.

    Args:
        trainer: BaseCompressionTrainer instance.
        loader: Test data loader.
        checkpoint_name: Checkpoint to load ("best", "latest", or None).

    Returns:
        Dict with test metrics.
    """
    if not trainer.is_main_process:
        return {}

    evaluator = trainer._create_test_evaluator()
    return evaluator.evaluate(
        loader,
        checkpoint_name=checkpoint_name,
        get_eval_model=trainer._get_model_for_eval,
    )
