"""
Unified metrics system for compression trainers.

Eliminates duplicated code across VAE, VQ-VAE, DC-AE (2D and 3D) trainers
by providing:

1. TrainerMetricsConfig: Defines what metrics each trainer mode uses
2. LossAccumulator: Unified epoch loss tracking
3. MetricsLogger: Unified TensorBoard + console logging

Usage:
    # In trainer __init__
    self._init_unified_metrics('vae')  # Uses helper in BaseCompressionTrainer

    # In train_epoch
    self._loss_accumulator.reset()
    for batch in loader:
        result = self.train_step(batch)
        self._loss_accumulator.update(result.to_dict_with_key('kl'))
    avg = self._loss_accumulator.compute()
    self._metrics_logger.log_training(epoch, avg)

    # Epoch summary
    self._metrics_logger.log_epoch_summary(epoch, epochs, avg, val_metrics, elapsed)
"""
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Set, Union

import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


# =============================================================================
# Trainer Mode Enumeration
# =============================================================================

class TrainerMode(Enum):
    """Defines trainer type for metrics configuration."""
    VAE = auto()       # KL regularization
    VQVAE = auto()     # VQ loss
    DCAE = auto()       # No regularization (deterministic)
    SEG = auto()       # Segmentation mode (BCE + Dice + Boundary)


# =============================================================================
# Canonical Keys
# =============================================================================

class LossKey:
    """Canonical loss key names for consistency.

    Use these constants instead of string literals to prevent typos
    and ensure consistency across all trainers.
    """
    # Generator/reconstruction
    GEN = 'gen'        # Total generator loss
    RECON = 'recon'    # Reconstruction loss (L1 or seg composite)
    PERC = 'perc'      # Perceptual loss

    # Regularization (mutually exclusive per trainer)
    KL = 'kl'          # VAE KL divergence
    VQ = 'vq'          # VQ-VAE commitment loss

    # GAN
    DISC = 'disc'      # Discriminator loss
    ADV = 'adv'        # Adversarial loss (generator side)

    # Segmentation mode
    BCE = 'bce'        # Binary cross entropy
    DICE = 'dice'      # Dice loss (1 - dice_score)
    BOUNDARY = 'boundary'  # Boundary loss


class MetricKey:
    """Canonical validation metric key names.

    Quality metrics (higher = better) go under Validation/ in TensorBoard.
    Loss metrics (lower = better) go under Loss/.
    """
    # Image quality
    PSNR = 'psnr'
    LPIPS = 'lpips'
    MSSSIM = 'msssim'
    MSSSIM_3D = 'msssim_3d'
    L1 = 'l1'

    # Segmentation quality
    DICE_SCORE = 'dice_score'  # Dice score (not loss) - different from LossKey.DICE
    IOU = 'iou'


# =============================================================================
# Trainer Metrics Configuration
# =============================================================================

@dataclass
class TrainerMetricsConfig:
    """Configuration for trainer-specific metrics.

    Defines which losses and metrics are tracked based on trainer mode.
    Use factory methods for common configurations.

    Attributes:
        mode: Trainer mode (VAE, VQVAE, DCAE, SEG).
        loss_keys: Set of loss keys to track during training.
        regularization_key: Key for regularization loss (KL, VQ, or None).
        validation_metrics: Set of validation metric keys to log.
        has_gan: Whether GAN training is enabled.
        spatial_dims: 2 or 3 for 2D or 3D models.

    Example:
        # VAE trainer
        config = TrainerMetricsConfig.for_vae()

        # VQ-VAE 3D trainer with seg mode
        config = TrainerMetricsConfig.for_vqvae(spatial_dims=3, seg_mode=True)

        # DC-AE without GAN
        config = TrainerMetricsConfig.for_dcae(has_gan=False)
    """
    mode: TrainerMode
    loss_keys: Set[str] = field(default_factory=set)
    regularization_key: Optional[str] = None
    validation_metrics: Set[str] = field(default_factory=set)
    has_gan: bool = True
    spatial_dims: int = 2

    @classmethod
    def for_vae(
        cls,
        has_gan: bool = True,
        spatial_dims: int = 2,
        log_msssim: bool = True,
        log_lpips: bool = True,
    ) -> 'TrainerMetricsConfig':
        """Create config for VAE trainer.

        Args:
            has_gan: Whether GAN training is enabled.
            spatial_dims: 2 or 3 for 2D or 3D models.
            log_msssim: Whether to log MS-SSIM metrics.
            log_lpips: Whether to log LPIPS metrics.

        Returns:
            TrainerMetricsConfig for VAE.
        """
        loss_keys = {LossKey.GEN, LossKey.RECON, LossKey.PERC, LossKey.KL}
        if has_gan:
            loss_keys.update({LossKey.DISC, LossKey.ADV})

        val_metrics = {MetricKey.PSNR, MetricKey.L1}
        if log_msssim:
            val_metrics.add(MetricKey.MSSSIM)
            if spatial_dims == 3:
                val_metrics.add(MetricKey.MSSSIM_3D)
        if log_lpips:
            val_metrics.add(MetricKey.LPIPS)

        return cls(
            mode=TrainerMode.VAE,
            loss_keys=loss_keys,
            regularization_key=LossKey.KL,
            validation_metrics=val_metrics,
            has_gan=has_gan,
            spatial_dims=spatial_dims,
        )

    @classmethod
    def for_vqvae(
        cls,
        has_gan: bool = True,
        spatial_dims: int = 2,
        seg_mode: bool = False,
        log_msssim: bool = True,
        log_lpips: bool = True,
    ) -> 'TrainerMetricsConfig':
        """Create config for VQ-VAE trainer.

        Args:
            has_gan: Whether GAN training is enabled.
            spatial_dims: 2 or 3 for 2D or 3D models.
            seg_mode: Whether segmentation mode is enabled.
            log_msssim: Whether to log MS-SSIM metrics.
            log_lpips: Whether to log LPIPS metrics.

        Returns:
            TrainerMetricsConfig for VQ-VAE.
        """
        if seg_mode:
            return cls._for_seg_mode(
                base_mode=TrainerMode.VQVAE,
                regularization_key=LossKey.VQ,
                spatial_dims=spatial_dims,
            )

        loss_keys = {LossKey.GEN, LossKey.RECON, LossKey.PERC, LossKey.VQ}
        if has_gan:
            loss_keys.update({LossKey.DISC, LossKey.ADV})

        val_metrics = {MetricKey.PSNR, MetricKey.L1}
        if log_msssim:
            val_metrics.add(MetricKey.MSSSIM)
            if spatial_dims == 3:
                val_metrics.add(MetricKey.MSSSIM_3D)
        if log_lpips:
            val_metrics.add(MetricKey.LPIPS)

        return cls(
            mode=TrainerMode.VQVAE,
            loss_keys=loss_keys,
            regularization_key=LossKey.VQ,
            validation_metrics=val_metrics,
            has_gan=has_gan,
            spatial_dims=spatial_dims,
        )

    @classmethod
    def for_dcae(
        cls,
        has_gan: bool = True,
        spatial_dims: int = 2,
        seg_mode: bool = False,
        log_msssim: bool = True,
        log_lpips: bool = True,
    ) -> 'TrainerMetricsConfig':
        """Create config for DC-AE trainer.

        DC-AE is deterministic (no regularization loss).

        Args:
            has_gan: Whether GAN training is enabled.
            spatial_dims: 2 or 3 for 2D or 3D models.
            seg_mode: Whether segmentation mode is enabled.
            log_msssim: Whether to log MS-SSIM metrics.
            log_lpips: Whether to log LPIPS metrics.

        Returns:
            TrainerMetricsConfig for DC-AE.
        """
        if seg_mode:
            return cls._for_seg_mode(
                base_mode=TrainerMode.DCAE,
                regularization_key=None,
                spatial_dims=spatial_dims,
            )

        loss_keys = {LossKey.GEN, LossKey.RECON, LossKey.PERC}
        if has_gan:
            loss_keys.update({LossKey.DISC, LossKey.ADV})

        val_metrics = {MetricKey.PSNR, MetricKey.L1}
        if log_msssim:
            val_metrics.add(MetricKey.MSSSIM)
            if spatial_dims == 3:
                val_metrics.add(MetricKey.MSSSIM_3D)
        if log_lpips:
            val_metrics.add(MetricKey.LPIPS)

        return cls(
            mode=TrainerMode.DCAE,
            loss_keys=loss_keys,
            regularization_key=None,
            validation_metrics=val_metrics,
            has_gan=has_gan,
            spatial_dims=spatial_dims,
        )

    @classmethod
    def _for_seg_mode(
        cls,
        base_mode: TrainerMode,
        regularization_key: Optional[str],
        spatial_dims: int,
    ) -> 'TrainerMetricsConfig':
        """Create seg mode config (shared by VQVAE and DCAE).

        Seg mode uses BCE + Dice + Boundary loss instead of L1 + Perceptual,
        and Dice/IoU metrics instead of PSNR/LPIPS/MS-SSIM.

        Args:
            base_mode: Original trainer mode (VQVAE or DCAE).
            regularization_key: Key for regularization (VQ or None).
            spatial_dims: 2 or 3 for 2D or 3D models.

        Returns:
            TrainerMetricsConfig for seg mode.
        """
        # Seg mode: BCE + Dice + Boundary, no GAN
        loss_keys = {
            LossKey.GEN, LossKey.RECON,
            LossKey.BCE, LossKey.DICE, LossKey.BOUNDARY,
        }
        if regularization_key:
            loss_keys.add(regularization_key)

        return cls(
            mode=TrainerMode.SEG,
            loss_keys=loss_keys,
            regularization_key=regularization_key,
            validation_metrics={MetricKey.DICE_SCORE, MetricKey.IOU},
            has_gan=False,
            spatial_dims=spatial_dims,
        )

    @property
    def is_seg_mode(self) -> bool:
        """Check if this is segmentation mode."""
        return self.mode == TrainerMode.SEG

    @property
    def uses_image_quality_metrics(self) -> bool:
        """Check if image quality metrics (PSNR/LPIPS/MS-SSIM) are used."""
        return MetricKey.PSNR in self.validation_metrics


# =============================================================================
# Loss Accumulator
# =============================================================================

class LossAccumulator:
    """Unified epoch loss accumulation for compression trainers.

    Eliminates duplicated accumulation code across trainers.
    Automatically tracks only the losses defined in the config.

    Usage:
        accumulator = LossAccumulator(metrics_config)
        accumulator.reset()

        for batch in loader:
            result = train_step(batch)
            losses = result.to_dict_with_key('kl')
            accumulator.update(losses)

        avg_losses = accumulator.compute()

    Attributes:
        config: TrainerMetricsConfig defining which losses to track.
    """

    def __init__(self, config: TrainerMetricsConfig) -> None:
        """Initialize accumulator with metrics config.

        Args:
            config: TrainerMetricsConfig defining losses to track.
        """
        self.config = config
        self._accumulators: Dict[str, float] = {}
        self._count: int = 0
        self.reset()

    def reset(self) -> None:
        """Reset accumulators for new epoch."""
        self._accumulators = {key: 0.0 for key in self.config.loss_keys}
        self._count = 0

    def update(self, losses: Dict[str, Union[float, torch.Tensor]]) -> None:
        """Accumulate losses from a single step.

        Args:
            losses: Dictionary of loss values. Keys not in config are ignored.
                   Values can be float or 0-dim tensors.
        """
        for key in self._accumulators:
            if key in losses:
                val = losses[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                self._accumulators[key] += val
        self._count += 1

    def compute(self) -> Dict[str, float]:
        """Compute average losses over accumulated steps.

        Returns:
            Dictionary of averaged losses.
        """
        if self._count == 0:
            return {key: 0.0 for key in self._accumulators}
        return {key: val / self._count for key, val in self._accumulators.items()}


# =============================================================================
# Metrics Logger
# =============================================================================

class MetricsLogger:
    """Unified TensorBoard and console logging for compression trainers.

    Eliminates duplicated logging code by providing consistent
    metric naming and logging patterns.

    Usage:
        logger = MetricsLogger(writer, config)

        # After train epoch
        logger.log_training(epoch, avg_losses)

        # After validation
        logger.log_validation(epoch, val_metrics)

        # Epoch summary
        logger.log_epoch_summary(epoch, total_epochs, avg_losses, val_metrics, elapsed)
    """

    # TensorBoard tag mappings for training losses
    _TRAIN_LOSS_TAGS = {
        LossKey.GEN: 'Loss/Generator_train',
        LossKey.RECON: 'Loss/L1_train',
        LossKey.PERC: 'Loss/Perceptual_train',
        LossKey.KL: 'Loss/KL_train',
        LossKey.VQ: 'Loss/VQ_train',
        LossKey.DISC: 'Loss/Discriminator',
        LossKey.ADV: 'Loss/Adversarial',
        LossKey.BCE: 'Loss/BCE_train',
        LossKey.DICE: 'Loss/Dice_train',
        LossKey.BOUNDARY: 'Loss/Boundary_train',
    }

    # TensorBoard tag mappings for validation losses
    _VAL_LOSS_TAGS = {
        LossKey.GEN: 'Loss/Generator_val',
        LossKey.RECON: 'Loss/L1_val',
        LossKey.PERC: 'Loss/Perceptual_val',
        LossKey.KL: 'Loss/KL_val',
        LossKey.VQ: 'Loss/VQ_val',
        LossKey.BCE: 'Loss/BCE_val',
        LossKey.DICE: 'Loss/Dice_val',
        LossKey.BOUNDARY: 'Loss/Boundary_val',
    }

    # TensorBoard tag mappings for validation metrics
    _VAL_METRIC_TAGS = {
        MetricKey.PSNR: 'Validation/PSNR',
        MetricKey.LPIPS: 'Validation/LPIPS',
        MetricKey.MSSSIM: 'Validation/MS-SSIM',
        MetricKey.MSSSIM_3D: 'Validation/MS-SSIM-3D',
        MetricKey.L1: 'Loss/L1_val',  # L1 goes under Loss/
        MetricKey.DICE_SCORE: 'Validation/Dice',
        MetricKey.IOU: 'Validation/IoU',
    }

    def __init__(
        self,
        writer: Optional[SummaryWriter],
        config: TrainerMetricsConfig,
        use_multi_gpu: bool = False,
    ) -> None:
        """Initialize metrics logger.

        Args:
            writer: TensorBoard SummaryWriter (may be None).
            config: TrainerMetricsConfig defining what to log.
            use_multi_gpu: If True, skip logging (only rank 0 logs).
        """
        self.writer = writer
        self.config = config
        self.use_multi_gpu = use_multi_gpu

    def log_training(self, epoch: int, avg_losses: Dict[str, float]) -> None:
        """Log training losses to TensorBoard.

        Args:
            epoch: Current epoch number.
            avg_losses: Dictionary of averaged losses.
        """
        if self.writer is None or self.use_multi_gpu:
            return

        for key, value in avg_losses.items():
            tag = self._TRAIN_LOSS_TAGS.get(key)
            if tag:
                self.writer.add_scalar(tag, value, epoch)

    def log_validation(
        self,
        epoch: int,
        metrics: Dict[str, float],
        prefix: str = '',
    ) -> None:
        """Log validation metrics to TensorBoard.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of validation metrics.
            prefix: Optional prefix for per-modality logging (e.g., 't1_pre').
        """
        if self.writer is None:
            return

        for key, value in metrics.items():
            # Check if it's a loss or a quality metric
            if key in self._VAL_LOSS_TAGS:
                tag = self._VAL_LOSS_TAGS[key]
            elif key in self._VAL_METRIC_TAGS:
                tag = self._VAL_METRIC_TAGS[key]
            else:
                # Unknown key - skip (don't log arbitrary keys)
                continue

            # Apply prefix for per-modality logging
            if prefix:
                # e.g., Validation/PSNR -> Validation_t1_pre/PSNR
                parts = tag.split('/')
                tag = f'{parts[0]}_{prefix}/{parts[1]}'

            self.writer.add_scalar(tag, value, epoch)

    def log_regularization(
        self,
        epoch: int,
        value: float,
        weight: Optional[float] = None,
        suffix: str = 'val',
    ) -> None:
        """Log regularization metric (KL or VQ) to TensorBoard.

        Handles unnormalization for KL (divides by weight if provided).

        Args:
            epoch: Current epoch number.
            value: Regularization loss value (weighted).
            weight: Optional weight to unnormalize by (e.g., kl_weight).
            suffix: 'train' or 'val'.
        """
        if self.writer is None or not self.config.regularization_key:
            return

        # Unnormalize if weight provided
        display_value = value / weight if weight and weight > 0 else value

        key = self.config.regularization_key
        tag_map = self._TRAIN_LOSS_TAGS if suffix == 'train' else self._VAL_LOSS_TAGS
        tag = tag_map.get(key, f'Loss/{key}_{suffix}')

        self.writer.add_scalar(tag, display_value, epoch)

    def log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
        elapsed_time: float,
    ) -> None:
        """Log epoch completion summary to console.

        Unified logging that handles all trainer modes.

        Args:
            epoch: Current epoch number (0-indexed).
            total_epochs: Total number of epochs.
            avg_losses: Dictionary of averaged training losses.
            val_metrics: Dictionary of validation metrics (may be None).
            elapsed_time: Time taken for epoch in seconds.
        """
        timestamp = time.strftime("%H:%M:%S")
        epoch_pct = ((epoch + 1) / total_epochs) * 100

        # Build validation suffix for losses
        val_gen = f"(v:{val_metrics.get('gen', 0):.4f})" if val_metrics else ""

        if self.config.is_seg_mode:
            self._log_seg_epoch_summary(
                timestamp, epoch, total_epochs, epoch_pct,
                avg_losses, val_metrics, val_gen, elapsed_time
            )
        else:
            self._log_image_epoch_summary(
                timestamp, epoch, total_epochs, epoch_pct,
                avg_losses, val_metrics, val_gen, elapsed_time
            )

    def _log_image_epoch_summary(
        self,
        timestamp: str,
        epoch: int,
        total_epochs: int,
        epoch_pct: float,
        avg_losses: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
        val_gen: str,
        elapsed_time: float,
    ) -> None:
        """Log epoch summary for image reconstruction mode."""
        val_l1 = f"(v:{val_metrics.get('l1', 0):.4f})" if val_metrics else ""

        # Build regularization string
        reg_str = ""
        if self.config.regularization_key:
            key = self.config.regularization_key
            if key in avg_losses:
                reg_str = f"{key.upper()}: {avg_losses[key]:.4f} | "

        # Build quality metrics string
        metrics_parts = []
        if val_metrics:
            if val_metrics.get(MetricKey.MSSSIM):
                metrics_parts.append(f"MS-SSIM: {val_metrics[MetricKey.MSSSIM]:.3f}")
            if val_metrics.get(MetricKey.MSSSIM_3D):
                metrics_parts.append(f"MS-SSIM-3D: {val_metrics[MetricKey.MSSSIM_3D]:.3f}")
            if val_metrics.get(MetricKey.PSNR):
                metrics_parts.append(f"PSNR: {val_metrics[MetricKey.PSNR]:.2f}")
        metric_str = " | ".join(metrics_parts) if metrics_parts else ""

        logger.info(
            f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
            f"G: {avg_losses.get(LossKey.GEN, 0):.4f}{val_gen} | "
            f"L1: {avg_losses.get(LossKey.RECON, 0):.4f}{val_l1} | "
            f"{reg_str}"
            f"D: {avg_losses.get(LossKey.DISC, 0):.4f} | "
            f"{metric_str} | "
            f"Time: {elapsed_time:.1f}s"
        )

    def _log_seg_epoch_summary(
        self,
        timestamp: str,
        epoch: int,
        total_epochs: int,
        epoch_pct: float,
        avg_losses: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
        val_gen: str,
        elapsed_time: float,
    ) -> None:
        """Log epoch summary for segmentation mode."""
        # Seg mode: show BCE, Dice, Boundary instead of L1, Perceptual
        bce_str = f"BCE: {avg_losses.get(LossKey.BCE, 0):.4f}"
        dice_str = f"Dice: {avg_losses.get(LossKey.DICE, 0):.4f}"
        boundary_str = f"Bound: {avg_losses.get(LossKey.BOUNDARY, 0):.4f}"

        # Build regularization string (VQ for VQVAE seg mode)
        reg_str = ""
        if self.config.regularization_key:
            key = self.config.regularization_key
            if key in avg_losses:
                reg_str = f"{key.upper()}: {avg_losses[key]:.4f} | "

        # Build validation quality metrics
        metrics_parts = []
        if val_metrics:
            if val_metrics.get(MetricKey.DICE_SCORE):
                metrics_parts.append(f"Dice: {val_metrics[MetricKey.DICE_SCORE]:.3f}")
            if val_metrics.get(MetricKey.IOU):
                metrics_parts.append(f"IoU: {val_metrics[MetricKey.IOU]:.3f}")
        metric_str = " | ".join(metrics_parts) if metrics_parts else ""

        logger.info(
            f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
            f"G: {avg_losses.get(LossKey.GEN, 0):.4f}{val_gen} | "
            f"{bce_str} | {dice_str} | {boundary_str} | "
            f"{reg_str}"
            f"{metric_str} | "
            f"Time: {elapsed_time:.1f}s"
        )


# =============================================================================
# Helper Functions
# =============================================================================

def create_metrics_config(
    trainer_type: str,
    has_gan: bool = True,
    spatial_dims: int = 2,
    seg_mode: bool = False,
    log_msssim: bool = True,
    log_lpips: bool = True,
) -> TrainerMetricsConfig:
    """Factory function to create metrics config from string trainer type.

    Args:
        trainer_type: One of 'vae', 'vqvae', 'dcae'.
        has_gan: Whether GAN training is enabled.
        spatial_dims: 2 or 3 for 2D or 3D models.
        seg_mode: Whether segmentation mode is enabled.
        log_msssim: Whether to log MS-SSIM metrics.
        log_lpips: Whether to log LPIPS metrics.

    Returns:
        TrainerMetricsConfig for the specified trainer type.

    Raises:
        ValueError: If trainer_type is not recognized.
    """
    trainer_type = trainer_type.lower()

    if trainer_type == 'vae':
        return TrainerMetricsConfig.for_vae(
            has_gan=has_gan,
            spatial_dims=spatial_dims,
            log_msssim=log_msssim,
            log_lpips=log_lpips,
        )
    elif trainer_type == 'vqvae':
        return TrainerMetricsConfig.for_vqvae(
            has_gan=has_gan,
            spatial_dims=spatial_dims,
            seg_mode=seg_mode,
            log_msssim=log_msssim,
            log_lpips=log_lpips,
        )
    elif trainer_type == 'dcae':
        return TrainerMetricsConfig.for_dcae(
            has_gan=has_gan,
            spatial_dims=spatial_dims,
            seg_mode=seg_mode,
            log_msssim=log_msssim,
            log_lpips=log_lpips,
        )
    else:
        raise ValueError(
            f"Unknown trainer type: {trainer_type}. "
            "Expected 'vae', 'vqvae', or 'dcae'."
        )
