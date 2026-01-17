"""
Unified metrics system for ALL trainers (compression + diffusion).

Eliminates duplicated code across VAE, VQ-VAE, DC-AE, and Diffusion trainers
by providing:

1. TrainerMetricsConfig: Defines what metrics each trainer mode uses
2. LossAccumulator: Unified epoch loss tracking
3. MetricsLogger: Unified TensorBoard + console logging

Usage:
    # In trainer __init__
    self._init_unified_metrics('vae')       # Compression trainers
    self._init_unified_metrics('diffusion') # Diffusion trainer

    # In train_epoch
    self._loss_accumulator.reset()
    for batch in loader:
        result = train_step(batch)
        self._loss_accumulator.update(result.to_dict())
    avg = self._loss_accumulator.compute()
    self._metrics_logger.log_training(epoch, avg)

    # Validation with per-modality support
    self._metrics_logger.log_validation(epoch, metrics)
    self._metrics_logger.log_validation_per_modality(epoch, 't1_pre', modality_metrics)

    # Epoch summary
    self._metrics_logger.log_epoch_summary(epoch, epochs, avg, val_metrics, elapsed)
"""
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
    DCAE = auto()      # No regularization (deterministic)
    SEG = auto()       # Segmentation mode (BCE + Dice + Boundary)
    DIFFUSION = auto() # Diffusion model (MSE noise prediction)


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

    # Diffusion-specific
    MSE = 'mse'        # Mean squared error (noise prediction)
    TOTAL = 'total'    # Alias for total loss (diffusion naming convention)


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
        mode: Trainer mode (VAE, VQVAE, DCAE, SEG, DIFFUSION).
        loss_keys: Set of loss keys to track during training.
        regularization_key: Key for regularization loss (KL, VQ, or None).
        validation_metrics: Set of validation metric keys to log.
        has_gan: Whether GAN training is enabled.
        spatial_dims: 2 or 3 for 2D or 3D models.
        modality_keys: List of modality names for per-modality logging.

    Example:
        # VAE trainer
        config = TrainerMetricsConfig.for_vae()

        # VQ-VAE 3D trainer with seg mode
        config = TrainerMetricsConfig.for_vqvae(spatial_dims=3, seg_mode=True)

        # DC-AE without GAN
        config = TrainerMetricsConfig.for_dcae(has_gan=False)

        # Diffusion with per-modality logging
        config = TrainerMetricsConfig.for_diffusion(modality_keys=['t1_pre', 'bravo'])
    """
    mode: TrainerMode
    loss_keys: Set[str] = field(default_factory=set)
    regularization_key: Optional[str] = None
    validation_metrics: Set[str] = field(default_factory=set)
    has_gan: bool = True
    spatial_dims: int = 2
    modality_keys: Optional[List[str]] = None

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

    @classmethod
    def for_diffusion(
        cls,
        spatial_dims: int = 2,
        log_msssim: bool = True,
        log_lpips: bool = False,
        log_psnr: bool = True,
        modality_keys: Optional[List[str]] = None,
    ) -> 'TrainerMetricsConfig':
        """Create config for diffusion trainer.

        Diffusion models use MSE loss for noise prediction, plus optional
        perceptual loss. Validation metrics are computed on denoised samples.

        Args:
            spatial_dims: 2 or 3 for 2D or 3D models.
            log_msssim: Whether to log MS-SSIM metrics.
            log_lpips: Whether to log LPIPS metrics (slower).
            log_psnr: Whether to log PSNR metrics.
            modality_keys: List of modality names for per-modality logging.

        Returns:
            TrainerMetricsConfig for diffusion.
        """
        # Diffusion: MSE + optional perceptual, no GAN
        loss_keys = {LossKey.TOTAL, LossKey.MSE, LossKey.PERC}

        val_metrics: Set[str] = set()
        if log_psnr:
            val_metrics.add(MetricKey.PSNR)
        if log_msssim:
            val_metrics.add(MetricKey.MSSSIM)
            if spatial_dims == 3:
                val_metrics.add(MetricKey.MSSSIM_3D)
        if log_lpips:
            val_metrics.add(MetricKey.LPIPS)

        return cls(
            mode=TrainerMode.DIFFUSION,
            loss_keys=loss_keys,
            regularization_key=None,
            validation_metrics=val_metrics,
            has_gan=False,
            spatial_dims=spatial_dims,
            modality_keys=modality_keys,
        )

    @property
    def is_seg_mode(self) -> bool:
        """Check if this is segmentation mode."""
        return self.mode == TrainerMode.SEG

    @property
    def is_diffusion_mode(self) -> bool:
        """Check if this is diffusion mode."""
        return self.mode == TrainerMode.DIFFUSION

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


class SimpleLossAccumulator:
    """Simple loss accumulator that doesn't require a predefined config.

    Dynamically tracks any loss keys it sees during training.
    Use this for trainers that don't use TrainerMetricsConfig.

    Usage:
        accumulator = SimpleLossAccumulator()
        accumulator.reset()

        for batch in loader:
            losses = {'l1': 0.5, 'perc': 0.1, 'kl': 0.01}
            accumulator.update(losses)

        avg_losses = accumulator.compute()
    """

    def __init__(self) -> None:
        """Initialize empty accumulator."""
        self._accumulators: Dict[str, float] = {}
        self._count: int = 0

    def reset(self) -> None:
        """Reset accumulators for new epoch."""
        self._accumulators.clear()
        self._count = 0

    def update(self, losses: Dict[str, Union[float, torch.Tensor]]) -> None:
        """Accumulate losses from a single step.

        Args:
            losses: Dictionary of loss values. All keys are tracked dynamically.
                   Values can be float or 0-dim tensors.
        """
        for key, val in losses.items():
            if isinstance(val, torch.Tensor):
                val = val.item()
            if key not in self._accumulators:
                self._accumulators[key] = 0.0
            self._accumulators[key] += val
        self._count += 1

    def compute(self) -> Dict[str, float]:
        """Compute average losses over accumulated steps.

        Returns:
            Dictionary of averaged losses.
        """
        if self._count == 0:
            return {}
        return {key: val / self._count for key, val in self._accumulators.items()}


# =============================================================================
# Metrics Logger
# =============================================================================

class MetricsLogger:
    """Unified TensorBoard and console logging for ALL trainers.

    Eliminates duplicated logging code by providing consistent
    metric naming and logging patterns across compression and diffusion trainers.

    Usage:
        logger = MetricsLogger(writer, config)

        # After train epoch
        logger.log_training(epoch, avg_losses)
        logger.log_learning_rate(epoch, lr)

        # After validation
        logger.log_validation(epoch, val_metrics)
        logger.log_validation_per_modality(epoch, 't1_pre', modality_metrics)

        # Timestep losses (diffusion)
        logger.log_timestep_losses(epoch, timestep_bins)

        # Test evaluation
        logger.log_test(metrics, prefix='test_best')

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
        # Diffusion-specific
        LossKey.MSE: 'Loss/MSE_train',
        LossKey.TOTAL: 'Loss/Total_train',
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
        # Diffusion-specific
        LossKey.MSE: 'Loss/MSE_val',
        LossKey.TOTAL: 'Loss/Total_val',
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

    def log_learning_rate(
        self,
        epoch: int,
        lr: float,
        name: str = 'Generator',
    ) -> None:
        """Log learning rate to TensorBoard.

        Args:
            epoch: Current epoch number.
            lr: Learning rate value.
            name: Name for the learning rate (e.g., 'Generator', 'Discriminator').
        """
        if self.writer is None or self.use_multi_gpu:
            return
        self.writer.add_scalar(f'LR/{name}', lr, epoch)

    def log_validation_per_modality(
        self,
        epoch: int,
        modality: str,
        metrics: Dict[str, float],
    ) -> None:
        """Log validation metrics for a specific modality.

        Tags are formatted as: Validation/{MetricName}_{modality}
        e.g., Validation/PSNR_t1_pre, Validation/MS-SSIM_bravo

        Args:
            epoch: Current epoch number.
            modality: Modality name (e.g., 't1_pre', 'bravo', 'flair').
            metrics: Dictionary of validation metrics for this modality.
        """
        if self.writer is None:
            return

        for key, value in metrics.items():
            if key in self._VAL_METRIC_TAGS:
                base_tag = self._VAL_METRIC_TAGS[key]
                # e.g., Validation/PSNR -> Validation/PSNR_t1_pre
                tag = f'{base_tag}_{modality}'
                self.writer.add_scalar(tag, value, epoch)

    def log_timestep_losses(
        self,
        epoch: int,
        timestep_bins: Dict[str, float],
    ) -> None:
        """Log per-timestep loss distribution (diffusion trainer).

        Args:
            epoch: Current epoch number.
            timestep_bins: Dictionary mapping bin names to average losses.
                Keys should be formatted as '{start:04d}-{end:04d}'.
                e.g., {'0000-0099': 0.5, '0100-0199': 0.3, ...}
        """
        if self.writer is None or self.use_multi_gpu:
            return

        for bin_name, loss in timestep_bins.items():
            tag = f'Timestep/{bin_name}'
            self.writer.add_scalar(tag, loss, epoch)

    def log_test(
        self,
        metrics: Dict[str, float],
        prefix: str = 'test_best',
        modality: Optional[str] = None,
    ) -> None:
        """Log test evaluation metrics to TensorBoard.

        Args:
            epoch: Epoch number (usually 0 for test).
            metrics: Dictionary of test metrics.
            prefix: Prefix for tags (e.g., 'test_best', 'test_latest').
            modality: Optional modality name for per-modality test metrics.
        """
        if self.writer is None:
            return

        for key, value in metrics.items():
            # Get display name from validation metric tags
            if key in self._VAL_METRIC_TAGS:
                base_tag = self._VAL_METRIC_TAGS[key]
                # Extract metric name (e.g., 'Validation/PSNR' -> 'PSNR')
                metric_name = base_tag.split('/')[-1]
            elif key in self._VAL_LOSS_TAGS:
                base_tag = self._VAL_LOSS_TAGS[key]
                metric_name = base_tag.split('/')[-1]
            else:
                # Use key as-is for unknown metrics
                metric_name = key

            # Build tag with optional modality suffix
            if modality:
                tag = f'{prefix}/{metric_name}_{modality}'
            else:
                tag = f'{prefix}/{metric_name}'

            self.writer.add_scalar(tag, value, 0)

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
        if val_metrics:
            # Handle different key names for total loss
            val_total = val_metrics.get('gen') or val_metrics.get('total', 0)
            val_gen = f"(v:{val_total:.4f})"
        else:
            val_gen = ""

        if self.config.is_seg_mode:
            self._log_seg_epoch_summary(
                timestamp, epoch, total_epochs, epoch_pct,
                avg_losses, val_metrics, val_gen, elapsed_time
            )
        elif self.config.is_diffusion_mode:
            self._log_diffusion_epoch_summary(
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

    def _log_diffusion_epoch_summary(
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
        """Log epoch summary for diffusion trainer."""
        # Get MSE and perceptual losses
        mse_loss = avg_losses.get(LossKey.MSE, 0)
        perc_loss = avg_losses.get(LossKey.PERC, 0)
        total_loss = avg_losses.get(LossKey.TOTAL, 0) or avg_losses.get(LossKey.GEN, 0)

        # Build validation metrics string
        metrics_parts = []
        if val_metrics:
            if val_metrics.get(MetricKey.MSSSIM):
                metrics_parts.append(f"MS-SSIM: {val_metrics[MetricKey.MSSSIM]:.3f}")
            if val_metrics.get(MetricKey.MSSSIM_3D):
                metrics_parts.append(f"MS-SSIM-3D: {val_metrics[MetricKey.MSSSIM_3D]:.3f}")
            if val_metrics.get(MetricKey.PSNR):
                metrics_parts.append(f"PSNR: {val_metrics[MetricKey.PSNR]:.2f}")
            if val_metrics.get(MetricKey.LPIPS):
                metrics_parts.append(f"LPIPS: {val_metrics[MetricKey.LPIPS]:.3f}")
        metric_str = " | ".join(metrics_parts) if metrics_parts else ""

        # Format: Loss: 0.1234(v:0.0987) | MSE: 0.0567 | Perc: 0.0654 | MS-SSIM: 0.95 | PSNR: 25.3
        logger.info(
            f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
            f"Loss: {total_loss:.4f}{val_gen} | "
            f"MSE: {mse_loss:.4f} | "
            f"Perc: {perc_loss:.4f} | "
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
    log_psnr: bool = True,
    modality_keys: Optional[List[str]] = None,
) -> TrainerMetricsConfig:
    """Factory function to create metrics config from string trainer type.

    Args:
        trainer_type: One of 'vae', 'vqvae', 'dcae', 'diffusion'.
        has_gan: Whether GAN training is enabled.
        spatial_dims: 2 or 3 for 2D or 3D models.
        seg_mode: Whether segmentation mode is enabled.
        log_msssim: Whether to log MS-SSIM metrics.
        log_lpips: Whether to log LPIPS metrics.
        log_psnr: Whether to log PSNR metrics.
        modality_keys: List of modality names for per-modality logging.

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
    elif trainer_type == 'diffusion':
        return TrainerMetricsConfig.for_diffusion(
            spatial_dims=spatial_dims,
            log_msssim=log_msssim,
            log_lpips=log_lpips,
            log_psnr=log_psnr,
            modality_keys=modality_keys,
        )
    else:
        raise ValueError(
            f"Unknown trainer type: {trainer_type}. "
            "Expected 'vae', 'vqvae', 'dcae', or 'diffusion'."
        )


# =============================================================================
# Unified Metrics (NEW - Single Entry Point for All Trainers)
# =============================================================================

class UnifiedMetrics:
    """
    Single entry point for all metrics across ALL trainers.

    Provides:
    - Per-metric method calls (update_psnr, update_lpips, etc.)
    - Mode-aware behavior (auto-skip PSNR for seg mode)
    - Dimension-aware (3D metrics computed slice-wise if metric is 2D-only)
    - Modality suffix on all validation metrics (LPIPS_bravo, PSNR_flair)
    - No manual add_scalar in trainers - all logging through this class

    Usage:
        metrics = UnifiedMetrics(
            writer=tensorboard_writer,
            mode='bravo',           # or 'seg', 'dual', 'multi', etc.
            spatial_dims=3,         # 2 or 3
            modality='bravo',       # For TensorBoard suffix
            device='cuda',
        )

        # Training loop
        for batch in train_loader:
            loss = train_step(batch)
            metrics.update_loss('MSE', loss.item())
            metrics.update_grad_norm(grad_norm)

        metrics.update_vram()
        metrics.log_training(epoch)
        metrics.reset_training()

        # Validation loop
        for batch in val_loader:
            pred, gt = validate_step(batch)
            metrics.update_psnr(pred, gt)
            metrics.update_lpips(pred, gt)
            metrics.update_msssim(pred, gt)
            metrics.update_regional(pred, gt, mask)

        metrics.log_validation(epoch)
        metrics.reset_validation()

        # Test evaluation
        metrics.evaluate_test(test_loader, prefix='test_best')
    """

    def __init__(
        self,
        writer: Optional[SummaryWriter],
        mode: str,
        spatial_dims: int = 2,
        modality: Optional[str] = None,
        device: Optional[torch.device] = None,
        # Optional feature flags
        enable_regional: bool = False,
        enable_codebook: bool = False,
        codebook_size: int = 512,
        num_timestep_bins: int = 10,
        # Regional tracker params (2D)
        image_size: int = 256,
        fov_mm: float = 240.0,
        # Regional tracker params (3D)
        volume_size: Optional[Tuple[int, int, int]] = None,
    ):
        """Initialize unified metrics.

        Args:
            writer: TensorBoard SummaryWriter (may be None for DDP non-main).
            mode: Training mode ('seg', 'bravo', 'dual', 'multi', etc.).
            spatial_dims: 2 or 3 for 2D or 3D models.
            modality: Modality name for TensorBoard suffix (e.g., 'bravo', 'flair').
            device: Device for computation (defaults to cuda if available).
            enable_regional: Enable regional metrics tracking.
            enable_codebook: Enable codebook tracking (VQ-VAE).
            codebook_size: Size of VQ codebook (if enable_codebook=True).
            num_timestep_bins: Number of bins for timestep loss tracking.
            image_size: Image size for 2D regional tracker (default 256).
            fov_mm: Field of view in mm for regional tracker (default 240.0).
            volume_size: (H, W, D) for 3D regional tracker.
        """
        self.writer = writer
        self.mode = mode
        self.spatial_dims = spatial_dims
        self.modality = modality
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_timestep_bins = num_timestep_bins
        self.image_size = image_size
        self.fov_mm = fov_mm
        self.volume_size = volume_size or (256, 256, 160)

        # Mode-aware flags
        self.is_seg_mode = mode in ('seg', 'seg_compression', 'seg_conditioned_3d')
        self.uses_image_quality = not self.is_seg_mode

        # Initialize accumulators
        self._init_accumulators()

        # Initialize optional components
        self._regional_tracker = None
        if enable_regional:
            self._init_regional()

        self._codebook_tracker = None
        if enable_codebook:
            self._init_codebook(codebook_size)

    def _init_accumulators(self):
        """Initialize all metric accumulators."""
        # Training accumulators
        self._train_losses: Dict[str, Dict[str, float]] = {}
        self._grad_norm_sum = 0.0
        self._grad_norm_count = 0
        self._current_lr: Optional[float] = None

        # Validation quality metrics
        self._val_psnr_sum = 0.0
        self._val_psnr_count = 0
        self._val_lpips_sum = 0.0
        self._val_lpips_count = 0
        self._val_msssim_sum = 0.0
        self._val_msssim_count = 0
        self._val_msssim_3d_sum = 0.0
        self._val_msssim_3d_count = 0

        # Validation seg metrics
        self._val_dice_sum = 0.0
        self._val_dice_count = 0
        self._val_iou_sum = 0.0
        self._val_iou_count = 0

        # Validation losses
        self._val_losses: Dict[str, Dict[str, float]] = {}

        # Timestep losses (validation only)
        self._val_timesteps = self._create_timestep_storage()

        # Resource metrics
        self._vram_allocated = 0.0
        self._vram_reserved = 0.0
        self._vram_max = 0.0
        self._flops_epoch = 0.0
        self._flops_total = 0.0

    def _create_timestep_storage(self) -> Dict[str, Any]:
        """Create storage for timestep losses."""
        return {
            'sums': [0.0] * self.num_timestep_bins,
            'counts': [0] * self.num_timestep_bins,
        }

    def _init_regional(self):
        """Initialize regional metrics tracker."""
        if self.spatial_dims == 3:
            from .regional_3d import RegionalMetricsTracker3D
            self._regional_tracker = RegionalMetricsTracker3D(
                volume_size=self.volume_size,
                fov_mm=self.fov_mm,
                device=self.device,
            )
        else:
            from .regional import RegionalMetricsTracker
            self._regional_tracker = RegionalMetricsTracker(
                image_size=self.image_size,
                fov_mm=self.fov_mm,
                device=self.device,
            )

    def _init_codebook(self, codebook_size: int):
        """Initialize codebook tracker for VQ-VAE."""
        try:
            from ..tracking.codebook import CodebookTracker
            self._codebook_tracker = CodebookTracker(
                num_embeddings=codebook_size,
                device=self.device,
            )
        except ImportError:
            logger.warning("CodebookTracker not available, codebook metrics disabled")

    # =========================================================================
    # Quality Metric Methods
    # =========================================================================

    def update_psnr(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute and accumulate PSNR.

        Works for both 2D and 3D data.
        Auto-skips for seg mode.

        Args:
            pred: Predicted tensor [B, C, (D), H, W].
            gt: Ground truth tensor [B, C, (D), H, W].

        Returns:
            Computed PSNR value (0.0 if skipped).
        """
        if not self.uses_image_quality:
            return 0.0
        from .quality import compute_psnr
        val = compute_psnr(pred, gt)
        self._val_psnr_sum += val
        self._val_psnr_count += 1
        return val

    def update_lpips(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute and accumulate LPIPS.

        3D data uses slice-wise computation (2.5D approach).
        Auto-skips for seg mode.

        Args:
            pred: Predicted tensor [B, C, (D), H, W].
            gt: Ground truth tensor [B, C, (D), H, W].

        Returns:
            Computed LPIPS value (0.0 if skipped).
        """
        if not self.uses_image_quality:
            return 0.0
        from .quality import compute_lpips, compute_lpips_3d
        if self.spatial_dims == 3:
            val = compute_lpips_3d(pred, gt, device=self.device)
        else:
            val = compute_lpips(pred, gt)
        self._val_lpips_sum += val
        self._val_lpips_count += 1
        return val

    def update_msssim(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute and accumulate MS-SSIM.

        3D data uses 2D slice-wise computation for efficiency.
        Auto-skips for seg mode.

        Args:
            pred: Predicted tensor [B, C, (D), H, W].
            gt: Ground truth tensor [B, C, (D), H, W].

        Returns:
            Computed MS-SSIM value (0.0 if skipped).
        """
        if not self.uses_image_quality:
            return 0.0
        from .quality import compute_msssim, compute_msssim_2d_slicewise
        if self.spatial_dims == 3:
            val = compute_msssim_2d_slicewise(pred, gt)
        else:
            val = compute_msssim(pred, gt, spatial_dims=2)
        self._val_msssim_sum += val
        self._val_msssim_count += 1
        return val

    def update_msssim_3d(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute true 3D MS-SSIM (native, not slice-wise).

        Only works for 3D data. Auto-skips for seg mode or 2D.

        Args:
            pred: Predicted tensor [B, C, D, H, W].
            gt: Ground truth tensor [B, C, D, H, W].

        Returns:
            Computed 3D MS-SSIM value (0.0 if skipped).
        """
        if not self.uses_image_quality or self.spatial_dims != 3:
            return 0.0
        from .quality import compute_msssim
        val = compute_msssim(pred, gt, spatial_dims=3)
        self._val_msssim_3d_sum += val
        self._val_msssim_3d_count += 1
        return val

    def update_dice(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute and accumulate Dice coefficient.

        Works for both image and seg modes.

        Args:
            pred: Predicted tensor [B, C, (D), H, W].
            gt: Ground truth tensor [B, C, (D), H, W].

        Returns:
            Computed Dice value.
        """
        from .quality import compute_dice
        val = compute_dice(pred, gt)
        self._val_dice_sum += val
        self._val_dice_count += 1
        return val

    def update_iou(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute and accumulate IoU.

        Works for both image and seg modes.

        Args:
            pred: Predicted tensor [B, C, (D), H, W].
            gt: Ground truth tensor [B, C, (D), H, W].

        Returns:
            Computed IoU value.
        """
        from .quality import compute_iou
        val = compute_iou(pred, gt)
        self._val_iou_sum += val
        self._val_iou_count += 1
        return val

    # =========================================================================
    # Loss Methods
    # =========================================================================

    def update_loss(self, key: str, value: float, phase: str = 'train'):
        """Accumulate a loss value.

        Args:
            key: Loss key (e.g., 'MSE', 'Total', 'KL', 'Perceptual').
            value: Loss value to accumulate.
            phase: 'train' or 'val'.
        """
        storage = self._train_losses if phase == 'train' else self._val_losses
        if key not in storage:
            storage[key] = {'sum': 0.0, 'count': 0}
        storage[key]['sum'] += value
        storage[key]['count'] += 1

    # =========================================================================
    # Training Diagnostic Methods
    # =========================================================================

    def update_grad_norm(self, value: float):
        """Accumulate gradient norm.

        Args:
            value: Gradient norm value.
        """
        self._grad_norm_sum += value
        self._grad_norm_count += 1

    def update_lr(self, value: float):
        """Store current learning rate.

        Args:
            value: Learning rate value.
        """
        self._current_lr = value

    def update_vram(self):
        """Capture current VRAM usage."""
        if torch.cuda.is_available():
            self._vram_allocated = torch.cuda.memory_allocated(self.device) / 1e9
            self._vram_reserved = torch.cuda.memory_reserved(self.device) / 1e9
            self._vram_max = torch.cuda.max_memory_allocated(self.device) / 1e9

    def update_flops(self, tflops_epoch: float, tflops_total: float):
        """Store FLOPs values.

        Args:
            tflops_epoch: TFLOPs for this epoch.
            tflops_total: Cumulative TFLOPs.
        """
        self._flops_epoch = tflops_epoch
        self._flops_total = tflops_total

    def update_timestep_loss(self, t: float, loss: float):
        """Accumulate validation loss for timestep bin.

        Args:
            t: Timestep value (0.0 to 1.0).
            loss: Loss value at this timestep.
        """
        bin_idx = min(int(t * self.num_timestep_bins), self.num_timestep_bins - 1)
        self._val_timesteps['sums'][bin_idx] += loss
        self._val_timesteps['counts'][bin_idx] += 1

    # =========================================================================
    # Regional and Codebook Methods
    # =========================================================================

    def update_regional(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Update regional metrics tracker.

        Args:
            pred: Predicted tensor.
            gt: Ground truth tensor.
            mask: Segmentation mask for regional analysis.
        """
        if self._regional_tracker is not None:
            self._regional_tracker.update(pred, gt, mask)

    def update_codebook(self, indices: torch.Tensor):
        """Update codebook usage tracker.

        Args:
            indices: Codebook indices from VQ-VAE quantization.
        """
        if self._codebook_tracker is not None:
            self._codebook_tracker.update_fast(indices)

    # =========================================================================
    # Logging Methods
    # =========================================================================

    def log_training(self, epoch: int):
        """Log all training metrics to TensorBoard.

        Args:
            epoch: Current epoch number.
        """
        if self.writer is None:
            return

        # Losses
        for key, data in self._train_losses.items():
            if data['count'] > 0:
                self.writer.add_scalar(
                    f'Loss/{key}_train',
                    data['sum'] / data['count'],
                    epoch,
                )

        # LR
        if self._current_lr is not None:
            self.writer.add_scalar('LR/Generator', self._current_lr, epoch)

        # Grad norm
        if self._grad_norm_count > 0:
            self.writer.add_scalar(
                'training/grad_norm_epoch',
                self._grad_norm_sum / self._grad_norm_count,
                epoch,
            )

        # VRAM
        if self._vram_allocated > 0:
            self.writer.add_scalar('VRAM/allocated_GB', self._vram_allocated, epoch)
            self.writer.add_scalar('VRAM/reserved_GB', self._vram_reserved, epoch)
            self.writer.add_scalar('VRAM/max_allocated_GB', self._vram_max, epoch)

        # FLOPs
        if self._flops_epoch > 0:
            self.writer.add_scalar('FLOPs/TFLOPs_epoch', self._flops_epoch, epoch)
            self.writer.add_scalar('FLOPs/TFLOPs_total', self._flops_total, epoch)

        # Codebook
        if self._codebook_tracker is not None:
            self._codebook_tracker.log_to_tensorboard(self.writer, epoch)

    def log_validation(self, epoch: int):
        """Log all validation metrics to TensorBoard.

        Args:
            epoch: Current epoch number.
        """
        if self.writer is None:
            return

        suffix = f'_{self.modality}' if self.modality else ''

        # Losses
        for key, data in self._val_losses.items():
            if data['count'] > 0:
                self.writer.add_scalar(
                    f'Loss/{key}_val',
                    data['sum'] / data['count'],
                    epoch,
                )

        # Quality metrics (image modes)
        if self.uses_image_quality:
            if self._val_psnr_count > 0:
                self.writer.add_scalar(
                    f'Validation/PSNR{suffix}',
                    self._val_psnr_sum / self._val_psnr_count,
                    epoch,
                )
            if self._val_msssim_count > 0:
                self.writer.add_scalar(
                    f'Validation/MS-SSIM{suffix}',
                    self._val_msssim_sum / self._val_msssim_count,
                    epoch,
                )
            if self._val_lpips_count > 0:
                self.writer.add_scalar(
                    f'Validation/LPIPS{suffix}',
                    self._val_lpips_sum / self._val_lpips_count,
                    epoch,
                )
            if self._val_msssim_3d_count > 0:
                self.writer.add_scalar(
                    f'Validation/MS-SSIM-3D{suffix}',
                    self._val_msssim_3d_sum / self._val_msssim_3d_count,
                    epoch,
                )

        # Seg metrics (always log if computed)
        if self._val_dice_count > 0:
            self.writer.add_scalar(
                f'Validation/Dice{suffix}',
                self._val_dice_sum / self._val_dice_count,
                epoch,
            )
        if self._val_iou_count > 0:
            self.writer.add_scalar(
                f'Validation/IoU{suffix}',
                self._val_iou_sum / self._val_iou_count,
                epoch,
            )

        # Validation timesteps
        for i in range(self.num_timestep_bins):
            if self._val_timesteps['counts'][i] > 0:
                t_start = i / self.num_timestep_bins
                t_end = (i + 1) / self.num_timestep_bins
                avg = self._val_timesteps['sums'][i] / self._val_timesteps['counts'][i]
                self.writer.add_scalar(
                    f'val_timestep_losses/t_{t_start:.1f}_{t_end:.1f}',
                    avg,
                    epoch,
                )

        # Regional
        if self._regional_tracker is not None:
            prefix = f'regional{suffix}' if suffix else 'regional'
            self._regional_tracker.log_to_tensorboard(self.writer, epoch, prefix=prefix)

    def log_generation(self, epoch: int, results: Dict[str, float]):
        """Log generation metrics to TensorBoard.

        Args:
            epoch: Current epoch number.
            results: Dict of generation metric results (KID, CMMD, etc.).
        """
        if self.writer is None:
            return
        for key, value in results.items():
            self.writer.add_scalar(f'Generation/{key}', value, epoch)

    def log_test(self, metrics: Dict[str, float], prefix: str = 'test_best'):
        """Log test evaluation metrics.

        Args:
            metrics: Dict of metric name -> value.
            prefix: 'test_best' or 'test_latest'.
        """
        if self.writer is None:
            return
        suffix = f'_{self.modality}' if self.modality else ''
        for key, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{key}{suffix}', value, 0)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def get_validation_metrics(self) -> Dict[str, float]:
        """Collect current validation metrics as a dict.

        Returns:
            Dict of metric name -> averaged value.
        """
        metrics = {}

        if self.uses_image_quality:
            if self._val_psnr_count > 0:
                metrics['PSNR'] = self._val_psnr_sum / self._val_psnr_count
            if self._val_msssim_count > 0:
                metrics['MS-SSIM'] = self._val_msssim_sum / self._val_msssim_count
            if self._val_lpips_count > 0:
                metrics['LPIPS'] = self._val_lpips_sum / self._val_lpips_count
            if self._val_msssim_3d_count > 0:
                metrics['MS-SSIM-3D'] = self._val_msssim_3d_sum / self._val_msssim_3d_count

        if self._val_dice_count > 0:
            metrics['Dice'] = self._val_dice_sum / self._val_dice_count
        if self._val_iou_count > 0:
            metrics['IoU'] = self._val_iou_sum / self._val_iou_count

        # Add losses
        for key, data in self._val_losses.items():
            if data['count'] > 0:
                metrics[key] = data['sum'] / data['count']

        return metrics

    def get_training_losses(self) -> Dict[str, float]:
        """Collect current training losses as a dict.

        Returns:
            Dict of loss name -> averaged value.
        """
        return {
            key: data['sum'] / data['count']
            for key, data in self._train_losses.items()
            if data['count'] > 0
        }

    # =========================================================================
    # Reset Methods
    # =========================================================================

    def reset_training(self):
        """Reset training accumulators for next epoch."""
        self._train_losses.clear()
        self._grad_norm_sum = 0.0
        self._grad_norm_count = 0
        self._current_lr = None
        self._vram_allocated = 0.0
        self._vram_reserved = 0.0
        self._vram_max = 0.0
        if self._codebook_tracker is not None:
            self._codebook_tracker.reset()

    def reset_validation(self):
        """Reset validation accumulators for next epoch."""
        self._val_losses.clear()
        self._val_psnr_sum = self._val_psnr_count = 0
        self._val_lpips_sum = self._val_lpips_count = 0
        self._val_msssim_sum = self._val_msssim_count = 0
        self._val_msssim_3d_sum = self._val_msssim_3d_count = 0
        self._val_dice_sum = self._val_dice_count = 0
        self._val_iou_sum = self._val_iou_count = 0
        self._val_timesteps = self._create_timestep_storage()
        if self._regional_tracker is not None:
            self._regional_tracker.reset()

    def reset_all(self):
        """Reset everything."""
        self.reset_training()
        self.reset_validation()

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def log_reconstruction_figure(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        epoch: int,
        mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        tag: str = 'Figures/reconstruction',
        max_samples: int = 8,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Log reconstruction comparison figure to TensorBoard.

        Args:
            original: Ground truth tensor [B, C, (D), H, W].
            reconstructed: Reconstructed tensor [B, C, (D), H, W].
            epoch: Current epoch number.
            mask: Optional segmentation mask for contour overlay.
            timesteps: Optional timesteps for column titles (diffusion).
            tag: TensorBoard tag for the figure.
            max_samples: Maximum number of samples to display.
            metrics: Optional dict of metrics to show in subtitle.
        """
        if self.writer is None:
            return

        from .figures import create_reconstruction_figure
        import matplotlib.pyplot as plt

        # Handle 3D volumes - take center slice
        if self.spatial_dims == 3:
            original = self._extract_center_slice(original)
            reconstructed = self._extract_center_slice(reconstructed)
            if mask is not None:
                mask = self._extract_center_slice(mask)

        fig = create_reconstruction_figure(
            original=original,
            generated=reconstructed,
            timesteps=timesteps,
            mask=mask,
            max_samples=max_samples,
            metrics=metrics,
        )
        self.writer.add_figure(tag, fig, epoch)
        plt.close(fig)

    def log_worst_batch(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        loss: float,
        epoch: int,
        phase: str = 'train',
        mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ):
        """Log worst batch visualization to TensorBoard.

        Args:
            original: Ground truth tensor.
            reconstructed: Reconstructed tensor.
            loss: Loss value for this batch.
            epoch: Current epoch number.
            phase: 'train' or 'val'.
            mask: Optional segmentation mask.
            timesteps: Optional timesteps (diffusion).
        """
        if self.writer is None:
            return

        tag = f'worst_batch/{phase}'
        metrics = {'loss': loss}

        self.log_reconstruction_figure(
            original=original,
            reconstructed=reconstructed,
            epoch=epoch,
            mask=mask,
            timesteps=timesteps,
            tag=tag,
            metrics=metrics,
        )

    def log_denoising_trajectory(
        self,
        trajectory: list,
        epoch: int,
        tag: str = 'denoising_trajectory',
    ):
        """Log denoising step visualization to TensorBoard.

        Args:
            trajectory: List of tensors showing denoising progress.
            epoch: Current epoch number.
            tag: TensorBoard tag prefix.
        """
        if self.writer is None or not trajectory:
            return

        import matplotlib.pyplot as plt
        import numpy as np

        # Stack trajectory into single tensor
        steps = len(trajectory)
        sample = trajectory[0]

        # Handle 3D - take center slice
        if self.spatial_dims == 3 and sample.dim() == 5:
            trajectory = [self._extract_center_slice(t) for t in trajectory]
            sample = trajectory[0]

        # Create figure showing progression
        fig, axes = plt.subplots(1, steps, figsize=(2.5 * steps, 3))
        if steps == 1:
            axes = [axes]

        for i, step_tensor in enumerate(trajectory):
            # Take first sample, first channel
            if isinstance(step_tensor, torch.Tensor):
                img = step_tensor[0, 0].cpu().float().numpy()
            else:
                img = step_tensor[0, 0]
            axes[i].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f'Step {i}', fontsize=8)
            axes[i].axis('off')

        fig.tight_layout()
        self.writer.add_figure(f'{tag}/progression', fig, epoch)
        plt.close(fig)

    def log_test_figure(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        prefix: str = 'test_best',
        mask: Optional[torch.Tensor] = None,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Log test evaluation figure to TensorBoard.

        Args:
            original: Ground truth tensor.
            reconstructed: Reconstructed tensor.
            prefix: 'test_best' or 'test_latest'.
            mask: Optional segmentation mask.
            metrics: Optional dict of metrics for subtitle.
        """
        if self.writer is None:
            return

        self.log_reconstruction_figure(
            original=original,
            reconstructed=reconstructed,
            epoch=0,  # Test uses step 0
            mask=mask,
            tag=f'{prefix}/reconstruction',
            metrics=metrics,
        )

    def _extract_center_slice(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract center slice from 3D volume.

        Args:
            tensor: 5D tensor [B, C, D, H, W].

        Returns:
            4D tensor [B, C, H, W] with center depth slice.
        """
        if tensor.dim() == 5:
            depth = tensor.shape[2]
            center_idx = depth // 2
            return tensor[:, :, center_idx, :, :]
        return tensor

    # =========================================================================
    # Console Logging
    # =========================================================================

    def log_console_summary(
        self,
        epoch: int,
        total_epochs: int,
        elapsed_time: float,
    ):
        """Log epoch completion summary to console.

        Args:
            epoch: Current epoch number (0-indexed).
            total_epochs: Total number of epochs.
            elapsed_time: Time taken for epoch in seconds.
        """
        import time as time_module

        timestamp = time_module.strftime("%H:%M:%S")
        epoch_pct = ((epoch + 1) / total_epochs) * 100

        train_losses = self.get_training_losses()
        val_metrics = self.get_validation_metrics()

        # Build loss string
        loss_parts = []
        total_loss = train_losses.get('Total') or train_losses.get('MSE') or 0
        val_total = val_metrics.get('MSE') or val_metrics.get('Total') or 0
        loss_parts.append(f"Loss: {total_loss:.4f}")
        if val_total > 0:
            loss_parts[-1] += f"(v:{val_total:.4f})"

        for key in ['MSE', 'Perceptual', 'KL', 'VQ', 'BCE', 'Dice']:
            if key in train_losses and key != 'Total':
                loss_parts.append(f"{key}: {train_losses[key]:.4f}")

        # Build validation metrics string
        metric_parts = []
        if self.uses_image_quality:
            if 'MS-SSIM' in val_metrics:
                metric_parts.append(f"MS-SSIM: {val_metrics['MS-SSIM']:.3f}")
            if 'MS-SSIM-3D' in val_metrics:
                metric_parts.append(f"MS-SSIM-3D: {val_metrics['MS-SSIM-3D']:.3f}")
            if 'PSNR' in val_metrics:
                metric_parts.append(f"PSNR: {val_metrics['PSNR']:.2f}")
            if 'LPIPS' in val_metrics:
                metric_parts.append(f"LPIPS: {val_metrics['LPIPS']:.3f}")
        else:
            if 'Dice' in val_metrics:
                metric_parts.append(f"Dice: {val_metrics['Dice']:.3f}")
            if 'IoU' in val_metrics:
                metric_parts.append(f"IoU: {val_metrics['IoU']:.3f}")

        # Combine all parts
        all_parts = loss_parts + metric_parts
        all_parts.append(f"Time: {elapsed_time:.1f}s")

        logger.info(
            f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
            + " | ".join(all_parts)
        )
