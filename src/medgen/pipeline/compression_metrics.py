"""Metrics logging for compression training.

This module provides:
- CompressionMetricsLogger: Handles metrics logging for compression training
- compute_volume_3d_msssim: Compute 3D MS-SSIM by reconstructing full volumes

Consolidates the metrics logging logic from BaseCompressionTrainer into
a reusable class with proper TensorBoard integration.
"""
import logging
from typing import TYPE_CHECKING

import torch
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from medgen.metrics import (
    GradientNormTracker,
    SimpleLossAccumulator,
    UnifiedMetrics,
    compute_msssim,
)

if TYPE_CHECKING:
    from medgen.pipeline.compression_trainer import BaseCompressionTrainer

logger = logging.getLogger(__name__)


class CompressionMetricsLogger:
    """Handles metrics logging for compression training.

    Provides unified metrics logging through UnifiedMetrics, including:
    - Training loss logging
    - Validation metrics logging (PSNR, MS-SSIM, LPIPS, etc.)
    - Epoch summary logging
    - Gradient norm logging
    - Regional metrics logging
    """

    def __init__(
        self,
        writer: SummaryWriter,
        spatial_dims: int,
        mode_name: str,
        device,
        trainer_type: str = 'vae',
        image_size: int = 256,
        fov_mm: float = 240.0,
        enable_regional: bool = False,
        seg_mode: bool = False,
    ) -> None:
        """Initialize compression metrics logger.

        Args:
            writer: TensorBoard SummaryWriter.
            spatial_dims: Spatial dimensions (2 or 3).
            mode_name: Training mode name (e.g., 'bravo', 'multi_modality').
            device: Compute device.
            trainer_type: Type of trainer ('vae', 'vqvae', 'dcae').
            image_size: Image size for regional tracker.
            fov_mm: Field of view in mm for regional tracker.
            enable_regional: Whether to enable regional metrics.
            seg_mode: Whether in segmentation mode.
        """
        self.writer = writer
        self.spatial_dims = spatial_dims
        self.mode_name = mode_name
        self.device = device
        self.trainer_type = trainer_type
        self.seg_mode = seg_mode

        # Determine modality suffix
        # seg_conditioned modes: no suffix (distinguish by TensorBoard run color)
        is_single_modality = mode_name not in ('multi_modality', 'dual', 'multi')
        is_seg_conditioned = mode_name.startswith('seg_conditioned')
        modality = None if is_seg_conditioned else (mode_name if is_single_modality else None)

        # Initialize unified metrics
        self._unified_metrics = UnifiedMetrics(
            writer=writer,
            mode=mode_name,
            spatial_dims=spatial_dims,
            modality=modality,
            device=device,
            enable_regional=enable_regional,
            enable_codebook=(trainer_type == 'vqvae'),
            image_size=image_size,
            fov_mm=fov_mm,
        )

        # Create loss accumulator for epoch-level loss tracking
        self._loss_accumulator = SimpleLossAccumulator()

    @property
    def unified_metrics(self) -> UnifiedMetrics:
        """Access underlying UnifiedMetrics."""
        return self._unified_metrics

    @property
    def loss_accumulator(self) -> SimpleLossAccumulator:
        """Access loss accumulator."""
        return self._loss_accumulator

    def log_training(self, epoch: int, avg_losses: dict[str, float]) -> None:
        """Log training metrics.

        Args:
            epoch: Current epoch number.
            avg_losses: Dictionary of averaged losses from training.
        """
        if self._unified_metrics is None:
            return

        if self.seg_mode:
            # Use dedicated seg training logging
            self._unified_metrics.log_seg_training(avg_losses, epoch)
        else:
            # Standard loss logging
            for key, value in avg_losses.items():
                self._unified_metrics.update_loss(key, value, phase='train')
            self._unified_metrics.log_training(epoch)
            self._unified_metrics.reset_training()

    def log_validation(
        self,
        epoch: int,
        metrics: dict[str, float],
        prefix: str = '',
    ) -> None:
        """Log validation metrics.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of validation metrics.
            prefix: Optional prefix for per-modality logging.
        """
        if self._unified_metrics is None:
            return

        # Update metric accumulators
        if 'psnr' in metrics:
            self._unified_metrics._val_psnr_sum = metrics['psnr']
            self._unified_metrics._val_psnr_count = 1
        if 'msssim' in metrics:
            self._unified_metrics._val_msssim_sum = metrics['msssim']
            self._unified_metrics._val_msssim_count = 1
        if 'lpips' in metrics:
            self._unified_metrics._val_lpips_sum = metrics['lpips']
            self._unified_metrics._val_lpips_count = 1
        if 'msssim_3d' in metrics:
            self._unified_metrics._val_msssim_3d_sum = metrics['msssim_3d']
            self._unified_metrics._val_msssim_3d_count = 1
        if 'dice_score' in metrics:
            self._unified_metrics._val_dice_sum = metrics['dice_score']
            self._unified_metrics._val_dice_count = 1
        if 'iou' in metrics:
            self._unified_metrics._val_iou_sum = metrics['iou']
            self._unified_metrics._val_iou_count = 1

        # Log validation losses
        if 'val_loss' in metrics:
            self._unified_metrics.update_loss('Total', metrics['val_loss'], phase='val')

        self._unified_metrics.log_validation(epoch)
        self._unified_metrics.reset_validation()

    def log_epoch_summary(
        self,
        epoch: int,
        n_epochs: int,
        elapsed_time: float,
    ) -> None:
        """Log epoch summary to console.

        Args:
            epoch: Current epoch number.
            n_epochs: Total number of epochs.
            elapsed_time: Time taken for epoch in seconds.
        """
        if self._unified_metrics is not None:
            self._unified_metrics.log_console_summary(epoch, n_epochs, elapsed_time)

    def log_grad_norms(
        self,
        epoch: int,
        g_tracker: GradientNormTracker,
        d_tracker: GradientNormTracker | None = None,
        gan_enabled: bool = True,
    ) -> None:
        """Log gradient norm statistics.

        Args:
            epoch: Current epoch number.
            g_tracker: Generator gradient norm tracker.
            d_tracker: Discriminator gradient norm tracker (optional).
            gan_enabled: Whether GAN training is enabled.
        """
        if self._unified_metrics is None:
            return

        # Use _g suffix only when discriminator exists for clarity
        gen_prefix = 'training/grad_norm_g' if gan_enabled else 'training/grad_norm'
        self._unified_metrics.log_grad_norm_from_tracker(g_tracker, epoch, prefix=gen_prefix)

        if gan_enabled and d_tracker is not None:
            self._unified_metrics.log_grad_norm_from_tracker(
                d_tracker, epoch, prefix='training/grad_norm_d'
            )

    def log_worst_batch(
        self,
        original,
        reconstructed,
        loss: float,
        epoch: int,
        phase: str = 'val',
    ) -> None:
        """Log worst batch figure.

        Args:
            original: Original images.
            reconstructed: Reconstructed images.
            loss: Loss value for the worst batch.
            epoch: Current epoch number.
            phase: Phase ('train' or 'val').
        """
        if self._unified_metrics is not None:
            self._unified_metrics.log_worst_batch(
                original=original,
                reconstructed=reconstructed,
                loss=loss,
                epoch=epoch,
                phase=phase,
            )

    def log_per_modality_validation(
        self,
        metrics: dict[str, float | None],
        modality: str,
        epoch: int,
    ) -> None:
        """Log per-modality validation metrics.

        Args:
            metrics: Dictionary of metrics (psnr, msssim, lpips, msssim_3d).
            modality: Modality name for suffix.
            epoch: Current epoch number.
        """
        if self._unified_metrics is not None:
            self._unified_metrics.log_per_modality_validation(metrics, modality, epoch)

    def log_per_channel_validation(
        self,
        per_channel_data: dict[str, dict[str, float]],
        epoch: int,
    ) -> None:
        """Log per-channel validation metrics for dual mode.

        Args:
            per_channel_data: Dictionary mapping channel names to metrics.
            epoch: Current epoch number.
        """
        if self._unified_metrics is not None:
            self._unified_metrics.log_per_channel_validation(per_channel_data, epoch)

    def log_validation_regional(
        self,
        regional_tracker,
        epoch: int,
        modality_override: str | None = None,
    ) -> None:
        """Log regional validation metrics.

        Args:
            regional_tracker: Regional metrics tracker.
            epoch: Current epoch number.
            modality_override: Optional modality for suffix.
        """
        if self._unified_metrics is not None:
            self._unified_metrics.log_validation_regional(
                regional_tracker, epoch, modality_override=modality_override
            )


def compute_volume_3d_msssim(
    trainer: 'BaseCompressionTrainer',
    epoch: int,
    data_split: str = 'val',
    modality_override: str | None = None,
) -> float | None:
    """Compute 3D MS-SSIM by reconstructing full volumes slice-by-slice.

    For 2D trainers, this loads full 3D volumes, processes each slice through
    the model, stacks reconstructed slices back into a volume, and computes
    3D MS-SSIM. This shows how well 2D models maintain cross-slice consistency.

    Optimizations applied:
    - inference_mode instead of no_grad (faster)
    - Tensor slicing instead of loop for batch extraction
    - Non-blocking GPU transfers
    - Pre-allocated output tensor

    Args:
        trainer: Compression trainer instance.
        epoch: Current epoch number.
        data_split: Which data split to use ('val' or 'test_new').
        modality_override: Optional specific modality to compute for
            (e.g., 'bravo', 't1_pre'). If None, uses mode from config.

    Returns:
        Average 3D MS-SSIM across all volumes, or None if unavailable.
    """
    if not trainer.log_msssim:
        return None

    # Import here to avoid circular imports
    from medgen.data.loaders.vae import create_vae_volume_validation_dataloader

    # Determine modality - use override if provided, else from config
    if modality_override is not None:
        modality = modality_override
    else:
        mode_name = trainer.cfg.mode.get('name', 'bravo')
        n_channels = trainer.cfg.mode.get('in_channels', 1)
        # Use subdir for file loading (e.g., 'seg' instead of 'seg_conditioned')
        subdir = trainer.cfg.mode.get('subdir', mode_name)
        modality = 'dual' if n_channels > 1 else subdir

    # Create volume dataloader
    result = create_vae_volume_validation_dataloader(trainer.cfg, modality, data_split)
    if result is None:
        return None

    volume_loader, _ = result

    model_to_use = trainer._get_model_for_eval()
    model_to_use.eval()

    total_msssim_3d = 0.0
    n_volumes = 0
    slice_batch_size = trainer.cfg.training.batch_size  # Reuse training batch size

    with torch.inference_mode():  # Faster than no_grad
        for batch in volume_loader:
            # batch['image'] is [1, C, H, W, D] (batch_size=1 for volumes)
            # Non-blocking transfer to GPU
            volume = batch['image'].to(trainer.device, non_blocking=True)  # [1, C, H, W, D]
            volume = volume.squeeze(0)  # [C, H, W, D]

            n_channels_vol, height, width, depth = volume.shape

            # Pre-allocate output tensor on GPU
            all_recon = torch.empty(
                (depth, n_channels_vol, height, width),
                dtype=trainer.weight_dtype,
                device=trainer.device
            )

            # Process slices in batches using tensor slicing (no Python loop for extraction)
            for start_idx in range(0, depth, slice_batch_size):
                end_idx = min(start_idx + slice_batch_size, depth)

                # Direct tensor slicing: [C, H, W, D] -> [B, C, H, W]
                # Transpose to get slices along last dim, then slice
                slice_tensor = volume[:, :, :, start_idx:end_idx].permute(3, 0, 1, 2)

                with autocast('cuda', enabled=True, dtype=trainer.weight_dtype):
                    # Forward through model
                    recon, _ = trainer._forward_for_validation(model_to_use, slice_tensor)

                # Write directly to pre-allocated tensor
                all_recon[start_idx:end_idx] = recon

            # Reshape: [D, C, H, W] -> [1, C, D, H, W]
            recon_3d = all_recon.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, D, H, W]
            volume_3d = volume.permute(0, 3, 1, 2).unsqueeze(0)  # [1, C, D, H, W]

            # Compute 3D MS-SSIM
            msssim_3d = compute_msssim(recon_3d.float(), volume_3d.float(), spatial_dims=3)
            total_msssim_3d += msssim_3d
            n_volumes += 1

    model_to_use.train()

    if n_volumes == 0:
        return None

    avg_msssim_3d = total_msssim_3d / n_volumes

    # Note: Logging is handled by caller through _log_validation_metrics_core
    # which applies proper modality suffix for single-modality modes
    return avg_msssim_3d
