"""3D compression test evaluator and factory function.

Contains the Compression3DTestEvaluator class for volumetric model
evaluation and the create_compression_test_evaluator factory.

Moved from evaluation.py during file split.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .evaluation import BaseTestEvaluator, CompressionTestEvaluator, MetricsConfig
from .evaluation_logging import log_test_per_modality

if TYPE_CHECKING:
    from medgen.metrics.unified import UnifiedMetrics
    from medgen.pipeline.compression_trainer import BaseCompressionTrainer

logger = logging.getLogger(__name__)


class Compression3DTestEvaluator(BaseTestEvaluator):
    """Test evaluator for 3D volumetric compression models (VAE-3D, VQVAE-3D).

    Computes: L1, MS-SSIM-3D (volumetric), MS-SSIM (2D slicewise), PSNR, LPIPS-3D.
    For seg_mode: Dice, IoU instead of image metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        save_dir: str,
        forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
        weight_dtype: torch.dtype = torch.bfloat16,
        writer: SummaryWriter | None = None,
        metrics_config: MetricsConfig | None = None,
        is_cluster: bool = False,
        regional_tracker_factory: Callable[[], Any] | None = None,
        worst_batch_figure_fn: Callable[[dict[str, Any]], Any] | None = None,
        image_keys: list[str] | None = None,
        seg_loss_fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict[str, float]]] | None = None,
        modality_name: str | None = None,
        unified_metrics: "UnifiedMetrics | None" = None,
    ):
        """Initialize 3D compression evaluator.

        Args:
            model: Model to evaluate.
            device: Device for evaluation.
            save_dir: Directory for saving results.
            forward_fn: Callable that takes (model, volumes) and returns reconstruction.
            weight_dtype: Data type for autocast.
            writer: Optional TensorBoard writer.
            metrics_config: Which metrics to compute.
            is_cluster: If True, disable tqdm progress bar.
            regional_tracker_factory: Optional factory to create 3D regional tracker.
            worst_batch_figure_fn: Optional callable to create 3D worst batch figure.
            image_keys: Optional list of channel names for per-channel metrics.
            seg_loss_fn: Optional segmentation loss function for seg_mode.
            modality_name: Optional modality name for single-modality suffix (e.g., 'bravo', 'seg').
            unified_metrics: Optional UnifiedMetrics instance for centralized logging.
        """
        super().__init__(
            model, device, save_dir, writer, metrics_config, is_cluster,
            worst_batch_figure_fn=worst_batch_figure_fn,
            modality_name=modality_name,
            unified_metrics=unified_metrics,
        )
        self.forward_fn = forward_fn
        self.weight_dtype = weight_dtype
        self.regional_tracker_factory = regional_tracker_factory
        self.image_keys = image_keys
        self.seg_loss_fn = seg_loss_fn
        self._regional_tracker: Any | None = None
        self._per_channel_metrics: dict[str, dict[str, float]] = {}
        self._per_channel_count: int = 0

    def _init_accumulators(self) -> dict[str, float]:
        """Initialize accumulators including 3D MS-SSIM."""
        accumulators = super()._init_accumulators()
        if self.metrics_config.compute_msssim_3d:
            accumulators['msssim_3d'] = 0.0
        return accumulators

    def evaluate(
        self,
        test_loader: DataLoader,
        checkpoint_name: str | None = None,
        get_eval_model: Callable[[], nn.Module] | None = None,
    ) -> dict[str, float]:
        """Run test evaluation with optional regional tracking."""
        if self.regional_tracker_factory is not None:
            self._regional_tracker = self.regional_tracker_factory()

        # Reset per-channel accumulators
        self._per_channel_metrics = {}
        self._per_channel_count = 0

        result = super().evaluate(test_loader, checkpoint_name, get_eval_model)

        label = checkpoint_name or "current"

        # Log regional metrics with modality suffix for single-modality modes
        if self._regional_tracker is not None and (self.writer is not None or self._unified_metrics is not None):
            is_single_modality = (
                self.modality_name is not None
                and self.modality_name != ''
                and self.modality_name not in ('multi_modality', 'dual')
            )
            if is_single_modality:
                regional_prefix = f'test_{label}_regional_{self.modality_name}'
            else:
                regional_prefix = f'test_{label}_regional'
            # Use UnifiedMetrics if available (preferred)
            if self._unified_metrics is not None:
                self._unified_metrics.log_test_regional(
                    self._regional_tracker, prefix=regional_prefix
                )
            else:
                self._regional_tracker.log_to_tensorboard(
                    self.writer, 0, prefix=regional_prefix
                )

        # Log per-channel metrics to TensorBoard using unified helper
        if self._per_channel_count > 0 and self.writer is not None:
            for key, channel_metrics in self._per_channel_metrics.items():
                modality_metrics = {}
                if self.metrics_config.compute_msssim:
                    modality_metrics['msssim'] = channel_metrics['msssim'] / self._per_channel_count
                if self.metrics_config.compute_msssim_3d:
                    modality_metrics['msssim_3d'] = channel_metrics['msssim_3d'] / self._per_channel_count
                if self.metrics_config.compute_psnr:
                    modality_metrics['psnr'] = channel_metrics['psnr'] / self._per_channel_count
                if self.metrics_config.compute_lpips:
                    modality_metrics['lpips'] = channel_metrics['lpips'] / self._per_channel_count
                log_test_per_modality(self.writer, modality_metrics, f'test_{label}', key)

        return result

    def _get_batch_size(self, batch_data: tuple[torch.Tensor, torch.Tensor | None]) -> int:
        """Get batch size from prepared batch."""
        images, _ = batch_data
        return images.shape[0]

    def _compute_batch_metrics(
        self,
        model: nn.Module,
        batch_data: tuple[torch.Tensor, torch.Tensor | None],
    ) -> dict[str, float]:
        """Compute metrics for a single 3D batch."""
        from torch.amp import autocast

        from medgen.metrics import (
            compute_dice,
            compute_iou,
            compute_lpips_3d,
            compute_msssim,
            compute_msssim_2d_slicewise,
            compute_psnr,
        )

        images, mask = batch_data

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            reconstructed = self.forward_fn(model, images)

        metrics = {}

        # Segmentation mode: compute Dice/IoU instead of image metrics
        if self.metrics_config.seg_mode:
            metrics['dice'] = compute_dice(reconstructed, images, apply_sigmoid=True)
            metrics['iou'] = compute_iou(reconstructed, images, apply_sigmoid=True)
            # Compute seg loss breakdown if available
            if self.seg_loss_fn is not None:
                seg_loss, seg_breakdown = self.seg_loss_fn(reconstructed, images)
                metrics['bce'] = seg_breakdown.get('bce', 0.0)
                metrics['boundary'] = seg_breakdown.get('boundary', 0.0)
                metrics['gen'] = seg_loss.item()
        else:
            # Standard image metrics
            # L1 loss
            if self.metrics_config.compute_l1:
                metrics['l1'] = torch.abs(reconstructed - images).mean().item()

            # MS-SSIM (2D slicewise for comparison with 2D trainers)
            if self.metrics_config.compute_msssim:
                metrics['msssim'] = compute_msssim_2d_slicewise(
                    reconstructed.float(), images.float()
                )

            # MS-SSIM-3D (volumetric)
            if self.metrics_config.compute_msssim_3d:
                metrics['msssim_3d'] = compute_msssim(
                    reconstructed.float(), images.float(), spatial_dims=3
                )

            # PSNR
            if self.metrics_config.compute_psnr:
                metrics['psnr'] = compute_psnr(reconstructed, images)

            # LPIPS (3D batched)
            if self.metrics_config.compute_lpips:
                metrics['lpips'] = compute_lpips_3d(
                    reconstructed.float(), images.float(), device=self.device
                )

            # Per-channel metrics for multi-modality mode
            if self.image_keys is not None and len(self.image_keys) > 1:
                n_channels = images.shape[1]
                if n_channels == len(self.image_keys):
                    for i, key in enumerate(self.image_keys):
                        img_ch = images[:, i:i+1]
                        rec_ch = reconstructed[:, i:i+1]

                        if key not in self._per_channel_metrics:
                            self._per_channel_metrics[key] = {'msssim': 0.0, 'msssim_3d': 0.0, 'psnr': 0.0, 'lpips': 0.0}

                        if self.metrics_config.compute_msssim:
                            self._per_channel_metrics[key]['msssim'] += compute_msssim_2d_slicewise(rec_ch.float(), img_ch.float())
                        if self.metrics_config.compute_msssim_3d:
                            self._per_channel_metrics[key]['msssim_3d'] += compute_msssim(rec_ch.float(), img_ch.float(), spatial_dims=3)
                        if self.metrics_config.compute_psnr:
                            self._per_channel_metrics[key]['psnr'] += compute_psnr(rec_ch, img_ch)
                        if self.metrics_config.compute_lpips:
                            self._per_channel_metrics[key]['lpips'] += compute_lpips_3d(rec_ch.float(), img_ch.float(), device=self.device)

                    self._per_channel_count += 1

        # Regional tracking
        if self._regional_tracker is not None:
            if self.metrics_config.seg_mode:
                # Seg mode: images IS the target mask, use 2-arg update
                self._regional_tracker.update(reconstructed, images, apply_sigmoid=True)
            elif mask is not None:
                # Standard mode: separate mask for regional tracking
                self._regional_tracker.update(reconstructed.float(), images.float(), mask)

        # Store for worst batch capture
        self._current_batch = {
            'images': images,
            'reconstructed': reconstructed,
        }

        return metrics

    def _capture_worst_batch(
        self,
        batch_data: tuple[torch.Tensor, torch.Tensor | None],
        batch_metrics: dict[str, float],
    ) -> dict[str, Any] | None:
        """Capture worst batch data for 3D visualization."""
        if not hasattr(self, '_current_batch'):
            return None

        # Build loss breakdown based on mode
        if self.metrics_config.seg_mode:
            loss = 1.0 - batch_metrics.get('dice', 1.0)  # 1-Dice as loss
            loss_breakdown = {
                'Dice': batch_metrics.get('dice', 0.0),
                'IoU': batch_metrics.get('iou', 0.0),
            }
        else:
            loss = batch_metrics.get('l1', 0.0)
            loss_breakdown = {
                'L1': batch_metrics.get('l1', 0.0),
            }

        return {
            'original': self._current_batch['images'].cpu(),
            'generated': self._current_batch['reconstructed'].float().cpu(),
            'loss': loss,
            'loss_breakdown': loss_breakdown,
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_compression_test_evaluator(
    trainer: 'BaseCompressionTrainer',
) -> 'CompressionTestEvaluator | Compression3DTestEvaluator':
    """Create test evaluator for compression trainer.

    Factory that creates appropriate evaluator based on trainer.spatial_dims.
    Creates a CompressionTestEvaluator (2D) or Compression3DTestEvaluator (3D)
    with trainer-specific callbacks.

    Args:
        trainer: Compression trainer instance.

    Returns:
        CompressionTestEvaluator for 2D, Compression3DTestEvaluator for 3D.
    """
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
