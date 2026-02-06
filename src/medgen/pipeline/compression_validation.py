"""Extracted validation helpers for BaseCompressionTrainer.

Module-level functions that implement validation logic.
Each function takes a `trainer` (BaseCompressionTrainer instance) as its first
argument and accesses trainer attributes like `trainer.device`, `trainer.cfg`, etc.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.amp import autocast

if TYPE_CHECKING:
    from medgen.evaluation import ValidationRunner
    from medgen.metrics import RegionalMetricsTracker

    from .compression_trainer import BaseCompressionTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regional tracker creation
# ---------------------------------------------------------------------------

def create_regional_tracker(
    trainer: BaseCompressionTrainer,
):
    """Create regional metrics tracker (2D or 3D).

    Args:
        trainer: BaseCompressionTrainer instance.

    Returns:
        RegionalMetricsTracker (2D) or RegionalMetricsTracker3D (3D).
    """
    from medgen.metrics import RegionalMetricsTracker

    if trainer.spatial_dims == 3:
        from medgen.metrics import RegionalMetricsTracker3D
        return RegionalMetricsTracker3D(
            volume_size=(trainer.volume_height, trainer.volume_width, trainer.volume_depth),
            fov_mm=trainer._paths_config.fov_mm,
            loss_fn='l1',
            device=trainer.device,
        )
    return RegionalMetricsTracker(
        image_size=trainer.cfg.model.image_size,
        fov_mm=trainer._paths_config.fov_mm,
        loss_fn='l1',
        device=trainer.device,
    )


# ---------------------------------------------------------------------------
# Worst batch figure
# ---------------------------------------------------------------------------

def create_worst_batch_figure(
    trainer: BaseCompressionTrainer,
    worst_batch_data: dict[str, Any],
) -> plt.Figure:
    """Create worst batch figure for TensorBoard (2D or 3D).

    Args:
        trainer: BaseCompressionTrainer instance.
        worst_batch_data: Dict with 'original', 'generated', 'loss', 'loss_breakdown'.

    Returns:
        Matplotlib figure.
    """
    figure_fn = trainer._create_worst_batch_figure_fn()
    return figure_fn(
        original=worst_batch_data['original'],
        generated=worst_batch_data['generated'],
        loss=worst_batch_data['loss'],
        loss_breakdown=worst_batch_data.get('loss_breakdown'),
    )


# ---------------------------------------------------------------------------
# Validation runner factory
# ---------------------------------------------------------------------------

def create_validation_runner(
    trainer: BaseCompressionTrainer,
) -> ValidationRunner:
    """Create ValidationRunner for this trainer (2D or 3D).

    Factory method that creates a ValidationRunner with trainer-specific
    configuration and callbacks.

    Args:
        trainer: BaseCompressionTrainer instance.

    Returns:
        Configured ValidationRunner instance.
    """
    from medgen.evaluation import ValidationConfig, ValidationRunner

    config = ValidationConfig(
        log_msssim=trainer.log_msssim,
        log_psnr=trainer.log_psnr,
        log_lpips=trainer.log_lpips,
        log_regional_losses=trainer.log_regional_losses,
        weight_dtype=trainer.weight_dtype,
        use_compile=trainer.use_compile,
        spatial_dims=trainer.spatial_dims,
    )

    regional_factory = None
    if trainer.log_regional_losses:
        regional_factory = trainer._create_regional_tracker

    return ValidationRunner(
        config=config,
        device=trainer.device,
        forward_fn=trainer._forward_for_validation,
        perceptual_loss_fn=trainer._compute_perceptual_loss,
        regional_tracker_factory=regional_factory,
        prepare_batch_fn=trainer._prepare_batch,
    )


# ---------------------------------------------------------------------------
# Main validation computation
# ---------------------------------------------------------------------------

def compute_validation_losses(
    trainer: BaseCompressionTrainer,
    epoch: int,
    log_figures: bool = True,
) -> dict[str, float]:
    """Compute validation losses using ValidationRunner (2D or 3D).

    Args:
        trainer: BaseCompressionTrainer instance.
        epoch: Current epoch number.
        log_figures: Whether to log figures (worst_batch).

    Returns:
        Dictionary of validation metrics.
    """
    if trainer.val_loader is None:
        return {}

    # Get model for evaluation
    model_to_use = trainer._get_model_for_eval()
    model_to_use.eval()

    # Run validation using extracted runner
    runner = trainer._create_validation_runner()
    result = runner.run(
        val_loader=trainer.val_loader,
        model=model_to_use,
        perceptual_weight=trainer.perceptual_weight,
        log_figures=log_figures,
    )

    model_to_use.train()

    # Compute 3D MS-SSIM on full volumes (2D trainers only, skip for seg_mode)
    # 3D trainers compute their own volumetric MS-SSIM in the runner
    # Must be done BEFORE logging so the metric gets logged with modality suffix
    if trainer.spatial_dims == 2 and not getattr(trainer, 'seg_mode', False):
        msssim_3d = trainer._compute_volume_3d_msssim(epoch, data_split='val')
        if msssim_3d is not None:
            result.metrics['msssim_3d'] = msssim_3d

    # Log to TensorBoard
    trainer._log_validation_metrics(
        epoch, result.metrics, result.worst_batch_data,
        result.regional_tracker, log_figures
    )

    return result.metrics


# ---------------------------------------------------------------------------
# Worst batch capture
# ---------------------------------------------------------------------------

def capture_worst_batch(
    trainer: BaseCompressionTrainer,
    images: torch.Tensor,
    reconstruction: torch.Tensor,
    loss: float,
    l1_loss: torch.Tensor,
    p_loss: torch.Tensor,
    reg_loss: torch.Tensor,
) -> dict[str, Any]:
    """Capture worst batch data for visualization (2D or 3D).

    Args:
        trainer: BaseCompressionTrainer instance.
        images: Original images/volumes.
        reconstruction: Reconstructed images/volumes.
        loss: Total loss value.
        l1_loss: L1 loss tensor.
        p_loss: Perceptual loss tensor.
        reg_loss: Regularization loss tensor.

    Returns:
        Dictionary with worst batch data.
    """
    # 3D: Simple capture without dict conversion
    if trainer.spatial_dims == 3:
        return {
            'original': images.cpu(),
            'generated': reconstruction.float().cpu(),
            'loss': loss,
            'loss_breakdown': {
                'L1': l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss,
                'Perc': p_loss.item() if isinstance(p_loss, torch.Tensor) else p_loss,
                'Reg': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            },
        }

    # 2D: Handle dual mode with dict conversion
    n_channels = trainer.cfg.mode.in_channels

    # Convert to dict format for dual mode
    if n_channels == 2:
        image_keys = trainer.cfg.mode.image_keys
        orig_dict = {
            image_keys[0]: images[:, 0:1].cpu(),
            image_keys[1]: images[:, 1:2].cpu(),
        }
        gen_dict = {
            image_keys[0]: reconstruction[:, 0:1].float().cpu(),
            image_keys[1]: reconstruction[:, 1:2].float().cpu(),
        }
    else:
        orig_dict = images.cpu()
        gen_dict = reconstruction.float().cpu()

    return {
        'original': orig_dict,
        'generated': gen_dict,
        'loss': loss,
        'loss_breakdown': {
            'L1': l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss,
            'Perc': p_loss.item() if isinstance(p_loss, torch.Tensor) else p_loss,
            'Reg': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
        },
    }


# ---------------------------------------------------------------------------
# Validation metrics logging (core)
# ---------------------------------------------------------------------------

def log_validation_metrics_core(
    trainer: BaseCompressionTrainer,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    """Log validation metrics with modality suffix handling.

    This method handles the core metrics logging with proper modality suffixes.
    Uses UnifiedMetrics for consistent TensorBoard paths.
    Subclasses should call this for metrics logging, then add their own
    worst batch and regional tracker handling.

    Args:
        trainer: BaseCompressionTrainer instance.
        epoch: Current epoch number.
        metrics: Dictionary of validation metrics.
    """
    if trainer.writer is None:
        return

    # Check for seg_mode - use dedicated seg validation logging
    seg_mode = getattr(trainer, 'seg_mode', False)
    if seg_mode:
        # Use dedicated seg validation logging for consistent paths
        trainer._unified_metrics.log_seg_validation(metrics, epoch)
        return

    # Get mode name for modality suffix
    mode_name = trainer.cfg.mode.name
    n_channels = trainer.cfg.mode.in_channels
    is_multi_modality = mode_name == 'multi_modality'
    is_dual = n_channels == 2 and mode_name == 'dual'
    is_seg_conditioned = mode_name.startswith('seg_conditioned')

    # For single-modality modes (not multi_modality or dual), use modality suffix
    # Multi-modality and dual modes are handled by their respective per-modality loops
    # seg_conditioned modes: no suffix needed (can distinguish by TensorBoard run color)
    if not is_multi_modality and not is_dual:
        modality_metrics = {
            'psnr': metrics.get('psnr'),
            'msssim': metrics.get('msssim'),
            'lpips': metrics.get('lpips'),
            'msssim_3d': metrics.get('msssim_3d'),
            'dice': metrics.get('dice_score'),
            'iou': metrics.get('iou'),
        }
        # No suffix for seg_conditioned modes (distinguish by TensorBoard run color)
        modality = '' if is_seg_conditioned else mode_name
        trainer._unified_metrics.log_per_modality_validation(modality_metrics, modality, epoch)

        # Also log losses without suffix (gen, l1, perc, reg)
        loss_metrics = {k: v for k, v in metrics.items()
                      if k in ('gen', 'l1', 'perc', 'reg', 'bce', 'dice', 'boundary')}
        if loss_metrics:
            trainer._log_validation_metrics_unified(epoch, loss_metrics)
    else:
        # Multi-modality/dual: log aggregate metrics, per-modality handled separately
        trainer._log_validation_metrics_unified(epoch, metrics)


# ---------------------------------------------------------------------------
# Validation metrics logging (full)
# ---------------------------------------------------------------------------

def log_validation_metrics(
    trainer: BaseCompressionTrainer,
    epoch: int,
    metrics: dict[str, float],
    worst_batch_data: dict[str, Any] | None,
    regional_tracker: RegionalMetricsTracker | None,
    log_figures: bool,
) -> None:
    """Log validation metrics to TensorBoard using unified system.

    Args:
        trainer: BaseCompressionTrainer instance.
        epoch: Current epoch number.
        metrics: Dictionary of validation metrics.
        worst_batch_data: Worst batch data for visualization.
        regional_tracker: Regional metrics tracker.
        log_figures: Whether to log figures.
    """
    if trainer.writer is None:
        return

    # Log metrics with modality suffix handling
    trainer._log_validation_metrics_core(epoch, metrics)

    # Log worst batch figure (uses unified metrics)
    if log_figures and worst_batch_data is not None:
        trainer._unified_metrics.log_worst_batch(
            original=worst_batch_data['original'],
            reconstructed=worst_batch_data['generated'],
            loss=worst_batch_data['loss'],
            epoch=epoch,
            phase='val',
        )

    # Log regional metrics with modality suffix for single-modality modes
    if regional_tracker is not None:
        mode_name = trainer.cfg.mode.name
        is_multi_modality = mode_name == 'multi_modality'
        is_dual = trainer.cfg.mode.in_channels == 2 and mode_name == 'dual'
        is_seg_conditioned = mode_name.startswith('seg_conditioned')
        # No suffix for multi_modality, dual, or seg_conditioned modes
        if is_multi_modality or is_dual or is_seg_conditioned:
            modality_override = None
        else:
            modality_override = mode_name
        trainer._unified_metrics.log_validation_regional(regional_tracker, epoch, modality_override=modality_override)


# ---------------------------------------------------------------------------
# Per-modality validation
# ---------------------------------------------------------------------------

def compute_per_modality_validation(
    trainer: BaseCompressionTrainer,
    epoch: int,
) -> None:
    """Compute per-modality validation metrics (2D or 3D).

    For 3D volumes:
    - Uses compute_lpips_3d (slice-by-slice)
    - Computes both volumetric MS-SSIM-3D and slice-wise MS-SSIM-2D

    Args:
        trainer: BaseCompressionTrainer instance.
        epoch: Current epoch number.
    """
    from medgen.metrics import (
        compute_lpips,
        compute_lpips_3d,
        compute_msssim,
        compute_msssim_2d_slicewise,
        compute_psnr,
    )

    if not trainer.per_modality_val_loaders:
        return

    model_to_use = trainer._get_model_for_eval()
    model_to_use.eval()

    # Get appropriate LPIPS function for dimensionality
    lpips_fn = trainer._create_lpips_fn()

    for modality, loader in trainer.per_modality_val_loaders.items():
        total_psnr = 0.0
        total_lpips = 0.0
        total_msssim = 0.0
        total_msssim_3d = 0.0  # Volumetric 3D MS-SSIM (only for 3D)
        n_batches = 0

        # Regional tracker for this modality
        regional_tracker = None
        if trainer.log_regional_losses:
            regional_tracker = trainer._create_regional_tracker()

        with torch.inference_mode():
            for batch in loader:
                images, mask = trainer._prepare_batch(batch)

                with autocast('cuda', enabled=True, dtype=trainer.weight_dtype):
                    reconstruction, _ = trainer._forward_for_validation(model_to_use, images)

                # Compute metrics
                if trainer.log_psnr:
                    total_psnr += compute_psnr(reconstruction, images)
                if trainer.log_lpips:
                    total_lpips += lpips_fn(
                        reconstruction.float(), images.float(), device=trainer.device
                    )
                if trainer.log_msssim:
                    if trainer.spatial_dims == 3:
                        # 3D: Compute both volumetric and slice-wise
                        total_msssim_3d += compute_msssim(
                            reconstruction.float(), images.float(), spatial_dims=3
                        )
                        total_msssim += compute_msssim_2d_slicewise(
                            reconstruction.float(), images.float()
                        )
                    else:
                        total_msssim += compute_msssim(reconstruction, images)

                # Regional tracking
                if regional_tracker is not None and mask is not None:
                    regional_tracker.update(reconstruction.float(), images.float(), mask)

                n_batches += 1

        # Log metrics using unified system
        if n_batches > 0 and trainer.writer is not None:
            # Compute 3D MS-SSIM for 2D trainers (full volume reconstruction)
            msssim_3d = None
            if trainer.spatial_dims == 2:
                if trainer.log_msssim and not getattr(trainer, 'seg_mode', False):
                    msssim_3d = trainer._compute_volume_3d_msssim(
                        epoch, data_split='val', modality_override=modality
                    )
            else:
                # 3D trainers already computed volumetric MS-SSIM above
                msssim_3d = total_msssim_3d / n_batches if trainer.log_msssim else None

            # Build metrics dict for unified logging
            modality_metrics = {
                'psnr': total_psnr / n_batches if trainer.log_psnr else None,
                'msssim': total_msssim / n_batches if trainer.log_msssim else None,
                'lpips': total_lpips / n_batches if trainer.log_lpips else None,
                'msssim_3d': msssim_3d,
            }
            trainer._unified_metrics.log_per_modality_validation(modality_metrics, modality, epoch)

            # Log regional metrics using unified system
            if regional_tracker is not None:
                trainer._unified_metrics.log_validation_regional(regional_tracker, epoch, modality_override=modality)

    model_to_use.train()


# ---------------------------------------------------------------------------
# Per-channel validation
# ---------------------------------------------------------------------------

def compute_per_channel_validation(
    trainer: BaseCompressionTrainer,
    epoch: int,
) -> None:
    """Compute per-channel validation metrics for dual mode.

    For dual mode (2 channels), computes metrics separately for each channel
    (e.g., t1_pre, t1_gd) and logs them to TensorBoard.

    Args:
        trainer: BaseCompressionTrainer instance.
        epoch: Current epoch number.
    """
    from medgen.metrics import (
        compute_lpips,
        compute_msssim,
        compute_psnr,
    )

    n_channels = trainer.cfg.mode.in_channels
    if n_channels != 2 or trainer.val_loader is None:
        return

    image_keys = trainer.cfg.mode.image_keys
    model_to_use = trainer._get_model_for_eval()
    model_to_use.eval()

    # Per-channel accumulators
    channel_metrics = {key: {'psnr': 0.0, 'lpips': 0.0, 'msssim': 0.0} for key in image_keys}
    n_batches = 0

    with torch.inference_mode():
        for batch in trainer.val_loader:
            images, _ = trainer._prepare_batch(batch)

            with autocast('cuda', enabled=True, dtype=trainer.weight_dtype):
                reconstruction, _ = trainer._forward_for_validation(model_to_use, images)

            # Compute per-channel metrics
            for i, key in enumerate(image_keys):
                img_ch = images[:, i:i+1]
                rec_ch = reconstruction[:, i:i+1]

                if trainer.log_psnr:
                    channel_metrics[key]['psnr'] += compute_psnr(rec_ch, img_ch)
                if trainer.log_lpips:
                    channel_metrics[key]['lpips'] += compute_lpips(rec_ch, img_ch, device=trainer.device)
                if trainer.log_msssim:
                    channel_metrics[key]['msssim'] += compute_msssim(rec_ch, img_ch)

            n_batches += 1

    model_to_use.train()

    # Log per-channel metrics using unified system
    if n_batches > 0 and trainer.writer is not None:
        # Build per-channel data for unified logging
        per_channel_data = {}
        for key in image_keys:
            # Compute 3D MS-SSIM for this channel if needed
            msssim_3d = None
            if trainer.log_msssim and not getattr(trainer, 'seg_mode', False):
                msssim_3d = trainer._compute_volume_3d_msssim(
                    epoch, data_split='val', modality_override=key
                )

            per_channel_data[key] = {
                'psnr': channel_metrics[key]['psnr'] if trainer.log_psnr else 0,
                'msssim': channel_metrics[key]['msssim'] if trainer.log_msssim else 0,
                'lpips': channel_metrics[key]['lpips'] if trainer.log_lpips else 0,
                'count': n_batches,
            }

            # Log 3D MS-SSIM separately per channel (not part of standard per-channel)
            if msssim_3d is not None:
                trainer._unified_metrics.log_per_modality_validation(
                    {'msssim_3d': msssim_3d}, key, epoch
                )

        trainer._unified_metrics.log_per_channel_validation(per_channel_data, epoch)


# ---------------------------------------------------------------------------
# Volume 3D MS-SSIM (2D trainers reconstruct full volumes)
# ---------------------------------------------------------------------------

def compute_volume_3d_msssim(
    trainer: BaseCompressionTrainer,
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
        trainer: BaseCompressionTrainer instance.
        epoch: Current epoch number.
        data_split: Which data split to use ('val' or 'test_new').
        modality_override: Optional specific modality to compute for
            (e.g., 'bravo', 't1_pre'). If None, uses mode from config.

    Returns:
        Average 3D MS-SSIM across all volumes, or None if unavailable.
    """
    from medgen.metrics import compute_msssim

    if not trainer.log_msssim:
        return None

    # Import here to avoid circular imports
    from medgen.data.loaders.volume_3d import create_vae_volume_validation_dataloader

    # Determine modality - use override if provided, else from config
    if modality_override is not None:
        modality = modality_override
    else:
        mode_name = trainer.cfg.mode.name
        n_channels = trainer.cfg.mode.in_channels
        # Use subdir for file loading (e.g., 'seg' instead of 'seg_conditioned')
        subdir = getattr(trainer.cfg.mode, 'subdir', mode_name)
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
