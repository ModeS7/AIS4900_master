"""Validation utilities for diffusion training.

This module provides validation loop functionality:
- Main validation loss computation with regional tracking
- Per-modality validation for multi-modality mode
- Timestep bin loss tracking
"""
import logging
from typing import TYPE_CHECKING, Any

import torch

from medgen.metrics import (
    RegionalMetricsTracker,
    compute_dice,
    compute_iou,
    compute_lpips,
    compute_lpips_3d,
    compute_msssim,
    compute_psnr,
)

if TYPE_CHECKING:
    from medgen.pipeline.trainer import DiffusionTrainer

logger = logging.getLogger(__name__)


def compute_validation_losses(
    trainer: 'DiffusionTrainer',
    epoch: int,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    """Compute losses and metrics on validation set.

    Args:
        trainer: The DiffusionTrainer instance.
        epoch: Current epoch number (for TensorBoard logging).

    Returns:
        Tuple of (metrics dict, worst_batch_data or None).
        Metrics dict contains: mse, perceptual, total, msssim, psnr.
        Worst batch data contains: original, generated, mask, timesteps, loss.
    """
    if trainer.val_loader is None:
        return {}, None

    # Save random state - validation uses torch.randn_like() which would otherwise
    # shift the global RNG and cause training to diverge across epochs
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(trainer.device) if torch.cuda.is_available() else None

    model_to_use = trainer.ema.ema_model if trainer.ema is not None else trainer.model_raw
    model_to_use.eval()

    total_mse = 0.0
    total_perc = 0.0
    total_loss = 0.0
    total_msssim = 0.0
    total_psnr = 0.0
    total_lpips = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = 0

    # Check if seg mode (use Dice/IoU instead of perceptual metrics)
    is_seg_mode = trainer.mode_name in ('seg', 'seg_conditioned')

    # Per-channel metrics for dual/multi modes
    per_channel_metrics: dict[str, dict[str, float]] = {}

    # Track worst validation batch (only from full-sized batches)
    worst_loss = 0.0
    worst_batch_data: dict[str, Any] | None = None
    min_batch_size = trainer.batch_size  # Don't track small last batches

    # Regional tracking now uses unified metrics internal tracker
    # (initialized with enable_regional=trainer.log_regional_losses)

    # Initialize timestep loss tracking for validation
    num_timestep_bins = 10
    timestep_loss_sum = torch.zeros(num_timestep_bins, device=trainer.device)
    timestep_loss_count = torch.zeros(num_timestep_bins, device=trainer.device, dtype=torch.long)

    # Mark CUDA graph step boundary to prevent tensor caching issues
    torch.compiler.cudagraph_mark_step_begin()

    with torch.no_grad():
        for batch in trainer.val_loader:
            prepared = trainer.mode.prepare_batch(batch, trainer.device)
            images = prepared['images']
            labels = prepared.get('labels')
            mode_id = prepared.get('mode_id')  # For multi-modality mode
            size_bins = prepared.get('size_bins')  # For seg_conditioned mode
            bin_maps = prepared.get('bin_maps')  # For seg_conditioned_input mode
            is_latent = prepared.get('is_latent', False)  # Latent dataloader flag
            labels_is_latent = prepared.get('labels_is_latent', False)  # Labels already encoded

            # Get current batch size
            if isinstance(images, dict):
                first_key = list(images.keys())[0]
                current_batch_size = images[first_key].shape[0]
            else:
                current_batch_size = images.shape[0]

            # Keep original pixel-space labels for regional metrics (before encoding)
            # Regional metrics need pixel-space masks to identify tumor/background regions
            # For bravo_seg_cond mode, use seg_mask from prepared batch
            labels_pixel = prepared.get('seg_mask', labels) if is_latent else labels

            # Encode to diffusion space (identity for PixelSpace)
            # Skip encoding if data is already in latent space (from latent dataloader)
            if not is_latent:
                images = trainer.space.encode_batch(images)
            if labels is not None and not labels_is_latent:
                labels = trainer.space.encode(labels)

            labels_dict = {'labels': labels, 'bin_maps': bin_maps}

            # Sample timesteps and noise
            if isinstance(images, dict):
                noise = {key: torch.randn_like(img).to(trainer.device) for key, img in images.items()}
            else:
                noise = torch.randn_like(images).to(trainer.device)

            timesteps = trainer.strategy.sample_timesteps(images)
            noisy_images = trainer.strategy.add_noise(images, noise, timesteps)

            # For ControlNet (Stage 1 or 2): use only noisy images (no concatenation)
            if trainer.use_controlnet or trainer.controlnet_stage1:
                model_input = noisy_images
            else:
                model_input = trainer.mode.format_model_input(noisy_images, labels_dict)

            # Apply mode intensity scaling for validation consistency
            if trainer.use_mode_intensity_scaling and mode_id is not None:
                model_input, _ = trainer._apply_mode_intensity_scale(model_input, mode_id)

            # Predict and compute loss
            if trainer.use_mode_embedding and mode_id is not None:
                prediction = model_to_use(model_input, timesteps, mode_id=mode_id)
            elif trainer.use_size_bin_embedding:
                prediction = model_to_use(model_input, timesteps, size_bins=size_bins)
            else:
                prediction = trainer.strategy.predict_noise_or_velocity(model_to_use, model_input, timesteps)
            mse_loss, predicted_clean = trainer.strategy.compute_loss(prediction, images, noise, noisy_images, timesteps)

            # Compute perceptual loss
            if trainer.perceptual_weight > 0:
                if trainer.space.scale_factor > 1:
                    pred_decoded = trainer.space.decode_batch(predicted_clean)
                    images_decoded = trainer.space.decode_batch(images)
                else:
                    pred_decoded = predicted_clean
                    images_decoded = images
                p_loss = trainer.perceptual_loss_fn(pred_decoded.float(), images_decoded.float())
            else:
                p_loss = torch.tensor(0.0, device=trainer.device)

            loss = mse_loss + trainer.perceptual_weight * p_loss
            loss_val = loss.item()

            total_mse += mse_loss.item()
            total_perc += p_loss.item()
            total_loss += loss_val

            # Track worst batch (only from full-sized batches)
            if loss_val > worst_loss and current_batch_size >= min_batch_size:
                worst_loss = loss_val
                if isinstance(images, dict):
                    worst_batch_data = {
                        'original': {k: v.cpu() for k, v in images.items()},
                        'generated': {k: v.cpu() for k, v in predicted_clean.items()},
                        'mask': labels.cpu() if labels is not None else None,
                        'timesteps': timesteps.cpu(),
                        'loss': loss_val,
                    }
                else:
                    worst_batch_data = {
                        'original': images.cpu(),
                        'generated': predicted_clean.cpu(),
                        'mask': labels.cpu() if labels is not None else None,
                        'timesteps': timesteps.cpu(),
                        'loss': loss_val,
                    }

            # Quality metrics (decode to pixel space for latent diffusion)
            if trainer.space.scale_factor > 1:
                metrics_pred = trainer.space.decode_batch(predicted_clean)
                metrics_gt = trainer.space.decode_batch(images)
            else:
                metrics_pred = predicted_clean
                metrics_gt = images

            if isinstance(metrics_pred, dict):
                # Dual/multi mode: compute per-channel AND average metrics
                keys = list(metrics_pred.keys())
                channel_msssim = {}
                channel_psnr = {}
                channel_lpips = {}

                for key in keys:
                    channel_msssim[key] = compute_msssim(metrics_pred[key], metrics_gt[key])
                    channel_psnr[key] = compute_psnr(metrics_pred[key], metrics_gt[key])
                    if trainer.log_lpips:
                        channel_lpips[key] = compute_lpips(metrics_pred[key], metrics_gt[key], trainer.device)

                    # Accumulate per-channel metrics
                    if key not in per_channel_metrics:
                        per_channel_metrics[key] = {'msssim': 0.0, 'psnr': 0.0, 'lpips': 0.0, 'count': 0}
                    per_channel_metrics[key]['msssim'] += channel_msssim[key]
                    per_channel_metrics[key]['psnr'] += channel_psnr[key]
                    if trainer.log_lpips:
                        per_channel_metrics[key]['lpips'] += channel_lpips[key]
                    per_channel_metrics[key]['count'] += 1

                # Average across channels for combined metrics
                msssim_val = sum(channel_msssim.values()) / len(keys)
                psnr_val = sum(channel_psnr.values()) / len(keys)
                lpips_val = sum(channel_lpips.values()) / len(keys) if trainer.log_lpips else 0.0
            else:
                # Use dimension-appropriate metric functions
                msssim_val = compute_msssim(metrics_pred, metrics_gt, spatial_dims=trainer.spatial_dims)
                psnr_val = compute_psnr(metrics_pred, metrics_gt)
                if trainer.log_lpips:
                    if trainer.spatial_dims == 3:
                        # 3D: use center-slice LPIPS (2.5D approach)
                        lpips_val = compute_lpips_3d(metrics_pred, metrics_gt, trainer.device)
                    else:
                        lpips_val = compute_lpips(metrics_pred, metrics_gt, trainer.device)
                else:
                    lpips_val = 0.0

            total_msssim += msssim_val
            total_psnr += psnr_val
            total_lpips += lpips_val

            # Compute Dice/IoU for seg modes
            # For diffusion output: predicted_clean is already in [0, 1] range (not logits)
            if is_seg_mode:
                dice_val = compute_dice(metrics_pred, metrics_gt, apply_sigmoid=False)
                iou_val = compute_iou(metrics_pred, metrics_gt, apply_sigmoid=False)
                total_dice += dice_val
                total_iou += iou_val

            n_batches += 1

            # Regional tracking via unified metrics (tumor vs background)
            # IMPORTANT: Use DECODED tensors (metrics_pred, metrics_gt) and PIXEL-SPACE labels
            # This ensures regional quality metrics are computed in pixel space for consistency
            # with pixel-space diffusion (same PSNR/MSE interpretation regardless of latent vs pixel)
            if trainer.log_regional_losses and labels_pixel is not None:
                trainer._unified_metrics.update_regional(metrics_pred, metrics_gt, labels_pixel)

            # Timestep loss tracking (per-bin velocity/noise prediction MSE)
            # Uses the actual training target (velocity for RFlow, noise for DDPM)
            # Note: This is computed in TRAINING space (latent for latent diffusion)
            # because we're measuring model prediction quality, not reconstruction quality
            if trainer.log_timestep_losses:
                # Compute target based on strategy (velocity for RFlow, noise for DDPM)
                if isinstance(images, dict):
                    keys = list(images.keys())
                    if trainer.strategy_name == 'rflow':
                        target = torch.cat([images[k] - noise[k] for k in keys], dim=1)
                    else:
                        target = torch.cat([noise[k] for k in keys], dim=1)
                else:
                    target = images - noise if trainer.strategy_name == 'rflow' else noise

                # Mean over all non-batch dims: 2D [B,C,H,W] -> (1,2,3), 3D [B,C,D,H,W] -> (1,2,3,4)
                mse_per_sample = ((prediction.float() - target.float()) ** 2).flatten(1).mean(1)
                bin_size = trainer.num_timesteps // num_timestep_bins
                bin_indices = (timesteps // bin_size).clamp(max=num_timestep_bins - 1).long()
                timestep_loss_sum.scatter_add_(0, bin_indices, mse_per_sample)
                timestep_loss_count.scatter_add_(0, bin_indices, torch.ones_like(bin_indices))

                # Timestep-region tracking (for heatmap) - split by tumor/background
                # For latent diffusion: compute error in PIXEL space for consistent interpretation
                # Uses decoded predictions and pixel-space labels for masking
                if trainer.log_timestep_region_losses and labels_pixel is not None:
                    # Compute pixel-space error map for region tracking
                    if trainer.space.scale_factor > 1:
                        # Latent diffusion: decode prediction and compute pixel-space target
                        # For pixel-space region analysis, we compare decoded x0 prediction vs ground truth
                        error_for_region = ((metrics_pred.float() - metrics_gt.float()) ** 2)
                        region_labels = labels_pixel
                    else:
                        # Pixel space: use training-space error
                        error_for_region = ((prediction.float() - target.float()) ** 2)
                        region_labels = labels

                    if trainer.spatial_dims == 3:
                        # 3D: extract center slice for efficiency
                        center_idx = error_for_region.shape[2] // 2
                        error_map = error_for_region[:, :, center_idx, :, :].mean(dim=1)  # [B, H, W]
                        mask = region_labels[:, 0, center_idx, :, :] > 0.5  # [B, H, W]
                    else:
                        # 2D: use full images
                        error_map = error_for_region.mean(dim=1)  # [B, H, W]
                        mask = region_labels[:, 0] > 0.5  # [B, H, W]

                    for i in range(current_batch_size):
                        t_norm = timesteps[i].item() / trainer.num_timesteps
                        sample_error = error_map[i]  # [H, W]
                        sample_mask = mask[i]  # [H, W]
                        tumor_px = sample_mask.sum().item()
                        bg_px = (~sample_mask).sum().item()
                        tumor_loss = (sample_error * sample_mask.float()).sum().item() if tumor_px > 0 else 0.0
                        bg_loss = (sample_error * (~sample_mask).float()).sum().item() if bg_px > 0 else 0.0
                        trainer._unified_metrics.update_timestep_region_loss(
                            t_norm, tumor_loss, bg_loss, int(tumor_px), int(bg_px)
                        )

    # Restore model state and RNG using try/finally for robustness
    try:
        model_to_use.train()

        # Handle empty validation set
        if n_batches == 0:
            logger.warning("Validation set is empty, skipping metrics")
            return {}, None

        metrics = {
            'mse': total_mse / n_batches,
            'perceptual': total_perc / n_batches,
            'total': total_loss / n_batches,
            'msssim': total_msssim / n_batches,
            'psnr': total_psnr / n_batches,
        }
        if trainer.log_lpips:
            metrics['lpips'] = total_lpips / n_batches

        # Add Dice/IoU for seg modes
        if is_seg_mode:
            metrics['dice'] = total_dice / n_batches
            metrics['iou'] = total_iou / n_batches

        # Log to TensorBoard using unified system
        if trainer.writer is not None and trainer._unified_metrics is not None:
            # Update unified metrics with validation values
            trainer._unified_metrics.update_loss('MSE', metrics['mse'], phase='val')
            if trainer.perceptual_weight > 0:
                trainer._unified_metrics.update_loss('Total', metrics['total'], phase='val')
                trainer._unified_metrics.update_loss('Perceptual', metrics['perceptual'], phase='val')

            # Compute 3D MS-SSIM first so we can include it in metrics
            msssim_3d = None
            if trainer.log_msssim:
                msssim_3d = trainer._compute_volume_3d_msssim(epoch, data_split='val')
                if msssim_3d is not None:
                    metrics['msssim_3d'] = msssim_3d

            # Update quality metrics using unified system
            trainer._unified_metrics.update_validation_batch(
                psnr=metrics['psnr'],
                msssim=metrics['msssim'],
                lpips=metrics.get('lpips'),
                msssim_3d=msssim_3d,
                dice=metrics.get('dice'),
                iou=metrics.get('iou'),
            )

            # Log per-timestep validation losses using unified system
            # NOTE: Must happen BEFORE log_validation() so timesteps are included
            if trainer.log_timestep_losses:
                counts = timestep_loss_count.cpu()
                sums = timestep_loss_sum.cpu()
                for i in range(num_timestep_bins):
                    if counts[i] > 0:
                        avg_loss = (sums[i] / counts[i]).item()
                        t_normalized = (i + 0.5) / num_timestep_bins  # Center of bin
                        trainer._unified_metrics.update_timestep_loss(t_normalized, avg_loss)

            # Log validation metrics (now includes timesteps)
            trainer._unified_metrics.log_validation(epoch)

            # Log per-channel metrics for dual/multi modes (via unified metrics)
            if per_channel_metrics:
                trainer._unified_metrics.log_per_channel_validation(per_channel_metrics, epoch)

            # Regional metrics are now logged via log_validation() using internal tracker

            # Log timestep-region heatmap on figure epochs
            if (epoch + 1) % trainer.figure_interval == 0 and trainer.log_timestep_region_losses:
                trainer._unified_metrics.log_timestep_region_heatmap(epoch)

            # Compute generation quality metrics (KID, CMMD)
            if trainer._gen_metrics is not None:
                # Clear fragmented memory before generation (prevents OOM from reserved-but-unused memory)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                try:
                    gen_model = trainer.ema.ema_model if trainer.ema is not None else trainer.model_raw
                    gen_model.eval()

                    # Quick metrics every epoch
                    gen_results = trainer._gen_metrics.compute_epoch_metrics(
                        gen_model, trainer.strategy, trainer.mode
                    )
                    trainer._unified_metrics.log_generation(epoch, gen_results)

                    # Extended metrics at figure_interval
                    if (epoch + 1) % trainer.figure_interval == 0:
                        extended_results = trainer._gen_metrics.compute_extended_metrics(
                            gen_model, trainer.strategy, trainer.mode
                        )
                        trainer._unified_metrics.log_generation(epoch, extended_results)

                    gen_model.train()
                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(f"Generation metrics skipped due to OOM: {e}")
                    torch.cuda.empty_cache()
                except (RuntimeError, ValueError) as e:
                    logger.exception(f"Generation metrics computation failed at epoch {epoch}: {e}")
                finally:
                    # Always clean up after generation metrics to prevent memory buildup
                    torch.cuda.empty_cache()

            # Record epoch history for JSON export (before reset)
            trainer._unified_metrics.record_epoch_history(epoch)

            # Reset validation metrics for next epoch
            trainer._unified_metrics.reset_validation()

        return metrics, worst_batch_data

    finally:
        # Ensure RNG state is ALWAYS restored, even if an exception occurred
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, trainer.device)


def compute_per_modality_validation(
    trainer: 'DiffusionTrainer',
    epoch: int,
) -> None:
    """Compute and log validation metrics for each modality separately.

    For multi-modality training, this logs PSNR, LPIPS, MS-SSIM and regional
    metrics for each modality (bravo, t1_pre, t1_gd) to compare with
    single-modality experiments.

    Args:
        trainer: The DiffusionTrainer instance.
        epoch: Current epoch number.
    """
    if not hasattr(trainer, 'per_modality_val_loaders') or not trainer.per_modality_val_loaders:
        return

    # Save random state (same pattern as compute_validation_losses)
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(trainer.device) if torch.cuda.is_available() else None

    model_to_use = trainer.ema.ema_model if trainer.ema is not None else trainer.model_raw
    try:
        model_to_use.eval()

        for modality, loader in trainer.per_modality_val_loaders.items():
            total_psnr = 0.0
            total_lpips = 0.0
            total_msssim = 0.0
            n_batches = 0

            # Initialize regional tracker for this modality
            regional_tracker = None
            if trainer.log_regional_losses:
                regional_tracker = RegionalMetricsTracker(
                    image_size=trainer.image_size,
                    fov_mm=trainer._paths_config.fov_mm,
                    loss_fn='mse',
                    device=trainer.device,
                )

            with torch.no_grad():
                for batch in loader:
                    prepared = trainer.mode.prepare_batch(batch, trainer.device)
                    images = prepared['images']
                    labels = prepared.get('labels')
                    size_bins = prepared.get('size_bins')
                    bin_maps = prepared.get('bin_maps')

                    # Encode to diffusion space (identity for PixelSpace)
                    images = trainer.space.encode_batch(images)
                    if labels is not None:
                        labels = trainer.space.encode(labels)

                    labels_dict = {'labels': labels, 'bin_maps': bin_maps}

                    # Sample timesteps and noise
                    noise = torch.randn_like(images).to(trainer.device)
                    timesteps = trainer.strategy.sample_timesteps(images)
                    noisy_images = trainer.strategy.add_noise(images, noise, timesteps)
                    model_input = trainer.mode.format_model_input(noisy_images, labels_dict)

                    # Predict
                    if trainer.use_size_bin_embedding:
                        prediction = model_to_use(model_input, timesteps, size_bins=size_bins)
                    else:
                        prediction = trainer.strategy.predict_noise_or_velocity(model_to_use, model_input, timesteps)
                    _, predicted_clean = trainer.strategy.compute_loss(prediction, images, noise, noisy_images, timesteps)

                    # Compute metrics
                    total_psnr += compute_psnr(predicted_clean, images)
                    if trainer.log_lpips:
                        total_lpips += compute_lpips(predicted_clean, images, device=trainer.device)
                    total_msssim += compute_msssim(predicted_clean, images)

                    # Regional tracking (tumor vs background)
                    if regional_tracker is not None and labels is not None:
                        regional_tracker.update(predicted_clean, images, labels)

                    n_batches += 1

            # Compute averages and log per-modality metrics (via unified metrics)
            if n_batches > 0:
                modality_metrics = {
                    'psnr': total_psnr / n_batches,
                    'msssim': total_msssim / n_batches,
                }

                if trainer.log_lpips:
                    modality_metrics['lpips'] = total_lpips / n_batches

                # Compute 3D MS-SSIM for this modality
                if trainer.log_msssim:
                    msssim_3d = trainer._compute_volume_3d_msssim(
                        epoch, data_split='val', modality_override=modality
                    )
                    if msssim_3d is not None:
                        modality_metrics['msssim_3d'] = msssim_3d

                # Log via unified metrics
                trainer._unified_metrics.log_per_modality_validation(modality_metrics, modality, epoch)

                # Log regional metrics for this modality via unified system
                if regional_tracker is not None:
                    trainer._unified_metrics.log_validation_regional(
                        regional_tracker, epoch, modality_override=modality
                    )
    finally:
        model_to_use.train()

        # Restore RNG state
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, trainer.device)
