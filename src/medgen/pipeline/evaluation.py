"""Test evaluation utilities for diffusion training.

This module provides:
- Full test set evaluation with metrics
- 3D MS-SSIM computation for 2D and 3D models
- Test reconstruction figure creation
"""
import json
import logging
import os
from typing import TYPE_CHECKING, Any

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from medgen.metrics import (
    compute_msssim,
    compute_psnr,
    create_reconstruction_figure,
)
from medgen.metrics.dispatch import compute_lpips_dispatch, compute_msssim_dispatch

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    from medgen.pipeline.trainer import DiffusionTrainer

logger = logging.getLogger(__name__)


def evaluate_test_set(
    trainer: 'DiffusionTrainer',
    test_loader: DataLoader,
    checkpoint_name: str | None = None,
) -> dict[str, float]:
    """Evaluate diffusion model on test set.

    Runs inference on the entire test set and computes metrics:
    - MSE (prediction error)
    - MS-SSIM (Multi-Scale Structural Similarity)
    - PSNR (Peak Signal-to-Noise Ratio)

    Results are saved to test_results_{checkpoint_name}.json and logged to TensorBoard.

    Args:
        trainer: The DiffusionTrainer instance.
        test_loader: DataLoader for test set.
        checkpoint_name: Name of checkpoint to load ("best", "latest", or None
            for current model state).

    Returns:
        Dict with test metrics: 'mse', 'msssim', 'psnr', 'n_samples'.
    """
    if not trainer.is_main_process:
        return {}

    # Load checkpoint if specified
    if checkpoint_name is not None:
        checkpoint_path = os.path.join(trainer.save_dir, f"checkpoint_{checkpoint_name}.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
            trainer.model_raw.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded {checkpoint_name} checkpoint for test evaluation")

            # Load EMA state if available and EMA is configured
            if trainer.ema is not None and 'ema_state_dict' in checkpoint:
                trainer.ema.load_state_dict(checkpoint['ema_state_dict'])
                logger.info("Loaded EMA state from checkpoint")
        else:
            logger.warning(f"Checkpoint {checkpoint_path} not found, using current model state")
            checkpoint_name = "current"

    label = checkpoint_name or "current"
    logger.info("=" * 60)
    logger.info(f"EVALUATING ON TEST SET ({label.upper()} MODEL)")
    logger.info("=" * 60)

    # Use EMA model if available
    if trainer.ema is not None:
        model_to_use = trainer.ema.ema_model
        logger.info("Using EMA model for evaluation")
    else:
        model_to_use = trainer.model_raw
    model_to_use.eval()

    # Accumulators for metrics
    total_mse = 0.0
    total_msssim = 0.0
    total_psnr = 0.0
    total_lpips = 0.0
    n_batches = 0
    n_samples = 0

    # Track worst batch by loss
    worst_batch_loss = 0.0
    worst_batch_data: dict[str, Any] | None = None

    # Timestep bin accumulators (10 bins: 0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
    num_timestep_bins = 10
    timestep_loss_sum = torch.zeros(num_timestep_bins, device=trainer.device)
    timestep_loss_count = torch.zeros(num_timestep_bins, device=trainer.device, dtype=torch.long)

    # Initialize regional tracker for test (if enabled)
    regional_tracker = None
    if trainer.log_regional_losses:
        regional_tracker = trainer._create_regional_tracker(loss_fn='mse')

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test evaluation", ncols=100, disable=not trainer.verbose):
            prepared = trainer.mode.prepare_batch(batch, trainer.device)
            images = prepared['images']
            labels = prepared.get('labels')
            mode_id = prepared.get('mode_id')
            is_latent = prepared.get('is_latent', False)
            labels_is_latent = prepared.get('labels_is_latent', False)
            batch_size = images[list(images.keys())[0]].shape[0] if isinstance(images, dict) else images.shape[0]

            # Keep pixel-space originals for quality metrics
            images_pixel = images if not is_latent else None

            # Encode to diffusion space (skip if already latent)
            if not is_latent:
                images = trainer.space.encode_batch(images)
            if labels is not None and not labels_is_latent:
                labels = trainer.space.encode(labels)

            labels_dict = {'labels': labels}

            # Sample timesteps and noise
            if isinstance(images, dict):
                noise = {key: torch.randn_like(img).to(trainer.device) for key, img in images.items()}
            else:
                noise = torch.randn_like(images).to(trainer.device)

            timesteps = trainer.strategy.sample_timesteps(images)
            noisy_images = trainer.strategy.add_noise(images, noise, timesteps)

            # For ControlNet (Stage 1 or 2): use only noisy images
            if trainer.use_controlnet or trainer.controlnet_stage1:
                model_input = noisy_images
            else:
                model_input = trainer.mode.format_model_input(noisy_images, labels_dict)

            # Apply mode intensity scaling for test consistency
            if trainer.use_mode_intensity_scaling and mode_id is not None:
                model_input, _ = trainer._apply_mode_intensity_scale(model_input, mode_id)

            with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                if trainer.use_mode_embedding and mode_id is not None:
                    prediction = model_to_use(model_input, timesteps, mode_id=mode_id)
                else:
                    prediction = trainer.strategy.predict_noise_or_velocity(model_to_use, model_input, timesteps)
                mse_loss, predicted_clean = trainer.strategy.compute_loss(prediction, images, noise, noisy_images, timesteps)

            # Compute metrics
            loss_val = mse_loss.item()
            total_mse += loss_val

            # Track per-timestep-bin losses
            with torch.no_grad():
                if isinstance(predicted_clean, dict):
                    keys = list(predicted_clean.keys())
                    mse_per_sample = (
                        (predicted_clean[keys[0]] - images[keys[0]]).pow(2).flatten(1).mean(1) +
                        (predicted_clean[keys[1]] - images[keys[1]]).pow(2).flatten(1).mean(1)
                    ) / 2
                else:
                    mse_per_sample = (predicted_clean - images).pow(2).flatten(1).mean(1)
                bin_size = trainer.num_timesteps // num_timestep_bins
                bin_indices = (timesteps // bin_size).clamp(max=num_timestep_bins - 1).long()
                timestep_loss_sum.scatter_add_(0, bin_indices, mse_per_sample)
                timestep_loss_count.scatter_add_(0, bin_indices, torch.ones_like(bin_indices))

            # Decode prediction to pixel space, compare against original pixels
            if trainer.space.needs_decode:
                metrics_pred = trainer.space.decode_batch(predicted_clean)
            else:
                metrics_pred = predicted_clean
            if images_pixel is not None:
                metrics_gt = images_pixel
            elif trainer.space.needs_decode:
                metrics_gt = trainer.space.decode_batch(images)
            else:
                metrics_gt = images

            if isinstance(metrics_pred, dict):
                keys = list(metrics_pred.keys())
                msssim_val = (compute_msssim_dispatch(metrics_pred[keys[0]], metrics_gt[keys[0]], trainer.spatial_dims) +
                              compute_msssim_dispatch(metrics_pred[keys[1]], metrics_gt[keys[1]], trainer.spatial_dims)) / 2
                psnr_val = (compute_psnr(metrics_pred[keys[0]], metrics_gt[keys[0]]) +
                            compute_psnr(metrics_pred[keys[1]], metrics_gt[keys[1]])) / 2
                if trainer.log_lpips:
                    lpips_val = (compute_lpips_dispatch(metrics_pred[keys[0]], metrics_gt[keys[0]], trainer.spatial_dims, trainer.device) +
                                 compute_lpips_dispatch(metrics_pred[keys[1]], metrics_gt[keys[1]], trainer.spatial_dims, trainer.device)) / 2
                else:
                    lpips_val = 0.0
            else:
                msssim_val = compute_msssim_dispatch(metrics_pred, metrics_gt, trainer.spatial_dims)
                psnr_val = compute_psnr(metrics_pred, metrics_gt)
                if trainer.log_lpips:
                    lpips_val = compute_lpips_dispatch(metrics_pred, metrics_gt, trainer.spatial_dims, trainer.device)
                else:
                    lpips_val = 0.0

            # Regional metrics tracking (tumor vs background)
            if regional_tracker is not None and labels is not None:
                # Decode labels to pixel space if needed
                labels_pixel = trainer.space.decode(labels) if trainer.space.needs_decode else labels
                regional_tracker.update(metrics_pred, metrics_gt, labels_pixel)

            # Track worst batch
            if loss_val > worst_batch_loss and batch_size >= trainer.batch_size:
                worst_batch_loss = loss_val
                if isinstance(images, dict):
                    worst_batch_data = {
                        'original': {k: v.cpu() for k, v in images.items()},
                        'generated': {k: v.cpu() for k, v in predicted_clean.items()},
                        'timesteps': timesteps.cpu(),
                        'loss': loss_val,
                    }
                else:
                    worst_batch_data = {
                        'original': images.cpu(),
                        'generated': predicted_clean.cpu(),
                        'timesteps': timesteps.cpu(),
                        'loss': loss_val,
                    }

            total_msssim += msssim_val
            total_psnr += psnr_val
            total_lpips += lpips_val
            n_batches += 1
            n_samples += batch_size

    # Restore model to training mode (using try/finally for robustness)
    try:
        model_to_use.train()

        # Handle empty test set
        if n_batches == 0:
            logger.warning(f"Test set ({label}) is empty, skipping evaluation")
            return {}

        # Compute averages
        metrics = {
            'mse': total_mse / n_batches,
            'msssim': total_msssim / n_batches,
            'psnr': total_psnr / n_batches,
            'n_samples': n_samples,
        }
        if trainer.log_lpips:
            metrics['lpips'] = total_lpips / n_batches

        # Log results
        logger.info(f"Test Results - {label} ({n_samples} samples):")
        logger.info(f"  MSE:     {metrics['mse']:.6f}")
        logger.info(f"  MS-SSIM: {metrics['msssim']:.4f}")
        logger.info(f"  PSNR:    {metrics['psnr']:.2f} dB")
        if 'lpips' in metrics:
            logger.info(f"  LPIPS:   {metrics['lpips']:.4f}")

        # Save results to JSON
        results_path = os.path.join(trainer.save_dir, f'test_results_{label}.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Test results saved to: {results_path}")

        # Log to TensorBoard using unified system
        tb_prefix = f'test_{label}'
        if trainer.writer is not None and trainer._unified_metrics is not None:
            # Compute volume 3D MS-SSIM first so it can be included in test_metrics
            msssim_3d = None
            if trainer.log_msssim:
                msssim_3d = compute_volume_3d_msssim(trainer, 0, data_split='test_new')
                if msssim_3d is not None:
                    metrics['msssim_3d'] = msssim_3d

            # Build test metrics dict for unified logging
            test_metrics = {
                'PSNR': metrics['psnr'],
                'MS-SSIM': metrics['msssim'],
                'MSE': metrics['mse'],
            }
            if 'lpips' in metrics:
                test_metrics['LPIPS'] = metrics['lpips']
            if msssim_3d is not None:
                test_metrics['MS-SSIM-3D'] = msssim_3d

            trainer._unified_metrics.log_test(test_metrics, prefix=tb_prefix)

            # Log timestep bin losses using unified system
            counts = timestep_loss_count.cpu()
            sums = timestep_loss_sum.cpu()
            timestep_bins = {}
            for bin_idx in range(num_timestep_bins):
                bin_start = bin_idx / num_timestep_bins
                bin_end = (bin_idx + 1) / num_timestep_bins
                count = counts[bin_idx].item()
                if count > 0:
                    avg_loss = (sums[bin_idx] / count).item()
                    timestep_bins[f'{bin_start:.1f}-{bin_end:.1f}'] = avg_loss
            if timestep_bins:
                trainer._unified_metrics.log_test_timesteps(timestep_bins, prefix=tb_prefix)

            # Log regional metrics via unified system
            if regional_tracker is not None:
                trainer._unified_metrics.log_test_regional(regional_tracker, prefix=tb_prefix)

            # Create visualization of worst batch (uses unified metrics)
            if worst_batch_data is not None:
                # Build display metrics
                display_metrics = {'MS-SSIM': metrics['msssim'], 'PSNR': metrics['psnr']}
                if 'lpips' in metrics:
                    display_metrics['LPIPS'] = metrics['lpips']

                fig_path = os.path.join(trainer.save_dir, f'test_worst_batch_{label}.png')
                trainer._unified_metrics.log_worst_batch(
                    original=worst_batch_data['original'],
                    reconstructed=worst_batch_data['generated'],
                    loss=metrics.get('mse', 0.0),
                    epoch=0,
                    tag_prefix=tb_prefix,
                    timesteps=worst_batch_data['timesteps'],
                    save_path=fig_path,
                    display_metrics=display_metrics,
                )
                logger.info(f"Test worst batch saved to: {fig_path}")

            # Compute generation quality metrics (FID, KID, CMMD) if enabled
            if trainer._gen_metrics is not None:
                try:
                    logger.info("Computing generation metrics (FID, KID, CMMD)...")
                    test_gen_results = trainer._gen_metrics.compute_test_metrics(
                        model_to_use, trainer.strategy, trainer.mode, test_loader
                    )
                    # Log to TensorBoard via unified metrics
                    exported = trainer._unified_metrics.log_test_generation(test_gen_results, prefix=tb_prefix)
                    metrics.update(exported)
                    # Log to console
                    if 'FID' in test_gen_results:
                        logger.info(f"  FID:     {test_gen_results['FID']:.4f}")
                    if 'KID_mean' in test_gen_results:
                        logger.info(f"  KID:     {test_gen_results['KID_mean']:.4f} +/- {test_gen_results.get('KID_std', 0):.4f}")
                    if 'CMMD' in test_gen_results:
                        logger.info(f"  CMMD:    {test_gen_results['CMMD']:.4f}")
                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(f"Generation metrics skipped due to OOM: {e}")
                    torch.cuda.empty_cache()
                except (RuntimeError, ValueError) as e:
                    logger.exception(f"Generation metrics computation failed on test set: {e}")

        return metrics

    finally:
        # Ensure model is in train mode even if an exception occurred
        model_to_use.train()


def compute_volume_3d_msssim(
    trainer: 'DiffusionTrainer',
    epoch: int,
    data_split: str = 'val',
    modality_override: str | None = None,
) -> float | None:
    """Compute 3D MS-SSIM by reconstructing full volumes.

    For 2D diffusion models: processes slice-by-slice then stacks.
    For 3D diffusion models: delegates to compute_volume_3d_msssim_native.

    2D approach:
    1. Loads full 3D volumes
    2. Processes each 2D slice: add noise at mid-range timestep → denoise → get predicted clean
    3. Stacks slices back into a volume
    4. Computes 3D MS-SSIM between reconstructed and original volumes

    This measures cross-slice consistency of the 2D diffusion model's denoising quality.

    Args:
        trainer: The DiffusionTrainer instance.
        epoch: Current epoch number.
        data_split: Which data split to use ('val' or 'test_new').
        modality_override: Optional specific modality to compute for
            (e.g., 'bravo', 't1_pre'). If None, uses mode from config.

    Returns:
        Average 3D MS-SSIM across all volumes, or None if unavailable/unsupported.
    """
    if not trainer.log_msssim:
        return None

    # For 3D diffusion models, use native 3D volume processing
    spatial_dims = trainer.spatial_dims
    if spatial_dims == 3:
        return compute_volume_3d_msssim_native(trainer, epoch, data_split, modality_override)

    # Import here to avoid circular imports
    from medgen.data.loaders.volume_3d import create_vae_volume_validation_dataloader

    # Determine modality - use override if provided, else from config
    if modality_override is not None:
        modality = modality_override
    else:
        # Use out_channels to determine volume channels (excludes conditioning)
        mode_name = trainer.mode_name
        out_channels = trainer._mode_config.out_channels
        modality = 'dual' if out_channels > 1 else mode_name
        # Map mode names to actual file modalities
        if modality in ('seg_conditioned', 'seg_conditioned_input'):
            modality = 'seg'

    # Skip for multi_modality mode - volume loader doesn't support mixed modalities
    # and computing volume metrics on mixed slices doesn't make sense
    if modality == 'multi_modality':
        return None

    # Get or create cached volume dataloader (avoid recreating datasets every epoch)
    cache_key = f"2d_{data_split}_{modality}"
    if cache_key not in trainer._volume_loaders_cache:
        result = create_vae_volume_validation_dataloader(trainer.cfg, modality, data_split)
        if result is None:
            return None
        volume_loader, _ = result
        trainer._volume_loaders_cache[cache_key] = volume_loader
    volume_loader = trainer._volume_loaders_cache[cache_key]

    model_to_use = trainer.ema.ema_model if trainer.ema is not None else trainer.model_raw
    model_to_use.eval()

    total_msssim_3d = 0.0
    n_volumes = 0
    slice_batch_size = trainer.batch_size

    # Use mid-range timestep for reconstruction quality measurement
    mid_timestep = trainer.num_timesteps // 2

    # Save random state - this method generates random noise which would otherwise
    # shift the global RNG and cause training to diverge across epochs
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(trainer.device) if torch.cuda.is_available() else None

    with torch.inference_mode():
        for batch in volume_loader:
            # batch['image'] is [1, C, H, W, D] (batch_size=1 for volumes)
            volume = batch['image'].to(trainer.device, non_blocking=True)
            volume = volume.squeeze(0)  # [C, H, W, D]

            n_channels_vol, height, width, depth = volume.shape

            # Pre-allocate output tensor
            all_recon = torch.empty(
                (depth, n_channels_vol, height, width),
                dtype=torch.bfloat16,
                device=trainer.device
            )

            # Process slices in batches
            for start_idx in range(0, depth, slice_batch_size):
                end_idx = min(start_idx + slice_batch_size, depth)
                current_batch_size = end_idx - start_idx

                # [C, H, W, D] -> [B, C, H, W]
                slice_tensor = volume[:, :, :, start_idx:end_idx].permute(3, 0, 1, 2)

                # Encode to diffusion space (identity for PixelSpace)
                slice_encoded = trainer.space.encode_batch(slice_tensor)

                # Add noise at mid-range timestep
                noise = torch.randn_like(slice_encoded)
                timesteps = torch.full(
                    (current_batch_size,),
                    mid_timestep,
                    device=trainer.device,
                    dtype=torch.long
                )
                noisy_slices = trainer.strategy.add_noise(slice_encoded, noise, timesteps)

                with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    # For conditional modes, use zeros as conditioning (no tumor)
                    # This measures pure denoising ability without semantic guidance
                    if trainer.mode.is_conditional:
                        dummy_labels = torch.zeros_like(slice_encoded[:, :1])  # Single channel
                        model_input = trainer.mode.format_model_input(noisy_slices, {'labels': dummy_labels})
                    else:
                        model_input = trainer.mode.format_model_input(noisy_slices, {'labels': None})

                    # Predict noise/velocity
                    prediction = trainer.strategy.predict_noise_or_velocity(
                        model_to_use, model_input, timesteps
                    )

                    # Get predicted clean images
                    _, predicted_clean = trainer.strategy.compute_loss(
                        prediction, slice_encoded, noise, noisy_slices, timesteps
                    )

                # Decode from diffusion space if needed
                if trainer.space.needs_decode:
                    predicted_clean = trainer.space.decode_batch(predicted_clean)

                all_recon[start_idx:end_idx] = predicted_clean

            # Reshape for 3D MS-SSIM: [D, C, H, W] -> [1, C, D, H, W]
            recon_3d = all_recon.permute(1, 0, 2, 3).unsqueeze(0)
            volume_3d = volume.permute(0, 3, 1, 2).unsqueeze(0)

            # Compute 3D MS-SSIM
            msssim_3d = compute_msssim(recon_3d.float(), volume_3d.float(), spatial_dims=3)
            total_msssim_3d += msssim_3d
            n_volumes += 1

    # Restore model state and RNG (use try/finally for robustness)
    try:
        model_to_use.train()
        if n_volumes == 0:
            return None
        return total_msssim_3d / n_volumes
    finally:
        # Ensure RNG state is ALWAYS restored
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, trainer.device)


def compute_volume_3d_msssim_native(
    trainer: 'DiffusionTrainer',
    epoch: int,
    data_split: str = 'val',
    modality_override: str | None = None,
) -> float | None:
    """Compute 3D MS-SSIM for 3D diffusion models (native volume processing).

    For 3D diffusion models, this:
    1. Loads full 3D volumes
    2. Adds noise at mid-range timestep to the whole volume
    3. Denoises the whole volume at once
    4. Computes 3D MS-SSIM between reconstructed and original volumes

    Args:
        trainer: The DiffusionTrainer instance.
        epoch: Current epoch number.
        data_split: Which data split to use ('val' or 'test_new').
        modality_override: Optional specific modality to compute for
            (e.g., 'bravo', 't1_pre'). If None, uses mode from config.

    Returns:
        Average 3D MS-SSIM across all volumes, or None if unavailable.
    """
    # Skip for multi_modality mode
    mode_name = trainer.mode_name
    if mode_name == 'multi_modality':
        return None

    # Use pixel_val_loader if available (latent diffusion), else val_loader.
    # Exception: bravo_seg_cond uses dual encoders — can't encode pixel seg correctly.
    if data_split == 'val':
        use_pixel = (
            getattr(trainer, 'pixel_val_loader', None) is not None
            and mode_name != 'bravo_seg_cond'
        )
        loader = trainer.pixel_val_loader if use_pixel else trainer.val_loader
        if loader is None:
            return None
    else:
        # For other splits (test), create loader if needed
        if modality_override is not None:
            modality = modality_override
        else:
            out_channels = trainer._mode_config.out_channels
            modality = 'dual' if out_channels > 1 else mode_name
            if modality in ('seg_conditioned', 'seg_conditioned_input'):
                modality = 'seg'

        cache_key = f"{data_split}_{modality}"
        if cache_key not in trainer._volume_loaders_cache:
            from medgen.data.loaders.volume_3d import (
                create_vae_3d_single_modality_validation_loader,
            )
            loader = create_vae_3d_single_modality_validation_loader(trainer.cfg, modality)
            if loader is None:
                return None
            trainer._volume_loaders_cache[cache_key] = loader
        loader = trainer._volume_loaders_cache[cache_key]

    model_to_use = trainer.ema.ema_model if trainer.ema is not None else trainer.model_raw
    model_to_use.eval()

    total_msssim_3d = 0.0
    n_volumes = 0

    # Use mid-range timestep for reconstruction quality
    mid_timestep = trainer.num_timesteps // 2

    # Save RNG state
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(trainer.device) if torch.cuda.is_available() else None

    with torch.inference_mode():
        for batch in loader:
            # Handle tuple format (SegDataset), latent dict, and pixel dict formats
            if isinstance(batch, (tuple, list)):
                # SegDataset returns (seg_volume, size_bins) or (seg_volume, size_bins, bin_maps)
                volume = batch[0].to(trainer.device, non_blocking=True)
                labels = None  # seg_conditioned mode has no separate labels
            elif 'latent' in batch:
                # Latent dict format: {'latent': ..., 'latent_seg': ...}
                # For bravo_seg_cond: latent is bravo, latent_seg is conditioning
                volume = batch['latent'].to(trainer.device, non_blocking=True)
                labels = batch.get('latent_seg')
                if labels is not None:
                    labels = labels.to(trainer.device, non_blocking=True)
            else:
                # Pixel dict format: {'image': ..., 'seg': ...}
                volume = batch['image'].to(trainer.device, non_blocking=True)
                labels = batch.get('seg')
                if labels is not None:
                    labels = labels.to(trainer.device, non_blocking=True)
            labels_dict = {'labels': labels}

            # Check if data is already in latent space (from latent loader)
            is_latent_data = 'latent' in batch if isinstance(batch, dict) else False

            # Create timestep tensor
            timesteps = torch.full(
                (volume.shape[0],), mid_timestep, device=trainer.device, dtype=torch.long
            )

            # Handle latent vs pixel space data
            if is_latent_data:
                # Data is already in latent space - decode for pixel comparison
                volume_latent = volume
                volume_pixel = trainer.space.decode_batch(volume)
            else:
                # Data is in pixel space - encode for model
                volume_pixel = volume
                volume_latent = trainer.space.encode_batch(volume)
                # Encode labels to match (s2d/wavelet/latent space)
                if labels is not None:
                    labels = trainer.space.encode(labels)
                    labels_dict = {'labels': labels}

            # Add noise in latent/pixel space (model always operates here)
            noise = torch.randn_like(volume_latent)
            noisy_volume = trainer.strategy.add_noise(volume_latent, noise, timesteps)

            # Format input and denoise
            # For ControlNet (Stage 1 or 2): use only noisy images
            if trainer.use_controlnet or trainer.controlnet_stage1:
                model_input = noisy_volume
            else:
                model_input = trainer.mode.format_model_input(noisy_volume, labels_dict)
            prediction = trainer.strategy.predict_noise_or_velocity(model_to_use, model_input, timesteps)
            _, predicted_clean = trainer.strategy.compute_loss(prediction, volume_latent, noise, noisy_volume, timesteps)

            # Decode back to pixel space if needed
            if trainer.space.needs_decode:
                predicted_clean = trainer.space.decode_batch(predicted_clean)

            # Compute 3D MS-SSIM in pixel space
            msssim_3d = compute_msssim(predicted_clean.float(), volume_pixel.float(), spatial_dims=3)
            total_msssim_3d += msssim_3d
            n_volumes += 1

    # Restore model state and RNG (use try/finally for robustness)
    try:
        model_to_use.train()
        if n_volumes == 0:
            return None
        return total_msssim_3d / n_volumes
    finally:
        # Ensure RNG state is ALWAYS restored
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, trainer.device)


def create_test_reconstruction_figure(
    original: torch.Tensor,
    predicted: torch.Tensor,
    metrics: dict[str, float],
    label: str,
    timesteps: torch.Tensor | None = None,
) -> 'plt.Figure':
    """Create side-by-side test evaluation figure.

    Uses shared create_reconstruction_figure for consistent visualization.

    Args:
        original: Original images [B, C, H, W] (CPU).
        predicted: Predicted clean images [B, C, H, W] (CPU).
        metrics: Dict with test metrics (mse, msssim, psnr, optionally lpips).
        label: Checkpoint label (best, latest, current).
        timesteps: Optional timesteps for each sample.

    Returns:
        Matplotlib figure.
    """
    title = f"Worst Test Batch ({label})"
    display_metrics = {
        'MS-SSIM': metrics['msssim'],
        'PSNR': metrics['psnr'],
    }
    if 'lpips' in metrics:
        display_metrics['LPIPS'] = metrics['lpips']
    return create_reconstruction_figure(
        original=original,
        generated=predicted,
        title=title,
        max_samples=8,
        metrics=display_metrics,
        timesteps=timesteps,
    )
