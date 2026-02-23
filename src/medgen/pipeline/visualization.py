"""Visualization utilities for diffusion training.

This module provides sample generation and visualization:
- 2D/3D sample generation
- Denoising trajectory visualization
- Size-binned generation for seg_conditioned mode
"""
import logging
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from medgen.pipeline.trainer import DiffusionTrainer

logger = logging.getLogger(__name__)


@torch.no_grad()
def visualize_samples(
    trainer: 'DiffusionTrainer',
    model: nn.Module,
    epoch: int,
    train_dataset: Dataset | None = None,
) -> None:
    """Generate and visualize samples.

    For 3D: Uses center slice visualization with cached training conditioning.
    For 2D: Uses ValidationVisualizer for full image visualization.

    Args:
        trainer: The DiffusionTrainer instance.
        model: Model to use for generation (typically EMA model).
        epoch: Current epoch number.
        train_dataset: Training dataset (required for 2D, optional for 3D).
    """
    if not trainer.is_main_process:
        return

    try:
        model.eval()

        if trainer.spatial_dims == 3:
            # 3D: Generate samples using cached training batch for real conditioning
            visualize_samples_3d(trainer, model, epoch)
            # 3D: Also log denoising trajectory if enabled
            if trainer.log_intermediate_steps:
                visualize_denoising_trajectory_3d(trainer, model, epoch)
        else:
            # 2D: Delegate to ValidationVisualizer
            if trainer.visualizer is not None and train_dataset is not None:
                trainer.visualizer.generate_samples(model, train_dataset, epoch)
    finally:
        model.train()


@torch.no_grad()
def visualize_samples_3d(
    trainer: 'DiffusionTrainer',
    model: nn.Module,
    epoch: int,
) -> None:
    """Generate and visualize 3D samples (center slices).

    Uses REAL conditioning from cached TRAINING samples instead of zeros.
    This matches the 3D trainer approach and ensures the model gets proper
    conditioning for generation.

    Args:
        trainer: The DiffusionTrainer instance.
        model: Model to use for generation.
        epoch: Current epoch number.
    """
    from medgen.diffusion import SegmentationConditionedInputMode

    if trainer._cached_train_batch is None:
        if trainer.mode.is_conditional:
            logger.warning("Cannot visualize 3D samples: no cached training batch for conditioning")
            return
        # Unconditional mode can proceed with random noise
        batch_size = 4
        noise = torch.randn(
            batch_size, 1,
            trainer.volume_depth,
            trainer.volume_height,
            trainer.volume_width,
            device=trainer.device
        )
        model_input = noise
        size_bins = None
        bin_maps = None
    else:
        cached_images = trainer._cached_train_batch['images']
        cached_labels = trainer._cached_train_batch.get('labels')
        cached_size_bins = trainer._cached_train_batch.get('size_bins')
        batch_size = min(4, cached_images.shape[0])

        # Generate noise matching the cached batch shape
        is_latent = trainer._cached_train_batch.get('is_latent', False)
        if is_latent:
            # Data is already in latent space (from latent loader)
            noise = torch.randn_like(cached_images[:batch_size])
        elif trainer.space.scale_factor > 1:
            # Pixel space data, encode to get latent shape
            with torch.no_grad():
                encoded = trainer.space.encode(cached_images[:batch_size])
            noise = torch.randn_like(encoded)
        else:
            # Pixel space diffusion
            noise = torch.randn_like(cached_images[:batch_size])

        # Build model input with real conditioning
        # Initialize bin_maps for seg_conditioned_input mode (None for other modes)
        bin_maps = None
        # For ControlNet (Stage 1 or 2): use only noise (no concatenation)
        if trainer.use_controlnet or trainer.controlnet_stage1:
            model_input = noise
            size_bins = None
        elif isinstance(trainer.mode, SegmentationConditionedInputMode):
            # Input channel conditioning: pass noise as model_input, bin_maps separately
            cached_bin_maps = trainer._cached_train_batch.get('bin_maps')
            if cached_bin_maps is not None:
                bin_maps = cached_bin_maps[:batch_size]
            model_input = noise
            size_bins = None
        elif trainer.use_size_bin_embedding:
            model_input = noise
            size_bins = cached_size_bins[:batch_size] if cached_size_bins is not None else None
        elif trainer.mode.is_conditional and cached_labels is not None:
            labels = cached_labels[:batch_size]
            # Check if labels are already in latent space (bravo_seg_cond mode)
            labels_is_latent = trainer._cached_train_batch.get('labels_is_latent', False)
            if trainer.space.encode_conditioning and not labels_is_latent:
                labels_encoded = trainer.space.encode(labels)
            else:
                labels_encoded = labels
            model_input = torch.cat([noise, labels_encoded], dim=1)
            size_bins = None
        else:
            model_input = noise
            size_bins = None

    # Generate samples
    # Use CFG scale from generation metrics config (default 2.0)
    cfg_scale = trainer._gen_metrics_config.cfg_scale if trainer._gen_metrics_config is not None else 2.0
    num_train_timesteps = trainer.scheduler.num_train_timesteps
    if trainer.use_size_bin_embedding and size_bins is not None:
        samples = generate_with_size_bins(
            model, noise, size_bins, num_train_timesteps,
            num_steps=25, cfg_scale=cfg_scale,
        )
    else:
        samples = trainer.strategy.generate(
            model,
            model_input,
            num_steps=25,
            device=trainer.device,
            use_progress_bars=False,
            bin_maps=bin_maps,
            cfg_scale=cfg_scale,
            latent_channels=trainer.space.latent_channels,
        )

    # Log latent space visualization before decoding
    if trainer.space.needs_decode and trainer._unified_metrics is not None:
        trainer._unified_metrics.log_latent_samples(samples, epoch, tag='Generated_Samples_Latent')

    # Decode to pixel space if needed
    if trainer.space.needs_decode:
        samples = trainer.space.decode(samples)

    # Log using unified metrics (handles 3D center slice extraction)
    if trainer._unified_metrics is not None:
        trainer._unified_metrics.log_generated_samples(samples, epoch, tag='Generated_Samples', nrow=2)


@torch.no_grad()
def visualize_denoising_trajectory(
    trainer: 'DiffusionTrainer',
    model: nn.Module,
    epoch: int,
    num_steps: int = 5,
) -> None:
    """Visualize intermediate denoising steps.

    Shows the progression from noise to clean sample at multiple timesteps.
    For 3D: uses center slice visualization.

    Args:
        trainer: The DiffusionTrainer instance.
        model: Model to use for generation.
        epoch: Current epoch number.
        num_steps: Number of intermediate steps to visualize.
    """
    if not trainer.is_main_process:
        return

    try:
        model.eval()

        if trainer.spatial_dims == 3:
            visualize_denoising_trajectory_3d(trainer, model, epoch, num_steps)
        else:
            # 2D: Delegate to ValidationVisualizer (if it has this method)
            pass  # 2D trajectory visualization handled separately
    finally:
        model.train()


@torch.no_grad()
def visualize_denoising_trajectory_3d(
    trainer: 'DiffusionTrainer',
    model: nn.Module,
    epoch: int,
    num_steps: int = 5,
) -> None:
    """Visualize intermediate denoising steps for 3D volumes.

    Args:
        trainer: The DiffusionTrainer instance.
        model: Model to use for generation.
        epoch: Current epoch number.
        num_steps: Number of intermediate steps to capture.
    """
    from medgen.diffusion import SegmentationConditionedInputMode

    if trainer._cached_train_batch is None:
        logger.warning("No cached training batch for 3D denoising trajectory")
        return

    cached_images = trainer._cached_train_batch['images']
    cached_labels = trainer._cached_train_batch.get('labels')
    cached_size_bins = trainer._cached_train_batch.get('size_bins')

    # Generate noise for single sample
    is_latent = trainer._cached_train_batch.get('is_latent', False)
    if is_latent:
        # Data is already in latent space (from latent loader)
        noise = torch.randn_like(cached_images[:1])
    elif trainer.space.scale_factor > 1:
        # Pixel space data, encode to get latent shape
        with torch.no_grad():
            encoded = trainer.space.encode(cached_images[:1])
        noise = torch.randn_like(encoded)
    else:
        # Pixel space diffusion
        noise = torch.randn_like(cached_images[:1])

    # Extract parameters for decoupled generation functions
    num_train_timesteps = trainer.scheduler.num_train_timesteps
    is_conditional = trainer.mode.is_conditional
    latent_channels = trainer.space.latent_channels
    scale_factor = trainer.space.scale_factor

    # Build model input with conditioning
    # For ControlNet (Stage 1 or 2): use only noise (no concatenation)
    if trainer.use_controlnet or trainer.controlnet_stage1:
        trajectory = generate_trajectory(
            model, noise, num_train_timesteps,
            is_conditional=False,  # ControlNet: no channel concatenation
            latent_channels=latent_channels, scale_factor=scale_factor,
            num_steps=25, capture_every=5,
        )
    elif isinstance(trainer.mode, SegmentationConditionedInputMode):
        # Input channel conditioning: concatenate noise with bin_maps
        cached_bin_maps = trainer._cached_train_batch.get('bin_maps')
        if cached_bin_maps is not None:
            bin_maps = cached_bin_maps[:1]
            model_input = torch.cat([noise, bin_maps], dim=1)
        else:
            model_input = noise
        trajectory = generate_trajectory(
            model, model_input, num_train_timesteps,
            is_conditional=True,  # Has channel conditioning
            latent_channels=latent_channels, scale_factor=scale_factor,
            num_steps=25, capture_every=5,
        )
    elif trainer.use_size_bin_embedding and cached_size_bins is not None:
        size_bins = cached_size_bins[:1]
        trajectory = generate_trajectory_with_size_bins(
            model, noise, size_bins, num_train_timesteps,
            num_steps=25, capture_every=5,
        )
    elif trainer.mode.is_conditional and cached_labels is not None:
        labels = cached_labels[:1]
        # Check if labels are already in latent space (bravo_seg_cond mode)
        labels_is_latent = trainer._cached_train_batch.get('labels_is_latent', False)
        if trainer.space.encode_conditioning and not labels_is_latent:
            labels_encoded = trainer.space.encode(labels)
        else:
            labels_encoded = labels
        model_input = torch.cat([noise, labels_encoded], dim=1)
        trajectory = generate_trajectory(
            model, model_input, num_train_timesteps,
            is_conditional=True,
            latent_channels=latent_channels, scale_factor=scale_factor,
            num_steps=25, capture_every=5,
        )
    else:
        trajectory = generate_trajectory(
            model, noise, num_train_timesteps,
            is_conditional=False,
            latent_channels=latent_channels, scale_factor=scale_factor,
            num_steps=25, capture_every=5,
        )

    # Log latent trajectory before decoding
    if trainer.space.needs_decode and trainer._unified_metrics is not None:
        trainer._unified_metrics.log_latent_trajectory(trajectory, epoch, tag='denoising_trajectory')

    # Decode trajectory to pixel space if needed
    if trainer.space.needs_decode:
        trajectory = [trainer.space.decode(t) for t in trajectory]

    # Log using unified metrics (handles 3D center slice extraction)
    if trainer._unified_metrics is not None:
        trainer._unified_metrics.log_denoising_trajectory(trajectory, epoch, tag='denoising_trajectory')


@torch.no_grad()
def generate_trajectory(
    model: nn.Module,
    model_input: Tensor,
    num_train_timesteps: int,
    is_conditional: bool,
    latent_channels: int = 1,
    scale_factor: int = 1,
    num_steps: int = 25,
    capture_every: int = 5,
) -> list[Tensor]:
    """Generate samples while capturing intermediate states.

    Works for both 2D [B, C, H, W] and 3D [B, C, D, H, W] tensors.
    All tensor operations (torch.full, model forward, Euler step) are
    dimension-agnostic.

    Args:
        model: Model to use for generation.
        model_input: Starting noisy tensor (may include conditioning).
        num_train_timesteps: Number of training timesteps (from scheduler).
        is_conditional: Whether mode is conditional (from mode.is_conditional).
        latent_channels: Number of latent channels (from space.latent_channels).
        scale_factor: Compression scale factor (1 = pixel space).
        num_steps: Total denoising steps.
        capture_every: Capture state every N steps.

    Returns:
        List of intermediate tensors.
    """
    # Extract noise from model_input (first channels)
    if is_conditional:
        in_ch = 1 if scale_factor == 1 else latent_channels
        x = model_input[:, :in_ch].clone()
        conditioning = model_input[:, in_ch:]
    else:
        x = model_input.clone()
        conditioning = None

    trajectory = [x.clone()]
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = 1.0 - i * dt
        # Scale to training range for correct embeddings
        t_scaled = t * num_train_timesteps
        t_tensor = torch.full((x.shape[0],), t_scaled, device=x.device)

        # Prepare input with conditioning
        if conditioning is not None:
            model_in = torch.cat([x, conditioning], dim=1)
        else:
            model_in = x

        # Get velocity prediction
        v = model(x=model_in, timesteps=t_tensor)

        # Euler step: x = x + dt * v
        x = x + dt * v

        # Capture intermediate state
        if (i + 1) % capture_every == 0:
            trajectory.append(x.clone())

    return trajectory


@torch.no_grad()
def generate_with_size_bins(
    model: nn.Module,
    noise: Tensor,
    size_bins: Tensor,
    num_train_timesteps: int,
    num_steps: int = 25,
    cfg_scale: float = 1.0,
) -> Tensor:
    """Generate samples with size bin conditioning.

    Works for both 2D [B, C, H, W] and 3D [B, C, D, H, W] tensors.
    All tensor operations (Euler integration, model forward) are
    dimension-agnostic.

    Args:
        model: The model to use for generation (e.g., EMA model).
        noise: Starting noise tensor [B, C, H, W] or [B, C, D, H, W].
        size_bins: Size bin embedding tensor [B, num_bins].
        num_train_timesteps: Number of training timesteps (from scheduler).
        num_steps: Number of denoising steps.
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance).

    Returns:
        Generated samples tensor.
    """
    x = noise.clone()
    dt = 1.0 / num_steps
    use_cfg = cfg_scale > 1.0

    # Prepare unconditional size_bins for CFG
    if use_cfg:
        uncond_size_bins = torch.zeros_like(size_bins)

    for i in range(num_steps):
        t = 1.0 - i * dt
        t_scaled = t * num_train_timesteps
        t_tensor = torch.full((x.shape[0],), t_scaled, device=x.device)

        if use_cfg:
            # CFG: compute both conditional and unconditional predictions
            v_cond = model(x, t_tensor, size_bins=size_bins)
            v_uncond = model(x, t_tensor, size_bins=uncond_size_bins)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            # No CFG: just conditional prediction
            v = model(x, t_tensor, size_bins=size_bins)

        # Euler step
        x = x + dt * v

    return x


@torch.no_grad()
def generate_trajectory_with_size_bins(
    model: nn.Module,
    noise: Tensor,
    size_bins: Tensor,
    num_train_timesteps: int,
    num_steps: int = 25,
    capture_every: int = 5,
) -> list[Tensor]:
    """Generate samples with size bins while capturing trajectory.

    Works for both 2D [B, C, H, W] and 3D [B, C, D, H, W] tensors.
    All tensor operations are dimension-agnostic.

    Args:
        model: The model to use for generation (e.g., EMA model).
        noise: Starting noise tensor [B, C, H, W] or [B, C, D, H, W].
        size_bins: Size bin embedding tensor [B, num_bins].
        num_train_timesteps: Number of training timesteps (from scheduler).
        num_steps: Total denoising steps.
        capture_every: Capture state every N steps.

    Returns:
        List of intermediate tensors.
    """
    x = noise.clone()
    trajectory = [x.clone()]
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = 1.0 - i * dt
        t_scaled = t * num_train_timesteps
        t_tensor = torch.full((x.shape[0],), t_scaled, device=x.device)

        v = model(x, t_tensor, size_bins=size_bins)
        x = x + dt * v

        if (i + 1) % capture_every == 0:
            trajectory.append(x.clone())

    return trajectory


# =============================================================================
# Backward Compatibility Aliases (Deprecated)
# =============================================================================
# These aliases exist for backward compatibility with code that imported
# the old _3d suffixed function names. They emit DeprecationWarning and
# delegate to the unified functions above.


def generate_trajectory_3d(*args, **kwargs):
    """DEPRECATED: Use generate_trajectory() instead.

    This function is dimension-agnostic and works for both 2D and 3D tensors.
    """
    import warnings
    warnings.warn(
        "generate_trajectory_3d is deprecated, use generate_trajectory",
        DeprecationWarning,
        stacklevel=2,
    )
    return generate_trajectory(*args, **kwargs)


def generate_with_size_bins_3d(*args, **kwargs):
    """DEPRECATED: Use generate_with_size_bins() instead.

    This function is dimension-agnostic and works for both 2D and 3D tensors.
    """
    import warnings
    warnings.warn(
        "generate_with_size_bins_3d is deprecated, use generate_with_size_bins",
        DeprecationWarning,
        stacklevel=2,
    )
    return generate_with_size_bins(*args, **kwargs)


def generate_trajectory_with_size_bins_3d(*args, **kwargs):
    """DEPRECATED: Use generate_trajectory_with_size_bins() instead.

    This function is dimension-agnostic and works for both 2D and 3D tensors.
    """
    import warnings
    warnings.warn(
        "generate_trajectory_with_size_bins_3d is deprecated, use generate_trajectory_with_size_bins",
        DeprecationWarning,
        stacklevel=2,
    )
    return generate_trajectory_with_size_bins(*args, **kwargs)
