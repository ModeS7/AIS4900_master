"""Training tricks and regularization utilities for diffusion training.

This module provides various training tricks:
- Gradient noise injection
- Curriculum timestep scheduling
- Timestep jitter
- Noise augmentation
- Conditioning dropout (CFG)
- Feature perturbation hooks
- DC-AE 1.5 augmented diffusion
"""
import logging
import random
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from medgen.pipeline.trainer import DiffusionTrainer

logger = logging.getLogger(__name__)


def add_gradient_noise(
    trainer: 'DiffusionTrainer',
    step: int,
) -> None:
    """Add Gaussian noise to gradients for regularization.

    Noise decays over training as: sigma / (1 + step)^decay
    Reference: "Adding Gradient Noise Improves Learning" (Neelakantan et al., 2015)

    Args:
        trainer: The DiffusionTrainer instance.
        step: Current global training step.
    """
    grad_noise_cfg = trainer.cfg.training.get('gradient_noise', {})
    if not grad_noise_cfg.get('enabled', False):
        return

    sigma = grad_noise_cfg.get('sigma', 0.01)
    decay = grad_noise_cfg.get('decay', 0.55)

    # Decay noise over training
    noise_std = sigma / (1 + step) ** decay

    for param in trainer.model_raw.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_std
            param.grad.add_(noise)


def get_curriculum_range(
    trainer: 'DiffusionTrainer',
    epoch: int,
) -> tuple[float, float] | None:
    """Get timestep range for curriculum learning.

    Linearly interpolates from start range to end range over warmup_epochs.

    Args:
        trainer: The DiffusionTrainer instance.
        epoch: Current training epoch.

    Returns:
        Tuple of (min_t, max_t) or None if curriculum disabled.
    """
    curriculum_cfg = trainer.cfg.training.get('curriculum', {})
    if not curriculum_cfg.get('enabled', False):
        return None

    warmup_epochs = curriculum_cfg.get('warmup_epochs', 50)
    progress = min(1.0, epoch / warmup_epochs)

    # Linear interpolation from start to end range
    min_t_start = curriculum_cfg.get('min_t_start', 0.0)
    min_t_end = curriculum_cfg.get('min_t_end', 0.0)
    max_t_start = curriculum_cfg.get('max_t_start', 0.3)
    max_t_end = curriculum_cfg.get('max_t_end', 1.0)

    min_t = min_t_start + progress * (min_t_end - min_t_start)
    max_t = max_t_start + progress * (max_t_end - max_t_start)

    return (min_t, max_t)


def apply_timestep_jitter(
    trainer: 'DiffusionTrainer',
    timesteps: Tensor,
) -> Tensor:
    """Add Gaussian noise to timesteps for regularization.

    Increases noise-level diversity without changing output distribution.

    Args:
        trainer: The DiffusionTrainer instance.
        timesteps: Original timesteps tensor.

    Returns:
        Jittered timesteps (clamped to valid range).
    """
    jitter_cfg = trainer.cfg.training.get('timestep_jitter', {})
    if not jitter_cfg.get('enabled', False):
        return timesteps

    std = jitter_cfg.get('std', 0.05)
    # Detect if input is discrete (int) or continuous (float)
    is_discrete = timesteps.dtype in (torch.int32, torch.int64, torch.long)
    # Convert to float, normalize to [0, 1], add jitter, clamp, scale back
    t_float = timesteps.float() / trainer.num_timesteps
    jitter = torch.randn_like(t_float) * std
    t_jittered = (t_float + jitter).clamp(0.0, 1.0)
    t_scaled = t_jittered * trainer.num_timesteps
    # Preserve dtype: int for DDPM, float for RFlow
    if is_discrete:
        return t_scaled.long()
    else:
        return t_scaled


def apply_noise_augmentation(
    trainer: 'DiffusionTrainer',
    noise: Tensor | dict[str, Tensor],
) -> Tensor | dict[str, Tensor]:
    """Add perturbation to noise vector for regularization.

    Increases noise diversity without affecting what model learns to output.

    Args:
        trainer: The DiffusionTrainer instance.
        noise: Original noise tensor or dict of tensors.

    Returns:
        Perturbed noise (renormalized to maintain variance).
    """
    noise_aug_cfg = trainer.cfg.training.get('noise_augmentation', {})
    if not noise_aug_cfg.get('enabled', False):
        return noise

    std = noise_aug_cfg.get('std', 0.1)

    if isinstance(noise, dict):
        perturbed = {}
        for k, v in noise.items():
            perturbation = torch.randn_like(v) * std
            # Add perturbation and renormalize to maintain unit variance
            perturbed_v = v + perturbation
            perturbed[k] = perturbed_v / perturbed_v.std() * v.std()
        return perturbed
    else:
        perturbation = torch.randn_like(noise) * std
        perturbed = noise + perturbation
        return perturbed / perturbed.std() * noise.std()


def apply_conditioning_dropout(
    trainer: 'DiffusionTrainer',
    conditioning: Tensor | None,
    batch_size: int,
) -> Tensor | None:
    """Apply per-sample CFG dropout to conditioning tensor.

    Used for ControlNet conditioning to enable classifier-free guidance
    at inference time. Sets entire samples to zero with probability
    `controlnet_cfg_dropout_prob`.

    Args:
        trainer: The DiffusionTrainer instance.
        conditioning: Conditioning tensor [B, C, ...] or None.
        batch_size: Batch size for dropout mask.

    Returns:
        Conditioning with per-sample dropout applied, or None if input is None.
    """
    if conditioning is None or trainer.controlnet_cfg_dropout_prob <= 0:
        return conditioning

    if not trainer.training:
        return conditioning

    # Per-sample dropout mask
    dropout_mask = torch.rand(batch_size, device=conditioning.device)
    keep_mask = (dropout_mask >= trainer.controlnet_cfg_dropout_prob).float()

    # Expand to match conditioning dims [B, C, H, W] or [B, C, D, H, W]
    for _ in range(conditioning.dim() - 1):
        keep_mask = keep_mask.unsqueeze(-1)

    return conditioning * keep_mask


def setup_feature_perturbation(trainer: 'DiffusionTrainer') -> None:
    """Setup forward hooks for feature perturbation.

    Args:
        trainer: The DiffusionTrainer instance.
    """
    trainer._feature_hooks = []
    feat_cfg = trainer.cfg.training.get('feature_perturbation', {})

    if not feat_cfg.get('enabled', False):
        return

    std = feat_cfg.get('std', 0.1)
    layers = feat_cfg.get('layers', ['mid'])

    def make_hook(noise_std):
        def hook(module, input, output):
            if trainer.model.training:
                noise = torch.randn_like(output) * noise_std
                return output + noise
            return output
        return hook

    # Register hooks on specified layers
    # UNet structure: down_blocks, mid_block, up_blocks
    if hasattr(trainer.model_raw, 'mid_block') and 'mid' in layers:
        handle = trainer.model_raw.mid_block.register_forward_hook(make_hook(std))
        trainer._feature_hooks.append(handle)

    if hasattr(trainer.model_raw, 'down_blocks') and 'encoder' in layers:
        for block in trainer.model_raw.down_blocks:
            handle = block.register_forward_hook(make_hook(std))
            trainer._feature_hooks.append(handle)

    if hasattr(trainer.model_raw, 'up_blocks') and 'decoder' in layers:
        for block in trainer.model_raw.up_blocks:
            handle = block.register_forward_hook(make_hook(std))
            trainer._feature_hooks.append(handle)


def remove_feature_perturbation_hooks(trainer: 'DiffusionTrainer') -> None:
    """Remove feature perturbation hooks.

    Args:
        trainer: The DiffusionTrainer instance.
    """
    for handle in getattr(trainer, '_feature_hooks', []):
        handle.remove()
    trainer._feature_hooks = []


def get_aug_diff_channel_steps(
    trainer: 'DiffusionTrainer',
    num_channels: int,
) -> list[int]:
    """Get list of channel counts for augmented diffusion masking.

    Returns [min_channels, min+step, min+2*step, ..., num_channels].
    Paper uses [16, 20, 24, ..., c] with step=4, min=16.

    Args:
        trainer: The DiffusionTrainer instance.
        num_channels: Total number of latent channels.

    Returns:
        List of valid channel counts to sample from during training.
    """
    if trainer._aug_diff_channel_steps is None:
        steps = list(range(
            trainer.aug_diff_min_channels,
            num_channels + 1,
            trainer.aug_diff_channel_step
        ))
        # Ensure max channels is always included
        if not steps or steps[-1] != num_channels:
            steps.append(num_channels)
        trainer._aug_diff_channel_steps = steps
    return trainer._aug_diff_channel_steps


def create_aug_diff_mask(
    trainer: 'DiffusionTrainer',
    tensor: Tensor,
) -> Tensor:
    """Create channel mask for augmented diffusion training.

    From DC-AE 1.5 paper (Eq. 2):
    - Sample random channel count c' from [min_channels, ..., num_channels]
    - Create mask: [1,...,1 (c' times), 0,...,0]

    Args:
        trainer: The DiffusionTrainer instance.
        tensor: Input tensor [B, C, H, W] to get shape from.

    Returns:
        Mask tensor [1, C, 1, 1] for broadcasting.
    """
    C = tensor.shape[1]
    steps = get_aug_diff_channel_steps(trainer, C)
    c_prime = random.choice(steps)

    mask = torch.zeros(1, C, 1, 1, device=tensor.device, dtype=tensor.dtype)
    mask[:, :c_prime, :, :] = 1.0
    return mask
