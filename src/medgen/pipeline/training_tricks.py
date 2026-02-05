"""Training tricks and regularization utilities for diffusion training.

This module provides various training tricks:
- Gradient noise injection
- Curriculum timestep scheduling
- Timestep jitter
- Noise augmentation
- Conditioning dropout (CFG)
- Feature perturbation hooks
- DC-AE 1.5 augmented diffusion
- ScoreAug loss computation
- SDA loss computation
"""
import logging
import random
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
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
            perturbed[k] = perturbed_v / (perturbed_v.std() + 1e-8) * v.std()
        return perturbed
    else:
        perturbation = torch.randn_like(noise) * std
        perturbed = noise + perturbation
        return perturbed / (perturbed.std() + 1e-8) * noise.std()


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


# ─────────────────────────────────────────────────────────────────────────────
# ScoreAug Loss Computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_scoreaug_loss(
    trainer: 'DiffusionTrainer',
    model_input: Tensor,
    timesteps: Tensor,
    images: Tensor | dict[str, Tensor],
    noise: Tensor | dict[str, Tensor],
    noisy_images: Tensor | dict[str, Tensor],
    mode_id: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | dict[str, Tensor]]:
    """Compute loss using ScoreAug augmentation.

    ScoreAug transforms both the noisy input and the target together to maintain
    consistency. This function handles the full ScoreAug training loop including:
    - Computing velocity/noise target before ScoreAug
    - Applying ScoreAug transform
    - Optional mode intensity scaling
    - Model prediction with appropriate conditioning
    - MSE loss computation
    - Perceptual loss with inverse transform (when applicable)

    Args:
        trainer: DiffusionTrainer instance (for accessing score_aug, model, strategy, etc.)
        model_input: Formatted model input tensor (noisy + conditioning)
        timesteps: Sampled timesteps
        images: Clean images (tensor or dict for dual mode)
        noise: Noise tensor (tensor or dict for dual mode)
        noisy_images: Noisy images at timestep t
        mode_id: Optional mode ID for multi-modality conditioning

    Returns:
        Tuple of (mse_loss, perceptual_loss, predicted_clean):
        - mse_loss: MSE loss between prediction and augmented target
        - perceptual_loss: Perceptual loss (0 if non-invertible transform or weight=0)
        - predicted_clean: Predicted clean images for visualization

    Note:
        Requires trainer.score_aug to be set (i.e., ScoreAug is enabled).
    """
    # 1. Compute target BEFORE ScoreAug using strategy's compute_target method
    target = trainer.strategy.compute_target(images, noise)

    # 2. For dual mode, stack targets for joint transform
    is_dual = isinstance(target, dict)
    if is_dual:
        keys = list(target.keys())
        stacked_target = torch.cat([target[k] for k in keys], dim=1)
        aug_input, aug_target_stacked, omega = trainer.score_aug(model_input, stacked_target)
        # Unstack back to dict
        aug_target = {
            keys[0]: aug_target_stacked[:, 0:1],
            keys[1]: aug_target_stacked[:, 1:2],
        }
    else:
        aug_input, aug_target, omega = trainer.score_aug(model_input, target)

    # 3. Apply mode intensity scaling if enabled (after ScoreAug, before model)
    if trainer.use_mode_intensity_scaling and mode_id is not None:
        aug_input, _ = trainer._apply_mode_intensity_scale(aug_input, mode_id)

    # 4. Get prediction from augmented input with appropriate conditioning
    prediction = _call_model_with_conditioning(
        trainer, aug_input, timesteps, omega, mode_id
    )

    # 5. Compute MSE loss with augmented target
    if is_dual:
        pred_0 = prediction[:, 0:1, :, :]
        pred_1 = prediction[:, 1:2, :, :]
        mse_loss = (
            F.mse_loss(pred_0.float(), aug_target[keys[0]].float()) +
            F.mse_loss(pred_1.float(), aug_target[keys[1]].float())
        ) / 2
    else:
        mse_loss = F.mse_loss(prediction.float(), aug_target.float())

    # 6. Compute perceptual loss with inverse transform
    p_loss, predicted_clean = _compute_perceptual_with_inverse(
        trainer, prediction, noisy_images, images, timesteps, omega, is_dual
    )

    return mse_loss, p_loss, predicted_clean


def _call_model_with_conditioning(
    trainer: 'DiffusionTrainer',
    model_input: Tensor,
    timesteps: Tensor,
    omega: dict[str, Any] | None,
    mode_id: Tensor | None,
) -> Tensor:
    """Call model with appropriate omega/mode_id conditioning.

    Handles the different combinations of conditioning wrappers:
    - CombinedModelWrapper (omega + mode_id)
    - ScoreAugModelWrapper (omega only)
    - ModeEmbedModelWrapper (mode_id only)
    - Raw model (no conditioning)

    Args:
        trainer: DiffusionTrainer instance
        model_input: Formatted input tensor
        timesteps: Timestep tensor
        omega: ScoreAug omega parameters (dict with rotation, flip, etc.)
        mode_id: Mode ID tensor for multi-modality

    Returns:
        Model prediction tensor
    """
    if trainer.use_omega_conditioning:
        # Both CombinedModelWrapper and ScoreAugModelWrapper accept omega + mode_id
        return trainer.model(model_input, timesteps, omega=omega, mode_id=mode_id)
    elif trainer.use_mode_embedding:
        return trainer.model(model_input, timesteps, mode_id=mode_id)
    else:
        return trainer.strategy.predict_noise_or_velocity(trainer.model, model_input, timesteps)


def _reconstruct_clean(
    aug_noisy: Tensor | dict[str, Tensor],
    prediction: Tensor,
    timesteps: Tensor,
    strategy_name: str,
    num_timesteps: int,
) -> Tensor | dict[str, Tensor]:
    """Reconstruct clean images from augmented noisy and prediction.

    Handles both single tensor and dual-mode dict inputs.

    Args:
        aug_noisy: Augmented noisy images (tensor or dict of tensors).
        prediction: Model prediction (velocity for RFlow, noise for DDPM).
        timesteps: Timestep tensor.
        strategy_name: 'rflow' or 'ddpm'.
        num_timesteps: Number of training timesteps for normalization.

    Returns:
        Reconstructed clean images (same type as aug_noisy).
    """
    if strategy_name == 'rflow':
        t_norm = timesteps.float() / float(num_timesteps)
        t_exp = t_norm.view(-1, 1, 1, 1)

        if isinstance(aug_noisy, dict):
            keys = list(aug_noisy.keys())
            return {
                k: torch.clamp(aug_noisy[k] + t_exp * prediction[:, i:i+1], 0, 1)
                for i, k in enumerate(keys)
            }
        else:
            return torch.clamp(aug_noisy + t_exp * prediction, 0, 1)
    else:  # ddpm
        if isinstance(aug_noisy, dict):
            keys = list(aug_noisy.keys())
            return {
                k: torch.clamp(aug_noisy[k] - prediction[:, i:i+1], 0, 1)
                for i, k in enumerate(keys)
            }
        else:
            return torch.clamp(aug_noisy - prediction, 0, 1)


def _compute_perceptual_with_inverse(
    trainer: 'DiffusionTrainer',
    prediction: Tensor,
    noisy_images: Tensor | dict[str, Tensor],
    clean_images: Tensor | dict[str, Tensor],
    timesteps: Tensor,
    omega: dict[str, Any],
    is_dual: bool,
) -> tuple[Tensor, Tensor | dict[str, Tensor]]:
    """Compute perceptual loss, applying inverse transform if available.

    For ScoreAug, we need to:
    1. Apply the same omega transform to noisy_images
    2. Reconstruct clean from velocity/noise prediction
    3. Inverse transform to original space
    4. Compute perceptual loss if transform was invertible

    Args:
        trainer: DiffusionTrainer instance
        prediction: Model prediction (velocity or noise)
        noisy_images: Noisy images at timestep t (tensor or dict)
        clean_images: Original clean images for perceptual loss target
        timesteps: Timestep tensor
        omega: ScoreAug omega parameters
        is_dual: Whether this is dual-image mode

    Returns:
        Tuple of (perceptual_loss, predicted_clean):
        - perceptual_loss: Perceptual loss (0 if non-invertible or weight=0)
        - predicted_clean: Predicted clean images for visualization
    """
    device = prediction.device

    # If perceptual weight is 0, skip computation
    if trainer.perceptual_weight <= 0:
        # Return placeholder for predicted_clean
        return torch.tensor(0.0, device=device), clean_images

    # Compute predicted clean in augmented space, then inverse transform
    if is_dual:
        keys = list(noisy_images.keys())
        # Apply same transform to noisy_images for reconstruction
        stacked_noisy = torch.cat([noisy_images[k] for k in keys], dim=1)
        aug_noisy = trainer.score_aug.apply_omega(stacked_noisy, omega)
        aug_noisy_dict = {keys[0]: aug_noisy[:, 0:1], keys[1]: aug_noisy[:, 1:2]}

        # Reconstruct clean from prediction using shared helper
        aug_clean = _reconstruct_clean(
            aug_noisy_dict, prediction, timesteps,
            trainer.strategy_name, trainer.num_timesteps
        )

        # Inverse transform to original space
        inv_clean = {k: trainer.score_aug.inverse_apply_omega(v, omega) for k, v in aug_clean.items()}
        if any(v is None for v in inv_clean.values()):
            # Non-invertible transform (rotation/flip), skip perceptual loss
            logger.debug("Perceptual loss skipped: non-invertible ScoreAug transform applied")
            return torch.tensor(0.0, device=device), aug_clean
        else:
            # Compute perceptual loss with wrapper that handles dicts
            p_loss = trainer.perceptual_loss_fn(inv_clean.float(), clean_images.float())
            return p_loss, inv_clean
    else:
        # Single channel mode
        aug_noisy = trainer.score_aug.apply_omega(noisy_images, omega)

        # Reconstruct clean from prediction using shared helper
        aug_clean = _reconstruct_clean(
            aug_noisy, prediction, timesteps,
            trainer.strategy_name, trainer.num_timesteps
        )

        inv_clean = trainer.score_aug.inverse_apply_omega(aug_clean, omega)
        if inv_clean is None:
            # Non-invertible transform (rotation/flip), skip perceptual loss
            logger.debug("Perceptual loss skipped: non-invertible ScoreAug transform applied")
            return torch.tensor(0.0, device=device), aug_clean
        else:
            p_loss = trainer.perceptual_loss_fn(inv_clean.float(), clean_images.float())
            return p_loss, inv_clean


# ─────────────────────────────────────────────────────────────────────────────
# SDA (Shifted Data Augmentation) Loss Computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_sda_loss(
    trainer: 'DiffusionTrainer',
    images: Tensor | dict[str, Tensor],
    noise: Tensor | dict[str, Tensor],
    timesteps: Tensor,
    labels: Tensor | None,
    mode_id: Tensor | None = None,
) -> Tensor:
    """Compute SDA (Shifted Data Augmentation) loss.

    SDA transforms CLEAN images (unlike ScoreAug which transforms noisy data),
    then adds noise at SHIFTED timesteps to prevent temporal distribution leakage.
    The model learns to denoise both original and augmented data.

    The key insight is that if we transform clean images and add noise at the
    same timestep, the model could learn to "cheat" by detecting the transform
    from noise patterns. Shifting timesteps prevents this.

    Args:
        trainer: DiffusionTrainer instance
        images: Clean images (tensor or dict for dual mode)
        noise: Noise tensor (tensor or dict for dual mode)
        timesteps: Original sampled timesteps
        labels: Optional conditioning labels (e.g., seg mask)
        mode_id: Optional mode ID for multi-modality

    Returns:
        SDA loss component (weighted by trainer.sda_weight before adding to total).
        Returns 0 if SDA didn't apply a transform this call.

    Note:
        Requires trainer.sda to be set (i.e., SDA is enabled).
        SDA is mutually exclusive with ScoreAug.
    """
    device = images.device if not isinstance(images, dict) else next(iter(images.values())).device
    is_dual = isinstance(images, dict)

    if is_dual:
        return _compute_sda_loss_dual(trainer, images, noise, timesteps, labels, mode_id)
    else:
        return _compute_sda_loss_single(trainer, images, noise, timesteps, labels, mode_id)


def _compute_sda_loss_single(
    trainer: 'DiffusionTrainer',
    images: Tensor,
    noise: Tensor,
    timesteps: Tensor,
    labels: Tensor | None,
    mode_id: Tensor | None,
) -> Tensor:
    """Compute SDA loss for single-channel mode."""
    device = images.device

    # Apply SDA to clean images
    aug_images, sda_info = trainer.sda(images)

    # If no transform was applied, return zero loss
    if sda_info is None:
        return torch.tensor(0.0, device=device)

    # Shift timesteps for augmented path
    shifted_timesteps = trainer.sda.shift_timesteps(timesteps)

    # Transform noise to match transformed images
    aug_noise = trainer.sda.apply_to_target(noise, sda_info)

    # Add TRANSFORMED noise at SHIFTED timesteps
    aug_noisy = trainer.strategy.add_noise(aug_images, aug_noise, shifted_timesteps)

    # Format input and get prediction
    aug_labels_dict = {'labels': labels}
    aug_model_input = trainer.mode.format_model_input(aug_noisy, aug_labels_dict)

    if trainer.use_mode_embedding:
        aug_prediction = trainer.model(aug_model_input, shifted_timesteps, mode_id=mode_id)
    else:
        aug_prediction = trainer.strategy.predict_noise_or_velocity(
            trainer.model, aug_model_input, shifted_timesteps
        )

    # Compute augmented target using strategy's compute_target method
    aug_target = trainer.strategy.compute_target(aug_images, aug_noise)

    # Compute MSE loss
    return F.mse_loss(aug_prediction.float(), aug_target.float())


def _compute_sda_loss_dual(
    trainer: 'DiffusionTrainer',
    images: dict[str, Tensor],
    noise: dict[str, Tensor],
    timesteps: Tensor,
    labels: Tensor | None,
    mode_id: Tensor | None,
) -> Tensor:
    """Compute SDA loss for dual-image mode."""
    keys = list(images.keys())
    device = images[keys[0]].device

    # Stack images for joint transform
    stacked_images = torch.cat([images[k] for k in keys], dim=1)
    aug_stacked, sda_info = trainer.sda(stacked_images)

    # If no transform was applied, return zero loss
    if sda_info is None:
        return torch.tensor(0.0, device=device)

    # Unstack augmented images
    aug_images_dict = {
        keys[0]: aug_stacked[:, 0:1],
        keys[1]: aug_stacked[:, 1:2],
    }

    # Shift timesteps for augmented path
    shifted_timesteps = trainer.sda.shift_timesteps(timesteps)

    # Transform noise to match transformed images
    aug_noise_dict = {
        k: trainer.sda.apply_to_target(noise[k], sda_info)
        for k in keys
    }

    # Add TRANSFORMED noise at SHIFTED timesteps
    aug_noisy_dict = {
        k: trainer.strategy.add_noise(aug_images_dict[k], aug_noise_dict[k], shifted_timesteps)
        for k in keys
    }

    # Format input and get prediction
    aug_labels_dict = {'labels': labels}
    aug_model_input = trainer.mode.format_model_input(aug_noisy_dict, aug_labels_dict)

    if trainer.use_mode_embedding:
        aug_prediction = trainer.model(aug_model_input, shifted_timesteps, mode_id=mode_id)
    else:
        aug_prediction = trainer.strategy.predict_noise_or_velocity(
            trainer.model, aug_model_input, shifted_timesteps
        )

    # Compute augmented target using strategy's compute_target method
    aug_target_dict = trainer.strategy.compute_target(aug_images_dict, aug_noise_dict)

    # Compute MSE loss for each channel
    aug_mse = sum(
        F.mse_loss(aug_prediction[:, i:i+1].float(), aug_target_dict[k].float())
        for i, k in enumerate(keys)
    ) / len(keys)

    return aug_mse
