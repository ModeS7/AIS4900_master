"""Loss computation utilities for diffusion training.

This module provides loss functions for:
- Self-conditioning consistency loss
- Min-SNR weighted MSE loss
- Region-weighted MSE loss (tumor-focused)
- LPIPS-Huber premetric (Lee et al., NeurIPS 2024)
- Pseudo-Huber loss

Timestep convention (MONAI RFlowScheduler):
    t=0 → clean data, t=T (t_norm=1) → pure noise.
    All (1-t) weights in this file use this convention:
    (1-t_norm) = 1.0 near clean, 0.0 near noise.
"""
import logging
import random
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

if TYPE_CHECKING:
    from medgen.pipeline.trainer import DiffusionTrainer

logger = logging.getLogger(__name__)


def compute_self_conditioning_loss(
    trainer: 'DiffusionTrainer',
    model_input: Tensor,
    timesteps: Tensor,
    prediction: Tensor,
    mode_id: Tensor | None = None,
) -> Tensor:
    """Compute self-conditioning consistency loss.

    With probability `prob`, runs model a second time and computes
    consistency loss between the two predictions.

    Args:
        trainer: The DiffusionTrainer instance.
        model_input: Current model input tensor.
        timesteps: Current timesteps.
        prediction: Current prediction from main forward pass.
        mode_id: Optional mode ID for multi-modality.

    Returns:
        Consistency loss (0 if disabled or skipped this batch).
    """
    tt = trainer._training_tricks
    if not tt.self_cond.enabled:
        return torch.tensor(0.0, device=model_input.device)

    prob = tt.self_cond.prob

    # With probability (1-prob), skip self-conditioning
    if random.random() >= prob:
        return torch.tensor(0.0, device=model_input.device)

    # Get second prediction (detached first prediction as reference)
    with torch.no_grad():
        if trainer.use_mode_embedding:
            prediction_ref = trainer.model(model_input, timesteps, mode_id=mode_id)
        else:
            prediction_ref = trainer.model(x=model_input, timesteps=timesteps)
        prediction_ref = prediction_ref.detach()

    # Consistency loss: predictions should be similar
    consistency_loss = F.mse_loss(prediction.float(), prediction_ref.float())

    return consistency_loss


def compute_min_snr_weighted_mse(
    trainer: 'DiffusionTrainer',
    prediction: Tensor,
    images: Tensor | dict[str, Tensor],
    noise: Tensor | dict[str, Tensor],
    timesteps: Tensor,
) -> Tensor:
    """Compute MSE loss with Min-SNR weighting.

    Applies per-sample SNR-based weights to the MSE loss to prevent
    high-noise timesteps from dominating training.

    Args:
        trainer: The DiffusionTrainer instance.
        prediction: Model prediction (noise or velocity).
        images: Original clean images.
        noise: Added noise.
        timesteps: Diffusion timesteps for each sample.

    Returns:
        Weighted MSE loss scalar.
    """
    snr_weights = trainer._unified_metrics.compute_snr_weights(timesteps)

    # Cast to FP32 for MSE computation (BF16 underflow causes ~15-20% lower loss)
    if isinstance(images, dict):
        keys = list(images.keys())
        if trainer.strategy_name == 'rflow':
            target_0 = images[keys[0]] - noise[keys[0]]
            target_1 = images[keys[1]] - noise[keys[1]]
        else:
            target_0, target_1 = noise[keys[0]], noise[keys[1]]
        pred_0, pred_1 = prediction[:, 0:1], prediction[:, 1:2]
        mse_0 = ((pred_0.float() - target_0.float()) ** 2).flatten(1).mean(1)
        mse_1 = ((pred_1.float() - target_1.float()) ** 2).flatten(1).mean(1)
        mse_per_sample = (mse_0 + mse_1) / 2
    else:
        target = images - noise if trainer.strategy_name == 'rflow' else noise
        mse_per_sample = ((prediction.float() - target.float()) ** 2).flatten(1).mean(1)

    return (mse_per_sample * snr_weights).mean()


def compute_rflow_snr_weighted_mse(
    prediction: Tensor,
    target: Tensor | dict[str, Tensor],
    timesteps: Tensor,
    num_timesteps: int,
    gamma: float = 5.0,
) -> Tensor:
    """Compute MSE loss with Min-SNR-γ weighting for RFlow.

    For RFlow: SNR(t̃) = ((1-t̃)/t̃)². Weight: min(SNR, γ) / SNR.
    Downweights high-SNR timesteps (t̃≈0, near clean) where the model
    wastes capacity on trivially easy predictions.

    Args:
        prediction: Model velocity prediction [B, C, ...].
        target: Target velocity (images - noise). Tensor or dict for dual mode.
        timesteps: Diffusion timesteps [B] in [0, num_timesteps].
        num_timesteps: Total training timesteps (e.g. 1000).
        gamma: SNR clipping threshold. 5.0 is standard.

    Returns:
        Weighted MSE loss scalar.
    """
    t_norm = (timesteps.float() / num_timesteps).clamp(1e-5, 1 - 1e-5)
    snr = ((1 - t_norm) / t_norm) ** 2
    weight = snr.clamp(max=gamma) / snr.clamp(min=1e-8)

    if isinstance(target, dict):
        keys = list(target.keys())
        pred_0, pred_1 = prediction[:, 0:1], prediction[:, 1:2]
        mse_0 = ((pred_0.float() - target[keys[0]].float()) ** 2).flatten(1).mean(1)
        mse_1 = ((pred_1.float() - target[keys[1]].float()) ** 2).flatten(1).mean(1)
        per_sample_mse = (mse_0 + mse_1) / 2
    else:
        per_sample_mse = ((prediction.float() - target.float()) ** 2).flatten(1).mean(1)

    return (per_sample_mse * weight).mean()


def compute_pseudo_huber_loss(
    prediction: Tensor,
    target: Tensor | dict[str, Tensor],
) -> Tensor:
    """Pseudo-Huber loss for velocity prediction.

    From "Improving Training of Rectified Flows" (NeurIPS 2024).
    L = sqrt(||v_pred - v_target||² + c²) - c

    Uses per-sample squared L2 norm (not per-element mean), then batch mean.
    c = 0.00054 * sqrt(d) where d = total data dimensionality per sample.

    Args:
        prediction: Model velocity prediction [B, C, ...].
        target: Target velocity. Tensor or dict for dual mode.

    Returns:
        Pseudo-Huber loss scalar.
    """
    if isinstance(target, dict):
        keys = list(target.keys())
        pred_0, pred_1 = prediction[:, 0:1], prediction[:, 1:2]
        error_0 = (pred_0.float() - target[keys[0]].float()).flatten(1)
        error_1 = (pred_1.float() - target[keys[1]].float()).flatten(1)
        # d = dimensionality per sample per modality
        d = error_0.shape[1]
        c = 0.00054 * (d ** 0.5)
        sq_norm_0 = (error_0 ** 2).sum(1)
        sq_norm_1 = (error_1 ** 2).sum(1)
        huber_0 = (sq_norm_0 + c * c).sqrt() - c
        huber_1 = (sq_norm_1 + c * c).sqrt() - c
        return ((huber_0 + huber_1) / 2).mean()
    else:
        error = (prediction.float() - target.float()).flatten(1)
        d = error.shape[1]
        c = 0.00054 * (d ** 0.5)
        sq_norm = (error ** 2).sum(1)
        return ((sq_norm + c * c).sqrt() - c).mean()


def compute_lpips_huber_loss(
    prediction: Tensor,
    target: Tensor | dict[str, Tensor],
    timesteps: Tensor,
    num_timesteps: int,
) -> Tensor:
    """(1-t)-weighted Pseudo-Huber loss for LPIPS-Huber combination.

    From "Improving Training of Rectified Flows" (Lee et al., NeurIPS 2024).
    Paper formula (Eq. in Section 3.2):

        L = (1-t) * Huber(v_target, v_pred) + LPIPS(x₀, x̂₀)

    This function computes ONLY the (1-t)*Huber term.
    The LPIPS term is computed separately in the trainer (constant weight,
    no time dependence — matching the paper).

    Convention (MONAI RFlowScheduler, same as Lee et al.):
        t=0 → clean data,  t=T (t_norm=1) → pure noise.
        (1-t_norm) = 1.0 near clean, 0.0 near noise.
        NOTE: REVERSED from original RF paper (Liu 2023) where t=0 is noise.

    Behavior:
        At t≈0 (clean): (1-t)≈1 → full Huber + LPIPS
        At t≈1 (noise): (1-t)≈0 → LPIPS only (Huber fades out)

    Args:
        prediction: Model velocity prediction [B, C, ...].
        target: Target velocity. Tensor or dict for dual mode.
        timesteps: Diffusion timesteps [B] in [0, num_timesteps].
        num_timesteps: Total training timesteps (e.g. 1000).

    Returns:
        Time-weighted Pseudo-Huber loss scalar.
    """
    t_norm = (timesteps.float() / num_timesteps).clamp(0, 1)  # [B]
    weight = 1.0 - t_norm  # [B]

    if isinstance(target, dict):
        keys = list(target.keys())
        pred_0, pred_1 = prediction[:, 0:1], prediction[:, 1:2]
        error_0 = (pred_0.float() - target[keys[0]].float()).flatten(1)
        error_1 = (pred_1.float() - target[keys[1]].float()).flatten(1)
        d = error_0.shape[1]
        c = 0.00054 * (d ** 0.5)
        sq_norm_0 = (error_0 ** 2).sum(1)
        sq_norm_1 = (error_1 ** 2).sum(1)
        huber_0 = (sq_norm_0 + c * c).sqrt() - c
        huber_1 = (sq_norm_1 + c * c).sqrt() - c
        per_sample = (huber_0 + huber_1) / 2
    else:
        error = (prediction.float() - target.float()).flatten(1)
        d = error.shape[1]
        c = 0.00054 * (d ** 0.5)
        sq_norm = (error ** 2).sum(1)
        per_sample = (sq_norm + c * c).sqrt() - c

    return (per_sample * weight).mean()


def compute_region_weighted_mse(
    trainer: 'DiffusionTrainer',
    prediction: Tensor,
    images: Tensor | dict[str, Tensor],
    noise: Tensor | dict[str, Tensor],
    seg_mask: Tensor,
) -> Tensor:
    """Compute MSE loss with per-pixel regional weighting.

    Applies higher weights to small tumor regions based on RANO-BM
    size classification using Feret diameter.

    Args:
        trainer: The DiffusionTrainer instance.
        prediction: Model prediction (noise or velocity).
        images: Original clean images.
        noise: Added noise.
        seg_mask: Binary segmentation mask [B, 1, H, W].

    Returns:
        Region-weighted MSE loss scalar.
    """
    # Compute weight map from segmentation mask
    weight_map = trainer.regional_weight_computer(seg_mask)  # [B, 1, H, W]

    # Compute per-pixel MSE
    if isinstance(images, dict):
        keys = list(images.keys())
        if trainer.strategy_name == 'rflow':
            target_0 = images[keys[0]] - noise[keys[0]]
            target_1 = images[keys[1]] - noise[keys[1]]
        else:
            target_0, target_1 = noise[keys[0]], noise[keys[1]]
        pred_0, pred_1 = prediction[:, 0:1], prediction[:, 1:2]

        # Per-pixel MSE with weights
        mse_0 = (pred_0.float() - target_0.float()) ** 2  # [B, 1, ...]
        mse_1 = (pred_1.float() - target_1.float()) ** 2  # [B, 1, ...]

        # Apply regional weights
        weighted_mse_0 = (mse_0 * weight_map).mean()
        weighted_mse_1 = (mse_1 * weight_map).mean()
        return (weighted_mse_0 + weighted_mse_1) / 2
    else:
        target = images - noise if trainer.strategy_name == 'rflow' else noise
        mse = (prediction.float() - target.float()) ** 2  # [B, C, H, W]

        # Apply regional weights (broadcast over channels)
        weighted_mse = (mse * weight_map).mean()
        return weighted_mse
