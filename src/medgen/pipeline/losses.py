"""Loss computation utilities for diffusion training.

This module provides loss functions for:
- Self-conditioning consistency loss
- Min-SNR weighted MSE loss
- Region-weighted MSE loss (tumor-focused)
"""
import logging
import random
from typing import Dict, Optional, Union, TYPE_CHECKING

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
    mode_id: Optional[Tensor] = None,
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
    self_cond_cfg = trainer.cfg.training.get('self_conditioning', {})
    if not self_cond_cfg.get('enabled', False):
        return torch.tensor(0.0, device=model_input.device)

    prob = self_cond_cfg.get('prob', 0.5)

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
    images: Union[Tensor, Dict[str, Tensor]],
    noise: Union[Tensor, Dict[str, Tensor]],
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
        pred_0, pred_1 = prediction[:, 0:1, :, :], prediction[:, 1:2, :, :]
        mse_0 = ((pred_0.float() - target_0.float()) ** 2).mean(dim=(1, 2, 3))
        mse_1 = ((pred_1.float() - target_1.float()) ** 2).mean(dim=(1, 2, 3))
        mse_per_sample = (mse_0 + mse_1) / 2
    else:
        target = images - noise if trainer.strategy_name == 'rflow' else noise
        mse_per_sample = ((prediction.float() - target.float()) ** 2).flatten(1).mean(1)

    return (mse_per_sample * snr_weights).mean()


def compute_region_weighted_mse(
    trainer: 'DiffusionTrainer',
    prediction: Tensor,
    images: Union[Tensor, Dict[str, Tensor]],
    noise: Union[Tensor, Dict[str, Tensor]],
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
        pred_0, pred_1 = prediction[:, 0:1, :, :], prediction[:, 1:2, :, :]

        # Per-pixel MSE with weights
        mse_0 = (pred_0.float() - target_0.float()) ** 2  # [B, 1, H, W]
        mse_1 = (pred_1.float() - target_1.float()) ** 2  # [B, 1, H, W]

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
