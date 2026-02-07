"""Extracted training helpers for BaseCompressionTrainer.

Module-level functions that implement the core training logic.
Each function takes a `trainer` (BaseCompressionTrainer instance) as its first
argument and accesses trainer attributes like `trainer.device`, `trainer.cfg`, etc.

The class keeps thin wrappers that delegate here; these functions call back
into the class for hook methods (resolved at runtime, no circular imports).
"""
from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

import torch
from torch.amp import autocast

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from .compression_trainer import BaseCompressionTrainer
    from .results import BatchType, TrainingStepResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------

def prepare_batch(
    trainer: BaseCompressionTrainer,
    batch: BatchType,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Prepare batch for compression training (2D or 3D).

    Handles multiple batch formats:
    - Tuple of (images, mask)
    - Dict with image keys (2D) or 'image'/'images' key (3D)
    - Single tensor

    Args:
        trainer: BaseCompressionTrainer instance.
        batch: Input batch.

    Returns:
        Tuple of (images, mask).
    """
    from medgen.core.dict_utils import get_with_fallbacks

    from .compression_trainer import _tensor_to_device

    # 3D-specific handling
    if trainer.spatial_dims == 3:
        if isinstance(batch, dict):
            images = get_with_fallbacks(batch, 'image', 'images')
            mask = get_with_fallbacks(batch, 'seg', 'mask')
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
            mask = batch[1] if len(batch) > 1 else None
        else:
            images = batch
            mask = None

        images = _tensor_to_device(images, trainer.device)
        mask = _tensor_to_device(mask, trainer.device) if mask is not None else None
        return images, mask

    # 2D handling
    # Handle tuple of (image, seg)
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        images, mask = batch
        return _tensor_to_device(images, trainer.device), _tensor_to_device(mask, trainer.device)

    # Handle dict batches
    if isinstance(batch, dict):
        image_keys = trainer.cfg.mode.image_keys
        tensors = [_tensor_to_device(batch[k], trainer.device) for k in image_keys if k in batch]
        images = torch.cat(tensors, dim=1)
        mask = _tensor_to_device(batch['seg'], trainer.device) if 'seg' in batch else None
        return images, mask

    # Handle tensor input
    tensor = _tensor_to_device(batch, trainer.device)

    # Check if seg is stacked as last channel
    n_image_channels = trainer.cfg.mode.in_channels
    if tensor.shape[1] > n_image_channels:
        images = tensor[:, :n_image_channels, :, :]
        mask = tensor[:, n_image_channels:n_image_channels + 1, :, :]
        return images, mask

    return tensor, None


# ---------------------------------------------------------------------------
# Discriminator step
# ---------------------------------------------------------------------------

def train_discriminator_step(
    trainer: BaseCompressionTrainer,
    images: torch.Tensor,
    reconstruction: torch.Tensor,
) -> torch.Tensor:
    """Train discriminator on real vs fake images.

    Args:
        trainer: BaseCompressionTrainer instance.
        images: Real images [B, C, H, W].
        reconstruction: Generated images [B, C, H, W].

    Returns:
        Discriminator loss.
    """
    if trainer.disable_gan or trainer.discriminator is None:
        return torch.tensor(0.0, device=trainer.device)

    trainer.optimizer_d.zero_grad(set_to_none=True)

    with autocast('cuda', enabled=True, dtype=trainer.weight_dtype):
        # Real images -> discriminator should output 1
        logits_real = trainer.discriminator(images.contiguous())
        # Fake images -> discriminator should output 0
        # Detach to prevent gradient flow through generator (saves memory)
        logits_fake = trainer.discriminator(reconstruction.detach().contiguous())

        d_loss = 0.5 * (
            trainer.adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
            + trainer.adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
        )

    d_loss.backward()

    # Gradient clipping and tracking
    grad_clip = trainer._training_config.gradient_clip_norm
    grad_norm_d = 0.0
    if grad_clip > 0:
        grad_norm_d = torch.nn.utils.clip_grad_norm_(
            trainer.discriminator_raw.parameters(), max_norm=grad_clip
        ).item()

    trainer.optimizer_d.step()

    # Track discriminator gradient norm
    if trainer.log_grad_norm:
        trainer._grad_norm_tracker_d.update(grad_norm_d)

    return d_loss


# ---------------------------------------------------------------------------
# Adversarial loss (generator side)
# ---------------------------------------------------------------------------

def compute_adversarial_loss(
    trainer: BaseCompressionTrainer,
    reconstruction: torch.Tensor,
) -> torch.Tensor:
    """Compute adversarial loss for generator.

    Args:
        trainer: BaseCompressionTrainer instance.
        reconstruction: Generated images [B, C, H, W].

    Returns:
        Adversarial loss.
    """
    if trainer.disable_gan or trainer.discriminator is None:
        return torch.tensor(0.0, device=trainer.device)

    logits_fake = trainer.discriminator(reconstruction.contiguous())
    return trainer.adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)


# ---------------------------------------------------------------------------
# Perceptual losses
# ---------------------------------------------------------------------------

def compute_perceptual_loss(
    trainer: BaseCompressionTrainer,
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute perceptual loss (2D or 2.5D for 3D).

    For 3D volumes with use_2_5d_perceptual=True, computes loss on
    sampled 2D slices. Otherwise computes standard perceptual loss.

    Args:
        trainer: BaseCompressionTrainer instance.
        reconstruction: Generated images/volumes.
        target: Target images/volumes.

    Returns:
        Perceptual loss value.
    """
    if trainer.perceptual_loss_fn is None:
        return torch.tensor(0.0, device=trainer.device)

    # 3D with 2.5D perceptual loss
    if trainer.spatial_dims == 3 and trainer.use_2_5d_perceptual:
        return compute_2_5d_perceptual_loss(trainer, reconstruction, target)

    return trainer.perceptual_loss_fn(reconstruction, target)


def compute_2_5d_perceptual_loss(
    trainer: BaseCompressionTrainer,
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute perceptual loss on sampled 2D slices from 3D volumes.

    Args:
        trainer: BaseCompressionTrainer instance.
        reconstruction: Reconstructed volume [B, C, D, H, W].
        target: Target volume [B, C, D, H, W].

    Returns:
        Perceptual loss averaged over sampled slices.
    """
    if trainer.perceptual_loss_fn is None:
        return torch.tensor(0.0, device=trainer.device)

    depth = reconstruction.shape[2]
    slice_fraction = trainer.perceptual_slice_fraction
    n_slices = max(1, int(depth * slice_fraction))

    # Sample slice indices
    indices = torch.randperm(depth)[:n_slices].to(trainer.device)

    total_loss = 0.0
    for idx in indices:
        recon_slice = reconstruction[:, :, idx, :, :]
        target_slice = target[:, :, idx, :, :]
        total_loss += trainer.perceptual_loss_fn(recon_slice, target_slice)

    return total_loss / n_slices


# ---------------------------------------------------------------------------
# KL loss (standalone – no trainer needed)
# ---------------------------------------------------------------------------

def compute_kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence loss for VAE training.

    Works for both 2D [B, C, H, W] and 3D [B, C, D, H, W] tensors by
    summing over all spatial dimensions, then averaging over batch.

    Args:
        mean: Mean of latent distribution.
        logvar: Log variance of latent distribution.

    Returns:
        KL divergence loss (scalar).
    """
    # Sum over all spatial dimensions (everything except batch dim 0)
    spatial_dims = list(range(1, mean.dim()))
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=spatial_dims)
    return kl.mean()


# ---------------------------------------------------------------------------
# Full train step
# ---------------------------------------------------------------------------

def compression_train_step(
    trainer: BaseCompressionTrainer,
    batch: BatchType,
) -> TrainingStepResult:
    """Template train step for compression trainers.

    Implements the common training step pattern shared by VAE, VQ-VAE, and DC-AE.
    Subclasses customize via hook methods:
    - _forward_for_training(): Model-specific forward pass returning (reconstruction, reg_loss)
    - _get_reconstruction_loss_weight(): Return L1 weight (1.0 for VAE/VQ-VAE, configurable for DC-AE)
    - _use_discriminator_before_generator(): Whether to run D step before G (VAE/VQ-VAE 2D: True)
    - _track_seg_breakdown(): Track seg loss breakdown for epoch averaging

    Args:
        trainer: BaseCompressionTrainer instance.
        batch: Input batch.

    Returns:
        TrainingStepResult with all loss components.
    """
    from .results import TrainingStepResult

    images, mask = trainer._prepare_batch(batch)
    grad_clip = trainer._training_config.gradient_clip_norm

    d_loss = torch.tensor(0.0, device=trainer.device)
    adv_loss = torch.tensor(0.0, device=trainer.device)

    # ==================== Discriminator Step (before generator, if applicable) ====================
    if trainer._use_discriminator_before_generator() and not trainer.disable_gan:
        with torch.no_grad():
            with autocast('cuda', enabled=True, dtype=trainer.weight_dtype):
                reconstruction_for_d, _ = trainer._forward_for_training(images)
        d_loss = trainer._train_discriminator_step(images, reconstruction_for_d)

    # ==================== Generator Step ====================
    trainer.optimizer.zero_grad(set_to_none=True)

    with autocast('cuda', enabled=True, dtype=trainer.weight_dtype):
        # Model-specific forward pass
        reconstruction, reg_loss = trainer._forward_for_training(images)

        # Compute reconstruction loss (L1 or seg-specific)
        seg_mode = trainer.seg_mode
        seg_loss_fn = trainer.seg_loss_fn

        if seg_mode and seg_loss_fn is not None:
            seg_loss, seg_breakdown = seg_loss_fn(reconstruction, images)
            l1_loss = seg_loss
            p_loss = torch.tensor(0.0, device=trainer.device)
            trainer._track_seg_breakdown(seg_breakdown)
        else:
            l1_loss = torch.nn.functional.l1_loss(reconstruction.float(), images.float())
            p_loss = trainer._compute_perceptual_loss(reconstruction.float(), images.float())

        # Adversarial loss
        if not trainer.disable_gan:
            adv_loss = trainer._compute_adversarial_loss(reconstruction)

        # Total generator loss with configurable weights
        l1_weight = trainer._get_reconstruction_loss_weight()
        g_loss = (
            l1_weight * l1_loss
            + trainer.perceptual_weight * p_loss
            + reg_loss
            + trainer.adv_weight * adv_loss
        )

    g_loss.backward()

    # Gradient clipping
    grad_norm_g = 0.0
    if grad_clip > 0:
        grad_norm_g = torch.nn.utils.clip_grad_norm_(
            trainer.model_raw.parameters(), max_norm=grad_clip
        ).item()

    trainer.optimizer.step()

    # Track gradient norm
    if trainer.log_grad_norm:
        trainer._grad_norm_tracker.update(grad_norm_g)

    # Update EMA
    trainer._update_ema()

    # ==================== Discriminator Step (after generator, if applicable) ====================
    if not trainer._use_discriminator_before_generator() and not trainer.disable_gan:
        d_loss = trainer._train_discriminator_step(images, reconstruction.detach())

    return TrainingStepResult(
        total_loss=g_loss.item(),
        reconstruction_loss=l1_loss.item(),
        perceptual_loss=p_loss.item() if isinstance(p_loss, torch.Tensor) else p_loss,
        regularization_loss=reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
        adversarial_loss=adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
        discriminator_loss=d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss,
    )


# ---------------------------------------------------------------------------
# Full train epoch
# ---------------------------------------------------------------------------

def compression_train_epoch(
    trainer: BaseCompressionTrainer,
    loader: DataLoader,
    epoch: int,
) -> dict[str, float]:
    """Train for one epoch using template method pattern.

    Subclasses customize via hook methods:
    - _get_loss_key(): Return loss dictionary key for regularization ('kl', 'vq', None)
    - _get_postfix_metrics(): Return metrics dict for progress bar
    - _on_train_epoch_start(): Optional setup (e.g., seg breakdown tracking)
    - _on_train_epoch_end(): Optional teardown (e.g., seg breakdown averaging)

    Args:
        trainer: BaseCompressionTrainer instance.
        loader: Training data loader.
        epoch: Current epoch number.

    Returns:
        Dict with average losses.
    """
    from tqdm import tqdm

    from .utils import create_epoch_iterator, get_vram_usage

    trainer.model.train()
    if not trainer.disable_gan and trainer.discriminator is not None:
        trainer.discriminator.train()

    trainer._loss_accumulator.reset()
    trainer._on_train_epoch_start(epoch)

    # Create epoch iterator (handles 2D vs 3D differences)
    if trainer.spatial_dims == 3:
        disable_pbar = not trainer.is_main_process or trainer.is_cluster
        total = trainer.limit_train_batches if trainer.limit_train_batches else len(loader)
        iterator = itertools.islice(loader, trainer.limit_train_batches) if trainer.limit_train_batches else loader
        epoch_iter = tqdm(iterator, desc=f"Epoch {epoch}", disable=disable_pbar, total=total)
    else:
        epoch_iter = create_epoch_iterator(
            loader, epoch, trainer.is_cluster, trainer.is_main_process,
            limit_batches=trainer.limit_train_batches
        )

    for step, batch in enumerate(epoch_iter):
        result = trainer.train_step(batch)
        losses = result.to_legacy_dict(trainer._get_loss_key())

        # Step profiler to mark training step boundary
        trainer._profiler_step()

        # Accumulate with unified system
        trainer._loss_accumulator.update(losses)

        if hasattr(epoch_iter, 'set_postfix'):
            avg_so_far = trainer._loss_accumulator.compute()
            epoch_iter.set_postfix(trainer._get_postfix_metrics(avg_so_far, losses))

        if epoch == 1 and step == 0 and trainer.is_main_process:
            logger.info(get_vram_usage(trainer.device))

    # Compute average losses using unified system
    avg_losses = trainer._loss_accumulator.compute()

    # Track batch count for seg breakdown averaging
    trainer._last_epoch_batch_count = trainer._loss_accumulator._count

    # Call subclass hook for post-epoch processing
    trainer._on_train_epoch_end(epoch, avg_losses)

    # Log training metrics using unified system
    trainer._log_training_metrics_unified(epoch, avg_losses)

    return avg_losses
