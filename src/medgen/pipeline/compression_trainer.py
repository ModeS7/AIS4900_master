"""
Compression trainer module for VAE, VQ-VAE, DC-AE, and 3D variants.

This module provides:
- BaseCompressionTrainer: Base class for all compression model trainers (2D)
- BaseCompression3DTrainer: Base class for 3D volumetric trainers

All compression trainers share:
- GAN training (optional discriminator)
- Perceptual loss
- EMA support
- Gradient norm tracking for G and D
- Worst batch tracking and visualization
- Regional metrics (tumor vs background)
"""
import json
import logging
import os
import time
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ema_pytorch import EMA
from omegaconf import DictConfig
from torch import nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from monai.losses import PatchAdversarialLoss
from monai.networks.nets import PatchDiscriminator

from medgen.core import create_warmup_cosine_scheduler, wrap_model_for_training
from .base_trainer import BaseTrainer
from .losses import PerceptualLoss
from .metrics import (
    RegionalMetricsTracker,
    compute_lpips,
    compute_lpips_3d,
    compute_msssim,
    compute_msssim_2d_slicewise,
    compute_psnr,
    reset_msssim_nan_warning,
)
from .tracking import GradientNormTracker, create_worst_batch_figure
from .utils import create_epoch_iterator, get_vram_usage, save_full_checkpoint

logger = logging.getLogger(__name__)


class BaseCompressionTrainer(BaseTrainer):
    """Base trainer for compression models (VAE, VQ-VAE, DC-AE).

    Extends BaseTrainer with:
    - GAN training infrastructure (discriminator, adversarial loss)
    - Perceptual loss
    - EMA support for generator
    - Dual gradient tracking (G and D)
    - Worst batch visualization
    - Per-modality validation

    Subclasses must implement:
    - setup_model(): Create model, discriminator, optimizers
    - _forward_for_validation(): Model-specific forward pass
    - _get_model_config(): Return config dict for checkpoint
    - train_step(): Model-specific training step

    Args:
        cfg: Hydra configuration object.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize compression trainer.

        Args:
            cfg: Hydra configuration object.
        """
        super().__init__(cfg)

        # ─────────────────────────────────────────────────────────────────────
        # Extract compression-specific config
        # ─────────────────────────────────────────────────────────────────────
        self.disc_lr: float = self._get_disc_lr(cfg)
        self.perceptual_weight: float = self._get_perceptual_weight(cfg)
        self.adv_weight: float = self._get_adv_weight(cfg)

        # GAN config
        self.disable_gan: bool = self._get_disable_gan(cfg)
        self.disc_num_layers: int = self._get_disc_num_layers(cfg)
        self.disc_num_channels: int = self._get_disc_num_channels(cfg)

        # EMA config
        self.use_ema: bool = cfg.training.get('use_ema', True)
        self.ema_decay: float = cfg.training.ema.get('decay', 0.999)

        # torch.compile option
        self.use_compile: bool = cfg.training.get('use_compile', True)
        self.compile_mode: str = "default"  # Subclasses can override

        # Precision config
        precision_cfg = cfg.training.get('precision', {})
        self.pure_weights: bool = precision_cfg.get('pure_weights', False)
        dtype_str = precision_cfg.get('dtype', 'bf16')
        self.weight_dtype: torch.dtype = {
            'bf16': torch.bfloat16,
            'fp16': torch.float16,
            'fp32': torch.float32,
        }.get(dtype_str, torch.bfloat16)

        # Constant LR option (for progressive training)
        progressive_cfg = cfg.get('progressive', {})
        self.use_constant_lr: bool = progressive_cfg.get('use_constant_lr', False)

        # ─────────────────────────────────────────────────────────────────────
        # Initialize compression-specific trackers
        # ─────────────────────────────────────────────────────────────────────
        self._grad_norm_tracker_d = GradientNormTracker()

        # ─────────────────────────────────────────────────────────────────────
        # Model placeholders (set in setup_model)
        # ─────────────────────────────────────────────────────────────────────
        self.discriminator: Optional[nn.Module] = None
        self.discriminator_raw: Optional[nn.Module] = None
        self.optimizer_d: Optional[AdamW] = None
        self.lr_scheduler_d: Optional[LRScheduler] = None
        self.perceptual_loss_fn: Optional[PerceptualLoss] = None
        self.adv_loss_fn: Optional[PatchAdversarialLoss] = None
        self.ema: Optional[EMA] = None

        # Per-modality validation loaders (set in train())
        self.per_modality_val_loaders: Optional[Dict[str, DataLoader]] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Config extraction methods (can be overridden by subclasses)
    # ─────────────────────────────────────────────────────────────────────────

    # List of config sections to check for trainer-specific settings
    _CONFIG_SECTIONS = ('vae', 'vqvae', 'dcae', 'vae_3d', 'vqvae_3d')

    def _get_config_value(self, cfg: DictConfig, key: str, default: Any) -> Any:
        """Get a value from any trainer config section.

        Searches through vae, vqvae, dcae, vae_3d, vqvae_3d sections
        for the specified key.

        Args:
            cfg: Hydra configuration object.
            key: Config key to look for.
            default: Default value if key not found.

        Returns:
            Value from config or default.
        """
        for section in self._CONFIG_SECTIONS:
            if section in cfg:
                return cfg[section].get(key, default)
        return default

    def _get_disc_lr(self, cfg: DictConfig) -> float:
        """Get discriminator learning rate from config."""
        return self._get_config_value(cfg, 'disc_lr', 5e-4)

    def _get_perceptual_weight(self, cfg: DictConfig) -> float:
        """Get perceptual loss weight from config."""
        return self._get_config_value(cfg, 'perceptual_weight', 0.001)

    def _get_adv_weight(self, cfg: DictConfig) -> float:
        """Get adversarial loss weight from config."""
        return self._get_config_value(cfg, 'adv_weight', 0.01)

    def _get_disable_gan(self, cfg: DictConfig) -> bool:
        """Get disable_gan flag from config."""
        # Check progressive config first (for staged training)
        progressive_cfg = cfg.get('progressive', {})
        if progressive_cfg.get('disable_gan', False):
            return True
        return self._get_config_value(cfg, 'disable_gan', False)

    def _get_disc_num_layers(self, cfg: DictConfig) -> int:
        """Get discriminator number of layers from config."""
        return self._get_config_value(cfg, 'disc_num_layers', 3)

    def _get_disc_num_channels(self, cfg: DictConfig) -> int:
        """Get discriminator number of channels from config."""
        return self._get_config_value(cfg, 'disc_num_channels', 64)

    # ─────────────────────────────────────────────────────────────────────────
    # Model setup helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _create_discriminator(
        self,
        n_channels: int,
        spatial_dims: int = 2,
    ) -> nn.Module:
        """Create PatchDiscriminator.

        Args:
            n_channels: Number of input channels.
            spatial_dims: Spatial dimensions (2 for 2D, 3 for 3D).

        Returns:
            PatchDiscriminator model on device.
        """
        return PatchDiscriminator(
            spatial_dims=spatial_dims,
            in_channels=n_channels,
            channels=self.disc_num_channels,
            num_layers_d=self.disc_num_layers,
        ).to(self.device)

    def _create_perceptual_loss(self, spatial_dims: int = 2) -> PerceptualLoss:
        """Create perceptual loss function.

        Args:
            spatial_dims: Spatial dimensions (2 for 2D, 3 for 3D).

        Returns:
            PerceptualLoss instance.
        """
        cache_dir = getattr(self.cfg.paths, 'cache_dir', None)
        return PerceptualLoss(
            spatial_dims=spatial_dims,
            network_type="radimagenet_resnet50",
            cache_dir=cache_dir,
            pretrained=True,
            device=self.device,
            use_compile=self.use_compile,
        )

    def _setup_optimizers(self, n_channels: int) -> None:
        """Setup optimizers and schedulers for generator and discriminator.

        Args:
            n_channels: Number of input channels (for discriminator).
        """
        # Generator optimizer
        self.optimizer = AdamW(self.model_raw.parameters(), lr=self.learning_rate)

        # Discriminator optimizer (only if GAN enabled)
        if not self.disable_gan and self.discriminator_raw is not None:
            self.optimizer_d = AdamW(self.discriminator_raw.parameters(), lr=self.disc_lr)

        # LR schedulers (only if not using constant LR)
        if not self.use_constant_lr:
            self.lr_scheduler = create_warmup_cosine_scheduler(
                self.optimizer,
                warmup_epochs=self.warmup_epochs,
                total_epochs=self.n_epochs,
            )
            if not self.disable_gan and self.optimizer_d is not None:
                self.lr_scheduler_d = create_warmup_cosine_scheduler(
                    self.optimizer_d,
                    warmup_epochs=self.warmup_epochs,
                    total_epochs=self.n_epochs,
                )
        else:
            if self.is_main_process:
                logger.info(f"Using constant LR: {self.learning_rate} (scheduler disabled)")

    def _setup_ema(self) -> None:
        """Setup EMA for generator if enabled."""
        if self.use_ema and self.model_raw is not None:
            self.ema = EMA(
                self.model_raw,
                beta=self.ema_decay,
                update_after_step=self.cfg.training.ema.get('update_after_step', 100),
                update_every=self.cfg.training.ema.get('update_every', 10),
            )
            if self.is_main_process:
                logger.info(f"EMA enabled with decay={self.ema_decay}")

    def _wrap_models(
        self,
        raw_model: nn.Module,
        raw_disc: Optional[nn.Module] = None,
    ) -> None:
        """Wrap models with DDP and/or torch.compile.

        Args:
            raw_model: Raw generator model.
            raw_disc: Raw discriminator model (optional).
        """
        # Convert weights to target dtype if pure_weights is enabled
        if self.pure_weights and self.weight_dtype != torch.float32:
            raw_model = raw_model.to(self.weight_dtype)
            if raw_disc is not None:
                raw_disc = raw_disc.to(self.weight_dtype)
            if self.is_main_process:
                logger.info(f"Converted model weights to {self.weight_dtype}")

        # Wrap generator
        self.model, self.model_raw = wrap_model_for_training(
            raw_model,
            use_multi_gpu=self.use_multi_gpu,
            local_rank=self.local_rank if self.use_multi_gpu else 0,
            use_compile=self.use_compile,
            compile_mode=self.compile_mode,
            is_main_process=self.is_main_process,
        )

        # Wrap discriminator if provided
        if raw_disc is not None:
            self.discriminator, self.discriminator_raw = wrap_model_for_training(
                raw_disc,
                use_multi_gpu=self.use_multi_gpu,
                local_rank=self.local_rank if self.use_multi_gpu else 0,
                use_compile=self.use_compile,
                compile_mode=self.compile_mode,
                is_main_process=False,  # Suppress duplicate logging
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Training helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _prepare_batch(self, batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare batch for 2D compression training.

        Handles multiple batch formats:
        - Tuple of (images, mask)
        - Dict with image keys
        - Single tensor

        Uses non_blocking=True for async GPU transfers to overlap with computation.

        Args:
            batch: Input batch.

        Returns:
            Tuple of (images, mask).
        """
        # Handle tuple of (image, seg)
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            images, mask = batch
            if hasattr(images, 'as_tensor'):
                images = images.as_tensor()
            if hasattr(mask, 'as_tensor'):
                mask = mask.as_tensor()
            return (
                images.to(self.device, non_blocking=True),
                mask.to(self.device, non_blocking=True)
            )

        # Handle dict batches
        if isinstance(batch, dict):
            image_keys = self.cfg.mode.get('image_keys', ['t1_pre', 't1_gd'])
            tensors = []
            for key in image_keys:
                if key in batch:
                    tensors.append(batch[key].to(self.device, non_blocking=True))
            images = torch.cat(tensors, dim=1)
            mask = batch['seg'].to(self.device, non_blocking=True) if 'seg' in batch else None
            return images, mask

        # Handle tensor input
        if hasattr(batch, 'as_tensor'):
            tensor = batch.as_tensor().to(self.device, non_blocking=True)
        else:
            tensor = batch.to(self.device, non_blocking=True)

        # Check if seg is stacked as last channel
        n_image_channels = self.cfg.mode.get('in_channels', 2)
        if tensor.shape[1] > n_image_channels:
            images = tensor[:, :n_image_channels, :, :]
            mask = tensor[:, n_image_channels:n_image_channels + 1, :, :]
            return images, mask

        return tensor, None

    def _train_discriminator_step(
        self,
        images: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        """Train discriminator on real vs fake images.

        Args:
            images: Real images [B, C, H, W].
            reconstruction: Generated images [B, C, H, W].

        Returns:
            Discriminator loss.
        """
        if self.disable_gan or self.discriminator is None:
            return torch.tensor(0.0, device=self.device)

        self.optimizer_d.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            # Real images -> discriminator should output 1
            logits_real = self.discriminator(images.contiguous())
            # Fake images -> discriminator should output 0
            logits_fake = self.discriminator(reconstruction.contiguous())

            d_loss = 0.5 * (
                self.adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                + self.adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
            )

        d_loss.backward()

        # Gradient clipping and tracking
        grad_clip = self.cfg.training.get('gradient_clip_norm', 1.0)
        grad_norm_d = 0.0
        if grad_clip > 0:
            grad_norm_d = torch.nn.utils.clip_grad_norm_(
                self.discriminator_raw.parameters(), max_norm=grad_clip
            ).item()

        self.optimizer_d.step()

        # Track discriminator gradient norm
        if self.log_grad_norm:
            self._grad_norm_tracker_d.update(grad_norm_d)

        return d_loss

    def _compute_adversarial_loss(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Compute adversarial loss for generator.

        Args:
            reconstruction: Generated images [B, C, H, W].

        Returns:
            Adversarial loss.
        """
        if self.disable_gan or self.discriminator is None:
            return torch.tensor(0.0, device=self.device)

        logits_fake = self.discriminator(reconstruction.contiguous())
        return self.adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)

    def _compute_perceptual_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss.

        Args:
            reconstruction: Generated images.
            target: Target images.

        Returns:
            Perceptual loss value.
        """
        if self.perceptual_loss_fn is None:
            return torch.tensor(0.0, device=self.device)
        return self.perceptual_loss_fn(reconstruction, target)

    def _update_ema(self) -> None:
        """Update EMA if enabled."""
        if self.ema is not None:
            self.ema.update()

    def _get_model_for_eval(self) -> nn.Module:
        """Get model for evaluation (EMA or regular).

        Returns:
            Model to use for evaluation.
        """
        if self.ema is not None:
            return self.ema.ema_model
        return self.model_raw

    # ─────────────────────────────────────────────────────────────────────────
    # Gradient norm tracking
    # ─────────────────────────────────────────────────────────────────────────

    def _on_epoch_start(self, epoch: int) -> None:
        """Reset gradient trackers at start of epoch."""
        super()._on_epoch_start(epoch)
        self._grad_norm_tracker.reset()
        self._grad_norm_tracker_d.reset()

    def _log_grad_norms(self, epoch: int) -> None:
        """Log gradient norm statistics to TensorBoard."""
        if not self.log_grad_norm or self.writer is None:
            return
        self._grad_norm_tracker.log(self.writer, epoch, prefix='training/grad_norm_g')
        if not self.disable_gan and self.discriminator is not None:
            self._grad_norm_tracker_d.log(self.writer, epoch, prefix='training/grad_norm_d')

    # ─────────────────────────────────────────────────────────────────────────
    # Epoch hooks
    # ─────────────────────────────────────────────────────────────────────────

    def _on_epoch_end(
        self,
        epoch: int,
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """Handle end of epoch: step schedulers, log metrics."""
        # Step schedulers (only if not using constant LR)
        if not self.use_constant_lr:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if not self.disable_gan and self.lr_scheduler_d is not None:
                self.lr_scheduler_d.step()

        # Log learning rates
        self._log_learning_rate(epoch, self.lr_scheduler, "LR/Generator")
        if not self.disable_gan and self.lr_scheduler_d is not None:
            self._log_learning_rate(epoch, self.lr_scheduler_d, "LR/Discriminator")

        # Log gradient norms
        self._log_grad_norms(epoch)

        # Log VRAM and FLOPs
        self._log_vram(epoch)
        self._log_flops(epoch)

        # Per-modality validation
        if self.per_modality_val_loaders:
            self._compute_per_modality_validation(epoch)

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _create_regional_tracker(self) -> RegionalMetricsTracker:
        """Create regional metrics tracker.

        Returns:
            RegionalMetricsTracker instance.
        """
        return RegionalMetricsTracker(
            image_size=self.cfg.model.image_size,
            fov_mm=self.cfg.paths.get('fov_mm', 240.0),
            loss_fn='l1',
            device=self.device,
        )

    def _create_worst_batch_figure(
        self,
        worst_batch_data: Dict[str, Any],
    ) -> plt.Figure:
        """Create worst batch figure for TensorBoard.

        Args:
            worst_batch_data: Dict with 'original', 'generated', 'loss', 'loss_breakdown'.

        Returns:
            Matplotlib figure.
        """
        return create_worst_batch_figure(
            original=worst_batch_data['original'],
            generated=worst_batch_data['generated'],
            loss=worst_batch_data['loss'],
            loss_breakdown=worst_batch_data.get('loss_breakdown'),
        )

    def compute_validation_losses(
        self,
        epoch: int,
        log_figures: bool = True,
    ) -> Dict[str, float]:
        """Compute validation losses and metrics.

        Args:
            epoch: Current epoch number.
            log_figures: Whether to log figures (worst_batch).

        Returns:
            Dictionary of validation metrics.
        """
        if self.val_loader is None:
            return {}

        model_to_use = self._get_model_for_eval()
        model_to_use.eval()

        # Accumulators
        total_l1 = 0.0
        total_perc = 0.0
        total_reg = 0.0  # Regularization (KL/VQ/none)
        total_gen = 0.0
        total_msssim = 0.0
        total_psnr = 0.0
        total_lpips = 0.0
        n_batches = 0

        # Worst batch tracking
        worst_loss = 0.0
        worst_batch_data = None

        # Regional tracker
        regional_tracker = None
        if self.log_regional_losses:
            regional_tracker = self._create_regional_tracker()

        # Mark CUDA graph step boundary
        if self.use_compile:
            torch.compiler.cudagraph_mark_step_begin()

        # Reset MS-SSIM NaN warning
        reset_msssim_nan_warning()

        with torch.inference_mode():  # Faster than no_grad for inference
            for batch in self.val_loader:
                images, mask = self._prepare_batch(batch)

                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    # Subclass-specific forward pass
                    reconstruction, reg_loss = self._forward_for_validation(model_to_use, images)

                    # Common loss computation
                    l1_loss = torch.abs(reconstruction - images).mean()
                    p_loss = self._compute_perceptual_loss(reconstruction, images)
                    g_loss = l1_loss + self.perceptual_weight * p_loss + reg_loss

                loss_val = g_loss.item()
                total_l1 += l1_loss.item()
                total_perc += p_loss.item()
                total_reg += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
                total_gen += loss_val

                # Track worst batch
                if loss_val > worst_loss:
                    worst_loss = loss_val
                    worst_batch_data = self._capture_worst_batch(
                        images, reconstruction, loss_val, l1_loss, p_loss, reg_loss
                    )

                # Quality metrics
                if self.log_msssim:
                    total_msssim += compute_msssim(reconstruction, images)
                if self.log_psnr:
                    total_psnr += compute_psnr(reconstruction, images)
                if self.log_lpips:
                    total_lpips += compute_lpips(reconstruction, images, device=self.device)

                # Regional tracking
                if regional_tracker is not None and mask is not None:
                    regional_tracker.update(reconstruction, images, mask)

                n_batches += 1

        model_to_use.train()

        # Compute averages
        metrics = {
            'l1': total_l1 / n_batches if n_batches > 0 else 0.0,
            'perc': total_perc / n_batches if n_batches > 0 else 0.0,
            'reg': total_reg / n_batches if n_batches > 0 else 0.0,
            'gen': total_gen / n_batches if n_batches > 0 else 0.0,
        }
        if self.log_psnr:
            metrics['psnr'] = total_psnr / n_batches if n_batches > 0 else 0.0
        if self.log_lpips:
            metrics['lpips'] = total_lpips / n_batches if n_batches > 0 else 0.0
        if self.log_msssim:
            metrics['msssim'] = total_msssim / n_batches if n_batches > 0 else 0.0

        # Log to TensorBoard
        self._log_validation_metrics(epoch, metrics, worst_batch_data, regional_tracker, log_figures)

        # Compute 3D MS-SSIM on full volumes (2D trainers only)
        msssim_3d = self._compute_volume_3d_msssim(epoch, data_split='val')
        if msssim_3d is not None:
            metrics['msssim_3d'] = msssim_3d

        return metrics

    def _capture_worst_batch(
        self,
        images: torch.Tensor,
        reconstruction: torch.Tensor,
        loss: float,
        l1_loss: torch.Tensor,
        p_loss: torch.Tensor,
        reg_loss: torch.Tensor,
    ) -> Dict[str, Any]:
        """Capture worst batch data for visualization.

        Args:
            images: Original images.
            reconstruction: Reconstructed images.
            loss: Total loss value.
            l1_loss: L1 loss tensor.
            p_loss: Perceptual loss tensor.
            reg_loss: Regularization loss tensor.

        Returns:
            Dictionary with worst batch data.
        """
        n_channels = self.cfg.mode.get('in_channels', 1)

        # Convert to dict format for dual mode
        if n_channels == 2:
            image_keys = self.cfg.mode.get('image_keys', ['t1_pre', 't1_gd'])
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

    def _log_validation_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float],
        worst_batch_data: Optional[Dict[str, Any]],
        regional_tracker: Optional[RegionalMetricsTracker],
        log_figures: bool,
    ) -> None:
        """Log validation metrics to TensorBoard.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of validation metrics.
            worst_batch_data: Worst batch data for visualization.
            regional_tracker: Regional metrics tracker.
            log_figures: Whether to log figures.
        """
        if self.writer is None:
            return

        # Log scalar metrics
        self.writer.add_scalar('Loss/L1_val', metrics['l1'], epoch)
        self.writer.add_scalar('Loss/Perceptual_val', metrics['perc'], epoch)
        self.writer.add_scalar('Loss/Generator_val', metrics['gen'], epoch)

        if 'psnr' in metrics:
            self.writer.add_scalar('Validation/PSNR', metrics['psnr'], epoch)
        if 'lpips' in metrics:
            self.writer.add_scalar('Validation/LPIPS', metrics['lpips'], epoch)
        if 'msssim' in metrics:
            self.writer.add_scalar('Validation/MS-SSIM', metrics['msssim'], epoch)
        if 'msssim_3d' in metrics:
            self.writer.add_scalar('Validation/MS-SSIM-3D', metrics['msssim_3d'], epoch)

        # Log worst batch figure
        if log_figures and worst_batch_data is not None:
            fig = self._create_worst_batch_figure(worst_batch_data)
            self.writer.add_figure('Validation/worst_batch', fig, epoch)
            plt.close(fig)

        # Log regional metrics
        if regional_tracker is not None:
            regional_tracker.log_to_tensorboard(self.writer, epoch, prefix='regional')

    def _compute_per_modality_validation(self, epoch: int) -> None:
        """Compute per-modality validation metrics.

        Args:
            epoch: Current epoch number.
        """
        if not self.per_modality_val_loaders:
            return

        model_to_use = self._get_model_for_eval()
        model_to_use.eval()

        for modality, loader in self.per_modality_val_loaders.items():
            total_psnr = 0.0
            total_lpips = 0.0
            total_msssim = 0.0
            n_batches = 0

            # Regional tracker for this modality
            regional_tracker = None
            if self.log_regional_losses:
                regional_tracker = self._create_regional_tracker()

            with torch.inference_mode():
                for batch in loader:
                    images, mask = self._prepare_batch(batch)

                    with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                        reconstruction, _ = self._forward_for_validation(model_to_use, images)

                    # Compute metrics
                    if self.log_psnr:
                        total_psnr += compute_psnr(reconstruction, images)
                    if self.log_lpips:
                        total_lpips += compute_lpips(reconstruction, images, device=self.device)
                    if self.log_msssim:
                        total_msssim += compute_msssim(reconstruction, images)

                    # Regional tracking
                    if regional_tracker is not None and mask is not None:
                        regional_tracker.update(reconstruction, images, mask)

                    n_batches += 1

            # Log metrics
            if n_batches > 0 and self.writer is not None:
                if self.log_psnr:
                    self.writer.add_scalar(f'Validation/PSNR_{modality}', total_psnr / n_batches, epoch)
                if self.log_lpips:
                    self.writer.add_scalar(f'Validation/LPIPS_{modality}', total_lpips / n_batches, epoch)
                if self.log_msssim:
                    self.writer.add_scalar(f'Validation/MS-SSIM_{modality}', total_msssim / n_batches, epoch)

                if regional_tracker is not None:
                    regional_tracker.log_to_tensorboard(
                        self.writer, epoch, prefix=f'regional_{modality}'
                    )

        model_to_use.train()

    def _compute_volume_3d_msssim(self, epoch: int, data_split: str = 'val') -> Optional[float]:
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
            epoch: Current epoch number.
            data_split: Which data split to use ('val' or 'test_new').

        Returns:
            Average 3D MS-SSIM across all volumes, or None if unavailable.
        """
        if not self.log_msssim:
            return None

        # Import here to avoid circular imports
        from medgen.data.loaders.vae import create_vae_volume_validation_dataloader

        # Determine modality from config
        mode_name = self.cfg.mode.get('name', 'bravo')
        n_channels = self.cfg.mode.get('in_channels', 1)
        modality = 'dual' if n_channels > 1 else mode_name

        # Create volume dataloader
        result = create_vae_volume_validation_dataloader(self.cfg, modality, data_split)
        if result is None:
            return None

        volume_loader, _ = result

        model_to_use = self._get_model_for_eval()
        model_to_use.eval()

        total_msssim_3d = 0.0
        n_volumes = 0
        slice_batch_size = self.cfg.training.batch_size  # Reuse training batch size

        with torch.inference_mode():  # Faster than no_grad
            for batch in volume_loader:
                # batch['image'] is [1, C, H, W, D] (batch_size=1 for volumes)
                # Non-blocking transfer to GPU
                volume = batch['image'].to(self.device, non_blocking=True)  # [1, C, H, W, D]
                volume = volume.squeeze(0)  # [C, H, W, D]

                n_channels_vol, height, width, depth = volume.shape

                # Pre-allocate output tensor on GPU
                all_recon = torch.empty(
                    (depth, n_channels_vol, height, width),
                    dtype=self.weight_dtype,
                    device=self.device
                )

                # Process slices in batches using tensor slicing (no Python loop for extraction)
                for start_idx in range(0, depth, slice_batch_size):
                    end_idx = min(start_idx + slice_batch_size, depth)

                    # Direct tensor slicing: [C, H, W, D] -> [B, C, H, W]
                    # Transpose to get slices along last dim, then slice
                    slice_tensor = volume[:, :, :, start_idx:end_idx].permute(3, 0, 1, 2)

                    with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                        # Forward through model
                        recon, _ = self._forward_for_validation(model_to_use, slice_tensor)

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

        # Log to TensorBoard
        if self.writer is not None:
            prefix = 'Validation' if data_split == 'val' else f'test_{data_split.replace("_new", "")}'
            self.writer.add_scalar(f'{prefix}/MS-SSIM-3D', avg_msssim_3d, epoch)

        return avg_msssim_3d

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, name: str) -> None:
        """Save checkpoint with standardized format.

        Args:
            epoch: Current epoch number.
            name: Checkpoint name ("latest" or "best").
        """
        if not self.is_main_process:
            return

        model_config = self._get_model_config()

        # Build extra state
        extra_state = {
            'disable_gan': self.disable_gan,
            'use_constant_lr': self.use_constant_lr,
        }

        # Add discriminator state if GAN is enabled
        if not self.disable_gan and self.discriminator_raw is not None:
            extra_state['discriminator_state_dict'] = self.discriminator_raw.state_dict()
            extra_state['disc_config'] = {
                'in_channels': self.cfg.mode.get('in_channels', 1),
                'channels': self.disc_num_channels,
                'num_layers_d': self.disc_num_layers,
            }
            if self.optimizer_d is not None:
                extra_state['optimizer_d_state_dict'] = self.optimizer_d.state_dict()
            if not self.use_constant_lr and self.lr_scheduler_d is not None:
                extra_state['scheduler_d_state_dict'] = self.lr_scheduler_d.state_dict()

        # Save using standardized format: checkpoint_{name}.pt
        save_full_checkpoint(
            model=self.model_raw,
            optimizer=self.optimizer,
            epoch=epoch,
            save_dir=self.save_dir,
            filename=f"checkpoint_{name}",
            model_config=model_config,
            scheduler=self.lr_scheduler if not self.use_constant_lr else None,
            ema=self.ema,
            extra_state=extra_state,
        )

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> int:
        """Load checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint file.
            load_optimizer: Whether to load optimizer state.

        Returns:
            Epoch number from checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model weights
        self.model_raw.load_state_dict(checkpoint['model_state_dict'])
        if self.is_main_process:
            logger.info(f"Loaded model weights from {checkpoint_path}")

        # Load discriminator weights
        if not self.disable_gan and self.discriminator_raw is not None:
            if 'discriminator_state_dict' in checkpoint:
                self.discriminator_raw.load_state_dict(checkpoint['discriminator_state_dict'])
                if self.is_main_process:
                    logger.info("Loaded discriminator weights")
            else:
                if self.is_main_process:
                    logger.warning("Checkpoint has no discriminator weights")

        # Load optimizer states
        if load_optimizer:
            if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Also try old key name for backwards compat
            elif 'optimizer_g_state_dict' in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_g_state_dict'])

            if not self.disable_gan and self.optimizer_d is not None:
                if 'optimizer_d_state_dict' in checkpoint:
                    self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

            # Load scheduler states
            if not self.use_constant_lr:
                if 'scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                elif 'scheduler_g_state_dict' in checkpoint and self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(checkpoint['scheduler_g_state_dict'])

                if not self.disable_gan and self.lr_scheduler_d is not None:
                    if 'scheduler_d_state_dict' in checkpoint:
                        self.lr_scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])

        # Load EMA state
        if self.use_ema and self.ema is not None:
            if 'ema_state_dict' in checkpoint:
                self.ema.load_state_dict(checkpoint['ema_state_dict'])
                if self.is_main_process:
                    logger.info("Loaded EMA state")

        epoch = checkpoint.get('epoch', 0)
        if self.is_main_process:
            logger.info(f"Resuming from epoch {epoch + 1}")

        return epoch

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
        elapsed_time: float,
    ) -> None:
        """Log epoch completion summary."""
        timestamp = time.strftime("%H:%M:%S")
        epoch_pct = ((epoch + 1) / total_epochs) * 100

        # Format validation metrics
        val_gen = f"(v:{val_metrics.get('gen', 0):.4f})" if val_metrics else ""
        val_l1 = f"(v:{val_metrics.get('l1', 0):.4f})" if val_metrics else ""
        msssim_str = f"MS-SSIM: {val_metrics.get('msssim', 0):.3f}" if val_metrics.get('msssim') else ""

        logger.info(
            f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
            f"G: {avg_losses.get('gen', 0):.4f}{val_gen} | "
            f"L1: {avg_losses.get('recon', avg_losses.get('l1', 0)):.4f}{val_l1} | "
            f"D: {avg_losses.get('disc', 0):.4f} | "
            f"{msssim_str} | "
            f"Time: {elapsed_time:.1f}s"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Abstract methods
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def _forward_for_validation(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass for validation.

        Returns (reconstruction, regularization_loss):
        - VAE: (recon, kl_loss)
        - VQ-VAE: (recon, vq_loss)
        - DC-AE: (recon, 0.0)

        Args:
            model: Model to use for inference.
            images: Input images.

        Returns:
            Tuple of (reconstruction, regularization_loss).
        """
        ...

    @abstractmethod
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for checkpoint.

        Returns:
            Dictionary of model configuration.
        """
        ...

    @abstractmethod
    def _test_forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Perform forward pass for test evaluation.

        Args:
            model: Model to use for inference.
            images: Input images.

        Returns:
            Reconstructed images tensor.
        """
        ...

    # ─────────────────────────────────────────────────────────────────────────
    # Test evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_test_set(
        self,
        test_loader: DataLoader,
        checkpoint_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate compression model on test set.

        Args:
            test_loader: Test data loader.
            checkpoint_name: Checkpoint to load ("best", "latest", or None).

        Returns:
            Dict with test metrics.
        """
        if not self.is_main_process:
            return {}

        # Load checkpoint if specified
        if checkpoint_name is not None:
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{checkpoint_name}.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model_raw.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded {checkpoint_name} checkpoint for test evaluation")
            else:
                logger.warning(f"Checkpoint {checkpoint_path} not found")
                checkpoint_name = "current"

        label = checkpoint_name or "current"
        logger.info("=" * 60)
        logger.info(f"EVALUATING ON TEST SET ({label.upper()} MODEL)")
        logger.info("=" * 60)

        model_to_use = self._get_model_for_eval() if checkpoint_name is None else self.model_raw
        model_to_use.eval()

        # Accumulators
        total_l1 = 0.0
        total_msssim = 0.0
        total_psnr = 0.0
        total_lpips = 0.0
        n_batches = 0
        n_samples = 0

        # Regional tracker
        regional_tracker = None
        if self.log_regional_losses:
            regional_tracker = self._create_regional_tracker()

        # Worst batch tracking
        worst_loss = 0.0
        worst_batch_data = None

        with torch.inference_mode():
            for batch in tqdm(test_loader, desc="Test evaluation", ncols=100, disable=self.is_cluster):
                images, mask = self._prepare_batch(batch)
                batch_size = images.shape[0]

                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    reconstructed = self._test_forward(model_to_use, images)

                # Compute metrics
                l1_loss = torch.abs(reconstructed - images).mean().item()
                total_l1 += l1_loss
                if self.log_msssim:
                    total_msssim += compute_msssim(reconstructed.float(), images.float())
                total_psnr += compute_psnr(reconstructed, images)
                total_lpips += compute_lpips(reconstructed.float(), images.float(), device=self.device)

                # Regional tracking
                if regional_tracker is not None and mask is not None:
                    regional_tracker.update(reconstructed, images, mask)

                # Track worst batch
                if l1_loss > worst_loss:
                    worst_loss = l1_loss
                    worst_batch_data = self._capture_worst_batch(
                        images, reconstructed, l1_loss,
                        torch.tensor(l1_loss), torch.tensor(0.0), torch.tensor(0.0)
                    )

                n_batches += 1
                n_samples += batch_size

        # Compute averages
        metrics = {
            'l1': total_l1 / n_batches,
            'psnr': total_psnr / n_batches,
            'lpips': total_lpips / n_batches,
            'n_samples': n_samples,
        }
        if self.log_msssim:
            metrics['msssim'] = total_msssim / n_batches

        # Compute 3D MS-SSIM on full volumes (2D trainers only)
        msssim_3d = self._compute_volume_3d_msssim(epoch=0, data_split='test_new')
        if msssim_3d is not None:
            metrics['msssim_3d'] = msssim_3d

        # Log results
        logger.info(f"Test Results - {label} ({n_samples} samples):")
        logger.info(f"  L1 Loss: {metrics['l1']:.6f}")
        if 'msssim' in metrics:
            logger.info(f"  MS-SSIM: {metrics['msssim']:.4f}")
        if 'msssim_3d' in metrics:
            logger.info(f"  MS-SSIM-3D: {metrics['msssim_3d']:.4f}")
        logger.info(f"  PSNR:    {metrics['psnr']:.2f} dB")
        logger.info(f"  LPIPS:   {metrics['lpips']:.4f}")

        # Save results
        results_path = os.path.join(self.save_dir, f'test_results_{label}.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Log to TensorBoard
        tb_prefix = f'test_{label}'
        if self.writer is not None:
            self.writer.add_scalar(f'{tb_prefix}/L1', metrics['l1'], 0)
            self.writer.add_scalar(f'{tb_prefix}/PSNR', metrics['psnr'], 0)
            self.writer.add_scalar(f'{tb_prefix}/LPIPS', metrics['lpips'], 0)
            if 'msssim' in metrics:
                self.writer.add_scalar(f'{tb_prefix}/MS-SSIM', metrics['msssim'], 0)
            if 'msssim_3d' in metrics:
                self.writer.add_scalar(f'{tb_prefix}/MS-SSIM-3D', metrics['msssim_3d'], 0)

            if regional_tracker is not None:
                regional_tracker.log_to_tensorboard(self.writer, 0, prefix=f'{tb_prefix}_regional')

            # Worst batch
            if worst_batch_data is not None:
                fig = self._create_worst_batch_figure(worst_batch_data)
                self.writer.add_figure(f'{tb_prefix}/worst_batch', fig, 0)
                plt.close(fig)

                # Also save as PNG file
                fig = self._create_worst_batch_figure(worst_batch_data)
                fig_path = os.path.join(self.save_dir, f'test_worst_batch_{label}.png')
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Test worst batch saved to: {fig_path}")

        model_to_use.train()
        return metrics


class BaseCompression3DTrainer(BaseCompressionTrainer):
    """Base trainer for 3D volumetric compression models.

    Extends BaseCompressionTrainer with:
    - Volume dimension handling
    - Gradient checkpointing support
    - 2.5D perceptual loss (slice sampling)
    - 3D regional metrics
    - 3D worst batch visualization

    Args:
        cfg: Hydra configuration object.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize 3D compression trainer.

        Args:
            cfg: Hydra configuration object.
        """
        super().__init__(cfg)

        # ─────────────────────────────────────────────────────────────────────
        # Extract 3D-specific config
        # ─────────────────────────────────────────────────────────────────────
        self.volume_depth: int = cfg.volume.get('depth', 160)
        self.volume_height: int = cfg.volume.get('height', 256)
        self.volume_width: int = cfg.volume.get('width', 256)

        # 2.5D perceptual loss config
        self.use_2_5d_perceptual: bool = self._get_2_5d_perceptual(cfg)
        self.perceptual_slice_fraction: float = self._get_perceptual_slice_fraction(cfg)

        # Gradient checkpointing
        self.gradient_checkpointing: bool = cfg.training.get('gradient_checkpointing', True)

        # torch.compile often causes issues with 3D models
        self.use_compile: bool = cfg.training.get('use_compile', False)

    def _get_2_5d_perceptual(self, cfg: DictConfig) -> bool:
        """Get 2.5D perceptual loss flag."""
        for section in ['vae_3d', 'vqvae_3d']:
            if section in cfg:
                return cfg[section].get('use_2_5d_perceptual', True)
        return True

    def _get_perceptual_slice_fraction(self, cfg: DictConfig) -> float:
        """Get perceptual slice fraction."""
        for section in ['vae_3d', 'vqvae_3d']:
            if section in cfg:
                return cfg[section].get('perceptual_slice_fraction', 0.25)
        return 0.25

    def _prepare_batch(self, batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare batch for 3D compression training.

        Uses non_blocking=True for async GPU transfers to overlap with computation.

        Args:
            batch: Input batch.

        Returns:
            Tuple of (images, mask).
        """
        if isinstance(batch, dict):
            images = batch.get('image', batch.get('images'))
            mask = batch.get('mask', batch.get('seg'))
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
            mask = batch[1] if len(batch) > 1 else None
        else:
            images = batch
            mask = None

        images = images.to(self.device, non_blocking=True)
        if mask is not None:
            mask = mask.to(self.device, non_blocking=True)

        return images, mask

    def _create_regional_tracker(self):
        """Create 3D regional metrics tracker."""
        from .metrics import RegionalMetricsTracker3D
        return RegionalMetricsTracker3D(
            volume_size=(self.volume_height, self.volume_width, self.volume_depth),
            fov_mm=self.cfg.paths.get('fov_mm', 240.0),
            loss_fn='l1',
            device=self.device,
        )

    def _create_worst_batch_figure(
        self,
        worst_batch_data: Dict[str, Any],
    ) -> plt.Figure:
        """Create 3D worst batch figure."""
        from .tracking import create_worst_batch_figure_3d
        return create_worst_batch_figure_3d(
            original=worst_batch_data['original'],
            generated=worst_batch_data['generated'],
            loss=worst_batch_data['loss'],
            loss_breakdown=worst_batch_data.get('loss_breakdown'),
        )

    def _compute_2_5d_perceptual_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss on sampled 2D slices.

        Args:
            reconstruction: Reconstructed volume [B, C, D, H, W].
            target: Target volume [B, C, D, H, W].

        Returns:
            Perceptual loss averaged over sampled slices.
        """
        if self.perceptual_loss_fn is None:
            return torch.tensor(0.0, device=self.device)

        depth = reconstruction.shape[2]
        n_slices = max(1, int(depth * self.perceptual_slice_fraction))

        # Sample slice indices
        indices = torch.randperm(depth)[:n_slices].to(self.device)

        total_loss = 0.0
        for idx in indices:
            recon_slice = reconstruction[:, :, idx, :, :]
            target_slice = target[:, :, idx, :, :]
            total_loss += self.perceptual_loss_fn(recon_slice, target_slice)

        return total_loss / n_slices

    def _compute_perceptual_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss (2.5D for 3D data).

        Args:
            reconstruction: Reconstructed volume.
            target: Target volume.

        Returns:
            Perceptual loss value.
        """
        if self.use_2_5d_perceptual:
            return self._compute_2_5d_perceptual_loss(reconstruction, target)
        return super()._compute_perceptual_loss(reconstruction, target)

    def _capture_worst_batch(
        self,
        images: torch.Tensor,
        reconstruction: torch.Tensor,
        loss: float,
        l1_loss: torch.Tensor,
        p_loss: torch.Tensor,
        reg_loss: torch.Tensor,
    ) -> Dict[str, Any]:
        """Capture worst batch data for 3D visualization."""
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

    def _create_fallback_save_dir(self) -> str:
        """Create fallback save directory for 3D models."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = self.cfg.training.get('name', '')
        mode_name = self.cfg.mode.get('name', 'multi_modality')
        run_name = f"{exp_name}{self.volume_height}_{timestamp}"
        return os.path.join(self.cfg.paths.model_dir, 'vae_3d', mode_name, run_name)

    def compute_validation_losses(
        self,
        epoch: int,
        log_figures: bool = True,
    ) -> Dict[str, float]:
        """Compute validation losses and metrics for 3D volumes.

        Overrides base class to use 3D-appropriate metrics:
        - MS-SSIM with spatial_dims=3
        - LPIPS computed slice-by-slice via compute_lpips_3d

        Args:
            epoch: Current epoch number.
            log_figures: Whether to log figures (worst_batch).

        Returns:
            Dictionary of validation metrics.
        """
        if self.val_loader is None:
            return {}

        model_to_use = self._get_model_for_eval()
        model_to_use.eval()

        # Accumulators
        total_l1 = 0.0
        total_perc = 0.0
        total_reg = 0.0
        total_gen = 0.0
        total_msssim_3d = 0.0  # Volumetric 3D MS-SSIM
        total_msssim_2d = 0.0  # Slice-by-slice 2D MS-SSIM
        total_psnr = 0.0
        total_lpips = 0.0
        n_batches = 0

        # Worst batch tracking
        worst_loss = 0.0
        worst_batch_data = None

        # Regional tracker
        regional_tracker = None
        if self.log_regional_losses:
            regional_tracker = self._create_regional_tracker()

        # Mark CUDA graph step boundary
        if self.use_compile:
            torch.compiler.cudagraph_mark_step_begin()

        # Reset MS-SSIM NaN warning
        reset_msssim_nan_warning()

        with torch.inference_mode():  # Faster than no_grad for inference
            for batch in self.val_loader:
                images, mask = self._prepare_batch(batch)

                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    reconstruction, reg_loss = self._forward_for_validation(model_to_use, images)

                    l1_loss = torch.abs(reconstruction - images).mean()
                    p_loss = self._compute_perceptual_loss(reconstruction, images)
                    g_loss = l1_loss + self.perceptual_weight * p_loss + reg_loss

                loss_val = g_loss.item()
                total_l1 += l1_loss.item()
                total_perc += p_loss.item()
                total_reg += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
                total_gen += loss_val

                # Track worst batch
                if loss_val > worst_loss:
                    worst_loss = loss_val
                    worst_batch_data = self._capture_worst_batch(
                        images, reconstruction, loss_val, l1_loss, p_loss, reg_loss
                    )

                # Quality metrics - 3D versions
                if self.log_msssim:
                    # Volumetric 3D MS-SSIM (captures cross-slice structure)
                    total_msssim_3d += compute_msssim(
                        reconstruction.float(), images.float(), spatial_dims=3
                    )
                    # Slice-by-slice 2D MS-SSIM (comparable to 2D trainers)
                    total_msssim_2d += compute_msssim_2d_slicewise(
                        reconstruction.float(), images.float()
                    )
                if self.log_psnr:
                    total_psnr += compute_psnr(reconstruction, images)
                if self.log_lpips:
                    total_lpips += compute_lpips_3d(
                        reconstruction.float(), images.float(), device=self.device
                    )

                # Regional tracking
                if regional_tracker is not None and mask is not None:
                    regional_tracker.update(reconstruction, images, mask)

                n_batches += 1

        model_to_use.train()

        # Compute averages
        metrics = {
            'l1': total_l1 / n_batches if n_batches > 0 else 0.0,
            'perc': total_perc / n_batches if n_batches > 0 else 0.0,
            'reg': total_reg / n_batches if n_batches > 0 else 0.0,
            'gen': total_gen / n_batches if n_batches > 0 else 0.0,
        }
        if self.log_psnr:
            metrics['psnr'] = total_psnr / n_batches if n_batches > 0 else 0.0
        if self.log_lpips:
            metrics['lpips'] = total_lpips / n_batches if n_batches > 0 else 0.0
        if self.log_msssim:
            # 2D MS-SSIM for comparison with 2D trainers
            metrics['msssim'] = total_msssim_2d / n_batches if n_batches > 0 else 0.0
            # 3D MS-SSIM for volumetric quality
            metrics['msssim_3d'] = total_msssim_3d / n_batches if n_batches > 0 else 0.0

        # Log to TensorBoard
        self._log_validation_metrics(epoch, metrics, worst_batch_data, regional_tracker, log_figures)

        return metrics

    def _compute_per_modality_validation(self, epoch: int) -> None:
        """Compute per-modality validation metrics for 3D volumes.

        Overrides base class to use 3D-appropriate metrics:
        - MS-SSIM-3D with spatial_dims=3 (volumetric)
        - MS-SSIM with slice-by-slice computation (comparable to 2D)
        - LPIPS computed slice-by-slice via compute_lpips_3d

        Args:
            epoch: Current epoch number.
        """
        if not self.per_modality_val_loaders:
            return

        model_to_use = self._get_model_for_eval()
        model_to_use.eval()

        for modality, loader in self.per_modality_val_loaders.items():
            total_psnr = 0.0
            total_lpips = 0.0
            total_msssim_3d = 0.0  # Volumetric 3D MS-SSIM
            total_msssim_2d = 0.0  # Slice-by-slice 2D MS-SSIM
            n_batches = 0

            # Regional tracker for this modality
            regional_tracker = None
            if self.log_regional_losses:
                regional_tracker = self._create_regional_tracker()

            with torch.inference_mode():
                for batch in loader:
                    images, mask = self._prepare_batch(batch)

                    with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                        reconstruction, _ = self._forward_for_validation(model_to_use, images)

                    # Compute metrics - 3D versions
                    if self.log_psnr:
                        total_psnr += compute_psnr(reconstruction, images)
                    if self.log_lpips:
                        total_lpips += compute_lpips_3d(
                            reconstruction.float(), images.float(), device=self.device
                        )
                    if self.log_msssim:
                        # Volumetric 3D MS-SSIM
                        total_msssim_3d += compute_msssim(
                            reconstruction.float(), images.float(), spatial_dims=3
                        )
                        # Slice-by-slice 2D MS-SSIM (comparable to 2D trainers)
                        total_msssim_2d += compute_msssim_2d_slicewise(
                            reconstruction.float(), images.float()
                        )

                    # Regional tracking
                    if regional_tracker is not None and mask is not None:
                        regional_tracker.update(reconstruction.float(), images.float(), mask)

                    n_batches += 1

            # Log metrics
            if n_batches > 0 and self.writer is not None:
                if self.log_psnr:
                    self.writer.add_scalar(f'Validation/PSNR_{modality}', total_psnr / n_batches, epoch)
                if self.log_lpips:
                    self.writer.add_scalar(f'Validation/LPIPS_{modality}', total_lpips / n_batches, epoch)
                if self.log_msssim:
                    self.writer.add_scalar(f'Validation/MS-SSIM_{modality}', total_msssim_2d / n_batches, epoch)
                    self.writer.add_scalar(f'Validation/MS-SSIM-3D_{modality}', total_msssim_3d / n_batches, epoch)

                if regional_tracker is not None:
                    regional_tracker.log_to_tensorboard(
                        self.writer, epoch, prefix=f'regional_{modality}'
                    )

        model_to_use.train()

    def evaluate_test_set(
        self,
        test_loader: DataLoader,
        checkpoint_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate 3D compression model on test set.

        Args:
            test_loader: Test data loader.
            checkpoint_name: Checkpoint to load ("best", "latest", or None).

        Returns:
            Dict with test metrics.
        """
        from .metrics import RegionalMetricsTracker3D
        from .tracking import create_worst_batch_figure_3d

        if not self.is_main_process:
            return {}

        # Load checkpoint if specified
        if checkpoint_name is not None:
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{checkpoint_name}.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model_raw.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded {checkpoint_name} checkpoint for test evaluation")
            else:
                logger.warning(f"Checkpoint {checkpoint_path} not found")
                checkpoint_name = "current"

        label = checkpoint_name or "current"
        logger.info("=" * 60)
        logger.info(f"EVALUATING ON TEST SET ({label.upper()} MODEL)")
        logger.info("=" * 60)

        model_to_use = self._get_model_for_eval() if checkpoint_name is None else self.model_raw
        model_to_use.eval()

        # Accumulators
        total_l1 = 0.0
        total_msssim_3d = 0.0
        total_msssim_2d = 0.0
        total_psnr = 0.0
        total_lpips = 0.0
        n_batches = 0
        n_samples = 0

        # Regional tracker
        regional_tracker = None
        if self.log_regional_losses:
            regional_tracker = self._create_regional_tracker()

        # Worst batch tracking
        worst_loss = 0.0
        worst_batch_data = None

        # Sample collection
        sample_inputs = []
        sample_outputs = []
        max_vis_samples = 4

        with torch.inference_mode():
            for batch in tqdm(test_loader, desc="Test evaluation", ncols=100, disable=self.is_cluster):
                images, mask = self._prepare_batch(batch)
                batch_size = images.shape[0]

                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    reconstructed = self._test_forward(model_to_use, images)

                # Compute metrics
                l1_loss = torch.abs(reconstructed - images).mean().item()
                total_l1 += l1_loss
                if self.log_msssim:
                    # 3D volumetric MS-SSIM
                    total_msssim_3d += compute_msssim(reconstructed.float(), images.float(), spatial_dims=3)
                    # 2D slice-by-slice MS-SSIM (for comparison with 2D trainers)
                    total_msssim_2d += compute_msssim_2d_slicewise(reconstructed.float(), images.float())
                total_psnr += compute_psnr(reconstructed, images)
                total_lpips += compute_lpips_3d(reconstructed.float(), images.float(), device=self.device)

                # Regional tracking
                if regional_tracker is not None and mask is not None:
                    regional_tracker.update(reconstructed.float(), images.float(), mask)

                # Track worst batch
                if l1_loss > worst_loss:
                    worst_loss = l1_loss
                    worst_batch_data = {
                        'original': images.cpu(),
                        'generated': reconstructed.float().cpu(),
                        'loss': l1_loss,
                    }

                n_batches += 1
                n_samples += batch_size

                # Collect samples
                if len(sample_inputs) < max_vis_samples:
                    remaining = max_vis_samples - len(sample_inputs)
                    sample_inputs.append(images[:remaining].cpu())
                    sample_outputs.append(reconstructed[:remaining].float().cpu())

        # Compute averages
        metrics = {
            'l1': total_l1 / n_batches,
            'psnr': total_psnr / n_batches,
            'lpips': total_lpips / n_batches,
            'n_samples': n_samples,
        }
        if self.log_msssim:
            metrics['msssim'] = total_msssim_2d / n_batches  # 2D slice-by-slice (comparable to 2D trainers)
            metrics['msssim_3d'] = total_msssim_3d / n_batches  # 3D volumetric

        # Log results
        logger.info(f"Test Results - {label} ({n_samples} samples):")
        logger.info(f"  L1 Loss:    {metrics['l1']:.6f}")
        if 'msssim' in metrics:
            logger.info(f"  MS-SSIM:    {metrics['msssim']:.4f} (2D)")
            logger.info(f"  MS-SSIM-3D: {metrics['msssim_3d']:.4f}")
        logger.info(f"  PSNR:       {metrics['psnr']:.2f} dB")
        logger.info(f"  LPIPS:      {metrics['lpips']:.4f}")

        # Save results
        results_path = os.path.join(self.save_dir, f'test_results_{label}.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Log to TensorBoard
        tb_prefix = f'test_{label}'
        if self.writer is not None:
            self.writer.add_scalar(f'{tb_prefix}/L1', metrics['l1'], 0)
            self.writer.add_scalar(f'{tb_prefix}/PSNR', metrics['psnr'], 0)
            self.writer.add_scalar(f'{tb_prefix}/LPIPS', metrics['lpips'], 0)
            if 'msssim' in metrics:
                self.writer.add_scalar(f'{tb_prefix}/MS-SSIM', metrics['msssim'], 0)
                self.writer.add_scalar(f'{tb_prefix}/MS-SSIM-3D', metrics['msssim_3d'], 0)

            if regional_tracker is not None:
                regional_tracker.log_to_tensorboard(self.writer, 0, prefix=f'{tb_prefix}_regional')

            # Worst batch figure
            if worst_batch_data is not None:
                fig = create_worst_batch_figure_3d(
                    original=worst_batch_data['original'],
                    generated=worst_batch_data['generated'],
                    loss=worst_batch_data['loss'],
                )
                self.writer.add_figure(f'{tb_prefix}/worst_batch', fig, 0)
                plt.close(fig)

                # Also save as PNG file
                fig = create_worst_batch_figure_3d(
                    original=worst_batch_data['original'],
                    generated=worst_batch_data['generated'],
                    loss=worst_batch_data['loss'],
                )
                fig_path = os.path.join(self.save_dir, f'test_worst_batch_{label}.png')
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Test worst batch saved to: {fig_path}")

        model_to_use.train()
        return metrics
