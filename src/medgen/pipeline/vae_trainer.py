"""
VAE trainer module for training AutoencoderKL models.

This module provides the VAETrainer class which inherits from BaseCompressionTrainer
and implements VAE-specific functionality:
- KL divergence regularization
- AutoencoderKL model creation
- VAE-specific forward pass (returns mean, logvar)
"""
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from monai.networks.nets import AutoencoderKL

from .compression_trainer import BaseCompressionTrainer
from .metrics import (
    compute_lpips,
    compute_msssim,
    compute_psnr,
    create_reconstruction_figure,
    RegionalMetricsTracker,
)
from .tracking import create_worst_batch_figure
from .utils import create_epoch_iterator, get_vram_usage, log_compression_epoch_summary

logger = logging.getLogger(__name__)


class VAETrainer(BaseCompressionTrainer):
    """AutoencoderKL trainer with KL divergence regularization.

    Inherits from BaseCompressionTrainer and adds:
    - KL divergence loss
    - AutoencoderKL model creation
    - VAE-specific forward pass returning (reconstruction, mean, logvar)

    Args:
        cfg: Hydra configuration object.

    Example:
        >>> trainer = VAETrainer(cfg)
        >>> trainer.setup_model()
        >>> trainer.train(train_loader, train_dataset, val_loader)
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize VAE trainer.

        Args:
            cfg: Hydra configuration object.
        """
        super().__init__(cfg)

        # ─────────────────────────────────────────────────────────────────────
        # VAE-specific config
        # ─────────────────────────────────────────────────────────────────────
        self.kl_weight: float = cfg.vae.get('kl_weight', 1e-6)
        self.latent_channels: int = cfg.vae.latent_channels
        self.vae_channels: Tuple[int, ...] = tuple(cfg.vae.channels)
        self.attention_levels: Tuple[bool, ...] = tuple(cfg.vae.attention_levels)
        self.num_res_blocks: int = cfg.vae.get('num_res_blocks', 2)
        self.image_size: int = cfg.model.image_size

    def _create_fallback_save_dir(self) -> str:
        """Create fallback save directory for VAE."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = self.cfg.training.get('name', '')
        mode_name = self.cfg.mode.get('name', 'dual')
        run_name = f"{exp_name}{self.image_size}_{timestamp}"
        return os.path.join(self.cfg.paths.model_dir, 'vae_2d', mode_name, run_name)

    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize VAE model, discriminator, optimizers, and loss functions.

        Args:
            pretrained_checkpoint: Optional path to checkpoint for loading
                pretrained weights (for progressive training).
        """
        n_channels = self.cfg.mode.get('in_channels', 1)

        # Create AutoencoderKL
        raw_model = AutoencoderKL(
            spatial_dims=2,
            in_channels=n_channels,
            out_channels=n_channels,
            channels=self.vae_channels,
            attention_levels=self.attention_levels,
            latent_channels=self.latent_channels,
            num_res_blocks=self.num_res_blocks,
            norm_num_groups=32,
            with_encoder_nonlocal_attn=True,
            with_decoder_nonlocal_attn=True,
        ).to(self.device)

        # Create discriminator if GAN enabled
        raw_disc = None
        if not self.disable_gan:
            raw_disc = self._create_discriminator(n_channels, spatial_dims=2)

        # Load pretrained weights if provided
        if pretrained_checkpoint:
            self._load_pretrained_weights(raw_model, raw_disc, pretrained_checkpoint, model_name="VAE")

        # Wrap models with DDP/compile
        self._wrap_models(raw_model, raw_disc)

        # Setup perceptual and adversarial loss
        self.perceptual_loss_fn = self._create_perceptual_loss(spatial_dims=2)
        if not self.disable_gan:
            from monai.losses import PatchAdversarialLoss
            self.adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

        # Setup optimizers and schedulers
        self._setup_optimizers(n_channels)

        # Setup EMA
        self._setup_ema()

        # Save metadata
        if self.is_main_process:
            self._save_metadata()

        # Log model info
        if self.is_main_process:
            vae_params = sum(p.numel() for p in self.model_raw.parameters())
            logger.info(f"VAE initialized: {vae_params / 1e6:.1f}M parameters")
            if not self.disable_gan:
                disc_params = sum(p.numel() for p in self.discriminator_raw.parameters())
                logger.info(f"Discriminator initialized: {disc_params / 1e6:.1f}M parameters")
            else:
                logger.info("GAN disabled - discriminator not created")
            logger.info(f"Latent shape: [{self.latent_channels}, {self.image_size // 8}, {self.image_size // 8}]")
            logger.info(f"Loss weights - Perceptual: {self.perceptual_weight}, KL: {self.kl_weight}, Adv: {self.adv_weight}")

    def _save_metadata(self) -> None:
        """Save training configuration and VAE config."""
        os.makedirs(self.save_dir, exist_ok=True)

        # Save full config
        config_path = os.path.join(self.save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        # Save VAE config separately (use _get_model_config for consistency)
        vae_config = self._get_model_config()

        metadata_path = os.path.join(self.save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'type': 'vae',
                'vae_config': vae_config,
                'image_size': self.image_size,
                'n_epochs': self.n_epochs,
            }, f, indent=2)

    def _compute_kl_loss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss.

        Args:
            mean: Mean of latent distribution [B, C, H, W].
            logvar: Log variance of latent distribution [B, C, H, W].

        Returns:
            KL divergence loss (scalar).
        """
        # KL divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=[1, 2, 3])
        return kl.mean()

    def train_step(self, batch: Any) -> Dict[str, float]:
        """Execute VAE training step with KL loss.

        Args:
            batch: Input batch.

        Returns:
            Dict with losses: 'gen', 'disc', 'recon', 'perc', 'kl', 'adv'.
        """
        images, mask = self._prepare_batch(batch)
        grad_clip = self.cfg.training.get('gradient_clip_norm', 1.0)

        d_loss = torch.tensor(0.0, device=self.device)
        adv_loss = torch.tensor(0.0, device=self.device)

        # ==================== Discriminator Step ====================
        if not self.disable_gan:
            with torch.no_grad():
                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    reconstruction_for_d, _, _ = self.model(images)

            d_loss = self._train_discriminator_step(images, reconstruction_for_d)

        # ==================== Generator Step ====================
        self.optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            reconstruction, mean, logvar = self.model(images)

            # L1 reconstruction loss
            l1_loss = torch.abs(reconstruction - images).mean()

            # Perceptual loss
            p_loss = self._compute_perceptual_loss(reconstruction, images)

            # KL divergence loss
            kl_loss = self._compute_kl_loss(mean, logvar)

            # Adversarial loss
            if not self.disable_gan:
                adv_loss = self._compute_adversarial_loss(reconstruction)

            # Total generator loss
            g_loss = (
                l1_loss
                + self.perceptual_weight * p_loss
                + self.kl_weight * kl_loss
                + self.adv_weight * adv_loss
            )

        g_loss.backward()

        # Gradient clipping
        grad_norm_g = 0.0
        if grad_clip > 0:
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                self.model_raw.parameters(), max_norm=grad_clip
            ).item()

        self.optimizer.step()

        # Track gradient norm
        if self.log_grad_norm:
            self._grad_norm_tracker.update(grad_norm_g)

        # Update EMA
        self._update_ema()

        return {
            'gen': g_loss.item(),
            'disc': d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss,
            'recon': l1_loss.item(),
            'perc': p_loss.item(),
            'kl': kl_loss.item(),
            'adv': adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
        }

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train VAE for one epoch.

        Args:
            data_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dict with average losses.
        """
        self.model.train()
        if not self.disable_gan and self.discriminator is not None:
            self.discriminator.train()

        epoch_losses = {'gen': 0, 'disc': 0, 'recon': 0, 'perc': 0, 'kl': 0, 'adv': 0}

        epoch_iter = create_epoch_iterator(
            data_loader, epoch, self.is_cluster, self.is_main_process,
            limit_batches=self.limit_train_batches
        )

        for step, batch in enumerate(epoch_iter):
            losses = self.train_step(batch)

            for key in epoch_losses:
                epoch_losses[key] += losses[key]

            if hasattr(epoch_iter, 'set_postfix'):
                epoch_iter.set_postfix(
                    G=f"{epoch_losses['gen'] / (step + 1):.4f}",
                    D=f"{epoch_losses['disc'] / (step + 1):.4f}"
                )

            if epoch == 1 and step == 0 and self.is_main_process:
                logger.info(get_vram_usage(self.device))

        # Average losses
        n_batches = self.limit_train_batches if self.limit_train_batches else len(data_loader)
        avg_losses = {key: val / n_batches for key, val in epoch_losses.items()}

        # Log training metrics (single-GPU only)
        if self.writer is not None and self.is_main_process and not self.use_multi_gpu:
            self.writer.add_scalar('Loss/Generator_train', avg_losses['gen'], epoch)
            self.writer.add_scalar('Loss/L1_train', avg_losses['recon'], epoch)
            self.writer.add_scalar('Loss/Perceptual_train', avg_losses['perc'], epoch)
            self.writer.add_scalar('Loss/KL_train', avg_losses['kl'], epoch)

            if not self.disable_gan:
                self.writer.add_scalar('Loss/Discriminator', avg_losses['disc'], epoch)
                self.writer.add_scalar('Loss/Adversarial', avg_losses['adv'], epoch)

        return avg_losses

    def _forward_for_validation(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """VAE forward pass for validation.

        Returns:
            Tuple of (reconstruction, weighted_kl_loss).
        """
        reconstruction, mean, logvar = model(images)
        kl_loss = self._compute_kl_loss(mean, logvar)
        return reconstruction, self.kl_weight * kl_loss

    def _get_model_config(self) -> Dict[str, Any]:
        """Get VAE model configuration for checkpoint."""
        n_channels = self.cfg.mode.get('in_channels', 1)
        return {
            'in_channels': n_channels,
            'out_channels': n_channels,
            'latent_channels': self.latent_channels,
            'channels': list(self.vae_channels),
            'attention_levels': list(self.attention_levels),
            'num_res_blocks': self.num_res_blocks,
            'norm_num_groups': 32,
            'with_encoder_nonlocal_attn': True,
            'with_decoder_nonlocal_attn': True,
        }

    def _log_validation_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float],
        worst_batch_data: Optional[Dict[str, Any]],
        regional_tracker: Optional[RegionalMetricsTracker],
        log_figures: bool,
    ) -> None:
        """Log VAE validation metrics including KL loss."""
        if self.writer is None:
            return

        # Call base class method first
        super()._log_validation_metrics(epoch, metrics, worst_batch_data, regional_tracker, log_figures)

        # Add VAE-specific KL loss logging
        if 'reg' in metrics:
            # Log as KL_val (reg is the weighted KL loss for VAE)
            self.writer.add_scalar('Loss/KL_val', metrics['reg'] / self.kl_weight if self.kl_weight > 0 else 0, epoch)

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
        elapsed_time: float,
    ) -> None:
        """Log VAE epoch summary."""
        log_compression_epoch_summary(
            epoch, total_epochs, avg_losses, val_metrics, elapsed_time,
            regularization_key=None,  # VAE's KL is already included in 'gen'
        )

    def _test_forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Perform VAE forward pass for test evaluation.

        Args:
            model: Model to use for inference.
            images: Input images.

        Returns:
            Reconstructed images tensor.
        """
        reconstructed, _, _ = model(images)
        return reconstructed
