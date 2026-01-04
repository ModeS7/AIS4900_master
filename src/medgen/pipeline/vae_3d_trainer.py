"""
3D VAE trainer module for training volumetric autoencoders.

This module provides the VAE3DTrainer class which inherits from BaseCompression3DTrainer
and implements 3D VAE-specific functionality:
- KL divergence regularization
- 3D AutoencoderKL model creation
- Gradient checkpointing for memory efficiency
"""
import itertools
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Disable MONAI MetaTensor tracking BEFORE importing MONAI modules
from monai.data import set_track_meta
set_track_meta(False)

from monai.networks.nets import AutoencoderKL

from .compression_trainer import BaseCompression3DTrainer
from .metrics import (
    compute_lpips_3d,
    compute_msssim,
    compute_psnr,
    RegionalMetricsTracker3D,
)
from .tracking import create_worst_batch_figure_3d
from .results import TrainingStepResult
from .utils import create_epoch_iterator, get_vram_usage, log_compression_epoch_summary

logger = logging.getLogger(__name__)


class CheckpointedAutoencoder(nn.Module):
    """Wrapper that applies gradient checkpointing to MONAI AutoencoderKL.

    MONAI's AutoencoderKL doesn't have built-in gradient checkpointing.
    This wrapper uses torch.utils.checkpoint to trade compute for memory,
    reducing activation memory by ~50% for 3D volumes.

    Args:
        model: The underlying AutoencoderKL model.
    """

    def __init__(self, model: AutoencoderKL):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with gradient checkpointing."""
        def encode_fn(x):
            h = self.model.encoder(x)
            z_mu = self.model.quant_conv_mu(h)
            z_log_var = self.model.quant_conv_log_sigma(h)
            return z_mu, z_log_var

        z_mu, z_log_var = grad_checkpoint(encode_fn, x, use_reentrant=False)
        z = self.model.sampling(z_mu, z_log_var)

        def decode_fn(z):
            z_post = self.model.post_quant_conv(z)
            return self.model.decoder(z_post)

        reconstruction = grad_checkpoint(decode_fn, z, use_reentrant=False)
        return reconstruction, z_mu, z_log_var

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space (for inference, no checkpointing)."""
        return self.model.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to output (for inference, no checkpointing)."""
        return self.model.decode(z)

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation for diffusion model."""
        return self.model.encode_stage_2_inputs(x)

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        """Decode diffusion model outputs."""
        return self.model.decode_stage_2_outputs(z)


class VAE3DTrainer(BaseCompression3DTrainer):
    """3D AutoencoderKL trainer with KL divergence regularization.

    Inherits from BaseCompression3DTrainer and adds:
    - KL divergence loss
    - 3D AutoencoderKL model creation
    - Gradient checkpointing for memory efficiency

    Args:
        cfg: Hydra configuration object.

    Example:
        >>> trainer = VAE3DTrainer(cfg)
        >>> trainer.setup_model()
        >>> trainer.train(train_loader, train_dataset, val_loader)
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize 3D VAE trainer.

        Args:
            cfg: Hydra configuration object.
        """
        super().__init__(cfg)

        # ─────────────────────────────────────────────────────────────────────
        # VAE-specific config
        # ─────────────────────────────────────────────────────────────────────
        self.kl_weight: float = cfg.vae_3d.get('kl_weight', 1e-6)
        self.latent_channels: int = cfg.vae_3d.latent_channels
        self.vae_channels: Tuple[int, ...] = tuple(cfg.vae_3d.channels)
        self.attention_levels: Tuple[bool, ...] = tuple(cfg.vae_3d.attention_levels)
        self.num_res_blocks: int = cfg.vae_3d.get('num_res_blocks', 2)

    def _get_disc_lr(self, cfg: DictConfig) -> float:
        """Get discriminator LR from vae_3d config."""
        return cfg.vae_3d.get('disc_lr', 1e-4)

    def _get_perceptual_weight(self, cfg: DictConfig) -> float:
        """Get perceptual weight from vae_3d config."""
        return cfg.vae_3d.get('perceptual_weight', 0.001)

    def _get_adv_weight(self, cfg: DictConfig) -> float:
        """Get adversarial weight from vae_3d config."""
        return cfg.vae_3d.get('adv_weight', 0.01)

    def _get_disable_gan(self, cfg: DictConfig) -> bool:
        """Determine if GAN is disabled from vae_3d config."""
        return cfg.vae_3d.get('disable_gan', False)

    def _create_fallback_save_dir(self) -> str:
        """Create fallback save directory for 3D VAE."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = self.cfg.training.get('name', '')
        return os.path.join(
            self.cfg.paths.model_dir, "compression_3d", self.cfg.mode.name,
            f"{exp_name}{self.volume_height}x{self.volume_depth}_{timestamp}"
        )

    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize 3D VAE model, discriminator, optimizers, and loss functions.

        Args:
            pretrained_checkpoint: Optional path to checkpoint for loading
                pretrained weights.
        """
        n_channels = self.cfg.mode.get('in_channels', 1)

        # Create 3D AutoencoderKL
        base_model = AutoencoderKL(
            spatial_dims=3,
            in_channels=n_channels,
            out_channels=n_channels,
            channels=self.vae_channels,
            attention_levels=self.attention_levels,
            latent_channels=self.latent_channels,
            num_res_blocks=self.num_res_blocks,
            norm_num_groups=32,
            with_encoder_nonlocal_attn=False,  # Disabled for 3D (memory)
            with_decoder_nonlocal_attn=False,
        ).to(self.device)

        # Load pretrained weights BEFORE wrapping
        if pretrained_checkpoint:
            self._load_pretrained_weights_base(base_model, pretrained_checkpoint)

        # Wrap with gradient checkpointing for memory efficiency
        if self.gradient_checkpointing:
            raw_model = CheckpointedAutoencoder(base_model)
            if self.is_main_process:
                logger.info("Gradient checkpointing enabled (CheckpointedAutoencoder wrapper)")
        else:
            raw_model = base_model

        # Create 3D discriminator if GAN enabled
        raw_disc = None
        if not self.disable_gan:
            raw_disc = self._create_discriminator(n_channels, spatial_dims=3)

        # Wrap models with DDP/compile
        self._wrap_models(raw_model, raw_disc)

        # Setup perceptual and adversarial loss
        if self.use_2_5d_perceptual and self.perceptual_weight > 0:
            self.perceptual_loss_fn = self._create_perceptual_loss(spatial_dims=2)  # 2D for 2.5D
        if not self.disable_gan:
            from monai.losses import PatchAdversarialLoss
            self.adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

        # Setup optimizers and schedulers
        self._setup_optimizers(n_channels)

        # Setup EMA (WARNING: doubles memory for 3D models)
        self._setup_ema()

        # Save metadata
        if self.is_main_process:
            self._save_metadata()

        # Log model info
        if self.is_main_process:
            vae_params = sum(p.numel() for p in self.model_raw.parameters())
            logger.info(f"3D VAE initialized: {vae_params / 1e6:.1f}M parameters")
            logger.info(f"  Channels: {self.vae_channels}")
            logger.info(f"  Latent channels: {self.latent_channels}")
            if not self.disable_gan:
                disc_params = sum(p.numel() for p in self.discriminator_raw.parameters())
                logger.info(f"3D Discriminator: {disc_params / 1e6:.1f}M parameters")
            else:
                logger.info("GAN disabled")
            logger.info(f"Volume: {self.volume_width}x{self.volume_height}x{self.volume_depth}")

            # Compute latent shape
            n_downsamples = len(self.vae_channels) - 1
            latent_h = self.volume_height // (2 ** n_downsamples)
            latent_w = self.volume_width // (2 ** n_downsamples)
            latent_d = self.volume_depth // (2 ** n_downsamples)
            logger.info(f"Latent shape: [{self.latent_channels}, {latent_d}, {latent_h}, {latent_w}]")

    def _load_pretrained_weights_base(
        self,
        base_model: nn.Module,
        checkpoint_path: str,
    ) -> None:
        """Load pretrained weights into base model (before checkpointing wrapper)."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Remove 'model.' prefix if present (from CheckpointedAutoencoder)
                # Check for consistent prefixing - warn if mixed (indicates corruption)
                keys_with_prefix = [k for k in state_dict.keys() if k.startswith('model.')]
                if keys_with_prefix:
                    if len(keys_with_prefix) != len(state_dict):
                        logger.warning(
                            f"Mixed prefix state: {len(keys_with_prefix)}/{len(state_dict)} keys "
                            "have 'model.' prefix. Stripping prefix from matching keys."
                        )
                    state_dict = {
                        k.replace('model.', '', 1) if k.startswith('model.') else k: v
                        for k, v in state_dict.items()
                    }
                base_model.load_state_dict(state_dict)
                if self.is_main_process:
                    logger.info(f"Loaded 3D VAE weights from {checkpoint_path}")
        except FileNotFoundError:
            if self.is_main_process:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")

    def _get_trainer_type(self) -> str:
        """Return trainer type for metadata."""
        return 'vae_3d'

    def _get_metadata_extra(self) -> Dict[str, Any]:
        """Return 3D VAE-specific metadata."""
        return {
            'vae_config': self._get_model_config(),
            'volume': {
                'height': self.volume_height,
                'width': self.volume_width,
                'depth': self.volume_depth,
            },
        }

    def _compute_kl_loss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss.

        Uses the same aggregation as 2D VAE: sum over spatial dimensions [C, D, H, W],
        then average over batch. This ensures KL regularization strength is consistent
        with 2D training (previous version used .mean() over all dims, making KL ~80,000x weaker).

        Args:
            mean: Mean of latent distribution [B, C, D, H, W].
            logvar: Log variance of latent distribution [B, C, D, H, W].

        Returns:
            KL divergence loss (scalar).
        """
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Sum over spatial dims [C, D, H, W], then average over batch
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=[1, 2, 3, 4])
        return kl.mean()

    def train_step(self, batch: Any) -> TrainingStepResult:
        """Execute 3D VAE training step with KL loss.

        Args:
            batch: Input batch.

        Returns:
            TrainingStepResult with all loss components.
        """
        images, _ = self._prepare_batch(batch)
        grad_clip = self.cfg.training.get('gradient_clip_norm', 1.0)

        d_loss = torch.tensor(0.0, device=self.device)
        adv_loss = torch.tensor(0.0, device=self.device)

        # ==================== Generator Step ====================
        self.optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            reconstruction, mean, logvar = self.model(images)

            # L1 reconstruction loss
            l1_loss = torch.nn.functional.l1_loss(reconstruction, images)

            # KL divergence loss
            kl_loss = self._compute_kl_loss(mean, logvar)

            # Perceptual loss (2.5D)
            if self.perceptual_weight > 0 and self.perceptual_loss_fn is not None:
                p_loss = self._compute_2_5d_perceptual_loss(reconstruction, images)
            else:
                p_loss = torch.tensor(0.0, device=self.device)

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

        # ==================== Discriminator Step ====================
        if not self.disable_gan:
            d_loss = self._train_discriminator_step(images, reconstruction.detach())

        return TrainingStepResult(
            total_loss=g_loss.item(),
            reconstruction_loss=l1_loss.item(),
            perceptual_loss=p_loss.item() if isinstance(p_loss, torch.Tensor) else p_loss,
            regularization_loss=kl_loss.item(),
            adversarial_loss=adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
            discriminator_loss=d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss,
        )

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train 3D VAE for one epoch.

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

        disable_pbar = not self.is_main_process or self.is_cluster
        total = self.limit_train_batches if self.limit_train_batches else len(data_loader)
        iterator = itertools.islice(data_loader, self.limit_train_batches) if self.limit_train_batches else data_loader
        pbar = tqdm(iterator, desc=f"Epoch {epoch}", disable=disable_pbar, total=total)

        for step, batch in enumerate(pbar):
            result = self.train_step(batch)
            losses = result.to_legacy_dict('kl')

            for key in epoch_losses:
                epoch_losses[key] += losses[key]

            if not disable_pbar:
                if not self.disable_gan:
                    pbar.set_postfix(
                        G=f"{losses['gen']:.4f}",
                        D=f"{losses['disc']:.4f}",
                        L1=f"{losses['recon']:.4f}",
                    )
                else:
                    pbar.set_postfix(
                        G=f"{losses['gen']:.4f}",
                        L1=f"{losses['recon']:.4f}",
                        KL=f"{losses['kl']:.4f}",
                    )

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
        """3D VAE forward pass for validation.

        Returns:
            Tuple of (reconstruction, weighted_kl_loss).
        """
        reconstruction, mean, logvar = model(images)
        kl_loss = self._compute_kl_loss(mean, logvar)
        return reconstruction, self.kl_weight * kl_loss

    def _get_model_config(self) -> Dict[str, Any]:
        """Get 3D VAE model configuration for checkpoint."""
        n_channels = self.cfg.mode.get('in_channels', 1)
        return {
            'in_channels': n_channels,
            'out_channels': n_channels,
            'latent_channels': self.latent_channels,
            'channels': list(self.vae_channels),
            'attention_levels': list(self.attention_levels),
            'num_res_blocks': self.num_res_blocks,
            'spatial_dims': 3,
        }

    def _log_validation_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float],
        worst_batch_data: Optional[Dict[str, Any]],
        regional_tracker: Optional[RegionalMetricsTracker3D],
        log_figures: bool,
    ) -> None:
        """Log 3D VAE validation metrics including KL loss."""
        if self.writer is None:
            return

        # Call base class method first
        super()._log_validation_metrics(epoch, metrics, worst_batch_data, regional_tracker, log_figures)

        # Add VAE-specific KL loss logging
        if 'reg' in metrics and self.kl_weight > 0:
            self.writer.add_scalar('Loss/KL_val', metrics['reg'] / self.kl_weight, epoch)

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
        elapsed_time: float,
    ) -> None:
        """Log 3D VAE epoch summary."""
        log_compression_epoch_summary(
            epoch, total_epochs, avg_losses, val_metrics, elapsed_time,
            regularization_key=None,  # VAE's KL is in 'gen' already
        )

    def _test_forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Perform 3D VAE forward pass for test evaluation.

        Args:
            model: Model to use for inference.
            images: Input images.

        Returns:
            Reconstructed images tensor.
        """
        reconstructed, _, _ = model(images)
        return reconstructed
