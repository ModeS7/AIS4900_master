"""
VAE trainer module for training AutoencoderKL models (2D and 3D).

This module provides the VAETrainer class which inherits from BaseCompressionTrainer
and implements VAE-specific functionality:
- KL divergence regularization
- AutoencoderKL model creation (2D or 3D)
- Gradient checkpointing for memory efficiency (3D)
"""
import itertools
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Disable MONAI MetaTensor tracking BEFORE importing MONAI modules
from monai.data import set_track_meta
set_track_meta(False)

from monai.networks.nets import AutoencoderKL

from .checkpointing import BaseCheckpointedModel
from .compression_trainer import BaseCompressionTrainer
from .results import TrainingStepResult
from .utils import create_epoch_iterator, get_vram_usage

logger = logging.getLogger(__name__)


class CheckpointedAutoencoder(BaseCheckpointedModel):
    """Wrapper that applies gradient checkpointing to MONAI AutoencoderKL.

    MONAI's AutoencoderKL doesn't have built-in gradient checkpointing.
    This wrapper uses torch.utils.checkpoint to trade compute for memory,
    reducing activation memory by ~50% for 3D volumes.

    Args:
        model: The underlying AutoencoderKL model.
    """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with gradient checkpointing."""
        def encode_fn(x):
            h = self.model.encoder(x)
            z_mu = self.model.quant_conv_mu(h)
            z_log_var = self.model.quant_conv_log_sigma(h)
            return z_mu, z_log_var

        z_mu, z_log_var = self.checkpoint(encode_fn, x)
        z = self.model.sampling(z_mu, z_log_var)

        def decode_fn(z):
            z_post = self.model.post_quant_conv(z)
            return self.model.decoder(z_post)

        reconstruction = self.checkpoint(decode_fn, z)
        return reconstruction, z_mu, z_log_var

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation for diffusion model."""
        return self.model.encode_stage_2_inputs(x)

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        """Decode diffusion model outputs."""
        return self.model.decode_stage_2_outputs(z)


class VAETrainer(BaseCompressionTrainer):
    """AutoencoderKL trainer with KL divergence regularization (2D and 3D).

    Supports both 2D images and 3D volumes via the spatial_dims parameter.

    Inherits from BaseCompressionTrainer and adds:
    - KL divergence loss
    - AutoencoderKL model creation
    - Gradient checkpointing for 3D memory efficiency

    Args:
        cfg: Hydra configuration object.
        spatial_dims: Spatial dimensions (2 or 3).

    Example:
        >>> trainer = VAETrainer(cfg, spatial_dims=2)
        >>> trainer.setup_model()
        >>> trainer.train(train_loader, train_dataset, val_loader)
    """

    def __init__(self, cfg: DictConfig, spatial_dims: int = 2) -> None:
        """Initialize VAE trainer.

        Args:
            cfg: Hydra configuration object.
            spatial_dims: Spatial dimensions (2 or 3).
        """
        super().__init__(cfg, spatial_dims=spatial_dims)

        # ─────────────────────────────────────────────────────────────────────
        # VAE-specific config (dimension-dependent)
        # ─────────────────────────────────────────────────────────────────────
        vae_cfg = cfg.vae_3d if spatial_dims == 3 else cfg.vae
        self.kl_weight: float = vae_cfg.get('kl_weight', 1e-6)
        self.latent_channels: int = vae_cfg.latent_channels
        self.vae_channels: Tuple[int, ...] = tuple(vae_cfg.channels)
        self.attention_levels: Tuple[bool, ...] = tuple(vae_cfg.attention_levels)
        self.num_res_blocks: int = vae_cfg.get('num_res_blocks', 2)

        # Initialize unified metrics system
        self._init_unified_metrics('vae')

    @classmethod
    def create_2d(cls, cfg: DictConfig, **kwargs) -> 'VAETrainer':
        """Create 2D VAETrainer.

        Args:
            cfg: Hydra configuration object.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            VAETrainer configured for 2D images.
        """
        return cls(cfg, spatial_dims=2, **kwargs)

    @classmethod
    def create_3d(cls, cfg: DictConfig, **kwargs) -> 'VAETrainer':
        """Create 3D VAETrainer.

        Args:
            cfg: Hydra configuration object.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            VAETrainer configured for 3D volumes.
        """
        return cls(cfg, spatial_dims=3, **kwargs)

    def _get_disc_lr(self, cfg: DictConfig) -> float:
        """Get discriminator LR from vae/vae_3d config."""
        section = 'vae_3d' if self.spatial_dims == 3 else 'vae'
        if section in cfg:
            return cfg[section].get('disc_lr', 1e-4 if self.spatial_dims == 3 else 5e-4)
        return 5e-4

    def _get_perceptual_weight(self, cfg: DictConfig) -> float:
        """Get perceptual weight from vae/vae_3d config."""
        section = 'vae_3d' if self.spatial_dims == 3 else 'vae'
        if section in cfg:
            return cfg[section].get('perceptual_weight', 0.001)
        return 0.001

    def _get_adv_weight(self, cfg: DictConfig) -> float:
        """Get adversarial weight from vae/vae_3d config."""
        section = 'vae_3d' if self.spatial_dims == 3 else 'vae'
        if section in cfg:
            return cfg[section].get('adv_weight', 0.01)
        return 0.01

    def _get_disable_gan(self, cfg: DictConfig) -> bool:
        """Determine if GAN is disabled from vae/vae_3d config."""
        section = 'vae_3d' if self.spatial_dims == 3 else 'vae'
        if section in cfg:
            return cfg[section].get('disable_gan', False)
        return False

    def _create_fallback_save_dir(self) -> str:
        """Create fallback save directory for VAE."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = self.cfg.training.get('name', '')
        mode_name = self.cfg.mode.get('name', 'dual')

        if self.spatial_dims == 3:
            run_name = f"{exp_name}{self.volume_height}x{self.volume_depth}_{timestamp}"
            return os.path.join(self.cfg.paths.model_dir, 'compression_3d', mode_name, run_name)
        else:
            run_name = f"{exp_name}{self.image_size}_{timestamp}"
            return os.path.join(self.cfg.paths.model_dir, 'vae_2d', mode_name, run_name)

    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize VAE model, discriminator, optimizers, and loss functions.

        Args:
            pretrained_checkpoint: Optional path to checkpoint for loading
                pretrained weights.
        """
        n_channels = self.cfg.mode.get('in_channels', 1)

        # 3D-specific: disable nonlocal attention for memory
        if self.spatial_dims == 3:
            with_nonlocal = False
        else:
            with_nonlocal = True

        # Create AutoencoderKL
        base_model = AutoencoderKL(
            spatial_dims=self.spatial_dims,
            in_channels=n_channels,
            out_channels=n_channels,
            channels=self.vae_channels,
            attention_levels=self.attention_levels,
            latent_channels=self.latent_channels,
            num_res_blocks=self.num_res_blocks,
            norm_num_groups=32,
            with_encoder_nonlocal_attn=with_nonlocal,
            with_decoder_nonlocal_attn=with_nonlocal,
        ).to(self.device)

        # Load pretrained weights BEFORE wrapping (for 3D with checkpointing)
        if pretrained_checkpoint and self.spatial_dims == 3:
            self._load_pretrained_weights_base(base_model, pretrained_checkpoint, "3D VAE")

        # 3D: Wrap with gradient checkpointing for memory efficiency
        if self.spatial_dims == 3 and self.gradient_checkpointing:
            raw_model = CheckpointedAutoencoder(base_model)
            if self.is_main_process:
                logger.info("Gradient checkpointing enabled (CheckpointedAutoencoder wrapper)")
        else:
            raw_model = base_model

        # Create discriminator if GAN enabled
        raw_disc = None
        if not self.disable_gan:
            raw_disc = self._create_discriminator(n_channels, spatial_dims=self.spatial_dims)

        # 2D: Load pretrained weights after wrapping (no prefix to strip)
        if pretrained_checkpoint and self.spatial_dims == 2:
            self._load_pretrained_weights(raw_model, raw_disc, pretrained_checkpoint, model_name="VAE")

        # Wrap models with DDP/compile
        self._wrap_models(raw_model, raw_disc)

        # Setup perceptual and adversarial loss
        # 3D: use 2D perceptual loss for 2.5D computation
        perceptual_spatial_dims = 2 if (self.spatial_dims == 3 and getattr(self, 'use_2_5d_perceptual', False)) else self.spatial_dims
        if self.perceptual_weight > 0:
            self.perceptual_loss_fn = self._create_perceptual_loss(spatial_dims=perceptual_spatial_dims)
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
            dim_str = "3D " if self.spatial_dims == 3 else ""
            logger.info(f"{dim_str}VAE initialized: {vae_params / 1e6:.1f}M parameters")
            if self.spatial_dims == 3:
                logger.info(f"  Channels: {self.vae_channels}")
            logger.info(f"  Latent channels: {self.latent_channels}")
            if not self.disable_gan:
                disc_params = sum(p.numel() for p in self.discriminator_raw.parameters())
                logger.info(f"{dim_str}Discriminator: {disc_params / 1e6:.1f}M parameters")
            else:
                logger.info("GAN disabled")

            # Log spatial info
            if self.spatial_dims == 3:
                logger.info(f"Volume: {self.volume_width}x{self.volume_height}x{self.volume_depth}")
                n_downsamples = len(self.vae_channels) - 1
                latent_h = self.volume_height // (2 ** n_downsamples)
                latent_w = self.volume_width // (2 ** n_downsamples)
                latent_d = self.volume_depth // (2 ** n_downsamples)
                logger.info(f"Latent shape: [{self.latent_channels}, {latent_d}, {latent_h}, {latent_w}]")
            else:
                logger.info(f"Latent shape: [{self.latent_channels}, {self.image_size // 8}, {self.image_size // 8}]")
                logger.info(f"Loss weights - Perceptual: {self.perceptual_weight}, KL: {self.kl_weight}, Adv: {self.adv_weight}")

    def _get_trainer_type(self) -> str:
        """Return trainer type for metadata."""
        return 'vae_3d' if self.spatial_dims == 3 else 'vae'

    def _get_metadata_extra(self) -> Dict[str, Any]:
        """Return VAE-specific metadata."""
        meta = {'vae_config': self._get_model_config()}
        if self.spatial_dims == 3:
            meta['volume'] = {
                'height': self.volume_height,
                'width': self.volume_width,
                'depth': self.volume_depth,
            }
        else:
            meta['image_size'] = self.image_size
        return meta

    def train_step(self, batch: Any) -> TrainingStepResult:
        """Execute VAE training step with KL loss.

        Args:
            batch: Input batch.

        Returns:
            TrainingStepResult with all loss components.
        """
        images, mask = self._prepare_batch(batch)
        grad_clip = self.cfg.training.get('gradient_clip_norm', 1.0)

        d_loss = torch.tensor(0.0, device=self.device)
        adv_loss = torch.tensor(0.0, device=self.device)

        # ==================== Discriminator Step (2D only, before generator) ====================
        if self.spatial_dims == 2 and not self.disable_gan:
            with torch.no_grad():
                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    reconstruction_for_d, _, _ = self.model(images)
            d_loss = self._train_discriminator_step(images, reconstruction_for_d)

        # ==================== Generator Step ====================
        self.optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            reconstruction, mean, logvar = self.model(images)

            # L1 reconstruction loss
            l1_loss = torch.nn.functional.l1_loss(reconstruction, images)

            # KL divergence loss
            kl_loss = self._compute_kl_loss(mean, logvar)

            # Perceptual loss (standard 2D or 2.5D for 3D)
            p_loss = self._compute_perceptual_loss(reconstruction, images)

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

        # ==================== Discriminator Step (3D: after generator) ====================
        if self.spatial_dims == 3 and not self.disable_gan:
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

        # Use unified loss accumulator
        self._loss_accumulator.reset()

        if self.spatial_dims == 3:
            # 3D: Use tqdm directly with itertools.islice for limit_batches
            disable_pbar = not self.is_main_process or self.is_cluster
            total = self.limit_train_batches if self.limit_train_batches else len(data_loader)
            iterator = itertools.islice(data_loader, self.limit_train_batches) if self.limit_train_batches else data_loader
            epoch_iter = tqdm(iterator, desc=f"Epoch {epoch}", disable=disable_pbar, total=total)
        else:
            # 2D: Use create_epoch_iterator helper
            epoch_iter = create_epoch_iterator(
                data_loader, epoch, self.is_cluster, self.is_main_process,
                limit_batches=self.limit_train_batches
            )

        for step, batch in enumerate(epoch_iter):
            result = self.train_step(batch)
            losses = result.to_legacy_dict('kl')

            # Step profiler to mark training step boundary
            self._profiler_step()

            # Accumulate with unified system
            self._loss_accumulator.update(losses)

            if hasattr(epoch_iter, 'set_postfix'):
                avg_so_far = self._loss_accumulator.compute()
                if not self.disable_gan:
                    epoch_iter.set_postfix(
                        G=f"{avg_so_far.get('gen', 0):.4f}",
                        D=f"{avg_so_far.get('disc', 0):.4f}",
                        L1=f"{losses.get('recon', 0):.4f}",
                    )
                else:
                    epoch_iter.set_postfix(
                        G=f"{avg_so_far.get('gen', 0):.4f}",
                        L1=f"{losses.get('recon', 0):.4f}",
                        KL=f"{losses.get('kl', 0):.6f}",
                    )

            if epoch == 1 and step == 0 and self.is_main_process:
                logger.info(get_vram_usage(self.device))

        # Compute average losses using unified system
        avg_losses = self._loss_accumulator.compute()

        # Log training metrics using unified system
        self._log_training_metrics_unified(epoch, avg_losses)

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
        config = {
            'in_channels': n_channels,
            'out_channels': n_channels,
            'latent_channels': self.latent_channels,
            'channels': list(self.vae_channels),
            'attention_levels': list(self.attention_levels),
            'num_res_blocks': self.num_res_blocks,
            'norm_num_groups': 32,
        }
        if self.spatial_dims == 3:
            config['spatial_dims'] = 3
        else:
            config['with_encoder_nonlocal_attn'] = True
            config['with_decoder_nonlocal_attn'] = True
        return config

    def _log_validation_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float],
        worst_batch_data: Optional[Dict[str, Any]],
        regional_tracker,
        log_figures: bool,
    ) -> None:
        """Log VAE validation metrics including KL loss."""
        if self.writer is None:
            return

        # Log metrics with modality suffix handling
        self._log_validation_metrics_core(epoch, metrics)

        # VAE-specific: log KL loss using unified system
        if 'reg' in metrics:
            unweighted_kl = metrics['reg'] / self.kl_weight if self.kl_weight > 0 else None
            self._unified_metrics.log_regularization_loss(
                loss_type='KL',
                weighted_loss=metrics['reg'],
                epoch=epoch,
                unweighted_loss=unweighted_kl,
            )

        # Log worst batch figure (uses unified metrics - handles 2D/3D automatically)
        if log_figures and worst_batch_data is not None:
            self._unified_metrics.log_worst_batch(
                original=worst_batch_data['original'],
                reconstructed=worst_batch_data['generated'],
                loss=worst_batch_data.get('loss', 0.0),
                epoch=epoch,
                phase='val',
            )

        # Log regional metrics with modality suffix for single-modality modes
        if regional_tracker is not None:
            mode_name = self.cfg.mode.get('name', 'bravo')
            is_multi_modality = mode_name == 'multi_modality'
            is_dual = self.cfg.mode.get('in_channels', 1) == 2 and mode_name == 'dual'
            modality_override = mode_name if not is_multi_modality and not is_dual else None
            self._unified_metrics.log_validation_regional(regional_tracker, epoch, modality_override=modality_override)

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
        elapsed_time: float,
    ) -> None:
        """Log VAE epoch summary using unified system."""
        self._log_epoch_summary_unified(epoch, avg_losses, val_metrics, elapsed_time)

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
