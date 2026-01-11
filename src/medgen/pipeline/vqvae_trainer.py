"""
VQ-VAE trainer module for training vector-quantized autoencoders.

This module provides the VQVAETrainer class which inherits from BaseCompressionTrainer
and implements VQ-VAE-specific functionality:
- Vector quantization with discrete latent codes
- VQ loss (commitment + codebook loss)
- VQVAE model creation
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

from monai.networks.nets import VQVAE

from .compression_trainer import BaseCompressionTrainer
from .results import TrainingStepResult
from .metrics import (
    compute_lpips,
    compute_msssim,
    compute_psnr,
    create_reconstruction_figure,
    RegionalMetricsTracker,
)
from .tracking import create_worst_batch_figure, CodebookTracker
from .utils import create_epoch_iterator, get_vram_usage, log_compression_epoch_summary

logger = logging.getLogger(__name__)


class VQVAETrainer(BaseCompressionTrainer):
    """VQ-VAE trainer with discrete latent space.

    Inherits from BaseCompressionTrainer and adds:
    - Vector quantization (VQ) loss
    - VQVAE model creation from MONAI
    - VQ-specific forward pass returning (reconstruction, vq_loss)

    Args:
        cfg: Hydra configuration object.

    Example:
        >>> trainer = VQVAETrainer(cfg)
        >>> trainer.setup_model()
        >>> trainer.train(train_loader, train_dataset, val_loader)
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize VQ-VAE trainer.

        Args:
            cfg: Hydra configuration object.
        """
        super().__init__(cfg)

        # ─────────────────────────────────────────────────────────────────────
        # VQ-VAE-specific config
        # ─────────────────────────────────────────────────────────────────────
        self.num_embeddings: int = cfg.vqvae.get('num_embeddings', 512)
        self.embedding_dim: int = cfg.vqvae.get('embedding_dim', 64)
        self.commitment_cost: float = cfg.vqvae.get('commitment_cost', 0.25)
        self.decay: float = cfg.vqvae.get('decay', 0.99)
        self.epsilon: float = cfg.vqvae.get('epsilon', 1e-5)

        # Architecture config
        self.channels: Tuple[int, ...] = tuple(cfg.vqvae.get('channels', [96, 96, 192]))
        self.num_res_layers: int = cfg.vqvae.get('num_res_layers', 3)
        self.num_res_channels: Tuple[int, ...] = tuple(
            cfg.vqvae.get('num_res_channels', [96, 96, 192])
        )
        self.downsample_parameters: Tuple[Tuple[int, ...], ...] = tuple(
            tuple(p) for p in cfg.vqvae.get('downsample_parameters', [[2, 4, 1, 1]] * 3)
        )
        self.upsample_parameters: Tuple[Tuple[int, ...], ...] = tuple(
            tuple(p) for p in cfg.vqvae.get('upsample_parameters', [[2, 4, 1, 1, 0]] * 3)
        )

        self.image_size: int = cfg.model.image_size

        # Codebook tracking (initialized after model setup)
        self._codebook_tracker: Optional[CodebookTracker] = None

        # Initialize unified metrics system
        self._init_unified_metrics('vqvae')

    def _create_fallback_save_dir(self) -> str:
        """Create fallback save directory for VQ-VAE."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = self.cfg.training.get('name', '')
        mode_name = self.cfg.mode.get('name', 'dual')
        run_name = f"{exp_name}{self.image_size}_{timestamp}"
        return os.path.join(self.cfg.paths.model_dir, 'vqvae_2d', mode_name, run_name)

    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize VQ-VAE model, discriminator, optimizers, and loss functions.

        Args:
            pretrained_checkpoint: Optional path to checkpoint for loading
                pretrained weights.
        """
        n_channels = self.cfg.mode.get('in_channels', 1)

        # Create VQVAE
        raw_model = VQVAE(
            spatial_dims=2,
            in_channels=n_channels,
            out_channels=n_channels,
            channels=self.channels,
            num_res_layers=self.num_res_layers,
            num_res_channels=self.num_res_channels,
            downsample_parameters=self.downsample_parameters,
            upsample_parameters=self.upsample_parameters,
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            commitment_cost=self.commitment_cost,
            decay=self.decay,
            epsilon=self.epsilon,
            ddp_sync=self.use_multi_gpu,
        ).to(self.device)

        # Create discriminator if GAN enabled
        raw_disc = None
        if not self.disable_gan:
            raw_disc = self._create_discriminator(n_channels, spatial_dims=2)

        # Load pretrained weights if provided
        if pretrained_checkpoint:
            self._load_pretrained_weights(raw_model, raw_disc, pretrained_checkpoint, model_name="VQ-VAE")

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

        # Initialize codebook tracker
        self._codebook_tracker = CodebookTracker(self.num_embeddings, self.device)

        # Save metadata
        if self.is_main_process:
            self._save_metadata()

        # Log model info
        if self.is_main_process:
            vqvae_params = sum(p.numel() for p in self.model_raw.parameters())
            logger.info(f"VQ-VAE initialized: {vqvae_params / 1e6:.1f}M parameters")
            logger.info(f"  Codebook: {self.num_embeddings} embeddings x {self.embedding_dim} dim")
            logger.info(f"  Channels: {self.channels}")
            if not self.disable_gan:
                disc_params = sum(p.numel() for p in self.discriminator_raw.parameters())
                logger.info(f"Discriminator initialized: {disc_params / 1e6:.1f}M parameters")
            else:
                logger.info("GAN disabled - discriminator not created")

            # Compute latent size
            n_downsamples = len(self.downsample_parameters)
            latent_size = self.image_size // (2 ** n_downsamples)
            logger.info(f"Latent shape: [{self.embedding_dim}, {latent_size}, {latent_size}]")
            logger.info(f"Loss weights - Perceptual: {self.perceptual_weight}, Adv: {self.adv_weight}")
            logger.info(f"VQ params - Commitment: {self.commitment_cost}, Decay: {self.decay}")

    def _get_trainer_type(self) -> str:
        """Return trainer type for metadata."""
        return 'vqvae'

    def _get_metadata_extra(self) -> Dict[str, Any]:
        """Return VQ-VAE-specific metadata."""
        return {
            'vqvae_config': self._get_model_config(),
            'image_size': self.image_size,
        }

    def train_step(self, batch: Any) -> TrainingStepResult:
        """Execute VQ-VAE training step.

        Args:
            batch: Input batch.

        Returns:
            TrainingStepResult with all loss components.
        """
        images, mask = self._prepare_batch(batch)
        grad_clip = self.cfg.training.get('gradient_clip_norm', 1.0)

        d_loss = torch.tensor(0.0, device=self.device)
        adv_loss = torch.tensor(0.0, device=self.device)

        # ==================== Discriminator Step ====================
        if not self.disable_gan:
            with torch.no_grad():
                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    reconstruction_for_d, _ = self.model(images)

            d_loss = self._train_discriminator_step(images, reconstruction_for_d)

        # ==================== Generator Step ====================
        self.optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            # VQVAE forward returns (reconstruction, vq_loss)
            reconstruction, vq_loss = self.model(images)

            # L1 reconstruction loss
            l1_loss = torch.abs(reconstruction - images).mean()

            # Perceptual loss
            p_loss = self._compute_perceptual_loss(reconstruction, images)

            # Adversarial loss
            if not self.disable_gan:
                adv_loss = self._compute_adversarial_loss(reconstruction)

            # Total generator loss
            # Note: vq_loss already includes commitment_cost weighting from the model
            g_loss = (
                l1_loss
                + self.perceptual_weight * p_loss
                + vq_loss
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

        return TrainingStepResult(
            total_loss=g_loss.item(),
            reconstruction_loss=l1_loss.item(),
            perceptual_loss=p_loss.item(),
            regularization_loss=vq_loss.item(),
            adversarial_loss=adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
            discriminator_loss=d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss,
        )

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train VQ-VAE for one epoch.

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

        epoch_iter = create_epoch_iterator(
            data_loader, epoch, self.is_cluster, self.is_main_process,
            limit_batches=self.limit_train_batches
        )

        for step, batch in enumerate(epoch_iter):
            result = self.train_step(batch)
            losses = result.to_legacy_dict('vq')

            # Step profiler to mark training step boundary
            self._profiler_step()

            # Accumulate with unified system
            self._loss_accumulator.update(losses)

            if hasattr(epoch_iter, 'set_postfix'):
                avg_so_far = self._loss_accumulator.compute()
                epoch_iter.set_postfix(
                    G=f"{avg_so_far.get('gen', 0):.4f}",
                    VQ=f"{avg_so_far.get('vq', 0):.4f}",
                    D=f"{avg_so_far.get('disc', 0):.4f}"
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
        """VQ-VAE forward pass for validation.

        Returns:
            Tuple of (reconstruction, vq_loss).
        """
        reconstruction, vq_loss = model(images)
        return reconstruction, vq_loss

    def _get_model_config(self) -> Dict[str, Any]:
        """Get VQ-VAE model configuration for checkpoint."""
        n_channels = self.cfg.mode.get('in_channels', 1)
        return {
            'in_channels': n_channels,
            'out_channels': n_channels,
            'channels': list(self.channels),
            'num_res_layers': self.num_res_layers,
            'num_res_channels': list(self.num_res_channels),
            'downsample_parameters': [list(p) for p in self.downsample_parameters],
            'upsample_parameters': [list(p) for p in self.upsample_parameters],
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embedding_dim,
            'commitment_cost': self.commitment_cost,
            'decay': self.decay,
            'epsilon': self.epsilon,
        }

    def _log_validation_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float],
        worst_batch_data: Optional[Dict[str, Any]],
        regional_tracker: Optional[RegionalMetricsTracker],
        log_figures: bool,
    ) -> None:
        """Log VQ-VAE validation metrics including VQ loss."""
        if self.writer is None:
            return

        # Use unified validation logging for common metrics
        self._log_validation_metrics_unified(epoch, metrics)

        # VQ-VAE-specific: log VQ loss (no unweighting needed)
        if 'reg' in metrics and hasattr(self, '_metrics_logger'):
            self._metrics_logger.log_regularization(epoch, metrics['reg'], suffix='val')

        # Log worst batch figure (keep existing logic)
        if log_figures and worst_batch_data is not None:
            fig = self._create_worst_batch_figure(worst_batch_data)
            self.writer.add_figure('Validation/worst_batch', fig, epoch)
            plt.close(fig)

        # Log regional metrics
        if regional_tracker is not None:
            regional_tracker.log_to_tensorboard(self.writer, epoch, prefix='regional')

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
        elapsed_time: float,
    ) -> None:
        """Log VQ-VAE epoch summary using unified system."""
        self._log_epoch_summary_unified(epoch, avg_losses, val_metrics, elapsed_time)

    def _test_forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Perform VQ-VAE forward pass for test evaluation.

        Args:
            model: Model to use for inference.
            images: Input images.

        Returns:
            Reconstructed images tensor.
        """
        reconstructed, _ = model(images)
        return reconstructed

    def _track_codebook_usage(
        self,
        data_loader: DataLoader,
        max_batches: int = 50,
    ) -> None:
        """Track codebook usage on validation data.

        Samples batches from validation data and tracks which codebook
        indices are selected. Results are stored in self._codebook_tracker.

        Args:
            data_loader: Validation data loader.
            max_batches: Maximum batches to process for tracking.
        """
        if self._codebook_tracker is None:
            return

        self._codebook_tracker.reset()
        model_to_use = self._get_model_for_eval()
        model_to_use.eval()

        # Get the raw model for index_quantize (unwrap DDP/compiled)
        raw_model = self.model_raw

        with torch.inference_mode():
            for i, batch in enumerate(data_loader):
                if i >= max_batches:
                    break

                images, _ = self._prepare_batch(batch)

                # Get codebook indices
                indices = raw_model.index_quantize(images)
                self._codebook_tracker.update_fast(indices)

    def compute_validation_losses(
        self,
        epoch: int,
        log_figures: bool = True,
    ) -> Dict[str, float]:
        """Compute validation losses with codebook tracking.

        Extends parent method to also track codebook utilization.

        Args:
            epoch: Current epoch number.
            log_figures: Whether to log figures.

        Returns:
            Dictionary of validation metrics.
        """
        # Call parent validation
        metrics = super().compute_validation_losses(epoch, log_figures)

        # Track codebook usage and log
        if self.is_main_process and self._codebook_tracker is not None:
            self._track_codebook_usage(self.val_loader, max_batches=50)

            if self.writer is not None:
                cb_metrics = self._codebook_tracker.log_to_tensorboard(
                    self.writer, epoch, prefix='Codebook'
                )

                # Add to returned metrics for logging
                metrics['codebook_perplexity'] = cb_metrics['perplexity']
                metrics['codebook_utilization'] = cb_metrics['utilization']

            # Log summary to console
            self._codebook_tracker.log_summary()

        return metrics
