"""
3D VQ-VAE trainer module for training volumetric vector-quantized autoencoders.

This module provides the VQVAE3DTrainer class which inherits from BaseCompression3DTrainer
and implements 3D VQ-VAE-specific functionality:
- Vector quantization with discrete latent codes
- VQ loss (commitment + codebook loss)
- 3D VQVAE model creation
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

from monai.networks.nets import VQVAE

from .compression_trainer import BaseCompression3DTrainer
from .metrics import (
    compute_lpips_3d,
    compute_msssim,
    compute_psnr,
    RegionalMetricsTracker3D,
)
from .tracking import create_worst_batch_figure_3d
from .results import TrainingStepResult
from .utils import get_vram_usage, log_compression_epoch_summary

logger = logging.getLogger(__name__)


class CheckpointedVQVAE(nn.Module):
    """Wrapper that applies gradient checkpointing to MONAI VQVAE.

    Reduces activation memory by ~50% for 3D volumes.

    Args:
        model: The underlying VQVAE model.
    """

    def __init__(self, model: VQVAE):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with gradient checkpointing."""
        def encode_fn(x):
            return self.model.encode(x)

        encoded = grad_checkpoint(encode_fn, x, use_reentrant=False)

        def quantize_fn(z):
            return self.model.quantize(z)

        quantized, vq_loss = grad_checkpoint(quantize_fn, encoded, use_reentrant=False)

        def decode_fn(z):
            return self.model.decode(z)

        reconstruction = grad_checkpoint(decode_fn, quantized, use_reentrant=False)
        return reconstruction, vq_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without checkpointing (for inference)."""
        return self.model.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode without checkpointing (for inference)."""
        return self.model.decode(z)

    def index_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Get codebook indices for input."""
        return self.model.index_quantize(x)

    def decode_samples(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from codebook indices."""
        return self.model.decode_samples(indices)


class VQVAE3DTrainer(BaseCompression3DTrainer):
    """3D VQ-VAE trainer with discrete latent space.

    Inherits from BaseCompression3DTrainer and adds:
    - Vector quantization (VQ) loss
    - 3D VQVAE model creation from MONAI
    - Gradient checkpointing for memory efficiency

    Args:
        cfg: Hydra configuration object.

    Example:
        >>> trainer = VQVAE3DTrainer(cfg)
        >>> trainer.setup_model()
        >>> trainer.train(train_loader, train_dataset, val_loader)
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize 3D VQ-VAE trainer.

        Args:
            cfg: Hydra configuration object.
        """
        super().__init__(cfg)

        # ─────────────────────────────────────────────────────────────────────
        # VQ-VAE-specific config
        # ─────────────────────────────────────────────────────────────────────
        self.num_embeddings: int = cfg.vqvae_3d.get('num_embeddings', 512)
        self.embedding_dim: int = cfg.vqvae_3d.get('embedding_dim', 3)
        self.commitment_cost: float = cfg.vqvae_3d.get('commitment_cost', 0.25)
        self.decay: float = cfg.vqvae_3d.get('decay', 0.99)
        self.epsilon: float = cfg.vqvae_3d.get('epsilon', 1e-5)

        # Architecture config
        self.channels: Tuple[int, ...] = tuple(cfg.vqvae_3d.get('channels', [64, 128]))
        self.num_res_layers: int = cfg.vqvae_3d.get('num_res_layers', 2)
        self.num_res_channels: Tuple[int, ...] = tuple(
            cfg.vqvae_3d.get('num_res_channels', [64, 128])
        )
        self.downsample_parameters: Tuple[Tuple[int, ...], ...] = tuple(
            tuple(p) for p in cfg.vqvae_3d.get('downsample_parameters', [[2, 4, 1, 1]] * 2)
        )
        self.upsample_parameters: Tuple[Tuple[int, ...], ...] = tuple(
            tuple(p) for p in cfg.vqvae_3d.get('upsample_parameters', [[2, 4, 1, 1, 0]] * 2)
        )

    def _get_disc_lr(self, cfg: DictConfig) -> float:
        """Get discriminator LR from vqvae_3d config."""
        return cfg.vqvae_3d.get('disc_lr', 5e-4)

    def _get_perceptual_weight(self, cfg: DictConfig) -> float:
        """Get perceptual weight from vqvae_3d config."""
        return cfg.vqvae_3d.get('perceptual_weight', 0.002)

    def _get_adv_weight(self, cfg: DictConfig) -> float:
        """Get adversarial weight from vqvae_3d config."""
        return cfg.vqvae_3d.get('adv_weight', 0.005)

    def _get_disable_gan(self, cfg: DictConfig) -> bool:
        """Determine if GAN is disabled from vqvae_3d config."""
        return cfg.vqvae_3d.get('disable_gan', False)

    def _create_fallback_save_dir(self) -> str:
        """Create fallback save directory for 3D VQ-VAE."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = self.cfg.training.get('name', '')
        return os.path.join(
            self.cfg.paths.model_dir, "compression_3d", self.cfg.mode.name,
            f"vqvae_{exp_name}{self.volume_height}x{self.volume_depth}_{timestamp}"
        )

    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize 3D VQ-VAE model, discriminator, optimizers, and loss functions.

        Args:
            pretrained_checkpoint: Optional path to checkpoint for loading
                pretrained weights.
        """
        n_channels = self.cfg.mode.get('in_channels', 1)

        # Create 3D VQVAE
        base_model = VQVAE(
            spatial_dims=3,
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

        # Load pretrained weights BEFORE wrapping
        if pretrained_checkpoint:
            self._load_pretrained_weights_base(base_model, pretrained_checkpoint)

        # Wrap with gradient checkpointing for memory efficiency
        if self.gradient_checkpointing:
            raw_model = CheckpointedVQVAE(base_model)
            if self.is_main_process:
                logger.info("Gradient checkpointing enabled (CheckpointedVQVAE wrapper)")
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

        # Setup EMA
        self._setup_ema()

        # Save metadata
        if self.is_main_process:
            self._save_metadata()

        # Log model info
        if self.is_main_process:
            vqvae_params = sum(p.numel() for p in self.model_raw.parameters())
            logger.info(f"3D VQ-VAE initialized: {vqvae_params / 1e6:.1f}M parameters")
            logger.info(f"  Codebook: {self.num_embeddings} embeddings x {self.embedding_dim} dim")
            logger.info(f"  Channels: {self.channels}")
            if not self.disable_gan:
                disc_params = sum(p.numel() for p in self.discriminator_raw.parameters())
                logger.info(f"3D Discriminator: {disc_params / 1e6:.1f}M parameters")
            else:
                logger.info("GAN disabled")
            logger.info(f"Volume: {self.volume_width}x{self.volume_height}x{self.volume_depth}")

            # Compute latent shape
            n_downsamples = len(self.downsample_parameters)
            latent_h = self.volume_height // (2 ** n_downsamples)
            latent_w = self.volume_width // (2 ** n_downsamples)
            latent_d = self.volume_depth // (2 ** n_downsamples)
            logger.info(f"Latent shape: [{self.embedding_dim}, {latent_d}, {latent_h}, {latent_w}]")

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
                # Remove 'model.' prefix if present (from CheckpointedVQVAE)
                if any(k.startswith('model.') for k in state_dict.keys()):
                    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                base_model.load_state_dict(state_dict)
                if self.is_main_process:
                    logger.info(f"Loaded 3D VQ-VAE weights from {checkpoint_path}")
        except FileNotFoundError:
            if self.is_main_process:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")

    def _get_trainer_type(self) -> str:
        """Return trainer type for metadata."""
        return 'vqvae_3d'

    def _get_metadata_extra(self) -> Dict[str, Any]:
        """Return 3D VQ-VAE-specific metadata."""
        return {
            'vqvae_config': self._get_model_config(),
            'volume': {
                'height': self.volume_height,
                'width': self.volume_width,
                'depth': self.volume_depth,
            },
        }

    def train_step(self, batch: Any) -> TrainingStepResult:
        """Execute 3D VQ-VAE training step with VQ loss.

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
            # VQVAE forward returns (reconstruction, vq_loss)
            reconstruction, vq_loss = self.model(images)

            # L1 reconstruction loss
            l1_loss = torch.nn.functional.l1_loss(reconstruction, images)

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

        # ==================== Discriminator Step ====================
        if not self.disable_gan:
            d_loss = self._train_discriminator_step(images, reconstruction.detach())

        return TrainingStepResult(
            total_loss=g_loss.item(),
            reconstruction_loss=l1_loss.item(),
            perceptual_loss=p_loss.item() if isinstance(p_loss, torch.Tensor) else p_loss,
            regularization_loss=vq_loss.item(),
            adversarial_loss=adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
            discriminator_loss=d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss,
        )

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train 3D VQ-VAE for one epoch.

        Args:
            data_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dict with average losses.
        """
        self.model.train()
        if not self.disable_gan and self.discriminator is not None:
            self.discriminator.train()

        epoch_losses = {'gen': 0, 'disc': 0, 'recon': 0, 'perc': 0, 'vq': 0, 'adv': 0}

        disable_pbar = not self.is_main_process or self.is_cluster
        total = self.limit_train_batches if self.limit_train_batches else len(data_loader)
        iterator = itertools.islice(data_loader, self.limit_train_batches) if self.limit_train_batches else data_loader
        pbar = tqdm(iterator, desc=f"Epoch {epoch}", disable=disable_pbar, total=total)

        for step, batch in enumerate(pbar):
            result = self.train_step(batch)
            losses = result.to_legacy_dict('vq')

            for key in epoch_losses:
                epoch_losses[key] += losses[key]

            if not disable_pbar:
                pbar.set_postfix(
                    G=f"{losses['gen']:.4f}",
                    VQ=f"{losses['vq']:.4f}",
                    D=f"{losses['disc']:.4f}"
                )

        # Average losses
        n_batches = self.limit_train_batches if self.limit_train_batches else len(data_loader)
        avg_losses = {key: val / n_batches for key, val in epoch_losses.items()}

        # Log training metrics (single-GPU only)
        if self.writer is not None and self.is_main_process and not self.use_multi_gpu:
            self.writer.add_scalar('Loss/Generator_train', avg_losses['gen'], epoch)
            self.writer.add_scalar('Loss/L1_train', avg_losses['recon'], epoch)
            self.writer.add_scalar('Loss/Perceptual_train', avg_losses['perc'], epoch)
            self.writer.add_scalar('Loss/VQ_train', avg_losses['vq'], epoch)

            if not self.disable_gan:
                self.writer.add_scalar('Loss/Discriminator', avg_losses['disc'], epoch)
                self.writer.add_scalar('Loss/Adversarial', avg_losses['adv'], epoch)

        return avg_losses

    def _forward_for_validation(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """3D VQ-VAE forward pass for validation.

        Returns:
            Tuple of (reconstruction, vq_loss).
        """
        reconstruction, vq_loss = model(images)
        return reconstruction, vq_loss

    def _get_model_config(self) -> Dict[str, Any]:
        """Get 3D VQ-VAE model configuration for checkpoint."""
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
        """Log 3D VQ-VAE validation metrics including VQ loss."""
        if self.writer is None:
            return

        # Call base class method first
        super()._log_validation_metrics(epoch, metrics, worst_batch_data, regional_tracker, log_figures)

        # Add VQ-VAE-specific VQ loss logging
        if 'reg' in metrics:
            self.writer.add_scalar('Loss/VQ_val', metrics['reg'], epoch)

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
        elapsed_time: float,
    ) -> None:
        """Log 3D VQ-VAE epoch summary."""
        log_compression_epoch_summary(
            epoch, total_epochs, avg_losses, val_metrics, elapsed_time,
            regularization_key='vq',
        )

    def _test_forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Perform 3D VQ-VAE forward pass for test evaluation.

        Args:
            model: Model to use for inference.
            images: Input images.

        Returns:
            Reconstructed images tensor.
        """
        reconstructed, _ = model(images)
        return reconstructed
