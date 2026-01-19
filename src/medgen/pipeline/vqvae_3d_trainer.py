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
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Disable MONAI MetaTensor tracking BEFORE importing MONAI modules
from monai.data import set_track_meta
set_track_meta(False)

from monai.networks.nets import VQVAE

from .checkpointing import BaseCheckpointedModel
from .compression_trainer import BaseCompression3DTrainer
from medgen.metrics import (
    compute_lpips_3d,
    compute_msssim,
    compute_psnr,
    RegionalMetricsTracker3D,
)
from medgen.metrics import create_worst_batch_figure_3d, CodebookTracker
from .results import TrainingStepResult
from .utils import get_vram_usage, log_compression_epoch_summary

logger = logging.getLogger(__name__)


class CheckpointedVQVAE(BaseCheckpointedModel):
    """Wrapper that applies gradient checkpointing to MONAI VQVAE.

    Reduces activation memory by ~50% for 3D volumes.

    Args:
        model: The underlying VQVAE model.
    """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with gradient checkpointing."""
        def encode_fn(x):
            return self.model.encode(x)

        encoded = self.checkpoint(encode_fn, x)

        def quantize_fn(z):
            return self.model.quantize(z)

        quantized, vq_loss = self.checkpoint(quantize_fn, encoded)

        def decode_fn(z):
            return self.model.decode(z)

        reconstruction = self.checkpoint(decode_fn, quantized)
        return reconstruction, vq_loss

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

        # Codebook tracking (initialized after model setup)
        self._codebook_tracker: Optional[CodebookTracker] = None

        # ─────────────────────────────────────────────────────────────────────
        # Segmentation mode (seg_mode)
        # ─────────────────────────────────────────────────────────────────────
        self.seg_mode: bool = cfg.vqvae_3d.get('seg_mode', False)
        self.seg_loss_fn: Optional['SegmentationLoss'] = None

        if self.seg_mode:
            from medgen.losses import SegmentationLoss
            seg_weights = cfg.vqvae_3d.get('seg_loss_weights', {})
            self.seg_loss_fn = SegmentationLoss(
                bce_weight=seg_weights.get('bce', 1.0),
                dice_weight=seg_weights.get('dice', 1.0),
                boundary_weight=seg_weights.get('boundary', 0.5),
                spatial_dims=3,  # 3D volumes
            )
            # Disable perceptual loss for binary masks
            self.perceptual_weight = 0.0
            # Disable GAN for binary masks
            self.disable_gan = True
            if self.is_main_process:
                logger.info("Seg mode enabled: BCE + Dice + Boundary loss")

        # Initialize unified metrics system
        self.spatial_dims = 3
        self._init_unified_metrics('vqvae')

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

        # Initialize codebook tracker
        self._codebook_tracker = CodebookTracker(self.num_embeddings, self.device)

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

            # Compute reconstruction loss based on mode
            if self.seg_mode and self.seg_loss_fn is not None:
                # Segmentation loss: BCE + Dice + Boundary
                seg_loss, seg_breakdown = self.seg_loss_fn(reconstruction, images)
                l1_loss = seg_loss  # Reuse variable for total seg loss
                p_loss = torch.tensor(0.0, device=self.device)
                # Track breakdown for epoch averaging
                if hasattr(self, '_epoch_seg_breakdown'):
                    for key in seg_breakdown:
                        self._epoch_seg_breakdown[key] += seg_breakdown[key]
            else:
                # Standard L1 reconstruction loss
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

        # Use unified loss accumulator
        self._loss_accumulator.reset()

        # Initialize seg breakdown tracking for this epoch
        if self.seg_mode:
            self._epoch_seg_breakdown = {'bce': 0.0, 'dice': 0.0, 'boundary': 0.0}

        disable_pbar = not self.is_main_process or self.is_cluster
        total = self.limit_train_batches if self.limit_train_batches else len(data_loader)
        iterator = itertools.islice(data_loader, self.limit_train_batches) if self.limit_train_batches else data_loader
        pbar = tqdm(iterator, desc=f"Epoch {epoch}", disable=disable_pbar, total=total)

        for step, batch in enumerate(pbar):
            result = self.train_step(batch)
            losses = result.to_legacy_dict('vq')

            # Step profiler to mark training step boundary
            self._profiler_step()

            # Accumulate with unified system
            self._loss_accumulator.update(losses)

            if not disable_pbar:
                avg_so_far = self._loss_accumulator.compute()
                pbar.set_postfix(
                    G=f"{avg_so_far.get('gen', 0):.4f}",
                    VQ=f"{avg_so_far.get('vq', 0):.4f}",
                    D=f"{avg_so_far.get('disc', 0):.4f}"
                )

        # Compute average losses using unified system
        avg_losses = self._loss_accumulator.compute()

        # Log training metrics using unified system
        self._log_training_metrics_unified(epoch, avg_losses)

        # Log seg breakdown if in seg_mode (supplement unified logging with breakdown)
        if self.seg_mode and self.writer is not None and self.is_main_process:
            n_batches = self.limit_train_batches if self.limit_train_batches else len(data_loader)
            seg_breakdown = {
                'bce': self._epoch_seg_breakdown['bce'] / n_batches,
                'dice': self._epoch_seg_breakdown['dice'] / n_batches,
                'boundary': self._epoch_seg_breakdown['boundary'] / n_batches,
            }
            # Log seg breakdown components
            for key, value in seg_breakdown.items():
                self.writer.add_scalar(f'Loss/{key}_train', value, epoch)

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

    def _create_validation_runner(self) -> 'ValidationRunner':
        """Create ValidationRunner for 3D VQ-VAE with seg_mode support.

        Overrides parent to add seg_mode configuration for segmentation
        mask compression training.

        Returns:
            Configured ValidationRunner instance.
        """
        from medgen.evaluation import ValidationRunner, ValidationConfig

        config = ValidationConfig(
            log_msssim=self.log_msssim and not self.seg_mode,
            log_psnr=self.log_psnr and not self.seg_mode,
            log_lpips=self.log_lpips and not self.seg_mode,
            log_regional_losses=self.log_regional_losses,
            weight_dtype=self.weight_dtype,
            use_compile=self.use_compile,
            spatial_dims=3,
            seg_mode=self.seg_mode,
        )

        # Regional tracker factory based on mode
        regional_factory = None
        seg_regional_factory = None
        if self.log_regional_losses:
            if self.seg_mode:
                seg_regional_factory = self._create_seg_regional_tracker
            else:
                regional_factory = self._create_regional_tracker

        return ValidationRunner(
            config=config,
            device=self.device,
            forward_fn=self._forward_for_validation,
            perceptual_loss_fn=self._compute_perceptual_loss if not self.seg_mode else None,
            seg_loss_fn=self.seg_loss_fn if self.seg_mode else None,
            regional_tracker_factory=regional_factory,
            seg_regional_tracker_factory=seg_regional_factory,
            prepare_batch_fn=self._prepare_batch,
        )

    def _create_seg_regional_tracker(self) -> 'SegRegionalMetricsTracker':
        """Create SegRegionalMetricsTracker for per-tumor Dice tracking.

        Returns:
            Configured SegRegionalMetricsTracker instance.
        """
        from medgen.metrics import SegRegionalMetricsTracker

        return SegRegionalMetricsTracker(
            image_size=self.volume_height,  # Use volume dimensions
            fov_mm=self.cfg.paths.get('fov_mm', 240.0),
            device=self.device,
        )

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
        """Log 3D VQ-VAE validation metrics including VQ loss.

        For seg_mode, logs Dice/IoU metrics instead of PSNR/LPIPS/MS-SSIM.
        """
        if self.writer is None:
            return

        # Log metrics with modality suffix handling
        self._log_validation_metrics_core(epoch, metrics)

        # VQ-VAE-specific: log VQ loss
        if 'reg' in metrics and self.writer is not None:
            self.writer.add_scalar('Loss/VQ_val', metrics['reg'], epoch)

        # Log worst batch figure if available (3D-specific)
        # Log worst batch figure (uses unified metrics - handles 3D automatically)
        if log_figures and worst_batch_data is not None and 'original' in worst_batch_data:
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
            if not is_multi_modality and not is_dual:
                regional_tracker.log_to_tensorboard(self.writer, epoch, prefix=f'regional_{mode_name}')
            else:
                regional_tracker.log_to_tensorboard(self.writer, epoch, prefix='regional')

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
        elapsed_time: float,
    ) -> None:
        """Log 3D VQ-VAE epoch summary using unified system."""
        self._log_epoch_summary_unified(epoch, avg_losses, val_metrics, elapsed_time)

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

    def _track_codebook_usage(
        self,
        data_loader: DataLoader,
        max_batches: int = 10,
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
        if hasattr(raw_model, 'model'):
            # CheckpointedVQVAE wrapper
            raw_model = raw_model.model

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
            self._track_codebook_usage(self.val_loader, max_batches=10)

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
