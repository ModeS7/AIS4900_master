"""
VQ-VAE trainer module for training vector-quantized autoencoders (2D and 3D).

This module provides the VQVAETrainer class which inherits from BaseCompressionTrainer
and implements VQ-VAE-specific functionality:
- Vector quantization with discrete latent codes
- VQ loss (commitment + codebook loss)
- VQVAE model creation (2D or 3D)
- Gradient checkpointing for memory efficiency (3D)
- Segmentation mode (seg_mode) for mask compression (3D only)
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

from monai.networks.nets import VQVAE

from .checkpointing import BaseCheckpointedModel
from .compression_trainer import BaseCompressionTrainer
from .results import TrainingStepResult
from .utils import create_epoch_iterator, get_vram_usage
from medgen.metrics import CodebookTracker

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


class VQVAETrainer(BaseCompressionTrainer):
    """VQ-VAE trainer with discrete latent space (2D and 3D).

    Supports both 2D images and 3D volumes via the spatial_dims parameter.

    Inherits from BaseCompressionTrainer and adds:
    - Vector quantization (VQ) loss
    - VQVAE model creation from MONAI
    - Gradient checkpointing for 3D memory efficiency
    - Segmentation mode (seg_mode) for mask compression (3D only)

    Args:
        cfg: Hydra configuration object.
        spatial_dims: Spatial dimensions (2 or 3).

    Example:
        >>> trainer = VQVAETrainer(cfg, spatial_dims=2)
        >>> trainer.setup_model()
        >>> trainer.train(train_loader, train_dataset, val_loader)
    """

    def __init__(self, cfg: DictConfig, spatial_dims: int = 2) -> None:
        """Initialize VQ-VAE trainer.

        Args:
            cfg: Hydra configuration object.
            spatial_dims: Spatial dimensions (2 or 3).
        """
        super().__init__(cfg, spatial_dims=spatial_dims)

        # ─────────────────────────────────────────────────────────────────────
        # VQ-VAE-specific config (dimension-dependent)
        # ─────────────────────────────────────────────────────────────────────
        vqvae_cfg = cfg.vqvae_3d if spatial_dims == 3 else cfg.vqvae
        self.num_embeddings: int = vqvae_cfg.get('num_embeddings', 512)
        self.embedding_dim: int = vqvae_cfg.get('embedding_dim', 64 if spatial_dims == 2 else 3)
        self.commitment_cost: float = vqvae_cfg.get('commitment_cost', 0.25)
        self.decay: float = vqvae_cfg.get('decay', 0.99)
        self.epsilon: float = vqvae_cfg.get('epsilon', 1e-5)

        # Architecture config
        default_channels = [96, 96, 192] if spatial_dims == 2 else [64, 128]
        self.channels: Tuple[int, ...] = tuple(vqvae_cfg.get('channels', default_channels))
        default_res_layers = 3 if spatial_dims == 2 else 2
        self.num_res_layers: int = vqvae_cfg.get('num_res_layers', default_res_layers)
        default_res_channels = [96, 96, 192] if spatial_dims == 2 else [64, 128]
        self.num_res_channels: Tuple[int, ...] = tuple(
            vqvae_cfg.get('num_res_channels', default_res_channels)
        )
        default_downsample = [[2, 4, 1, 1]] * (3 if spatial_dims == 2 else 2)
        self.downsample_parameters: Tuple[Tuple[int, ...], ...] = tuple(
            tuple(p) for p in vqvae_cfg.get('downsample_parameters', default_downsample)
        )
        default_upsample = [[2, 4, 1, 1, 0]] * (3 if spatial_dims == 2 else 2)
        self.upsample_parameters: Tuple[Tuple[int, ...], ...] = tuple(
            tuple(p) for p in vqvae_cfg.get('upsample_parameters', default_upsample)
        )

        # Codebook tracking (initialized after model setup)
        self._codebook_tracker: Optional[CodebookTracker] = None

        # ─────────────────────────────────────────────────────────────────────
        # Segmentation mode (seg_mode) - 3D only
        # ─────────────────────────────────────────────────────────────────────
        self.seg_mode: bool = vqvae_cfg.get('seg_mode', False) if spatial_dims == 3 else False
        self.seg_loss_fn = None

        if self.seg_mode:
            from medgen.losses import SegmentationLoss
            seg_weights = vqvae_cfg.get('seg_loss_weights', {})
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
        self._init_unified_metrics('vqvae')

    @classmethod
    def create_2d(cls, cfg: DictConfig, **kwargs) -> 'VQVAETrainer':
        """Create 2D VQVAETrainer.

        Args:
            cfg: Hydra configuration object.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            VQVAETrainer configured for 2D images.
        """
        return cls(cfg, spatial_dims=2, **kwargs)

    @classmethod
    def create_3d(cls, cfg: DictConfig, **kwargs) -> 'VQVAETrainer':
        """Create 3D VQVAETrainer.

        Args:
            cfg: Hydra configuration object.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            VQVAETrainer configured for 3D volumes.
        """
        return cls(cfg, spatial_dims=3, **kwargs)

    def _get_disc_lr(self, cfg: DictConfig) -> float:
        """Get discriminator LR from vqvae/vqvae_3d config."""
        section = 'vqvae_3d' if self.spatial_dims == 3 else 'vqvae'
        if section in cfg:
            return cfg[section].get('disc_lr', 5e-4)
        return 5e-4

    def _get_perceptual_weight(self, cfg: DictConfig) -> float:
        """Get perceptual weight from vqvae/vqvae_3d config."""
        section = 'vqvae_3d' if self.spatial_dims == 3 else 'vqvae'
        if section in cfg:
            return cfg[section].get('perceptual_weight', 0.002 if self.spatial_dims == 3 else 0.001)
        return 0.001

    def _get_adv_weight(self, cfg: DictConfig) -> float:
        """Get adversarial weight from vqvae/vqvae_3d config."""
        section = 'vqvae_3d' if self.spatial_dims == 3 else 'vqvae'
        if section in cfg:
            return cfg[section].get('adv_weight', 0.005 if self.spatial_dims == 3 else 0.01)
        return 0.01

    def _get_disable_gan(self, cfg: DictConfig) -> bool:
        """Determine if GAN is disabled from vqvae/vqvae_3d config."""
        section = 'vqvae_3d' if self.spatial_dims == 3 else 'vqvae'
        if section in cfg:
            return cfg[section].get('disable_gan', False)
        return False

    def _create_fallback_save_dir(self) -> str:
        """Create fallback save directory for VQ-VAE."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = self.cfg.training.get('name', '')
        mode_name = self.cfg.mode.get('name', 'dual')

        if self.spatial_dims == 3:
            run_name = f"vqvae_{exp_name}{self.volume_height}x{self.volume_depth}_{timestamp}"
            return os.path.join(self.cfg.paths.model_dir, 'compression_3d', mode_name, run_name)
        else:
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
        base_model = VQVAE(
            spatial_dims=self.spatial_dims,
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

        # Load pretrained weights BEFORE wrapping (for 3D with checkpointing)
        if pretrained_checkpoint and self.spatial_dims == 3:
            self._load_pretrained_weights_base(base_model, pretrained_checkpoint, "3D VQ-VAE")

        # 3D: Wrap with gradient checkpointing for memory efficiency
        if self.spatial_dims == 3 and self.gradient_checkpointing:
            raw_model = CheckpointedVQVAE(base_model)
            if self.is_main_process:
                logger.info("Gradient checkpointing enabled (CheckpointedVQVAE wrapper)")
        else:
            raw_model = base_model

        # Create discriminator if GAN enabled
        raw_disc = None
        if not self.disable_gan:
            raw_disc = self._create_discriminator(n_channels, spatial_dims=self.spatial_dims)

        # 2D: Load pretrained weights after wrapping (no prefix to strip)
        if pretrained_checkpoint and self.spatial_dims == 2:
            self._load_pretrained_weights(raw_model, raw_disc, pretrained_checkpoint, model_name="VQ-VAE")

        # Wrap models with DDP/compile
        self._wrap_models(raw_model, raw_disc)

        # Setup perceptual and adversarial loss
        # 3D: use 2D perceptual loss for 2.5D computation
        if self.perceptual_weight > 0:
            perceptual_spatial_dims = 2 if (self.spatial_dims == 3 and getattr(self, 'use_2_5d_perceptual', False)) else self.spatial_dims
            self.perceptual_loss_fn = self._create_perceptual_loss(spatial_dims=perceptual_spatial_dims)
        self._create_adversarial_loss()

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
            dim_str = "3D " if self.spatial_dims == 3 else ""
            logger.info(f"{dim_str}VQ-VAE initialized: {vqvae_params / 1e6:.1f}M parameters")
            logger.info(f"  Codebook: {self.num_embeddings} embeddings x {self.embedding_dim} dim")
            logger.info(f"  Channels: {self.channels}")
            if not self.disable_gan:
                disc_params = sum(p.numel() for p in self.discriminator_raw.parameters())
                logger.info(f"{dim_str}Discriminator: {disc_params / 1e6:.1f}M parameters")
            else:
                logger.info("GAN disabled")

            # Log spatial info
            n_downsamples = len(self.downsample_parameters)
            if self.spatial_dims == 3:
                logger.info(f"Volume: {self.volume_width}x{self.volume_height}x{self.volume_depth}")
                latent_h = self.volume_height // (2 ** n_downsamples)
                latent_w = self.volume_width // (2 ** n_downsamples)
                latent_d = self.volume_depth // (2 ** n_downsamples)
                logger.info(f"Latent shape: [{self.embedding_dim}, {latent_d}, {latent_h}, {latent_w}]")
            else:
                latent_size = self.image_size // (2 ** n_downsamples)
                logger.info(f"Latent shape: [{self.embedding_dim}, {latent_size}, {latent_size}]")
                logger.info(f"Loss weights - Perceptual: {self.perceptual_weight}, Adv: {self.adv_weight}")
                logger.info(f"VQ params - Commitment: {self.commitment_cost}, Decay: {self.decay}")

    def _get_trainer_type(self) -> str:
        """Return trainer type for metadata."""
        return 'vqvae_3d' if self.spatial_dims == 3 else 'vqvae'

    def _get_metadata_extra(self) -> Dict[str, Any]:
        """Return VQ-VAE-specific metadata."""
        meta = {'vqvae_config': self._get_model_config()}
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
        """Execute VQ-VAE training step with VQ loss.

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
                    reconstruction_for_d, _ = self.model(images)
            d_loss = self._train_discriminator_step(images, reconstruction_for_d)

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

                # Perceptual loss (standard 2D or 2.5D for 3D)
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

        # ==================== Discriminator Step (3D: after generator) ====================
        if self.spatial_dims == 3 and not self.disable_gan:
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

        # Initialize seg breakdown tracking for this epoch
        if self.seg_mode:
            self._epoch_seg_breakdown = {'bce': 0.0, 'dice': 0.0, 'boundary': 0.0}

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

        # Log seg breakdown if in seg_mode using unified system
        if self.seg_mode and self.is_main_process:
            n_batches = self.limit_train_batches if self.limit_train_batches else len(data_loader)
            seg_breakdown = {
                'bce': self._epoch_seg_breakdown['bce'] / n_batches,
                'dice': self._epoch_seg_breakdown['dice'] / n_batches,
                'boundary': self._epoch_seg_breakdown['boundary'] / n_batches,
            }
            for key, value in seg_breakdown.items():
                self._unified_metrics.update_loss(key, value, phase='train')

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

    def _create_validation_runner(self):
        """Create ValidationRunner for VQ-VAE with seg_mode support.

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
            spatial_dims=self.spatial_dims,
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

    def _create_seg_regional_tracker(self):
        """Create SegRegionalMetricsTracker for per-tumor Dice tracking.

        Returns:
            Configured SegRegionalMetricsTracker instance.
        """
        from medgen.metrics import SegRegionalMetricsTracker

        size = self.volume_height if self.spatial_dims == 3 else self.image_size
        return SegRegionalMetricsTracker(
            image_size=size,
            fov_mm=self.cfg.paths.get('fov_mm', 240.0),
            device=self.device,
        )

    def _get_model_config(self) -> Dict[str, Any]:
        """Get VQ-VAE model configuration for checkpoint."""
        n_channels = self.cfg.mode.get('in_channels', 1)
        config = {
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
        if self.spatial_dims == 3:
            config['spatial_dims'] = 3
        return config

    def _log_validation_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float],
        worst_batch_data: Optional[Dict[str, Any]],
        regional_tracker,
        log_figures: bool,
    ) -> None:
        """Log VQ-VAE validation metrics including VQ loss."""
        if self.writer is None:
            return

        # Log metrics with modality suffix handling
        self._log_validation_metrics_core(epoch, metrics)

        # VQ-VAE-specific: log VQ loss using unified system
        if 'reg' in metrics:
            self._unified_metrics.log_regularization_loss(
                loss_type='VQ',
                weighted_loss=metrics['reg'],
                epoch=epoch,
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

        # Get the raw model for index_quantize (unwrap DDP/compiled/checkpointed)
        raw_model = self.model_raw
        if hasattr(raw_model, 'model'):
            # CheckpointedVQVAE wrapper
            raw_model = raw_model.model

        # Adjust max_batches for 3D (smaller batches = less data)
        if self.spatial_dims == 3:
            max_batches = min(max_batches, 10)

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

        # Track codebook usage and log using unified system
        if self.is_main_process and self._codebook_tracker is not None:
            self._track_codebook_usage(self.val_loader)

            # Log codebook metrics using unified system
            cb_metrics = self._unified_metrics.log_codebook_metrics(
                self._codebook_tracker, epoch, prefix='Codebook'
            )

            if cb_metrics:
                # Add to returned metrics for logging
                metrics['codebook_perplexity'] = cb_metrics.get('perplexity', 0)
                metrics['codebook_utilization'] = cb_metrics.get('utilization', 0)

            # Log summary to console
            self._codebook_tracker.log_summary()

        return metrics
