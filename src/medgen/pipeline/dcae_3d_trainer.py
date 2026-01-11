"""
DC-AE 3D (Deep Compression Autoencoder 3D) trainer module.

This module provides the DCAE3DTrainer class which inherits from BaseCompression3DTrainer
and implements DC-AE-specific functionality for 3D volumes:
- Deterministic encoder (no KL/VQ regularization)
- Custom AutoencoderDC3D model with asymmetric compression
- Gradient checkpointing for memory efficiency
- High compression ratios (32x spatial, 4x depth)
"""
import itertools
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.autoencoder_dc_3d import AutoencoderDC3D, CheckpointedAutoencoderDC3D
from .compression_trainer import BaseCompression3DTrainer
from .results import TrainingStepResult
from .utils import log_compression_epoch_summary

logger = logging.getLogger(__name__)


class DCAE3DTrainer(BaseCompression3DTrainer):
    """3D DC-AE trainer for high-compression volumetric encoding.

    Inherits from BaseCompression3DTrainer and adds:
    - Deterministic encoder (no regularization loss)
    - Custom AutoencoderDC3D model
    - Asymmetric compression support (different spatial vs depth)
    - L1 reconstruction loss (DC-AE style)

    Args:
        cfg: Hydra configuration object.

    Example:
        >>> trainer = DCAE3DTrainer(cfg)
        >>> trainer.setup_model()
        >>> trainer.train(train_loader, train_dataset, val_loader)
    """

    # Add dcae_3d to config sections
    _CONFIG_SECTIONS = ('vae', 'vqvae', 'dcae', 'vae_3d', 'vqvae_3d', 'dcae_3d')

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize 3D DC-AE trainer.

        Args:
            cfg: Hydra configuration object.
        """
        super().__init__(cfg)

        # ─────────────────────────────────────────────────────────────────────
        # DC-AE specific config
        # ─────────────────────────────────────────────────────────────────────
        dcae_cfg = cfg.dcae_3d
        self.l1_weight: float = dcae_cfg.get('l1_weight', 1.0)
        self.latent_channels: int = dcae_cfg.latent_channels
        self.scaling_factor: float = dcae_cfg.get('scaling_factor', 1.0)

        # Architecture
        self.encoder_block_out_channels = tuple(dcae_cfg.encoder_block_out_channels)
        self.decoder_block_out_channels = tuple(dcae_cfg.decoder_block_out_channels)
        self.encoder_layers_per_block = tuple(dcae_cfg.encoder_layers_per_block)
        self.decoder_layers_per_block = tuple(dcae_cfg.decoder_layers_per_block)
        self.depth_factors = tuple(dcae_cfg.depth_factors)
        self.encoder_out_shortcut = dcae_cfg.get('encoder_out_shortcut', True)
        self.decoder_in_shortcut = dcae_cfg.get('decoder_in_shortcut', True)

        # Initialize unified metrics system
        self.spatial_dims = 3
        self._init_unified_metrics('dcae')

    def _get_2_5d_perceptual(self, cfg: DictConfig) -> bool:
        """Get 2.5D perceptual loss flag from dcae_3d config."""
        if 'dcae_3d' in cfg:
            return cfg.dcae_3d.get('use_2_5d_perceptual', True)
        return super()._get_2_5d_perceptual(cfg)

    def _get_perceptual_slice_fraction(self, cfg: DictConfig) -> float:
        """Get perceptual slice fraction from dcae_3d config."""
        if 'dcae_3d' in cfg:
            return cfg.dcae_3d.get('perceptual_slice_fraction', 0.25)
        return super()._get_perceptual_slice_fraction(cfg)

    def _get_disc_lr(self, cfg: DictConfig) -> float:
        """Get discriminator LR from dcae_3d config."""
        if 'dcae_3d' in cfg:
            return cfg.dcae_3d.get('disc_lr', 1e-4)
        return super()._get_disc_lr(cfg)

    def _get_perceptual_weight(self, cfg: DictConfig) -> float:
        """Get perceptual weight from dcae_3d config."""
        if 'dcae_3d' in cfg:
            return cfg.dcae_3d.get('perceptual_weight', 0.1)
        return super()._get_perceptual_weight(cfg)

    def _get_adv_weight(self, cfg: DictConfig) -> float:
        """Get adversarial weight from dcae_3d config."""
        if 'dcae_3d' in cfg:
            return cfg.dcae_3d.get('adv_weight', 0.0)
        return super()._get_adv_weight(cfg)

    def _get_disable_gan(self, cfg: DictConfig) -> bool:
        """Determine if GAN is disabled from dcae_3d config."""
        if 'dcae_3d' in cfg:
            return cfg.dcae_3d.get('disable_gan', True)
        return super()._get_disable_gan(cfg)

    def _create_fallback_save_dir(self) -> str:
        """Create fallback save directory for 3D DC-AE."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = self.cfg.training.get('name', '')
        return os.path.join(
            self.cfg.paths.model_dir, "compression_3d", self.cfg.mode.name,
            f"dcae_{exp_name}{self.volume_height}x{self.volume_depth}_{timestamp}"
        )

    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize 3D DC-AE model, discriminator, optimizers.

        Args:
            pretrained_checkpoint: Optional path to checkpoint for loading
                pretrained weights.
        """
        n_channels = self.cfg.mode.get('in_channels', 4)

        # Create AutoencoderDC3D
        base_model = AutoencoderDC3D(
            in_channels=n_channels,
            latent_channels=self.latent_channels,
            encoder_block_out_channels=self.encoder_block_out_channels,
            decoder_block_out_channels=self.decoder_block_out_channels,
            encoder_layers_per_block=self.encoder_layers_per_block,
            decoder_layers_per_block=self.decoder_layers_per_block,
            depth_factors=self.depth_factors,
            encoder_out_shortcut=self.encoder_out_shortcut,
            decoder_in_shortcut=self.decoder_in_shortcut,
            scaling_factor=self.scaling_factor,
        ).to(self.device)

        # Load pretrained weights BEFORE wrapping
        if pretrained_checkpoint:
            self._load_pretrained_weights_base(base_model, pretrained_checkpoint)

        # Wrap with gradient checkpointing for memory efficiency
        if self.gradient_checkpointing:
            raw_model = CheckpointedAutoencoderDC3D(base_model)
            if self.is_main_process:
                logger.info("Gradient checkpointing enabled (CheckpointedAutoencoderDC3D)")
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
            params = sum(p.numel() for p in self.model_raw.parameters())
            logger.info(f"3D DC-AE initialized: {params / 1e6:.1f}M parameters")
            logger.info(f"  Spatial compression: {base_model.spatial_compression}×")
            logger.info(f"  Depth compression: {base_model.depth_compression}×")
            logger.info(f"  Latent channels: {self.latent_channels}")
            if not self.disable_gan:
                disc_params = sum(p.numel() for p in self.discriminator_raw.parameters())
                logger.info(f"3D Discriminator: {disc_params / 1e6:.1f}M parameters")
            else:
                logger.info("GAN disabled")
            logger.info(f"Volume: {self.volume_width}x{self.volume_height}x{self.volume_depth}")

            # Compute latent shape
            latent_h = self.volume_height // base_model.spatial_compression
            latent_w = self.volume_width // base_model.spatial_compression
            latent_d = self.volume_depth // base_model.depth_compression
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
                # Remove 'model.' prefix if present (from CheckpointedAutoencoderDC3D)
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
                    logger.info(f"Loaded 3D DC-AE weights from {checkpoint_path}")
        except FileNotFoundError:
            if self.is_main_process:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")

    def _get_trainer_type(self) -> str:
        """Return trainer type for metadata."""
        return 'dcae_3d'

    def _get_metadata_extra(self) -> Dict[str, Any]:
        """Return 3D DC-AE-specific metadata."""
        return {
            'dcae_config': self._get_model_config(),
            'volume': {
                'height': self.volume_height,
                'width': self.volume_width,
                'depth': self.volume_depth,
            },
        }

    def _get_model_config(self) -> Dict[str, Any]:
        """Get 3D DC-AE model configuration for checkpoint."""
        n_channels = self.cfg.mode.get('in_channels', 4)
        return {
            'in_channels': n_channels,
            'latent_channels': self.latent_channels,
            'encoder_block_out_channels': list(self.encoder_block_out_channels),
            'decoder_block_out_channels': list(self.decoder_block_out_channels),
            'encoder_layers_per_block': list(self.encoder_layers_per_block),
            'decoder_layers_per_block': list(self.decoder_layers_per_block),
            'depth_factors': list(self.depth_factors),
            'encoder_out_shortcut': self.encoder_out_shortcut,
            'decoder_in_shortcut': self.decoder_in_shortcut,
            'scaling_factor': self.scaling_factor,
        }

    def train_step(self, batch: Any) -> TrainingStepResult:
        """Execute 3D DC-AE training step (deterministic, no regularization).

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
            # DC-AE forward (deterministic)
            reconstruction = self.model(images)

            # L1 reconstruction loss
            l1_loss = torch.nn.functional.l1_loss(reconstruction.float(), images.float())

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
                self.l1_weight * l1_loss
                + self.perceptual_weight * p_loss
                + self.adv_weight * adv_loss
            )

        g_loss.backward()

        # Gradient clipping
        grad_norm = 0.0
        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model_raw.parameters(), max_norm=grad_clip
            ).item()

        self.optimizer.step()

        # Track gradient norm
        if self.log_grad_norm:
            self._grad_norm_tracker.update(grad_norm)

        # Update EMA
        self._update_ema()

        # ==================== Discriminator Step ====================
        if not self.disable_gan:
            d_loss = self._train_discriminator_step(images, reconstruction.detach())

        return TrainingStepResult(
            total_loss=g_loss.item(),
            reconstruction_loss=l1_loss.item(),
            perceptual_loss=p_loss.item() if isinstance(p_loss, torch.Tensor) else p_loss,
            regularization_loss=0.0,  # DC-AE is deterministic, no regularization
            adversarial_loss=adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
            discriminator_loss=d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss,
        )

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train 3D DC-AE for one epoch.

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

        disable_pbar = not self.is_main_process or self.is_cluster
        total = self.limit_train_batches if self.limit_train_batches else len(data_loader)
        iterator = itertools.islice(data_loader, self.limit_train_batches) if self.limit_train_batches else data_loader
        pbar = tqdm(iterator, desc=f"Epoch {epoch}", disable=disable_pbar, total=total)

        for step, batch in enumerate(pbar):
            result = self.train_step(batch)
            # DC-AE uses None for reg key (deterministic, no regularization)
            losses = result.to_legacy_dict(None)

            # Step profiler to mark training step boundary
            self._profiler_step()

            # Accumulate with unified system
            self._loss_accumulator.update(losses)

            if not disable_pbar:
                avg_so_far = self._loss_accumulator.compute()
                if not self.disable_gan:
                    pbar.set_postfix(
                        G=f"{avg_so_far.get('gen', 0):.4f}",
                        D=f"{avg_so_far.get('disc', 0):.4f}",
                        L1=f"{avg_so_far.get('recon', 0):.4f}",
                    )
                else:
                    pbar.set_postfix(
                        G=f"{avg_so_far.get('gen', 0):.4f}",
                        L1=f"{avg_so_far.get('recon', 0):.4f}",
                        P=f"{avg_so_far.get('perc', 0):.4f}",
                    )

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
        """3D DC-AE forward pass for validation.

        Returns:
            Tuple of (reconstruction, regularization_loss).
            regularization_loss is always 0 for DC-AE (deterministic).
        """
        reconstruction = model(images)
        # DC-AE is deterministic - no regularization loss
        return reconstruction, torch.tensor(0.0, device=self.device)

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
        elapsed_time: float,
    ) -> None:
        """Log 3D DC-AE epoch summary using unified system."""
        self._log_epoch_summary_unified(epoch, avg_losses, val_metrics, elapsed_time)

    def _test_forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Perform 3D DC-AE forward pass for test evaluation.

        Args:
            model: Model to use for inference.
            images: Input images.

        Returns:
            Reconstructed images tensor.
        """
        return model(images)
