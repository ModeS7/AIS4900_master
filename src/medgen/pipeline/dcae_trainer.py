"""
DC-AE (Deep Compression Autoencoder) trainer module.

This module provides the DCAETrainer class which inherits from BaseCompressionTrainer
and implements DC-AE-specific functionality:
- Deterministic encoder (no KL divergence)
- diffusers AutoencoderDC with encode/decode API
- High compression (32× or 64× spatial)
- Pretrained ImageNet weights support
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
import torch.nn as nn
from diffusers import AutoencoderDC
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .compression_trainer import BaseCompressionTrainer
from .results import TrainingStepResult
from .metrics import (
    compute_lpips,
    compute_msssim,
    compute_psnr,
    create_reconstruction_figure,
    RegionalMetricsTracker,
)
from .tracking import create_worst_batch_figure, FLOPsTracker
from .utils import create_epoch_iterator, get_vram_usage, log_compression_epoch_summary

logger = logging.getLogger(__name__)


class DCAETrainer(BaseCompressionTrainer):
    """DC-AE trainer for high-compression 2D MRI encoding.

    Inherits from BaseCompressionTrainer and adds:
    - Deterministic encoder (no regularization loss)
    - diffusers AutoencoderDC model
    - Support for pretrained ImageNet weights

    Args:
        cfg: Hydra configuration object.

    Example:
        >>> trainer = DCAETrainer(cfg)
        >>> trainer.setup_model()
        >>> trainer.train(train_loader, train_dataset, val_loader)
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize DC-AE trainer.

        Args:
            cfg: Hydra configuration object.
        """
        super().__init__(cfg)

        # ─────────────────────────────────────────────────────────────────────
        # DC-AE-specific config
        # ─────────────────────────────────────────────────────────────────────
        self.l1_weight: float = cfg.dcae.get('l1_weight', 1.0)
        self.latent_channels: int = cfg.dcae.latent_channels
        self.compression_ratio: int = cfg.dcae.compression_ratio
        self.scaling_factor: float = cfg.dcae.get('scaling_factor', 1.0)

        # Pretrained model path (from HuggingFace or null)
        self.pretrained: Optional[str] = cfg.dcae.get('pretrained', None)

        # Training phase (1=no GAN, 3=with GAN)
        self.training_phase: int = cfg.training.get('phase', 1)

        # Override disable_gan based on training phase
        if self.training_phase == 1 or self.adv_weight == 0.0:
            self.disable_gan = True

        self.image_size: int = cfg.dcae.get('image_size', 256)

    def _get_disc_lr(self, cfg: DictConfig) -> float:
        """Get discriminator LR from dcae config."""
        return cfg.dcae.get('disc_lr', 5e-4)

    def _get_perceptual_weight(self, cfg: DictConfig) -> float:
        """Get perceptual weight from dcae config."""
        return cfg.dcae.get('perceptual_weight', 0.1)

    def _get_adv_weight(self, cfg: DictConfig) -> float:
        """Get adversarial weight from dcae config."""
        return cfg.dcae.get('adv_weight', 0.0)

    def _get_disable_gan(self, cfg: DictConfig) -> bool:
        """Determine if GAN is disabled."""
        training_phase = cfg.training.get('phase', 1)
        adv_weight = cfg.dcae.get('adv_weight', 0.0)
        return (adv_weight == 0.0) or (training_phase == 1)

    def _create_fallback_save_dir(self) -> str:
        """Create fallback save directory for DC-AE."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = self.cfg.training.get('name', '')
        mode_name = self.cfg.mode.get('name', 'multi_modality')
        return os.path.join(
            self.cfg.paths.model_dir, 'compression_2d', mode_name,
            f"{exp_name}{self.image_size}_{timestamp}"
        )

    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize DC-AE model, discriminator, optimizers, and loss functions.

        Args:
            pretrained_checkpoint: Optional path to checkpoint for resuming.
        """
        n_channels = self.cfg.mode.get('in_channels', 1)

        # Create AutoencoderDC
        if self.pretrained:
            raw_model = self._create_pretrained_model(n_channels)
        else:
            raw_model = self._create_model_from_scratch(n_channels)

        # Create discriminator if GAN enabled
        raw_disc = None
        if not self.disable_gan:
            raw_disc = self._create_discriminator(n_channels, spatial_dims=2)

        # Load checkpoint if provided
        if pretrained_checkpoint:
            self._load_pretrained_weights(raw_model, raw_disc, pretrained_checkpoint)

        # Wrap models with DDP/compile
        self._wrap_models(raw_model, raw_disc)

        # Setup perceptual and adversarial loss
        if self.perceptual_weight > 0:
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
            n_params = sum(p.numel() for p in self.model_raw.parameters()) / 1e6
            logger.info(f"DC-AE parameters: {n_params:.2f}M")
            logger.info(f"Compression: {self.compression_ratio}× | Latent channels: {self.latent_channels}")
            logger.info(f"GAN: {'Disabled' if self.disable_gan else 'Enabled'}")

    def _create_pretrained_model(self, n_channels: int) -> nn.Module:
        """Create DC-AE from pretrained HuggingFace weights."""
        if self.is_main_process:
            logger.info(f"Loading pretrained DC-AE from: {self.pretrained}")

        raw_model = AutoencoderDC.from_pretrained(
            self.pretrained,
            torch_dtype=torch.float32,
        )

        # Modify input layer for grayscale if needed
        if raw_model.encoder.conv_in.in_channels != n_channels:
            if self.is_main_process:
                logger.info(f"Replacing conv_in: {raw_model.encoder.conv_in.in_channels} -> {n_channels} channels")
            old_conv = raw_model.encoder.conv_in
            new_conv = nn.Conv2d(
                n_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
            )
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            raw_model.encoder.conv_in = new_conv

        # Modify output layer for grayscale
        if raw_model.decoder.conv_out.conv.out_channels != n_channels:
            if self.is_main_process:
                logger.info(f"Replacing conv_out: {raw_model.decoder.conv_out.conv.out_channels} -> {n_channels} channels")
            old_conv = raw_model.decoder.conv_out.conv
            new_conv = nn.Conv2d(
                old_conv.in_channels, n_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
            )
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=0, keepdim=True))
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias[:n_channels])
            raw_model.decoder.conv_out.conv = new_conv

        return raw_model.to(self.device)

    def _create_model_from_scratch(self, n_channels: int) -> nn.Module:
        """Create DC-AE from config (train from scratch)."""
        if self.is_main_process:
            logger.info("Creating DC-AE from scratch")

        return AutoencoderDC(
            in_channels=n_channels,
            latent_channels=self.latent_channels,
            encoder_block_out_channels=tuple(self.cfg.dcae.encoder_block_out_channels),
            decoder_block_out_channels=tuple(self.cfg.dcae.decoder_block_out_channels),
            encoder_layers_per_block=tuple(self.cfg.dcae.encoder_layers_per_block),
            decoder_layers_per_block=tuple(self.cfg.dcae.decoder_layers_per_block),
            encoder_qkv_multiscales=tuple(tuple(x) for x in self.cfg.dcae.encoder_qkv_multiscales),
            decoder_qkv_multiscales=tuple(tuple(x) for x in self.cfg.dcae.decoder_qkv_multiscales),
            encoder_block_types=self.cfg.dcae.encoder_block_types,
            decoder_block_types=self.cfg.dcae.decoder_block_types,
            downsample_block_type=self.cfg.dcae.downsample_block_type,
            upsample_block_type=self.cfg.dcae.upsample_block_type,
            encoder_out_shortcut=self.cfg.dcae.encoder_out_shortcut,
            decoder_in_shortcut=self.cfg.dcae.decoder_in_shortcut,
            scaling_factor=self.scaling_factor,
        ).to(self.device)

    def _load_pretrained_weights(
        self,
        raw_model: nn.Module,
        raw_disc: Optional[nn.Module],
        checkpoint_path: str,
    ) -> None:
        """Load pretrained weights from checkpoint."""
        if not os.path.exists(checkpoint_path):
            if self.is_main_process:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            raw_model.load_state_dict(checkpoint['model_state_dict'])
            if self.is_main_process:
                logger.info(f"Loaded DC-AE weights from {checkpoint_path}")

        if 'discriminator_state_dict' in checkpoint and raw_disc is not None:
            raw_disc.load_state_dict(checkpoint['discriminator_state_dict'])
            if self.is_main_process:
                logger.info(f"Loaded discriminator weights from {checkpoint_path}")

    def _get_trainer_type(self) -> str:
        """Return trainer type for metadata."""
        return 'dcae'

    def _get_metadata_extra(self) -> Dict[str, Any]:
        """Return DC-AE-specific metadata."""
        return {
            'compression_ratio': self.compression_ratio,
            'latent_channels': self.latent_channels,
            'scaling_factor': self.scaling_factor,
            'pretrained': self.pretrained,
            'image_size': self.image_size,
        }

    def _prepare_batch(self, batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare batch for DC-AE training.

        Args:
            batch: Input batch.

        Returns:
            Tuple of (images, mask).
        """
        mask = None
        if isinstance(batch, dict):
            images = batch.get('image', batch.get('images'))
            mask = batch.get('mask', batch.get('seg'))
        elif isinstance(batch, (tuple, list)):
            images = batch[0]
            if len(batch) > 1:
                mask = batch[1]
        else:
            images = batch

        # Handle MetaTensor
        if hasattr(images, 'as_tensor'):
            images = images.as_tensor()

        images = images.to(self.device, dtype=self.weight_dtype)
        if mask is not None:
            if hasattr(mask, 'as_tensor'):
                mask = mask.as_tensor()
            mask = mask.to(self.device)

        return images, mask

    def train_step(self, batch: Any) -> TrainingStepResult:
        """Execute DC-AE training step.

        Args:
            batch: Input batch.

        Returns:
            TrainingStepResult with all loss components (no regularization).
        """
        images, _ = self._prepare_batch(batch)
        grad_clip = self.cfg.training.get('gradient_clip_norm', 1.0)

        d_loss = torch.tensor(0.0, device=self.device)
        adv_loss = torch.tensor(0.0, device=self.device)

        # ==================== Generator Step ====================
        self.optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            # DC-AE forward: deterministic encoding
            latent = self.model.encode(images, return_dict=False)[0]
            reconstruction = self.model.decode(latent, return_dict=False)[0]

            # L1 reconstruction loss
            l1_loss = torch.nn.functional.l1_loss(reconstruction, images)

            # Perceptual loss
            if self.perceptual_weight > 0 and self.perceptual_loss_fn is not None:
                p_loss = self._compute_perceptual_loss(reconstruction.float(), images.float())
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
            adversarial_loss=adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
            discriminator_loss=d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss,
        )

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train DC-AE for one epoch.

        Args:
            data_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dict with average losses.
        """
        self.model.train()
        if not self.disable_gan and self.discriminator is not None:
            self.discriminator.train()

        epoch_losses = {'gen': 0, 'disc': 0, 'recon': 0, 'perc': 0, 'adv': 0}

        epoch_iter = create_epoch_iterator(
            data_loader, epoch, self.is_cluster, self.is_main_process,
            limit_batches=self.limit_train_batches
        )

        for step, batch in enumerate(epoch_iter):
            result = self.train_step(batch)
            losses = result.to_legacy_dict(None)  # DCAE has no regularization

            # Step profiler to mark training step boundary
            self._profiler_step()

            for key in epoch_losses:
                epoch_losses[key] += losses[key]

            if hasattr(epoch_iter, 'set_postfix'):
                epoch_iter.set_postfix(
                    G=f"{epoch_losses['gen'] / (step + 1):.4f}",
                    L1=f"{epoch_losses['recon'] / (step + 1):.4f}"
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

            if not self.disable_gan:
                self.writer.add_scalar('Loss/Discriminator', avg_losses['disc'], epoch)
                self.writer.add_scalar('Loss/Adversarial', avg_losses['adv'], epoch)

        return avg_losses

    def _forward_for_validation(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DC-AE forward pass for validation.

        Returns:
            Tuple of (reconstruction, zero_reg_loss).
        """
        latent = model.encode(images, return_dict=False)[0]
        reconstruction = model.decode(latent, return_dict=False)[0]
        # DC-AE is deterministic - no regularization loss
        return reconstruction, torch.tensor(0.0, device=self.device)

    def _get_model_config(self) -> Dict[str, Any]:
        """Get DC-AE model configuration for checkpoint."""
        return {
            'compression_ratio': self.compression_ratio,
            'latent_channels': self.latent_channels,
            'scaling_factor': self.scaling_factor,
            'pretrained': self.pretrained,
        }

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
        elapsed_time: float,
    ) -> None:
        """Log DC-AE epoch summary."""
        log_compression_epoch_summary(
            epoch, total_epochs, avg_losses, val_metrics, elapsed_time,
            regularization_key=None,  # DC-AE has no regularization
        )

    def _measure_model_flops(
        self,
        sample_images: torch.Tensor,
        steps_per_epoch: int,
    ) -> None:
        """Measure FLOPs for DC-AE (encode + decode cycle).

        Overrides base method to handle DC-AE's HuggingFace-style API
        which uses encode() and decode() instead of forward().

        Falls back to parameter-based estimation if torch.profiler fails
        (common with custom CUDA kernels or compiled operations).

        Args:
            sample_images: Sample input batch for measurement.
            steps_per_epoch: Number of training steps per epoch.
        """
        if not self.log_flops:
            return

        # DC-AE needs wrapper for encode-decode cycle measurement
        class DCAEForward(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                latent = self.model.encode(x, return_dict=False)[0]
                return self.model.decode(latent, return_dict=False)[0]

        wrapper = DCAEForward(self.model_raw)
        self._flops_tracker.measure(
            model=wrapper,
            sample_input=sample_images[:1],
            steps_per_epoch=steps_per_epoch,
            timesteps=None,
            is_main_process=self.is_main_process,
        )

        # Fallback: estimate FLOPs from parameter count if profiler failed
        if self._flops_tracker.forward_flops == 0 and self.is_main_process:
            num_params = sum(p.numel() for p in self.model_raw.parameters())
            # Rough estimate: 2 FLOPs per param per forward, x2 for encode+decode
            estimated_flops = num_params * 4
            self._flops_tracker.forward_flops = estimated_flops
            self._flops_tracker.steps_per_epoch = steps_per_epoch
            self._flops_tracker.mark_measured()
            gflops = estimated_flops / 1e9
            tflops_epoch = self._flops_tracker.get_tflops_epoch()
            logger.info(
                f"FLOPs estimated from params: {gflops:.2f} GFLOPs/forward, "
                f"{tflops_epoch:.2f} TFLOPs/epoch (based on {num_params/1e6:.1f}M params)"
            )

    def _test_forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Perform DC-AE forward pass for test evaluation.

        Args:
            model: Model to use for inference.
            images: Input images.

        Returns:
            Reconstructed images tensor.
        """
        latent = model.encode(images, return_dict=False)[0]
        return model.decode(latent, return_dict=False)[0]
