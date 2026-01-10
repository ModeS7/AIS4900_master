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
from .losses import SegmentationLoss
from .results import TrainingStepResult
from .metrics import (
    compute_dice,
    compute_iou,
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

        # Validate Phase 3 configuration
        if self.training_phase == 3:
            if self.adv_weight == 0.0:
                logger.warning(
                    "Phase 3 training with adv_weight=0.0 is unusual. "
                    "Phase 3 is meant for GAN refinement. Set dcae.adv_weight > 0."
                )
            if cfg.get('pretrained_checkpoint') is None and self.pretrained is None:
                logger.warning(
                    "Phase 3 training without pretrained checkpoint. "
                    "Phase 3 expects a model trained in Phase 1. "
                    "Set pretrained_checkpoint=/path/to/phase1/checkpoint_best.pt"
                )

        self.image_size: int = cfg.dcae.get('image_size', 256)

        # ─────────────────────────────────────────────────────────────────────
        # Segmentation mode (for mask compression)
        # ─────────────────────────────────────────────────────────────────────
        self.seg_mode: bool = cfg.dcae.get('seg_mode', False)
        self.seg_loss_fn: Optional[SegmentationLoss] = None

        if self.seg_mode:
            seg_weights = cfg.dcae.get('seg_loss_weights', {})
            self.seg_loss_fn = SegmentationLoss(
                bce_weight=seg_weights.get('bce', 1.0),
                dice_weight=seg_weights.get('dice', 1.0),
                boundary_weight=seg_weights.get('boundary', 0.5),
            )
            # Disable perceptual loss for seg mode (meaningless for binary masks)
            self.perceptual_weight = 0.0
            if self.is_main_process:
                logger.info("Seg mode enabled: BCE + Dice + Boundary loss")

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

        # Phase 3: Freeze all except decoder head (local refinement with GAN)
        # Paper: "we only tune the head layers of the decoder"
        if self.training_phase == 3:
            self._freeze_for_phase3()

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

    def _freeze_for_phase3(self) -> None:
        """Freeze all layers except decoder head for Phase 3 training.

        Paper (DC-AE, ICLR 2025): "we only tune the head layers of the decoder
        while freezing all the other layers"

        Decoder head layers (trainable):
        - norm_out: Final normalization
        - conv_act: Final activation
        - conv_out: Output projection

        All other layers (frozen):
        - Entire encoder (conv_in, down_blocks, conv_out)
        - Decoder body (conv_in, up_blocks)

        This design:
        1. Prevents latent space drift during GAN training
        2. GAN loss only improves local details, so only head layers needed
        3. Lower training cost and better accuracy than full GAN training
        """
        # Freeze entire encoder
        for param in self.model_raw.encoder.parameters():
            param.requires_grad = False

        # Freeze decoder body (conv_in + up_blocks)
        for param in self.model_raw.decoder.conv_in.parameters():
            param.requires_grad = False
        for param in self.model_raw.decoder.up_blocks.parameters():
            param.requires_grad = False

        # Decoder head layers remain trainable:
        # - decoder.norm_out
        # - decoder.conv_act
        # - decoder.conv_out

        if self.is_main_process:
            frozen = sum(p.numel() for p in self.model_raw.parameters()
                         if not p.requires_grad)
            trainable = sum(p.numel() for p in self.model_raw.parameters()
                            if p.requires_grad)
            logger.info(f"Phase 3: Frozen parameters: {frozen/1e6:.2f}M")
            logger.info(f"Phase 3: Trainable decoder head: {trainable:,} params ({trainable/1e3:.1f}K)")

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
            'seg_mode': self.seg_mode,
        }

    def _create_validation_runner(self) -> 'ValidationRunner':
        """Create ValidationRunner with seg_mode support.

        Overrides base method to pass seg_mode flag to ValidationConfig.

        Returns:
            Configured ValidationRunner instance.
        """
        from .validation import ValidationRunner, ValidationConfig

        config = ValidationConfig(
            log_msssim=self.log_msssim and not self.seg_mode,  # Disable for seg
            log_psnr=self.log_psnr and not self.seg_mode,       # Disable for seg
            log_lpips=self.log_lpips and not self.seg_mode,     # Disable for seg
            log_regional_losses=self.log_regional_losses,  # Enable for both modes
            weight_dtype=self.weight_dtype,
            use_compile=self.use_compile,
            seg_mode=self.seg_mode,  # Pass seg_mode flag
        )

        # Image mode: standard regional tracker
        regional_factory = None
        if self.log_regional_losses and not self.seg_mode:
            regional_factory = self._create_regional_tracker

        # Seg mode: per-tumor Dice tracker
        seg_regional_factory = None
        if self.log_regional_losses and self.seg_mode:
            seg_regional_factory = self._create_seg_regional_tracker

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

    def _create_seg_regional_tracker(self) -> 'SegRegionalMetricsTracker':
        """Create SegRegionalMetricsTracker for per-tumor Dice tracking.

        Returns:
            Configured SegRegionalMetricsTracker instance.
        """
        from .metrics import SegRegionalMetricsTracker

        return SegRegionalMetricsTracker(
            image_size=self.cfg.dcae.image_size,
            fov_mm=self.cfg.paths.get('fov_mm', 240.0),
            device=self.device,
        )

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

            if self.seg_mode and self.seg_loss_fn is not None:
                # Segmentation loss: BCE + Dice + Boundary on logits
                # Model outputs logits for numerically stable BCE
                seg_loss, seg_breakdown = self.seg_loss_fn(reconstruction, images)
                l1_loss = seg_loss  # Reuse variable for total seg loss
                p_loss = torch.tensor(0.0, device=self.device)
                # Accumulate breakdown for epoch averaging
                if hasattr(self, '_epoch_seg_breakdown'):
                    for key in seg_breakdown:
                        self._epoch_seg_breakdown[key] += seg_breakdown[key]
            else:
                # Standard L1 reconstruction loss
                l1_loss = torch.nn.functional.l1_loss(reconstruction, images)

                # Perceptual loss
                if self.perceptual_weight > 0 and self.perceptual_loss_fn is not None:
                    p_loss = self._compute_perceptual_loss(reconstruction.float(), images.float())
                else:
                    p_loss = torch.tensor(0.0, device=self.device)

            # Adversarial loss (typically disabled for seg mode)
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

        # Initialize seg breakdown accumulator for epoch averaging
        if self.seg_mode:
            self._epoch_seg_breakdown = {'bce': 0.0, 'dice': 0.0, 'boundary': 0.0}

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
                if self.seg_mode:
                    epoch_iter.set_postfix(
                        G=f"{epoch_losses['gen'] / (step + 1):.4f}",
                        Dice=f"{self._epoch_seg_breakdown['dice'] / (step + 1):.4f}"
                    )
                else:
                    epoch_iter.set_postfix(
                        G=f"{epoch_losses['gen'] / (step + 1):.4f}",
                        L1=f"{epoch_losses['recon'] / (step + 1):.4f}"
                    )

            if epoch == 1 and step == 0 and self.is_main_process:
                logger.info(get_vram_usage(self.device))

        # Average losses
        n_batches = self.limit_train_batches if self.limit_train_batches else len(data_loader)
        avg_losses = {key: val / n_batches for key, val in epoch_losses.items()}

        # Add seg breakdown to avg_losses for epoch summary
        if self.seg_mode and hasattr(self, '_epoch_seg_breakdown'):
            avg_seg = {k: v / n_batches for k, v in self._epoch_seg_breakdown.items()}
            avg_losses.update(avg_seg)

        # Log training metrics (single-GPU only)
        if self.writer is not None and self.is_main_process and not self.use_multi_gpu:
            self.writer.add_scalar('Loss/Generator_train', avg_losses['gen'], epoch)

            if self.seg_mode:
                # Seg mode: log BCE, Dice, Boundary losses
                self.writer.add_scalar('Loss/BCE_train', avg_losses.get('bce', 0), epoch)
                self.writer.add_scalar('Loss/Dice_train', avg_losses.get('dice', 0), epoch)
                self.writer.add_scalar('Loss/Boundary_train', avg_losses.get('boundary', 0), epoch)
            else:
                # Standard mode: log L1, Perceptual
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

    def _create_worst_batch_figure(
        self,
        worst_batch_data: Dict[str, Any],
    ) -> plt.Figure:
        """Create worst batch figure with seg_mode support.

        In seg_mode, applies sigmoid to generated logits for visualization.

        Args:
            worst_batch_data: Dict with 'original', 'generated', 'loss', 'loss_breakdown'.

        Returns:
            Matplotlib figure.
        """
        generated = worst_batch_data['generated']

        # Apply sigmoid for seg_mode visualization (model outputs logits)
        if self.seg_mode and isinstance(generated, torch.Tensor):
            generated = torch.sigmoid(generated)

        return create_worst_batch_figure(
            original=worst_batch_data['original'],
            generated=generated,
            loss=worst_batch_data['loss'],
            loss_breakdown=worst_batch_data.get('loss_breakdown'),
        )

    def _log_validation_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float],
        worst_batch_data: Optional[Dict[str, Any]],
        regional_tracker: Optional[RegionalMetricsTracker],
        log_figures: bool,
    ) -> None:
        """Log validation metrics with seg_mode support.

        In seg_mode, logs Dice/IoU instead of L1/Perceptual/PSNR/LPIPS/MS-SSIM.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of validation metrics.
            worst_batch_data: Worst batch data for visualization.
            regional_tracker: Regional metrics tracker.
            log_figures: Whether to log figures.
        """
        if self.writer is None:
            return

        if self.seg_mode:
            # Seg mode: losses under Loss/, metrics under Validation/
            if 'dice' in metrics:
                self.writer.add_scalar('Loss/Dice_val', metrics['dice'], epoch)
            if 'bce' in metrics:
                self.writer.add_scalar('Loss/BCE_val', metrics['bce'], epoch)
            if 'boundary' in metrics:
                self.writer.add_scalar('Loss/Boundary_val', metrics['boundary'], epoch)
            if 'gen' in metrics:
                self.writer.add_scalar('Loss/Generator_val', metrics['gen'], epoch)
            if 'iou' in metrics:
                self.writer.add_scalar('Validation/IoU', metrics['iou'], epoch)
        else:
            # Standard mode: loss metrics
            self.writer.add_scalar('Loss/L1_val', metrics.get('l1', 0), epoch)
            self.writer.add_scalar('Loss/Perceptual_val', metrics.get('perc', 0), epoch)
            self.writer.add_scalar('Loss/Generator_val', metrics.get('gen', 0), epoch)

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

        # Log regional metrics (different prefix for seg vs image mode)
        if regional_tracker is not None:
            prefix = 'regional_seg' if self.seg_mode else 'regional'
            regional_tracker.log_to_tensorboard(self.writer, epoch, prefix=prefix)

    def _create_test_evaluator(self) -> 'CompressionTestEvaluator':
        """Create test evaluator with seg_mode support.

        Overrides parent to pass seg_mode flag for Dice/IoU metrics.

        Returns:
            Configured CompressionTestEvaluator instance.
        """
        from .evaluation import CompressionTestEvaluator, MetricsConfig

        # Create metrics config - use seg metrics when seg_mode enabled
        metrics_config = MetricsConfig(
            compute_l1=not self.seg_mode,
            compute_psnr=not self.seg_mode,
            compute_lpips=not self.seg_mode,
            compute_msssim=self.log_msssim and not self.seg_mode,
            compute_msssim_3d=False,
            compute_regional=self.log_regional_losses and not self.seg_mode,
            seg_mode=self.seg_mode,
        )

        # Regional tracker factory (if configured and not seg_mode)
        regional_factory = None
        if self.log_regional_losses and not self.seg_mode:
            regional_factory = self._create_regional_tracker

        # Volume 3D MS-SSIM callback (disabled for seg_mode)
        def volume_3d_msssim() -> Optional[float]:
            if self.seg_mode:
                return None
            return self._compute_volume_3d_msssim(epoch=0, data_split='test_new')

        # Worst batch figure callback
        worst_batch_fig_fn = self._create_worst_batch_figure

        # Get image keys for per-channel metrics
        n_channels = self.cfg.mode.get('in_channels', 1)
        image_keys = None
        if n_channels > 1 and not self.seg_mode:
            image_keys = self.cfg.mode.get('image_keys', None)

        return CompressionTestEvaluator(
            model=self.model_raw,
            device=self.device,
            save_dir=self.save_dir,
            forward_fn=self._test_forward,
            weight_dtype=self.weight_dtype,
            writer=self.writer,
            metrics_config=metrics_config,
            is_cluster=self.is_cluster,
            regional_tracker_factory=regional_factory,
            volume_3d_msssim_fn=volume_3d_msssim,
            worst_batch_figure_fn=worst_batch_fig_fn,
            image_keys=image_keys,
            seg_loss_fn=self.seg_loss_fn if self.seg_mode else None,
        )

    def _get_model_config(self) -> Dict[str, Any]:
        """Get DC-AE model configuration for checkpoint."""
        return {
            'compression_ratio': self.compression_ratio,
            'latent_channels': self.latent_channels,
            'scaling_factor': self.scaling_factor,
            'pretrained': self.pretrained,
            'seg_mode': self.seg_mode,
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
        if self.seg_mode:
            # Seg mode: show Dice, BCE, Boundary, IoU
            import time
            timestamp = time.strftime("%H:%M:%S")
            epoch_pct = ((epoch + 1) / total_epochs) * 100

            val_gen = f"(v:{val_metrics.get('gen', 0):.4f})" if val_metrics else ""
            val_dice = f"(v:{val_metrics.get('dice', 0):.4f})" if val_metrics else ""

            # Seg metrics
            iou_str = f"IoU: {val_metrics.get('iou', 0):.3f}" if val_metrics and val_metrics.get('iou') else ""

            logger.info(
                f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
                f"G: {avg_losses['gen']:.4f}{val_gen} | "
                f"Dice: {avg_losses.get('dice', 0):.4f}{val_dice} | "
                f"BCE: {avg_losses.get('bce', 0):.4f} | "
                f"Bnd: {avg_losses.get('boundary', 0):.4f} | "
                f"{iou_str} | "
                f"Time: {elapsed_time:.1f}s"
            )
        else:
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
