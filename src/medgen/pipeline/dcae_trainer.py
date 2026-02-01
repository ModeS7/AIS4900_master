"""
DC-AE (Deep Compression Autoencoder) trainer module (2D and 3D).

This module provides the DCAETrainer class which inherits from BaseCompressionTrainer
and implements DC-AE-specific functionality:
- Deterministic encoder (no KL divergence)
- 2D: diffusers AutoencoderDC with encode/decode API
- 3D: Custom AutoencoderDC3D with forward() API
- High compression (32×, 64×, or 128× spatial)
- Pretrained ImageNet weights support (2D only)
- Structured latent space training (DC-AE 1.5, 2D only)
- Gradient checkpointing for 3D memory efficiency
"""
import itertools
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .compression_trainer import BaseCompressionTrainer
from .results import TrainingStepResult
from .utils import create_epoch_iterator, get_vram_usage
from medgen.metrics import create_worst_batch_figure

# Import 3D components for module-level export
from ..models.autoencoder_dc_3d import CheckpointedAutoencoderDC3D

logger = logging.getLogger(__name__)

__all__ = ['DCAETrainer', 'CheckpointedAutoencoderDC3D']


class DCAETrainer(BaseCompressionTrainer):
    """DC-AE trainer for high-compression encoding (2D and 3D).

    Supports both 2D images and 3D volumes via the spatial_dims parameter.

    2D uses diffusers AutoencoderDC with:
    - encode/decode API
    - Pretrained ImageNet weights support
    - Structured latent space training (DC-AE 1.5)
    - Training phases (1=no GAN, 3=with GAN)
    - Segmentation mode for mask compression

    3D uses custom AutoencoderDC3D with:
    - forward() API
    - Asymmetric compression (different spatial vs depth)
    - Gradient checkpointing

    Args:
        cfg: Hydra configuration object.
        spatial_dims: Spatial dimensions (2 or 3).

    Example:
        >>> trainer = DCAETrainer(cfg, spatial_dims=2)
        >>> trainer.setup_model()
        >>> trainer.train(train_loader, train_dataset, val_loader)
    """

    def __init__(self, cfg: DictConfig, spatial_dims: int = 2) -> None:
        """Initialize DC-AE trainer.

        Args:
            cfg: Hydra configuration object.
            spatial_dims: Spatial dimensions (2 or 3).
        """
        super().__init__(cfg, spatial_dims=spatial_dims)

        # ─────────────────────────────────────────────────────────────────────
        # DC-AE-specific config (dimension-dependent)
        # ─────────────────────────────────────────────────────────────────────
        dcae_cfg = cfg.dcae_3d if spatial_dims == 3 else cfg.dcae
        self.l1_weight: float = dcae_cfg.get('l1_weight', 1.0)
        self.latent_channels: int = dcae_cfg.latent_channels
        self.scaling_factor: float = dcae_cfg.get('scaling_factor', 1.0)

        if spatial_dims == 2:
            # 2D-specific config
            self.compression_ratio: int = dcae_cfg.compression_ratio
            self.pretrained: Optional[str] = dcae_cfg.get('pretrained', None)

            # Training phase (1=no GAN, 3=with GAN)
            self.training_phase: int = cfg.training.get('phase', 1)
            if self.training_phase == 1 or self.adv_weight == 0.0:
                self.disable_gan = True

            # Validate Phase 3 configuration
            if self.training_phase == 3:
                if self.adv_weight == 0.0:
                    logger.warning(
                        "Phase 3 training with adv_weight=0.0 is unusual. "
                        "Phase 3 is meant for GAN refinement."
                    )
                if cfg.get('pretrained_checkpoint') is None and self.pretrained is None:
                    logger.warning(
                        "Phase 3 training without pretrained checkpoint."
                    )

            # Segmentation mode (for mask compression)
            self.seg_mode: bool = dcae_cfg.get('seg_mode', False)
            self.seg_loss_fn = None
            if self.seg_mode:
                from medgen.losses import SegmentationLoss
                seg_weights = dcae_cfg.get('seg_loss_weights', {})
                self.seg_loss_fn = SegmentationLoss(
                    bce_weight=seg_weights.get('bce', 1.0),
                    dice_weight=seg_weights.get('dice', 1.0),
                    boundary_weight=seg_weights.get('boundary', 0.5),
                )
                self.perceptual_weight = 0.0
                if self.is_main_process:
                    logger.info("Seg mode enabled: BCE + Dice + Boundary loss")

            # DC-AE 1.5: Structured Latent Space
            structured_cfg = dcae_cfg.get('structured_latent', {})
            self.structured_latent_enabled: bool = structured_cfg.get('enabled', False)
            self.structured_latent_min: int = structured_cfg.get('min_channels', 16)
            self.structured_latent_step: int = structured_cfg.get('channel_step', 4)

            if self.structured_latent_enabled and self.is_main_process:
                steps = self._get_channel_steps()
                logger.info(f"DC-AE 1.5 Structured Latent Space enabled: channel_steps={steps}")
        else:
            # 3D-specific config
            self.encoder_block_out_channels = tuple(dcae_cfg.encoder_block_out_channels)
            self.decoder_block_out_channels = tuple(dcae_cfg.decoder_block_out_channels)
            self.encoder_layers_per_block = tuple(dcae_cfg.encoder_layers_per_block)
            self.decoder_layers_per_block = tuple(dcae_cfg.decoder_layers_per_block)
            self.depth_factors = tuple(dcae_cfg.depth_factors)
            self.encoder_out_shortcut = dcae_cfg.get('encoder_out_shortcut', True)
            self.decoder_in_shortcut = dcae_cfg.get('decoder_in_shortcut', True)
            # 3D doesn't have these features
            self.seg_mode = False
            self.seg_loss_fn = None
            self.training_phase = 1
            self.structured_latent_enabled = False

        # Initialize unified metrics system (after seg_mode is set)
        self._init_unified_metrics('dcae')

    @classmethod
    def create_2d(cls, cfg: DictConfig, **kwargs) -> 'DCAETrainer':
        """Create 2D DCAETrainer."""
        return cls(cfg, spatial_dims=2, **kwargs)

    @classmethod
    def create_3d(cls, cfg: DictConfig, **kwargs) -> 'DCAETrainer':
        """Create 3D DCAETrainer."""
        return cls(cfg, spatial_dims=3, **kwargs)

    def _get_disc_lr(self, cfg: DictConfig) -> float:
        """Get discriminator LR from dcae/dcae_3d config."""
        section = 'dcae_3d' if self.spatial_dims == 3 else 'dcae'
        if section in cfg:
            return cfg[section].get('disc_lr', 1e-4 if self.spatial_dims == 3 else 5e-4)
        return 5e-4

    def _get_perceptual_weight(self, cfg: DictConfig) -> float:
        """Get perceptual weight from dcae/dcae_3d config."""
        section = 'dcae_3d' if self.spatial_dims == 3 else 'dcae'
        if section in cfg:
            return cfg[section].get('perceptual_weight', 0.1)
        return 0.1

    def _get_adv_weight(self, cfg: DictConfig) -> float:
        """Get adversarial weight from dcae/dcae_3d config."""
        section = 'dcae_3d' if self.spatial_dims == 3 else 'dcae'
        if section in cfg:
            return cfg[section].get('adv_weight', 0.0)
        return 0.0

    def _get_disable_gan(self, cfg: DictConfig) -> bool:
        """Determine if GAN is disabled."""
        section = 'dcae_3d' if self.spatial_dims == 3 else 'dcae'
        if section in cfg:
            if self.spatial_dims == 3:
                return cfg[section].get('disable_gan', True)
            else:
                training_phase = cfg.training.get('phase', 1)
                adv_weight = cfg[section].get('adv_weight', 0.0)
                return (adv_weight == 0.0) or (training_phase == 1)
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # DC-AE 1.5: Structured Latent Space Methods (2D only)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_channel_steps(self) -> List[int]:
        """Get list of channel counts for structured latent training."""
        if not self.structured_latent_enabled:
            return [self.latent_channels]
        steps = list(range(
            self.structured_latent_min,
            self.latent_channels + 1,
            self.structured_latent_step
        ))
        if not steps or steps[-1] != self.latent_channels:
            steps.append(self.latent_channels)
        return steps

    def _sample_latent_channels(self) -> Optional[int]:
        """Sample random channel count for structured latent training.

        Returns:
            Number of channels to use, or None if structured latent is disabled.
        """
        if not self.structured_latent_enabled:
            return None
        return random.choice(self._get_channel_steps())

    def _create_fallback_save_dir(self) -> str:
        """Create fallback save directory for DC-AE."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = self.cfg.training.get('name', '')
        mode_name = self.cfg.mode.get('name', 'multi_modality')

        if self._spatial_dims == 3:
            # Read from config since this may be called before base class sets attributes
            volume_height = self.cfg.volume.get('height', 256)
            volume_depth = self.cfg.volume.get('depth', 160)
            run_name = f"dcae_{exp_name}{volume_height}x{volume_depth}_{timestamp}"
            return os.path.join(self.cfg.paths.model_dir, 'compression_3d', mode_name, run_name)
        else:
            image_size = self.cfg.dcae.get('image_size', self.cfg.model.get('image_size', 128))
            run_name = f"{exp_name}{image_size}_{timestamp}"
            return os.path.join(self.cfg.paths.model_dir, 'compression_2d', mode_name, run_name)

    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize DC-AE model, discriminator, optimizers.

        Args:
            pretrained_checkpoint: Optional path to checkpoint for loading
                pretrained weights.
        """
        n_channels = self.cfg.mode.get('in_channels', 1 if self.spatial_dims == 2 else 4)

        if self.spatial_dims == 3:
            raw_model = self._setup_model_3d(n_channels, pretrained_checkpoint)
        else:
            raw_model = self._setup_model_2d(n_channels, pretrained_checkpoint)

        # Create discriminator if GAN enabled
        raw_disc = None
        if not self.disable_gan:
            raw_disc = self._create_discriminator(n_channels, spatial_dims=self.spatial_dims)

        # 2D: Load checkpoint after model creation (handles both model + discriminator)
        if pretrained_checkpoint and self.spatial_dims == 2:
            self._load_pretrained_weights_2d(raw_model, raw_disc, pretrained_checkpoint)

        # Wrap models with DDP/compile
        self._wrap_models(raw_model, raw_disc)

        # Phase 3: Freeze all except decoder head (2D only)
        if self.spatial_dims == 2 and getattr(self, 'training_phase', 1) == 3:
            self._freeze_for_phase3()

        # Setup perceptual and adversarial loss
        if self.perceptual_weight > 0:
            perceptual_spatial_dims = 2 if (self.spatial_dims == 3 and getattr(self, 'use_2_5d_perceptual', False)) else self.spatial_dims
            self.perceptual_loss_fn = self._create_perceptual_loss(spatial_dims=perceptual_spatial_dims)
        self._create_adversarial_loss()

        # Setup optimizers and schedulers
        self._setup_optimizers(n_channels)

        # Setup EMA
        self._setup_ema()

        # Save metadata
        if self.is_main_process:
            self._save_metadata()

        # Log model info
        self._log_model_info()

    def _setup_model_2d(self, n_channels: int, pretrained_checkpoint: Optional[str] = None) -> nn.Module:
        """Setup 2D DC-AE model."""

        if self.pretrained:
            raw_model = self._create_pretrained_model_2d(n_channels)
        else:
            raw_model = self._create_model_from_scratch_2d(n_channels)

        # Wrap with structured latent wrapper if enabled (DC-AE 1.5)
        if self.structured_latent_enabled:
            from ..models.dcae_structured import StructuredAutoencoderDC
            channel_steps = self._get_channel_steps()
            raw_model = StructuredAutoencoderDC(raw_model, channel_steps)
            if self.is_main_process:
                logger.info(f"Wrapped model with StructuredAutoencoderDC (channel_steps={channel_steps})")

        return raw_model

    def _setup_model_3d(self, n_channels: int, pretrained_checkpoint: Optional[str] = None) -> nn.Module:
        """Setup 3D DC-AE model."""
        from ..models.autoencoder_dc_3d import AutoencoderDC3D, CheckpointedAutoencoderDC3D

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
            self._load_pretrained_weights_base(base_model, pretrained_checkpoint, "3D DC-AE")

        # Wrap with gradient checkpointing for memory efficiency
        if self.gradient_checkpointing:
            raw_model = CheckpointedAutoencoderDC3D(base_model)
            if self.is_main_process:
                logger.info("Gradient checkpointing enabled (CheckpointedAutoencoderDC3D)")
        else:
            raw_model = base_model

        return raw_model

    def _create_pretrained_model_2d(self, n_channels: int) -> nn.Module:
        """Create 2D DC-AE from pretrained HuggingFace weights."""
        from diffusers import AutoencoderDC

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

    def _create_model_from_scratch_2d(self, n_channels: int) -> nn.Module:
        """Create 2D DC-AE from config (train from scratch)."""
        from diffusers import AutoencoderDC

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

    def _load_pretrained_weights_2d(
        self,
        raw_model: nn.Module,
        raw_disc: Optional[nn.Module],
        checkpoint_path: str,
    ) -> None:
        """Load pretrained weights from checkpoint (2D)."""
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
        """Freeze all layers except decoder head for Phase 3 training (2D only)."""
        # Freeze entire encoder
        for param in self.model_raw.encoder.parameters():
            param.requires_grad = False

        # Freeze decoder body (conv_in + up_blocks)
        for param in self.model_raw.decoder.conv_in.parameters():
            param.requires_grad = False
        for param in self.model_raw.decoder.up_blocks.parameters():
            param.requires_grad = False

        if self.is_main_process:
            frozen = sum(p.numel() for p in self.model_raw.parameters() if not p.requires_grad)
            trainable = sum(p.numel() for p in self.model_raw.parameters() if p.requires_grad)
            logger.info(f"Phase 3: Frozen parameters: {frozen/1e6:.2f}M")
            logger.info(f"Phase 3: Trainable decoder head: {trainable:,} params ({trainable/1e3:.1f}K)")

    def _log_model_info(self) -> None:
        """Log model information."""
        if not self.is_main_process:
            return

        n_params = sum(p.numel() for p in self.model_raw.parameters()) / 1e6
        dim_str = "3D " if self.spatial_dims == 3 else ""
        logger.info(f"{dim_str}DC-AE parameters: {n_params:.2f}M")
        logger.info(f"Latent channels: {self.latent_channels}")
        logger.info(f"GAN: {'Disabled' if self.disable_gan else 'Enabled'}")

        if self.spatial_dims == 3:
            # Get compression info from model
            raw_model = self.model_raw
            if hasattr(raw_model, 'model'):
                raw_model = raw_model.model  # Unwrap CheckpointedAutoencoderDC3D
            logger.info(f"Volume: {self.volume_width}x{self.volume_height}x{self.volume_depth}")
            logger.info(f"Spatial compression: {raw_model.spatial_compression}×")
            logger.info(f"Depth compression: {raw_model.depth_compression}×")
            if not self.disable_gan:
                disc_params = sum(p.numel() for p in self.discriminator_raw.parameters())
                logger.info(f"3D Discriminator: {disc_params / 1e6:.1f}M parameters")
        else:
            logger.info(f"Compression: {self.compression_ratio}×")

    def _get_trainer_type(self) -> str:
        """Return trainer type for metadata."""
        return 'dcae_3d' if self.spatial_dims == 3 else 'dcae'

    def _get_metadata_extra(self) -> Dict[str, Any]:
        """Return DC-AE-specific metadata."""
        if self.spatial_dims == 3:
            return {
                'dcae_config': self._get_model_config(),
                'volume': {
                    'height': self.volume_height,
                    'width': self.volume_width,
                    'depth': self.volume_depth,
                },
            }
        else:
            return {
                'compression_ratio': self.compression_ratio,
                'latent_channels': self.latent_channels,
                'scaling_factor': self.scaling_factor,
                'pretrained': self.pretrained,
                'image_size': self.image_size,
                'seg_mode': self.seg_mode,
            }

    def _get_model_config(self) -> Dict[str, Any]:
        """Get DC-AE model configuration for checkpoint."""
        if self.spatial_dims == 3:
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
        else:
            return {
                'compression_ratio': self.compression_ratio,
                'latent_channels': self.latent_channels,
                'scaling_factor': self.scaling_factor,
                'pretrained': self.pretrained,
                'seg_mode': self.seg_mode,
            }

    def _prepare_batch(self, batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare batch for DC-AE training."""
        # Use base class for 3D
        if self.spatial_dims == 3:
            return super()._prepare_batch(batch)

        # 2D-specific handling
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
        """Execute DC-AE training step."""
        images, _ = self._prepare_batch(batch)
        grad_clip = self.cfg.training.get('gradient_clip_norm', 1.0)

        d_loss = torch.tensor(0.0, device=self.device)
        adv_loss = torch.tensor(0.0, device=self.device)

        # ==================== Generator Step ====================
        self.optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            # Forward pass (different API for 2D vs 3D)
            if self.spatial_dims == 3:
                reconstruction = self.model(images)
            else:
                # DC-AE 1.5: Sample random channel count for structured latent training
                # StructuredAutoencoderDC handles the weight slicing internally
                latent_channels = self._sample_latent_channels()
                if self.structured_latent_enabled:
                    latent = self.model.encode(images, latent_channels=latent_channels, return_dict=False)[0]
                else:
                    latent = self.model.encode(images, return_dict=False)[0]
                reconstruction = self.model.decode(latent, return_dict=False)[0]

            # Compute reconstruction loss
            if self.seg_mode and self.seg_loss_fn is not None:
                seg_loss, seg_breakdown = self.seg_loss_fn(reconstruction, images)
                l1_loss = seg_loss
                p_loss = torch.tensor(0.0, device=self.device)
                if hasattr(self, '_epoch_seg_breakdown'):
                    for key in seg_breakdown:
                        self._epoch_seg_breakdown[key] += seg_breakdown[key]
            else:
                l1_loss = torch.nn.functional.l1_loss(reconstruction.float(), images.float())
                p_loss = self._compute_perceptual_loss(reconstruction.float(), images.float())

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
            regularization_loss=0.0,  # DC-AE is deterministic
            adversarial_loss=adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
            discriminator_loss=d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss,
        )

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train DC-AE for one epoch."""
        self.model.train()
        if not self.disable_gan and self.discriminator is not None:
            self.discriminator.train()

        self._loss_accumulator.reset()

        # Initialize seg breakdown accumulator
        if self.seg_mode:
            self._epoch_seg_breakdown = {'bce': 0.0, 'dice': 0.0, 'boundary': 0.0}

        if self.spatial_dims == 3:
            disable_pbar = not self.is_main_process or self.is_cluster
            total = self.limit_train_batches if self.limit_train_batches else len(data_loader)
            iterator = itertools.islice(data_loader, self.limit_train_batches) if self.limit_train_batches else data_loader
            epoch_iter = tqdm(iterator, desc=f"Epoch {epoch}", disable=disable_pbar, total=total)
        else:
            epoch_iter = create_epoch_iterator(
                data_loader, epoch, self.is_cluster, self.is_main_process,
                limit_batches=self.limit_train_batches
            )

        for step, batch in enumerate(epoch_iter):
            result = self.train_step(batch)
            losses = result.to_legacy_dict(None)  # DC-AE has no regularization

            self._profiler_step()
            self._loss_accumulator.update(losses)

            if hasattr(epoch_iter, 'set_postfix'):
                avg_so_far = self._loss_accumulator.compute()
                if self.seg_mode:
                    epoch_iter.set_postfix(
                        G=f"{avg_so_far.get('gen', 0):.4f}",
                        Dice=f"{self._epoch_seg_breakdown['dice'] / (step + 1):.4f}"
                    )
                else:
                    epoch_iter.set_postfix(
                        G=f"{avg_so_far.get('gen', 0):.4f}",
                        L1=f"{avg_so_far.get('recon', 0):.4f}"
                    )

            if epoch == 1 and step == 0 and self.is_main_process:
                logger.info(get_vram_usage(self.device))

        avg_losses = self._loss_accumulator.compute()

        # Add seg breakdown to avg_losses
        if self.seg_mode and hasattr(self, '_epoch_seg_breakdown'):
            n_batches = self.limit_train_batches if self.limit_train_batches else len(data_loader)
            avg_seg = {k: v / n_batches for k, v in self._epoch_seg_breakdown.items()}
            avg_losses.update(avg_seg)

        self._log_training_metrics_unified(epoch, avg_losses)

        return avg_losses

    def _forward_for_validation(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DC-AE forward pass for validation."""
        if self.spatial_dims == 3:
            reconstruction = model(images)
        else:
            latent = model.encode(images, return_dict=False)[0]
            reconstruction = model.decode(latent, return_dict=False)[0]
        return reconstruction, torch.tensor(0.0, device=self.device)

    def _create_validation_runner(self):
        """Create ValidationRunner with seg_mode support."""
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
        """Create SegRegionalMetricsTracker for per-tumor Dice tracking."""
        from medgen.metrics import SegRegionalMetricsTracker
        size = self.volume_height if self.spatial_dims == 3 else self.image_size
        return SegRegionalMetricsTracker(
            image_size=size,
            fov_mm=self.cfg.paths.get('fov_mm', 240.0),
            device=self.device,
        )

    def _create_worst_batch_figure(
        self,
        worst_batch_data: Dict[str, Any],
    ) -> plt.Figure:
        """Create worst batch figure with seg_mode support."""
        generated = worst_batch_data['generated']
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
        regional_tracker,
        log_figures: bool,
    ) -> None:
        """Log validation metrics with seg_mode support."""
        if self.writer is None:
            return

        self._log_validation_metrics_core(epoch, metrics)

        if log_figures and worst_batch_data is not None:
            self._unified_metrics.log_worst_batch(
                original=worst_batch_data['original'],
                reconstructed=worst_batch_data['generated'],
                loss=worst_batch_data.get('loss', 0.0),
                epoch=epoch,
                phase='val',
            )

        if regional_tracker is not None:
            mode_name = self.cfg.mode.get('name', 'bravo')
            is_multi_modality = mode_name == 'multi_modality'
            is_dual = self.cfg.mode.get('in_channels', 1) == 2 and mode_name == 'dual'
            if self.seg_mode:
                modality_override = f'seg_{mode_name}'
            elif not is_multi_modality and not is_dual:
                modality_override = mode_name
            else:
                modality_override = None
            self._unified_metrics.log_validation_regional(regional_tracker, epoch, modality_override=modality_override)

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
        elapsed_time: float,
    ) -> None:
        """Log DC-AE epoch summary using unified system."""
        self._log_epoch_summary_unified(epoch, avg_losses, val_metrics, elapsed_time)

    def _test_forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Perform DC-AE forward pass for test evaluation."""
        if self.spatial_dims == 3:
            return model(images)
        else:
            latent = model.encode(images, return_dict=False)[0]
            return model.decode(latent, return_dict=False)[0]

    def _measure_model_flops(
        self,
        sample_images: torch.Tensor,
        steps_per_epoch: int,
    ) -> None:
        """Measure FLOPs for DC-AE."""
        if not self.log_flops:
            return

        if self.spatial_dims == 2:
            # DC-AE 2D needs wrapper for encode-decode cycle measurement
            class DCAEForward(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    latent = self.model.encode(x, return_dict=False)[0]
                    return self.model.decode(latent, return_dict=False)[0]

            wrapper = DCAEForward(self.model_raw)
        else:
            # 3D: model has direct forward()
            wrapper = self.model_raw

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
