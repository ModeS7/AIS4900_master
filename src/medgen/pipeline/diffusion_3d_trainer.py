"""
3D Diffusion model trainer for volumetric latent diffusion.

This module provides the Diffusion3DTrainer class for training diffusion models
on 3D volumetric data, supporting both pixel-space and latent-space training.

Key differences from 2D DiffusionTrainer:
- Handles 5D tensors [B, C, D, H, W]
- Uses 3D UNet with memory optimizations
- Computes 2D-only metrics (LPIPS, KID, CMMD, FID) slice-wise (2.5D)
- Visualizes center slices for TensorBoard
"""
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from ema_pytorch import EMA
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from monai.networks.nets import DiffusionModelUNet
from monai.metrics import SSIMMetric

from medgen.core import setup_distributed, create_warmup_cosine_scheduler
from .base_trainer import BaseTrainer
from .results import TrainingStepResult
from .modes import ConditionalSingleMode, SegmentationMode, TrainingMode
from .strategies import DDPMStrategy, RFlowStrategy, DiffusionStrategy
from .spaces import DiffusionSpace, PixelSpace, LatentSpace
from .utils import (
    get_vram_usage,
    log_vram_to_tensorboard,
    save_full_checkpoint,
)
from .metrics import (
    compute_psnr,
    compute_lpips_3d,
    compute_kid_3d,
    compute_cmmd_3d,
    ResNet50Features,
    BiomedCLIPFeatures,
    volumes_to_slices,
    extract_features_3d,
    compute_kid,
    compute_cmmd,
)
from .controlnet import (
    create_controlnet_for_unet,
    freeze_unet_for_controlnet,
    ControlNetConditionedUNet,
    ControlNetGenerationWrapper,
    load_controlnet_checkpoint,
)

logger = logging.getLogger(__name__)


class Diffusion3DTrainer(BaseTrainer):
    """3D Volumetric diffusion model trainer.

    Supports both pixel-space and latent-space training on 3D volumes.

    Key features:
    - Handles 5D tensors [B, C, D, H, W]
    - Memory-optimized (gradient checkpointing, batch_size=1)
    - 2D metrics (LPIPS, KID, CMMD, FID) computed slice-wise (2.5D)
    - Visualizations show center slices

    Args:
        cfg: Hydra configuration object.
        space: DiffusionSpace for pixel/latent operations. Defaults to PixelSpace.
    """

    def __init__(self, cfg: DictConfig, space: Optional[DiffusionSpace] = None) -> None:
        super().__init__(cfg)

        self.space = space if space is not None else PixelSpace()
        self.spatial_dims = 3

        # Volume dimensions
        self.volume_height = cfg.volume.height
        self.volume_width = cfg.volume.width
        self.volume_depth = cfg.volume.depth

        # Diffusion config
        self.strategy_name: str = cfg.strategy.name
        self.mode_name: str = cfg.mode.name
        self.num_timesteps: int = cfg.strategy.num_train_timesteps

        # Training config
        self.use_gradient_checkpointing: bool = cfg.training.get('gradient_checkpointing', True)

        # Initialize strategy and mode
        self.strategy = self._create_strategy(self.strategy_name)
        self.mode = self._create_mode(self.mode_name)

        # Setup scheduler with 3D dimensions
        self.scheduler = self.strategy.setup_scheduler(
            num_timesteps=self.num_timesteps,
            image_size=self.volume_height,
            depth_size=self.volume_depth,
            spatial_dims=3,
        )

        # EMA
        self.use_ema: bool = cfg.training.get('use_ema', True)
        self.ema_decay: float = cfg.training.get('ema', {}).get('decay', 0.9999)
        self.ema: Optional[EMA] = None

        # Model components (set in setup_model)
        self.model_raw: Optional[nn.Module] = None
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[AdamW] = None
        self.lr_scheduler: Optional[Any] = None

        # 3D MS-SSIM metric
        self.ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, win_size=7)

        # Validation loader (set in train())
        self.val_loader: Optional[DataLoader] = None

        # Tracking
        self._global_step: int = 0
        self._current_epoch: int = 0
        self.best_val_loss: float = float('inf')

        # Generation metrics (KID, CMMD) for overfitting detection
        # Defaults match 2D: 10 steps quick, 25 steps extended
        gen_cfg = cfg.training.get('generation_metrics', {})
        self._gen_metrics_enabled = gen_cfg.get('enabled', False)
        self._gen_metrics_samples_epoch = gen_cfg.get('samples_per_epoch', 1)  # Quick: 1 volume
        self._gen_metrics_samples_extended = gen_cfg.get('samples_extended', 4)  # Extended: 4 volumes
        self._gen_metrics_steps_epoch = gen_cfg.get('steps_per_epoch', 10)  # Same as 2D
        self._gen_metrics_steps_extended = gen_cfg.get('steps_extended', 25)  # Same as 2D
        self._gen_metrics_chunk_size = gen_cfg.get('chunk_size', 32)
        self._gen_metrics_interval = cfg.training.get('figure_interval', 10)  # Extended every N epochs

        # Feature extractors (lazy-initialized)
        self._resnet_extractor: Optional[ResNet50Features] = None
        self._biomed_extractor: Optional[BiomedCLIPFeatures] = None

        # Cached reference features from train/val
        self._train_resnet_features: Optional[torch.Tensor] = None
        self._train_biomed_features: Optional[torch.Tensor] = None
        self._val_resnet_features: Optional[torch.Tensor] = None
        self._val_biomed_features: Optional[torch.Tensor] = None

        # Fixed conditioning for generation (pixel-space seg masks)
        self._fixed_conditioning_volumes: Optional[torch.Tensor] = None

        # ControlNet configuration
        controlnet_cfg = cfg.get('controlnet', {})
        self.use_controlnet: bool = controlnet_cfg.get('enabled', False)
        self.controlnet_freeze_unet: bool = controlnet_cfg.get('freeze_unet', True)
        self.controlnet_scale: float = controlnet_cfg.get('conditioning_scale', 1.0)
        self.controlnet: Optional[nn.Module] = None

        if self.use_controlnet:
            logger.info("ControlNet enabled for pixel-space conditioning")
            if self.controlnet_freeze_unet:
                logger.info("UNet will be frozen (Stage 2 ControlNet training)")

        if self._gen_metrics_enabled and self.is_main_process:
            logger.info(
                f"Generation metrics enabled: {self._gen_metrics_samples_epoch} volumes/epoch "
                f"({self._gen_metrics_steps_epoch} steps), "
                f"{self._gen_metrics_samples_extended} volumes/extended "
                f"({self._gen_metrics_steps_extended} steps)"
            )

    def _create_strategy(self, name: str) -> DiffusionStrategy:
        """Create diffusion strategy based on name."""
        if name == 'ddpm':
            return DDPMStrategy()
        elif name == 'rflow':
            return RFlowStrategy()
        else:
            raise ValueError(f"Unknown strategy: {name}")

    def _create_mode(self, name: str) -> TrainingMode:
        """Create training mode based on name."""
        if name == 'seg':
            return SegmentationMode()
        elif name == 'bravo':
            return ConditionalSingleMode()
        else:
            # Default to conditional single for 3D (most common)
            return ConditionalSingleMode()

    # ─────────────────────────────────────────────────────────────────────────
    # Generation Metrics (KID, CMMD) - 2.5D Slice-Wise Approach
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_feature_extractors(self) -> None:
        """Lazy-initialize feature extractors."""
        if self._resnet_extractor is None:
            self._resnet_extractor = ResNet50Features(self.device)
        if self._biomed_extractor is None:
            self._biomed_extractor = BiomedCLIPFeatures(self.device)

    def _cache_reference_features_3d(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        max_volumes: int = 10,
    ) -> None:
        """Cache reference features from train/val volumes slice-wise.

        Extracts features from all slices of reference volumes to build
        the reference distribution for KID/CMMD computation.

        Args:
            train_loader: Training dataloader.
            val_loader: Optional validation dataloader.
            max_volumes: Maximum volumes to use per split.
        """
        self._ensure_feature_extractors()
        logger.info(f"Caching reference features from up to {max_volumes} volumes...")

        # Extract train features
        train_volumes = []
        for i, batch in enumerate(train_loader):
            if i >= max_volumes:
                break
            prepared = self.mode.prepare_batch(batch, self.device)
            images = prepared['images']  # [B, C, D, H, W]
            # Decode if in latent space
            if isinstance(self.space, LatentSpace):
                images = self.space.decode(images)
            train_volumes.append(images.cpu())

        if train_volumes:
            train_vols = torch.cat(train_volumes, dim=0)
            self._train_resnet_features = extract_features_3d(
                train_vols.to(self.device),
                self._resnet_extractor,
                self._gen_metrics_chunk_size,
            )
            self._train_biomed_features = extract_features_3d(
                train_vols.to(self.device),
                self._biomed_extractor,
                self._gen_metrics_chunk_size,
            )
            logger.info(f"Cached train features: {self._train_resnet_features.shape[0]} slices")

        # Extract val features
        if val_loader is not None:
            val_volumes = []
            for i, batch in enumerate(val_loader):
                if i >= max_volumes:
                    break
                prepared = self.mode.prepare_batch(batch, self.device)
                images = prepared['images']
                if isinstance(self.space, LatentSpace):
                    images = self.space.decode(images)
                val_volumes.append(images.cpu())

            if val_volumes:
                val_vols = torch.cat(val_volumes, dim=0)
                self._val_resnet_features = extract_features_3d(
                    val_vols.to(self.device),
                    self._resnet_extractor,
                    self._gen_metrics_chunk_size,
                )
                self._val_biomed_features = extract_features_3d(
                    val_vols.to(self.device),
                    self._biomed_extractor,
                    self._gen_metrics_chunk_size,
                )
                logger.info(f"Cached val features: {self._val_resnet_features.shape[0]} slices")

    def _set_fixed_conditioning_3d(
        self,
        train_loader: DataLoader,
        num_volumes: int = 4,
    ) -> None:
        """Sample fixed conditioning volumes (seg masks) for generation.

        For conditional modes, samples volumes with positive masks.

        Args:
            train_loader: Training dataloader.
            num_volumes: Number of volumes to sample.
        """
        if not self.mode.is_conditional:
            return

        conditioning_masks = []
        for batch in train_loader:
            prepared = self.mode.prepare_batch(batch, self.device)
            labels = prepared.get('labels')  # [B, 1, D, H, W] seg masks

            if labels is not None:
                # Check for positive masks
                for i in range(labels.shape[0]):
                    if labels[i].sum() > 0:
                        conditioning_masks.append(labels[i:i+1].cpu())
                        if len(conditioning_masks) >= num_volumes:
                            break

            if len(conditioning_masks) >= num_volumes:
                break

        if conditioning_masks:
            self._fixed_conditioning_volumes = torch.cat(conditioning_masks, dim=0).to(self.device)
            logger.info(f"Sampled {len(conditioning_masks)} fixed conditioning volumes")
        else:
            logger.warning("No positive conditioning masks found")

    @torch.no_grad()
    def _generate_samples_3d(
        self,
        model: nn.Module,
        num_samples: int,
        num_steps: int,
    ) -> torch.Tensor:
        """Generate 3D samples for metric computation.

        Supports two conditioning modes:
        - ControlNet: Full pixel-resolution conditioning (recommended for latent)
        - Concatenation: Conditioning encoded to latent space

        Args:
            model: Model to use for generation (can be EMA model).
            num_samples: Number of volumes to generate.
            num_steps: Denoising steps.

        Returns:
            Generated volumes [N, C, D, H, W] in pixel space.
        """
        # Determine output shape
        if isinstance(self.space, LatentSpace):
            # Latent space shape
            latent_depth = self.volume_depth // self.space.scale_factor
            latent_height = self.volume_height // self.space.scale_factor
            latent_width = self.volume_width // self.space.scale_factor
            noise_shape = (num_samples, self.space.latent_channels,
                          latent_depth, latent_height, latent_width)
        else:
            noise_shape = (num_samples, 1, self.volume_depth,
                          self.volume_height, self.volume_width)

        noise = torch.randn(noise_shape, device=self.device)

        # Handle conditioning based on mode
        if self.mode.is_conditional and self._fixed_conditioning_volumes is not None:
            # Use fixed conditioning (cycle if needed)
            n_cond = self._fixed_conditioning_volumes.shape[0]
            indices = [i % n_cond for i in range(num_samples)]
            conditioning = self._fixed_conditioning_volumes[indices]  # Pixel-space

            if self.use_controlnet:
                # ControlNet mode: wrap model with pixel-space conditioning
                # Conditioning stays at full resolution
                gen_model = ControlNetGenerationWrapper(model, conditioning)
                model_input = noise  # Just noise, no concatenation
            else:
                # Concatenation mode: encode conditioning to match latent space
                if isinstance(self.space, LatentSpace):
                    conditioning = self.space.encode(conditioning)
                model_input = torch.cat([noise, conditioning], dim=1)
                gen_model = model
        else:
            model_input = noise
            gen_model = model

        # Generate samples
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            samples = self.strategy.generate(
                gen_model,
                model_input,
                num_steps=num_steps,
                device=self.device,
                use_progress_bars=False,
            )

        # Decode if in latent space
        if isinstance(self.space, LatentSpace):
            samples = self.space.decode(samples)

        return torch.clamp(samples, 0, 1)

    @torch.no_grad()
    def _compute_generation_metrics_3d(
        self,
        model: nn.Module,
        epoch: int,
        extended: bool = False,
    ) -> Dict[str, float]:
        """Compute KID and CMMD using 2.5D slice-wise approach.

        Generates samples, extracts slice features, computes metrics
        against cached reference features.

        Args:
            model: Model to use for generation (can be EMA model).
            epoch: Current epoch number.
            extended: If True, use more samples and steps for thorough evaluation.

        Returns:
            Dictionary with KID/CMMD metrics (same key format as 2D).
        """
        if not self._gen_metrics_enabled:
            return {}

        if self._train_resnet_features is None:
            logger.warning("Reference features not cached, skipping generation metrics")
            return {}

        self._ensure_feature_extractors()

        # Select samples and steps based on mode
        if extended:
            num_samples = self._gen_metrics_samples_extended
            num_steps = self._gen_metrics_steps_extended
            prefix = "extended_"
        else:
            num_samples = self._gen_metrics_samples_epoch
            num_steps = self._gen_metrics_steps_epoch
            prefix = ""

        # Generate samples
        samples = self._generate_samples_3d(model, num_samples, num_steps)

        # Extract features slice-wise
        gen_resnet = extract_features_3d(
            samples, self._resnet_extractor, self._gen_metrics_chunk_size
        )
        gen_biomed = extract_features_3d(
            samples, self._biomed_extractor, self._gen_metrics_chunk_size
        )

        # Free samples
        del samples
        torch.cuda.empty_cache()

        results = {}

        # KID vs train (key format matches 2D: {prefix}KID_mean_train)
        kid_train_mean, kid_train_std = compute_kid(
            self._train_resnet_features.to(self.device),
            gen_resnet.to(self.device),
        )
        results[f'{prefix}KID_mean_train'] = kid_train_mean
        results[f'{prefix}KID_std_train'] = kid_train_std

        # CMMD vs train
        cmmd_train = compute_cmmd(
            self._train_biomed_features.to(self.device),
            gen_biomed.to(self.device),
        )
        results[f'{prefix}CMMD_train'] = cmmd_train

        # KID vs val (if available)
        if self._val_resnet_features is not None:
            kid_val_mean, kid_val_std = compute_kid(
                self._val_resnet_features.to(self.device),
                gen_resnet.to(self.device),
            )
            results[f'{prefix}KID_mean_val'] = kid_val_mean
            results[f'{prefix}KID_std_val'] = kid_val_std

            cmmd_val = compute_cmmd(
                self._val_biomed_features.to(self.device),
                gen_biomed.to(self.device),
            )
            results[f'{prefix}CMMD_val'] = cmmd_val

        return results

    def setup_model(self, train_dataset: Dataset) -> None:
        """Setup 3D diffusion model, optimizer, and scheduler.

        Supports two conditioning modes:
        - Concatenation (default): Labels concatenated to UNet input
        - ControlNet: Labels processed at pixel resolution, injected via zero convs

        Args:
            train_dataset: Training dataset for determining model config.
        """
        model_config = self.mode.get_model_config()

        # Get latent channels if in latent space
        if isinstance(self.space, LatentSpace):
            latent_channels = self.space.latent_channels
            out_channels = latent_channels

            if self.use_controlnet:
                # ControlNet handles conditioning separately (pixel space)
                in_channels = latent_channels
            elif self.mode.is_conditional:
                # Concatenation mode: add conditioning channel
                in_channels = latent_channels + 1
            else:
                in_channels = latent_channels
        else:
            in_channels = model_config['in_channels']
            out_channels = model_config['out_channels']
            latent_channels = in_channels  # For ControlNet

        # Create 3D UNet
        logger.info(f"Creating 3D DiffusionModelUNet: in={in_channels}, out={out_channels}")
        self.model_raw = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=self.cfg.model.channels,
            attention_levels=self.cfg.model.attention_levels,
            num_res_blocks=self.cfg.model.num_res_blocks,
            num_head_channels=self.cfg.model.num_head_channels,
        ).to(self.device)

        # Enable gradient checkpointing for memory
        if self.use_gradient_checkpointing:
            if hasattr(self.model_raw, 'enable_gradient_checkpointing'):
                self.model_raw.enable_gradient_checkpointing()
                logger.info("Enabled gradient checkpointing for 3D UNet")

        # Count UNet parameters
        num_params = sum(p.numel() for p in self.model_raw.parameters())
        logger.info(f"3D UNet parameters: {num_params:,}")

        # Setup ControlNet if enabled
        if self.use_controlnet:
            self.controlnet = create_controlnet_for_unet(
                unet=self.model_raw,
                cfg=self.cfg,
                device=self.device,
                spatial_dims=3,
                latent_channels=latent_channels,
            )

            # Load ControlNet checkpoint if provided
            controlnet_checkpoint = self.cfg.get('controlnet', {}).get('checkpoint')
            if controlnet_checkpoint:
                load_controlnet_checkpoint(self.controlnet, controlnet_checkpoint, self.device)

            # Freeze UNet for Stage 2 training
            if self.controlnet_freeze_unet:
                freeze_unet_for_controlnet(self.model_raw)

            # Create combined model for forward pass
            self.model = ControlNetConditionedUNet(
                unet=self.model_raw,
                controlnet=self.controlnet,
                conditioning_scale=self.controlnet_scale,
            )
            logger.info(f"ControlNet conditioning scale: {self.controlnet_scale}")
        else:
            self.model = self.model_raw

        # Setup optimizer with appropriate parameters
        if self.use_controlnet and self.controlnet_freeze_unet:
            # Only train ControlNet parameters
            trainable_params = self.controlnet.parameters()
            logger.info("Optimizer training ControlNet only (UNet frozen)")
        else:
            # Train all parameters
            trainable_params = self.model.parameters()

        self.optimizer = AdamW(
            trainable_params,
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.get('weight_decay', 0.0),
        )

        # Setup LR scheduler
        total_steps = self.cfg.training.epochs * len(train_dataset) // self.cfg.training.batch_size
        self.lr_scheduler = create_warmup_cosine_scheduler(
            self.optimizer,
            warmup_steps=self.cfg.training.warmup_steps,
            total_steps=total_steps,
            eta_min=self.cfg.training.get('eta_min', 1e-6),
        )

        # Setup EMA (on the appropriate model)
        ema_target = self.controlnet if (self.use_controlnet and self.controlnet_freeze_unet) else self.model
        if self.use_ema:
            self.ema = EMA(
                ema_target,
                beta=self.ema_decay,
                update_every=1,
            )
            logger.info(f"EMA enabled with decay={self.ema_decay}")

    def train(
        self,
        train_loader: DataLoader,
        train_dataset: Dataset,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """Main training loop.

        Args:
            train_loader: Training dataloader.
            train_dataset: Training dataset.
            val_loader: Optional validation dataloader.
        """
        self.val_loader = val_loader

        logger.info(f"Starting 3D diffusion training for {self.cfg.training.epochs} epochs")
        logger.info(f"Volume shape: {self.volume_depth}x{self.volume_height}x{self.volume_width}")
        logger.info(f"Strategy: {self.strategy_name}, Mode: {self.mode_name}")
        logger.info(f"Space: {type(self.space).__name__}")

        # Initialize generation metrics (cache reference features, set conditioning)
        if self._gen_metrics_enabled and self.is_main_process:
            logger.info("Initializing 3D generation metrics...")
            # Use extended count for conditioning (need enough for extended metrics)
            self._set_fixed_conditioning_3d(train_loader, num_volumes=self._gen_metrics_samples_extended)
            self._cache_reference_features_3d(train_loader, val_loader, max_volumes=10)

        for epoch in range(self.cfg.training.epochs):
            self._current_epoch = epoch
            epoch_start = time.time()

            # Training epoch
            train_loss = self._train_epoch(train_loader, epoch)

            # Validation
            val_loss = None
            if val_loader is not None and (epoch + 1) % self.cfg.training.val_interval == 0:
                val_loss = self._validate(val_loader, epoch)

                # Save best checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)

            # Save periodic checkpoint
            if (epoch + 1) % self.cfg.training.save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Compute generation metrics (KID, CMMD) - matches 2D trainer pattern
            if self._gen_metrics_enabled and self.is_main_process:
                try:
                    # Use EMA model if available (matches 2D)
                    model_to_use = self.ema.ema_model if self.ema is not None else self.model
                    model_to_use.eval()

                    # Quick metrics every epoch (1 volume)
                    gen_results = self._compute_generation_metrics_3d(model_to_use, epoch, extended=False)
                    for key, value in gen_results.items():
                        self.writer.add_scalar(f'Generation/{key}', value, epoch)

                    # Extended metrics at figure_interval (4 volumes)
                    if (epoch + 1) % self._gen_metrics_interval == 0:
                        ext_results = self._compute_generation_metrics_3d(model_to_use, epoch, extended=True)
                        for key, value in ext_results.items():
                            self.writer.add_scalar(f'Generation/{key}', value, epoch)

                    model_to_use.train()
                except Exception as e:
                    logger.warning(f"Generation metrics computation failed: {e}")

            # Log epoch summary
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{self.cfg.training.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f if val_loss else 'N/A'}, "
                f"Time: {epoch_time:.1f}s"
            )

        logger.info("Training complete!")

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training dataloader.
            epoch: Current epoch number.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not self.is_main_process)
        for batch in pbar:
            result = self._train_step(batch)
            total_loss += result.total_loss
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{result.total_loss:.4f}'})

            # Log to TensorBoard
            if self._global_step % 100 == 0 and self.is_main_process:
                self.writer.add_scalar('Loss/MSE_train', result.mse_loss, self._global_step)
                self.writer.add_scalar('LR/Generator', self.optimizer.param_groups[0]['lr'], self._global_step)

            self._global_step += 1

        avg_loss = total_loss / max(num_batches, 1)
        if self.is_main_process:
            self.writer.add_scalar('Loss/Total_train_epoch', avg_loss, epoch)

        return avg_loss

    def _train_step(self, batch: Any) -> TrainingStepResult:
        """Execute single training step.

        Supports two conditioning modes:
        - ControlNet: labels at pixel-space, processed via ControlNet
        - Concatenation: labels encoded to latent space, concatenated to input

        Args:
            batch: Batch from dataloader (latent format).

        Returns:
            TrainingStepResult with loss values.
        """
        # Prepare batch
        prepared = self.mode.prepare_batch(batch, self.device)
        images = prepared['images']  # [B, C, D, H, W] - latent or pixel space
        labels = prepared.get('labels')  # [B, 1, D, H, W] - pixel space from cache

        # Sample noise and timesteps
        noise = torch.randn_like(images)
        timesteps = self.strategy.sample_timesteps(images)

        # Add noise to get noisy images
        noisy_images = self.strategy.add_noise(images, noise, timesteps)

        # Forward pass with mixed precision
        self.optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            if self.use_controlnet and labels is not None:
                # ControlNet mode: labels at pixel-space, injected via ControlNet
                # The ControlNetConditionedUNet handles the residual injection
                prediction = self.model(
                    x=noisy_images,
                    timesteps=timesteps,
                    controlnet_cond=labels,  # Pixel-space conditioning
                )
            else:
                # Concatenation mode: labels must match image space
                if labels is not None:
                    # Encode labels to latent space if needed
                    if isinstance(self.space, LatentSpace):
                        labels = self.space.encode(labels)
                    model_input = torch.cat([noisy_images, labels], dim=1)
                else:
                    model_input = noisy_images

                prediction = self.model(x=model_input, timesteps=timesteps)

            mse_loss, _ = self.strategy.compute_loss(
                prediction, images, noise, noisy_images, timesteps
            )

        # Backward pass
        self.scaler.scale(mse_loss).backward()
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping (on trainable parameters only)
        if self.use_controlnet and self.controlnet_freeze_unet:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.controlnet.parameters(), 1.0)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # LR scheduler step
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # EMA update
        if self.ema is not None:
            self.ema.update()

        return TrainingStepResult(
            total_loss=mse_loss.item(),
            mse_loss=mse_loss.item(),
        )

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader, epoch: int) -> float:
        """Run validation.

        Args:
            val_loader: Validation dataloader.
            epoch: Current epoch number.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        num_batches = 0

        # Check if LPIPS logging is enabled (default True)
        compute_lpips = self.cfg.training.get('logging', {}).get('lpips', True)

        for batch in tqdm(val_loader, desc="Validation", disable=not self.is_main_process):
            prepared = self.mode.prepare_batch(batch, self.device)
            images = prepared['images']
            labels = prepared.get('labels')

            # Sample noise and timesteps
            noise = torch.randn_like(images)
            timesteps = self.strategy.sample_timesteps(images)
            noisy_images = self.strategy.add_noise(images, noise, timesteps)

            # Forward pass with appropriate conditioning
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                if self.use_controlnet and labels is not None:
                    # ControlNet mode: pixel-space conditioning
                    prediction = self.model(
                        x=noisy_images,
                        timesteps=timesteps,
                        controlnet_cond=labels,
                    )
                else:
                    # Concatenation mode
                    if labels is not None:
                        if isinstance(self.space, LatentSpace):
                            labels = self.space.encode(labels)
                        model_input = torch.cat([noisy_images, labels], dim=1)
                    else:
                        model_input = noisy_images
                    prediction = self.model(x=model_input, timesteps=timesteps)

                mse_loss, predicted_clean = self.strategy.compute_loss(
                    prediction, images, noise, noisy_images, timesteps
                )

            total_loss += mse_loss.item()

            # Decode if in latent space for quality metrics
            if isinstance(self.space, LatentSpace):
                images_pixel = self.space.decode(images)
                predicted_pixel = self.space.decode(predicted_clean)
            else:
                images_pixel = images
                predicted_pixel = predicted_clean

            # Compute PSNR (works for any dim)
            psnr = compute_psnr(predicted_pixel, images_pixel)
            total_psnr += psnr

            # Compute 3D MS-SSIM
            ssim = self.ssim_metric(predicted_pixel, images_pixel).mean().item()
            total_ssim += ssim

            # Compute LPIPS using 2.5D slice-wise approach
            # Reshape [B, C, D, H, W] → [B*D, C, H, W] internally
            if compute_lpips:
                lpips_val = compute_lpips_3d(predicted_pixel, images_pixel, device=self.device)
                total_lpips += lpips_val

            num_batches += 1

        # Average metrics
        avg_loss = total_loss / max(num_batches, 1)
        avg_psnr = total_psnr / max(num_batches, 1)
        avg_ssim = total_ssim / max(num_batches, 1)
        avg_lpips = total_lpips / max(num_batches, 1) if compute_lpips else 0.0

        # Log to TensorBoard
        if self.is_main_process:
            self.writer.add_scalar('Loss/MSE_val', avg_loss, epoch)
            self.writer.add_scalar('Validation/PSNR', avg_psnr, epoch)
            self.writer.add_scalar('Validation/MS-SSIM', avg_ssim, epoch)
            if compute_lpips:
                self.writer.add_scalar('Validation/LPIPS', avg_lpips, epoch)

            # Generate and visualize samples
            self._visualize_samples(epoch)

        return avg_loss

    @torch.no_grad()
    def _visualize_samples(self, epoch: int) -> None:
        """Generate and visualize 3D samples (center slices).

        Args:
            epoch: Current epoch number.
        """
        if not self.is_main_process:
            return

        self.model.eval()

        # Generate from noise
        batch_size = 2
        if isinstance(self.space, LatentSpace):
            # Generate in latent space
            latent_shape = (batch_size, self.space.latent_channels,
                           self.volume_depth // 8, self.volume_height // 8, self.volume_width // 8)
            noise = torch.randn(latent_shape, device=self.device)
        else:
            noise = torch.randn(batch_size, 1, self.volume_depth, self.volume_height, self.volume_width,
                               device=self.device)

        # Add conditioning if needed (zeros for unconditional generation)
        if self.mode.is_conditional:
            cond_shape = list(noise.shape)
            cond_shape[1] = 1
            conditioning = torch.zeros(cond_shape, device=self.device)
            model_input = torch.cat([noise, conditioning], dim=1)
        else:
            model_input = noise

        # Generate samples
        samples = self.strategy.generate(
            self.model,
            model_input,
            num_steps=25,
            device=self.device,
            use_progress_bars=False,
        )

        # Decode if in latent space
        if isinstance(self.space, LatentSpace):
            samples = self.space.decode(samples)

        # Extract center slice for visualization
        center_idx = samples.shape[2] // 2
        center_slices = samples[:, :, center_idx, :, :]  # [B, C, H, W]

        # Log to TensorBoard
        # Normalize to [0, 1] if needed
        center_slices = torch.clamp(center_slices, 0, 1)

        # Create grid
        from torchvision.utils import make_grid
        grid = make_grid(center_slices, nrow=2, normalize=False)
        self.writer.add_image('Generated_3D_CenterSlice', grid, epoch)

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best model so far.
        """
        if not self.is_main_process:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': OmegaConf.to_container(self.cfg, resolve=True),
            'best_val_loss': self.best_val_loss,
            'spatial_dims': 3,
        }

        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        # Save latest
        latest_path = os.path.join(self.save_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = os.path.join(self.save_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint (val_loss={self.best_val_loss:.4f})")
