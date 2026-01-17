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
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ema_pytorch import EMA
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm


@dataclass
class TrainStepResult:
    """Extended result for 3D training step with grad_norm."""
    total_loss: float
    mse_loss: float
    grad_norm: float
    timesteps: Optional[torch.Tensor] = None  # For timestep bin tracking

from monai.networks.nets import DiffusionModelUNet
from monai.metrics import SSIMMetric

from medgen.core import setup_distributed, create_warmup_cosine_scheduler
from .base_trainer import BaseTrainer
from .results import TrainingStepResult
from .modes import ConditionalSingleMode, SegmentationMode, SegmentationConditionedMode, TrainingMode
from .strategies import DDPMStrategy, RFlowStrategy, DiffusionStrategy
from .spaces import DiffusionSpace, PixelSpace, LatentSpace
from .utils import (
    get_vram_usage,
    log_vram_to_tensorboard,
    save_full_checkpoint,
)
from .tracking import FLOPsTracker
from .metrics import (
    compute_psnr,
    compute_msssim,
    compute_lpips_3d,
    compute_kid_3d,
    compute_cmmd_3d,
    ResNet50Features,
    BiomedCLIPFeatures,
    volumes_to_slices,
    extract_features_3d,
    compute_kid,
    compute_cmmd,
    compute_dice,
    compute_iou,
    # Unified metrics system
    UnifiedMetrics,
    # Regional tracking (still needed for conditional init)
    RegionalMetricsTracker3D,
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
        self.use_amp: bool = True  # Always use automatic mixed precision for 3D
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

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

        # Seg mode detection (use Dice/IoU instead of image quality metrics)
        self.is_seg_mode = self.mode_name in ('seg', 'seg_conditioned_3d')

        # Size bin embedding for seg_conditioned_3d mode
        self.use_size_bin_embedding = (self.mode_name == 'seg_conditioned_3d')
        if self.use_size_bin_embedding:
            size_bin_cfg = cfg.mode.get('size_bins', {})
            self.size_bin_num_bins = size_bin_cfg.get('num_bins', 7)
            self.size_bin_max_count = size_bin_cfg.get('max_count_per_bin', 10)
            self.size_bin_embed_dim = size_bin_cfg.get('embedding_dim', 32)
            if self.is_main_process:
                logger.info(
                    f"Size bin embedding enabled: num_bins={self.size_bin_num_bins}, "
                    f"max_count={self.size_bin_max_count}, embed_dim={self.size_bin_embed_dim}"
                )

        # Logging configuration
        log_cfg = cfg.training.get('logging', {})
        self.log_grad_norm = log_cfg.get('grad_norm', True)
        self.log_timestep_losses = log_cfg.get('timestep_losses', True)
        self.log_worst_batch = log_cfg.get('worst_batch', True)
        self.log_intermediate_steps = log_cfg.get('intermediate_steps', True)
        self.num_intermediate_steps = log_cfg.get('num_intermediate_steps', 5)
        self.log_regional_losses = log_cfg.get('regional_losses', True) and not self.is_seg_mode
        self.log_flops = log_cfg.get('flops', True)
        self.log_lpips = log_cfg.get('lpips', True) and not self.is_seg_mode  # Slice-wise LPIPS

        # FLOPs tracker (measured at start of training)
        self._flops_tracker = FLOPsTracker()

        # Limit batches per epoch (for fast debugging)
        self.limit_train_batches: Optional[int] = cfg.training.get('limit_train_batches', None)
        if self.limit_train_batches is not None and self.is_main_process:
            logger.info(f"Limiting training to {self.limit_train_batches} batches per epoch")

        # Min-SNR weighting (reduces impact of high-noise timesteps)
        self.use_min_snr: bool = cfg.training.get('use_min_snr', False)
        self.min_snr_gamma: float = cfg.training.get('min_snr_gamma', 5.0)
        if self.use_min_snr and self.is_main_process:
            logger.info(f"Min-SNR weighting enabled (gamma={self.min_snr_gamma})")

        # Clean regularization technique logging
        grad_noise_cfg = cfg.training.get('gradient_noise', {})
        if grad_noise_cfg.get('enabled', False) and self.is_main_process:
            logger.info(
                f"Gradient noise enabled: sigma={grad_noise_cfg.get('sigma', 0.01)}, "
                f"decay={grad_noise_cfg.get('decay', 0.55)}"
            )

        curriculum_cfg = cfg.training.get('curriculum', {})
        if curriculum_cfg.get('enabled', False) and self.is_main_process:
            logger.info(
                f"Curriculum timestep scheduling enabled: "
                f"warmup_epochs={curriculum_cfg.get('warmup_epochs', 50)}, "
                f"range [{curriculum_cfg.get('min_t_start', 0.0)}-{curriculum_cfg.get('max_t_start', 0.3)}] -> "
                f"[{curriculum_cfg.get('min_t_end', 0.0)}-{curriculum_cfg.get('max_t_end', 1.0)}]"
            )

        jitter_cfg = cfg.training.get('timestep_jitter', {})
        if jitter_cfg.get('enabled', False) and self.is_main_process:
            logger.info(f"Timestep jitter enabled: std={jitter_cfg.get('std', 0.05)}")

        noise_aug_cfg = cfg.training.get('noise_augmentation', {})
        if noise_aug_cfg.get('enabled', False) and self.is_main_process:
            logger.info(f"Noise augmentation enabled: std={noise_aug_cfg.get('std', 0.1)}")

        # ScoreAug 3D (transforms on noisy data)
        self.score_aug = None
        self.use_omega_conditioning = False
        score_aug_cfg = cfg.training.get('score_aug', {})
        if score_aug_cfg.get('enabled', False):
            from medgen.data.score_aug_3d import ScoreAugTransform3D
            self.score_aug = ScoreAugTransform3D(
                rotation=score_aug_cfg.get('rotation', True),
                flip=score_aug_cfg.get('flip', True),
                translation=score_aug_cfg.get('translation', False),
                cutout=score_aug_cfg.get('cutout', False),
                compose=score_aug_cfg.get('compose', False),
                compose_prob=score_aug_cfg.get('compose_prob', 0.5),
            )
            self.use_omega_conditioning = score_aug_cfg.get('use_omega_conditioning', True)
            if self.is_main_process:
                logger.info(
                    f"ScoreAug 3D enabled (omega_conditioning={self.use_omega_conditioning}, "
                    f"rotation={score_aug_cfg.get('rotation', True)}, "
                    f"flip={score_aug_cfg.get('flip', True)})"
                )

        # SDA 3D (Shifted Data Augmentation - transforms on CLEAN data with shifted timesteps)
        # Note: SDA and ScoreAug are mutually exclusive - use one OR the other
        self.sda = None
        self.sda_weight = 1.0
        sda_cfg = cfg.training.get('sda', {})
        if sda_cfg.get('enabled', False):
            if self.score_aug is not None:
                logger.warning("SDA and ScoreAug are mutually exclusive. Disabling SDA.")
            else:
                from medgen.data.sda_3d import SDATransform3D
                self.sda = SDATransform3D(
                    rotation=sda_cfg.get('rotation', True),
                    flip=sda_cfg.get('flip', True),
                    noise_shift=sda_cfg.get('noise_shift', 0.1),
                    prob=sda_cfg.get('prob', 0.5),
                )
                self.sda_weight = sda_cfg.get('weight', 1.0)
                if self.is_main_process:
                    logger.info(
                        f"SDA 3D enabled (noise_shift={sda_cfg.get('noise_shift', 0.1)}, "
                        f"prob={sda_cfg.get('prob', 0.5)}, weight={self.sda_weight})"
                    )

        # Regional loss weighting (higher weight for small tumors)
        # Only applicable for conditional modes with seg masks
        self.regional_weight_computer = None
        rw_cfg = cfg.training.get('regional_weighting', {})
        if rw_cfg.get('enabled', False) and not self.is_seg_mode:
            from .regional_weighting_3d import RegionalWeightComputer3D
            self.regional_weight_computer = RegionalWeightComputer3D(
                volume_size=(self.volume_height, self.volume_width, self.volume_depth),
                weights=dict(rw_cfg.get('weights', {})) if rw_cfg.get('weights') else None,
                background_weight=rw_cfg.get('background_weight', 1.0),
                fov_mm=rw_cfg.get('fov_mm', 240.0),
            )
            if self.is_main_process:
                weights = self.regional_weight_computer.weights
                logger.info(
                    f"Regional weighting 3D enabled: tiny={weights['tiny']}, "
                    f"small={weights['small']}, medium={weights['medium']}, large={weights['large']}"
                )

        # Timestep bins for per-bin loss tracking
        self.num_timestep_bins = 10

        # Figure interval for extended visualizations
        figure_count = cfg.training.get('figure_count', 20)
        total_epochs = cfg.training.epochs
        self.figure_interval = max(1, total_epochs // figure_count)

        # Unified metrics system (initialized in train() when writer is available)
        self._unified_metrics: Optional[UnifiedMetrics] = None
        self._last_val_metrics: Optional[Dict[str, float]] = None

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
        elif name == 'seg_conditioned_3d':
            size_bin_config = dict(self.cfg.mode.get('size_bins', {})) if 'size_bins' in self.cfg.mode else None
            return SegmentationConditionedMode(size_bin_config)
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
            channels=self.cfg.model.channels,
            attention_levels=self.cfg.model.attention_levels,
            num_res_blocks=self.cfg.model.num_res_blocks,
            num_head_channels=self.cfg.model.num_head_channels,
            norm_num_groups=self.cfg.model.get('norm_num_groups', 32),
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

        # Handle size bin embedding for seg_conditioned_3d mode
        if self.use_size_bin_embedding:
            from medgen.data import SizeBinModelWrapper

            # Get time_embed_dim from model
            channels = tuple(self.cfg.model.channels)
            time_embed_dim = 4 * channels[0]

            # Wrap with size bin embedding
            size_bin_wrapper = SizeBinModelWrapper(
                model=self.model_raw,
                embed_dim=time_embed_dim,
                num_bins=self.size_bin_num_bins,
                max_count=self.size_bin_max_count,
                per_bin_embed_dim=self.size_bin_embed_dim,
            ).to(self.device)

            if self.is_main_process:
                logger.info(
                    f"Size bin embedding: wrapper applied (num_bins={self.size_bin_num_bins}, "
                    f"embed_dim={time_embed_dim})"
                )

            self.model = size_bin_wrapper
            self.model_raw = size_bin_wrapper

        # Wrap with ScoreAug omega conditioning if enabled
        if self.use_omega_conditioning and self.score_aug is not None:
            from medgen.data.score_aug_3d import ScoreAugModelWrapper3D
            channels = tuple(self.cfg.model.channels)
            time_embed_dim = 4 * channels[0]
            self.model = ScoreAugModelWrapper3D(self.model_raw, embed_dim=time_embed_dim)
            if self.is_main_process:
                logger.info(f"Wrapped model with ScoreAugModelWrapper3D (embed_dim={time_embed_dim})")

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
        warmup_epochs = self.cfg.training.get('warmup_epochs', 5)
        total_epochs = self.cfg.training.epochs
        self.lr_scheduler = create_warmup_cosine_scheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
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

        # torch.compile for model optimization
        # Note: Incompatible with gradient_checkpointing - must disable one or the other
        use_compile = self.cfg.training.get('use_compile', False)
        if use_compile:
            if self.use_gradient_checkpointing:
                logger.warning(
                    "torch.compile is incompatible with gradient_checkpointing. "
                    "Disabling torch.compile. Set training.gradient_checkpointing=false to enable compilation."
                )
            else:
                logger.info("Compiling 3D UNet with torch.compile (mode=default)...")
                self.model = torch.compile(self.model, mode="default")

    def _measure_model_flops(self, train_loader: DataLoader) -> None:
        """Measure model FLOPs using batch_size=1 to avoid OOM during profiling.

        Args:
            train_loader: Training dataloader to get sample input.
        """
        if not self.log_flops:
            return

        try:
            batch = next(iter(train_loader))
            prepared = self.mode.prepare_batch(batch, self.device)
            images = prepared['images']
            labels = prepared.get('labels')

            # Slice to batch_size=1 for profiling
            images = images[:1]
            noise = torch.randn_like(images).to(self.device)

            if labels is not None:
                labels = labels[:1]

            timesteps = self.strategy.sample_timesteps(images)
            noisy_images = self.strategy.add_noise(images, noise, timesteps)

            # Build model input based on conditioning mode
            if self.use_controlnet and labels is not None:
                # ControlNet: model takes noisy_images, controlnet gets labels
                model_input = noisy_images
            elif labels is not None:
                # Concatenation mode
                if isinstance(self.space, LatentSpace):
                    labels = self.space.encode(labels)
                model_input = torch.cat([noisy_images, labels], dim=1)
            else:
                model_input = noisy_images

            self._flops_tracker.measure(
                self.model_raw if hasattr(self, 'model_raw') else self.model,
                model_input,
                steps_per_epoch=len(train_loader),
                timesteps=timesteps,
                is_main_process=self.is_main_process,
            )
        except Exception as e:
            if self.is_main_process:
                logger.warning(f"Could not measure FLOPs: {e}")

    def _compute_snr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute Min-SNR loss weights for given timesteps.

        For RFlow (our primary strategy), uses t/(1-t) as SNR approximation.
        For DDPM, uses alpha_bar/(1-alpha_bar) from the scheduler.

        Args:
            timesteps: Tensor of timestep values (normalized to [0, 1] for RFlow).

        Returns:
            Tensor of SNR-based loss weights [B].
        """
        if self.strategy_name == 'ddpm' and self.scheduler is not None:
            # DDPM: use alpha_bar from scheduler
            alphas_cumprod = self.scheduler.alphas_cumprod.to(timesteps.device)
            # timesteps are indices for DDPM
            alpha_bar = alphas_cumprod[timesteps.long()]
            snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
        else:
            # RFlow: timesteps are in [0, 1], use (1-t)/t as SNR
            t_normalized = timesteps.float()
            snr = (1.0 - t_normalized) / (t_normalized + 1e-8)

        # Clip SNR and compute weight: min(SNR, gamma) / SNR
        snr_clipped = torch.clamp(snr, max=self.min_snr_gamma)
        weights = snr_clipped / (snr + 1e-8)

        return weights

    def _compute_min_snr_weighted_mse(
        self,
        prediction: torch.Tensor,
        images: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss with Min-SNR weighting.

        Applies per-sample SNR-based weights to prevent high-noise timesteps
        from dominating training.

        Args:
            prediction: Model prediction (velocity for RFlow).
            images: Original clean images [B, C, D, H, W].
            noise: Added noise [B, C, D, H, W].
            timesteps: Diffusion timesteps for each sample [B].

        Returns:
            Weighted MSE loss scalar.
        """
        snr_weights = self._compute_snr_weights(timesteps)

        # Target is velocity (image - noise) for RFlow, or noise for DDPM
        if self.strategy_name == 'rflow':
            target = images - noise
        else:
            target = noise

        # Compute per-sample MSE (mean over spatial dims, keep batch dim)
        # Cast to FP32 to avoid BF16 underflow
        mse_per_sample = ((prediction.float() - target.float()) ** 2).mean(dim=(1, 2, 3, 4))

        # Apply SNR weights and reduce
        weighted_mse = (mse_per_sample * snr_weights).mean()

        return weighted_mse

    def _add_gradient_noise(self, step: int) -> None:
        """Add Gaussian noise to gradients for regularization.

        Implements the method from "Adding Gradient Noise Improves Learning
        for Very Deep Networks" (Neelakantan et al., 2015).

        Args:
            step: Current global training step.
        """
        grad_noise_cfg = self.cfg.training.get('gradient_noise', {})
        if not grad_noise_cfg.get('enabled', False):
            return

        sigma = grad_noise_cfg.get('sigma', 0.01)
        decay = grad_noise_cfg.get('decay', 0.55)

        # Decaying noise: sigma / (1 + step)^decay
        noise_scale = sigma / ((1 + step) ** decay)

        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad.add_(noise)

    def _get_curriculum_range(self, epoch: int) -> Optional[tuple]:
        """Get timestep range for curriculum learning.

        Linearly interpolates from start range to end range over warmup_epochs.

        Args:
            epoch: Current epoch.

        Returns:
            Tuple of (min_t, max_t) or None if curriculum disabled.
        """
        curriculum_cfg = self.cfg.training.get('curriculum', {})
        if not curriculum_cfg.get('enabled', False):
            return None

        warmup_epochs = curriculum_cfg.get('warmup_epochs', 50)
        progress = min(1.0, epoch / warmup_epochs)

        # Linear interpolation from start to end range
        min_t_start = curriculum_cfg.get('min_t_start', 0.0)
        min_t_end = curriculum_cfg.get('min_t_end', 0.0)
        max_t_start = curriculum_cfg.get('max_t_start', 0.3)
        max_t_end = curriculum_cfg.get('max_t_end', 1.0)

        min_t = min_t_start + progress * (min_t_end - min_t_start)
        max_t = max_t_start + progress * (max_t_end - max_t_start)

        return (min_t, max_t)

    def _apply_timestep_jitter(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to timesteps for regularization.

        Creates noise-level diversity during training, helping the model
        generalize across slightly different noise levels.

        Args:
            timesteps: Original timesteps [B].

        Returns:
            Jittered timesteps (clamped to valid range).
        """
        jitter_cfg = self.cfg.training.get('timestep_jitter', {})
        if not jitter_cfg.get('enabled', False):
            return timesteps

        std = jitter_cfg.get('std', 0.05)

        # Add Gaussian noise to normalized timesteps
        t_normalized = timesteps.float()
        t_jittered = t_normalized + torch.randn_like(t_normalized) * std

        return t_jittered.clamp(0.0, 1.0)

    def _apply_noise_augmentation(self, noise: torch.Tensor) -> torch.Tensor:
        """Perturb noise vector for diversity.

        Adds small perturbation and renormalizes to maintain variance.

        Args:
            noise: Original noise tensor [B, C, D, H, W].

        Returns:
            Perturbed noise (renormalized to maintain variance).
        """
        noise_aug_cfg = self.cfg.training.get('noise_augmentation', {})
        if not noise_aug_cfg.get('enabled', False):
            return noise

        std = noise_aug_cfg.get('std', 0.1)

        # Add perturbation and renormalize
        perturbed = noise + torch.randn_like(noise) * std
        return perturbed / perturbed.std() * noise.std()

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

        # Initialize unified metrics system
        self._unified_metrics = UnifiedMetrics(
            writer=self.writer,
            mode=self.mode_name,
            spatial_dims=3,
            modality=self.mode_name if self.mode_name not in ('seg', 'seg_conditioned_3d') else None,
            device=self.device,
            enable_regional=self.log_regional_losses and not self.is_seg_mode,
            num_timestep_bins=self.num_timestep_bins,
            fov_mm=self.cfg.paths.get('fov_mm', 240.0),
            volume_size=(self.volume_height, self.volume_width, self.volume_depth),
        )

        # Measure FLOPs at start of training
        self._measure_model_flops(train_loader)

        # Initialize generation metrics (cache reference features, set conditioning)
        # Skip for seg modes - feature extractors don't work on binary masks
        if self._gen_metrics_enabled and self.is_main_process and not self.is_seg_mode:
            logger.info("Initializing 3D generation metrics...")
            # Use extended count for conditioning (need enough for extended metrics)
            self._set_fixed_conditioning_3d(train_loader, num_volumes=self._gen_metrics_samples_extended)
            self._cache_reference_features_3d(train_loader, val_loader, max_volumes=10)
        elif self._gen_metrics_enabled and self.is_seg_mode and self.is_main_process:
            logger.info(f"{self.mode_name} mode: generation metrics disabled (binary masks)")

        for epoch in range(self.cfg.training.epochs):
            self._current_epoch = epoch
            epoch_start = time.time()

            # Training epoch
            train_loss = self._train_epoch(train_loader, epoch)

            # Validation (every epoch by default)
            val_loss = None
            val_interval = self.cfg.training.get('val_interval', 1)
            if val_loader is not None and (epoch + 1) % val_interval == 0:
                # Sync CUDA and clear cache before validation to prevent worker issues
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                val_loss = self._validate(val_loader, epoch)

                # Save best checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)

            # Save periodic checkpoint (every 10 epochs by default)
            save_interval = self.cfg.training.get('save_interval', 10)
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Compute generation metrics (KID, CMMD) - matches 2D trainer pattern
            # Skip for seg modes - feature extractors don't work on binary masks
            if self._gen_metrics_enabled and self.is_main_process and not self.is_seg_mode:
                try:
                    # Use EMA model if available (matches 2D)
                    model_to_use = self.ema.ema_model if self.ema is not None else self.model
                    model_to_use.eval()

                    # Quick metrics every epoch (1 volume)
                    gen_results = self._compute_generation_metrics_3d(model_to_use, epoch, extended=False)
                    self._unified_metrics.log_generation(epoch, gen_results)

                    # Extended metrics at figure_interval (4 volumes)
                    if (epoch + 1) % self._gen_metrics_interval == 0:
                        ext_results = self._compute_generation_metrics_3d(model_to_use, epoch, extended=True)
                        self._unified_metrics.log_generation(epoch, ext_results)

                    model_to_use.train()
                except Exception as e:
                    logger.warning(f"Generation metrics computation failed: {e}")

            # Log epoch summary using unified system
            epoch_time = time.time() - epoch_start
            self._unified_metrics.log_console_summary(
                epoch=epoch,
                total_epochs=self.cfg.training.epochs,
                elapsed_time=epoch_time,
            )

            # Log FLOPs (VRAM already logged via update_vram in _train_epoch)
            if self.is_main_process:
                self._flops_tracker.log_epoch(self.writer, epoch)

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
        self._unified_metrics.reset_training()

        # Worst batch tracking
        worst_batch_loss = 0.0
        worst_batch_data: Optional[Dict[str, Any]] = None

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not self.is_main_process)
        for batch_idx, batch in enumerate(pbar):
            # Limit batches per epoch (for fast debugging)
            if self.limit_train_batches is not None and batch_idx >= self.limit_train_batches:
                break

            result = self._train_step(batch)

            # Accumulate losses using unified system
            self._unified_metrics.update_loss('Total', result.total_loss)
            self._unified_metrics.update_loss('MSE', result.mse_loss)
            self._unified_metrics.update_grad_norm(result.grad_norm)

            # Update progress bar
            pbar.set_postfix({'loss': f'{result.total_loss:.4f}'})

            # Track worst batch (for visualization)
            if self.log_worst_batch and result.total_loss > worst_batch_loss:
                worst_batch_loss = result.total_loss
                # Store minimal info for worst batch
                prepared = self.mode.prepare_batch(batch, self.device)
                worst_batch_data = {
                    'images': prepared['images'][:2].detach().cpu(),  # Keep first 2 samples
                    'loss': result.total_loss,
                }

            self._global_step += 1

        # Log training metrics using unified system
        if self.is_main_process:
            self._unified_metrics.update_lr(self.optimizer.param_groups[0]['lr'])
            self._unified_metrics.update_vram()
            self._unified_metrics.log_training(epoch)

            # Log worst batch visualization (custom diagnostic)
            if self.log_worst_batch and worst_batch_data is not None and (epoch + 1) % self.figure_interval == 0:
                self._visualize_worst_batch(worst_batch_data, epoch, prefix='train')

        return self._unified_metrics.get_training_losses().get('Total', 0.0)

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
        size_bins = prepared.get('size_bins')  # [B, num_bins] - for seg_conditioned_3d mode

        # Sample noise (with optional augmentation)
        noise = torch.randn_like(images)
        noise = self._apply_noise_augmentation(noise)

        # Sample timesteps (with optional curriculum learning)
        curriculum_range = self._get_curriculum_range(self._current_epoch)
        timesteps = self.strategy.sample_timesteps(images, curriculum_range)

        # Apply timestep jitter (adds noise-level diversity)
        timesteps = self._apply_timestep_jitter(timesteps)

        # Add noise to get noisy images
        noisy_images = self.strategy.add_noise(images, noise, timesteps)

        # ScoreAug: apply transforms to noisy data (after noise addition)
        omega = None
        target = None
        if self.score_aug is not None:
            # Compute target before ScoreAug (velocity for rflow, noise for ddpm)
            if self.strategy_name == 'rflow':
                target = images - noise
            else:
                target = noise

            # Build noisy input (with optional labels concat for transformation)
            if labels is not None and not self.use_controlnet and not self.use_size_bin_embedding:
                if isinstance(self.space, LatentSpace):
                    labels_for_aug = self.space.encode(labels)
                else:
                    labels_for_aug = labels
                noisy_input = torch.cat([noisy_images, labels_for_aug], dim=1)
            else:
                noisy_input = noisy_images

            # Apply ScoreAug to noisy input and target
            aug_input, aug_target, omega = self.score_aug(noisy_input, target)

            # Split back if labels were concatenated
            if labels is not None and not self.use_controlnet and not self.use_size_bin_embedding:
                noisy_images = aug_input[:, :images.shape[1]]
                labels_encoded = aug_input[:, images.shape[1]:]
            else:
                noisy_images = aug_input
                labels_encoded = None

            target = aug_target
        else:
            labels_encoded = None

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
            elif self.use_size_bin_embedding:
                # Size bin embedding mode: model is SizeBinModelWrapper
                # Conditioning via embedding (added to timestep), NOT concatenation
                prediction = self.model(noisy_images, timesteps, size_bins=size_bins)
            elif self.use_omega_conditioning and omega is not None:
                # ScoreAug omega conditioning mode
                if labels_encoded is not None:
                    model_input = torch.cat([noisy_images, labels_encoded], dim=1)
                else:
                    model_input = noisy_images
                prediction = self.model(model_input, timesteps, omega=omega)
            else:
                # Standard concatenation mode
                if labels is not None:
                    # Encode labels to latent space if needed
                    if labels_encoded is None:
                        if isinstance(self.space, LatentSpace):
                            labels_encoded = self.space.encode(labels)
                        else:
                            labels_encoded = labels
                    model_input = torch.cat([noisy_images, labels_encoded], dim=1)
                else:
                    model_input = noisy_images

                prediction = self.model(x=model_input, timesteps=timesteps)

            # Compute loss with optional Min-SNR weighting and regional weighting
            # Use augmented target if ScoreAug was applied
            if self.score_aug is not None and target is not None:
                # Direct MSE against augmented target
                mse_loss = F.mse_loss(prediction.float(), target.float())
            elif self.regional_weight_computer is not None and labels is not None:
                # Regional weighting: higher weight for small tumors
                # Compute target (velocity for rflow)
                if self.strategy_name == 'rflow':
                    rw_target = images - noise
                else:
                    rw_target = noise

                # Compute per-voxel weights based on tumor sizes
                weight_map = self.regional_weight_computer(labels)  # [B, 1, D, H, W]

                # Compute weighted MSE
                per_voxel_mse = (prediction.float() - rw_target.float()) ** 2
                mse_loss = (per_voxel_mse * weight_map).mean()
            elif self.use_min_snr:
                mse_loss = self._compute_min_snr_weighted_mse(
                    prediction, images, noise, timesteps
                )
            else:
                mse_loss, _ = self.strategy.compute_loss(
                    prediction, images, noise, noisy_images, timesteps
                )

            # SDA: Shifted Data Augmentation - additional loss on augmented clean data
            total_loss = mse_loss
            if self.sda is not None:
                # Apply SDA to clean images (before noise)
                aug_images, transform_info = self.sda(images)

                if transform_info is not None:
                    # Shift timesteps for augmented path
                    shifted_t = self.sda.shift_timesteps(timesteps)

                    # Add noise at shifted timestep
                    aug_noisy = self.strategy.add_noise(aug_images, noise, shifted_t)

                    # Compute target for augmented path (velocity for rflow)
                    if self.strategy_name == 'rflow':
                        aug_target = aug_images - noise
                    else:
                        aug_target = noise
                    # Transform target to match augmented images
                    aug_target = self.sda.apply_to_target(aug_target, transform_info)

                    # Forward pass with augmented data
                    if labels is not None and not self.use_controlnet and not self.use_size_bin_embedding:
                        if isinstance(self.space, LatentSpace):
                            labels_sda = self.space.encode(labels)
                        else:
                            labels_sda = labels
                        # Apply same transform to labels
                        labels_sda = self.sda.apply_to_target(labels_sda, transform_info)
                        aug_input = torch.cat([aug_noisy, labels_sda], dim=1)
                    else:
                        aug_input = aug_noisy

                    aug_pred = self.model(x=aug_input, timesteps=shifted_t)
                    sda_loss = F.mse_loss(aug_pred.float(), aug_target.float())

                    # Combine losses
                    total_loss = mse_loss + self.sda_weight * sda_loss

        # Backward pass
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping (on trainable parameters only)
        if self.use_controlnet and self.controlnet_freeze_unet:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.controlnet.parameters(), 1.0)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Add gradient noise (for regularization, decays over training)
        self._add_gradient_noise(self._global_step)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # LR scheduler step
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # EMA update
        if self.ema is not None:
            self.ema.update()

        # Return extended result with grad_norm and timesteps
        return TrainStepResult(
            total_loss=total_loss.item(),  # Total loss (MSE + SDA if enabled)
            mse_loss=mse_loss.item(),
            grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            timesteps=timesteps.detach(),  # Keep for timestep bin tracking
        )

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader, epoch: int) -> float:
        """Run validation.

        For seg modes (seg, seg_conditioned_3d): computes Dice/IoU
        For image modes (bravo, etc.): computes PSNR/MS-SSIM

        Args:
            val_loader: Validation dataloader.
            epoch: Current epoch number.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        self._unified_metrics.reset_validation()
        num_batches = 0

        # Worst batch tracking
        worst_batch_loss = 0.0
        worst_batch_data: Optional[Dict[str, Any]] = None

        for batch in tqdm(val_loader, desc="Validation", disable=not self.is_main_process):
            prepared = self.mode.prepare_batch(batch, self.device)
            images = prepared['images']
            labels = prepared.get('labels')
            size_bins = prepared.get('size_bins')  # For seg_conditioned_3d mode

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
                elif self.use_size_bin_embedding:
                    # Size bin embedding mode
                    prediction = self.model(noisy_images, timesteps, size_bins=size_bins)
                else:
                    # Concatenation mode
                    if labels is not None:
                        if isinstance(self.space, LatentSpace):
                            labels_encoded = self.space.encode(labels)
                            model_input = torch.cat([noisy_images, labels_encoded], dim=1)
                        else:
                            model_input = torch.cat([noisy_images, labels], dim=1)
                    else:
                        model_input = noisy_images
                    prediction = self.model(x=model_input, timesteps=timesteps)

                mse_loss, predicted_clean = self.strategy.compute_loss(
                    prediction, images, noise, noisy_images, timesteps
                )

            batch_loss = mse_loss.item()
            self._unified_metrics.update_loss('MSE', batch_loss, phase='val')

            # Track timestep losses using unified system
            if self.log_timestep_losses:
                # Use average timestep for the batch
                avg_t = timesteps.float().mean().item()
                self._unified_metrics.update_timestep_loss(avg_t, batch_loss)

            # Track worst batch
            if self.log_worst_batch and batch_loss > worst_batch_loss:
                worst_batch_loss = batch_loss
                worst_batch_data = {
                    'images': images[:2].detach().cpu(),
                    'loss': batch_loss,
                }

            # Decode if in latent space for quality metrics
            if isinstance(self.space, LatentSpace):
                images_pixel = self.space.decode(images)
                predicted_pixel = self.space.decode(predicted_clean)
            else:
                images_pixel = images
                predicted_pixel = predicted_clean

            if self.is_seg_mode:
                # Seg mode: compute Dice/IoU using unified metrics
                self._unified_metrics.update_dice(predicted_pixel, images_pixel)
                self._unified_metrics.update_iou(predicted_pixel, images_pixel)
            else:
                # Image mode: compute PSNR/MS-SSIM/LPIPS using unified metrics
                self._unified_metrics.update_psnr(predicted_pixel, images_pixel)
                self._unified_metrics.update_msssim(predicted_pixel, images_pixel)
                self._unified_metrics.update_msssim_3d(predicted_pixel, images_pixel)

                # Slice-wise LPIPS (2.5D approach, consistent with KID/CMMD)
                if self.log_lpips:
                    self._unified_metrics.update_lpips(predicted_pixel, images_pixel)

                # Regional metrics tracking using unified system
                if self.log_regional_losses and labels is not None:
                    self._unified_metrics.update_regional(predicted_pixel, images_pixel, labels)

            num_batches += 1

        # Get validation metrics for return value and epoch summary
        val_metrics = self._unified_metrics.get_validation_metrics()
        self._last_val_metrics = val_metrics
        avg_loss = val_metrics.get('MSE', 0.0)

        # Log to TensorBoard
        if self.is_main_process:
            # Log validation metrics using unified system
            self._unified_metrics.log_validation(epoch)

            # Log worst batch visualization at figure_interval
            if self.log_worst_batch and worst_batch_data is not None and (epoch + 1) % self.figure_interval == 0:
                self._visualize_worst_batch(worst_batch_data, epoch, prefix='val')

            # Log info message for validation
            if self.is_seg_mode:
                dice = val_metrics.get('Dice', 0.0)
                iou = val_metrics.get('IoU', 0.0)
                logger.info(f"Validation - MSE: {avg_loss:.4f}, Dice: {dice:.4f}, IoU: {iou:.4f}")
            else:
                psnr = val_metrics.get('PSNR', 0.0)
                msssim = val_metrics.get('MS-SSIM', 0.0)
                msssim_3d = val_metrics.get('MS-SSIM-3D', 0.0)
                lpips_str = f", LPIPS: {val_metrics.get('LPIPS', 0.0):.4f}" if self.log_lpips else ""
                logger.info(f"Validation - MSE: {avg_loss:.4f}, PSNR: {psnr:.2f}, MS-SSIM: {msssim:.4f}, MS-SSIM-3D: {msssim_3d:.4f}{lpips_str}")

            # Generate and visualize samples
            self._visualize_samples(epoch)

            # Visualize denoising trajectory at figure_interval
            if self.log_intermediate_steps and (epoch + 1) % self.figure_interval == 0:
                self._visualize_denoising_trajectory(epoch, num_steps=self.num_intermediate_steps)

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

        # Add conditioning if needed
        # Note: seg_conditioned_3d uses embedding-based conditioning (not concatenation)
        if self.use_size_bin_embedding:
            # Size bin embedding mode: no concatenation, just noise
            model_input = noise
            # For visualization, we use zero size_bins (unconditional generation)
            # The model wrapper will use its default behavior
        elif self.mode.is_conditional:
            # Channel concatenation mode
            cond_shape = list(noise.shape)
            cond_shape[1] = 1
            conditioning = torch.zeros(cond_shape, device=self.device)
            model_input = torch.cat([noise, conditioning], dim=1)
        else:
            model_input = noise

        # Generate samples
        # For size_bin_embedding mode, we need a custom generation loop
        if self.use_size_bin_embedding:
            # Generate with zero size_bins (unconditional)
            batch_size = noise.shape[0]
            zero_size_bins = torch.zeros(batch_size, self.size_bin_num_bins, device=self.device)
            samples = self._generate_with_size_bins(noise, zero_size_bins, num_steps=25)
        else:
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

    def _visualize_worst_batch(
        self,
        worst_batch_data: Dict[str, Any],
        epoch: int,
        prefix: str = 'train',
    ) -> None:
        """Visualize the worst-performing batch from training/validation.

        Shows center slices of volumes that had the highest loss.

        Args:
            worst_batch_data: Dict with 'images' tensor and 'loss' value.
            epoch: Current epoch number.
            prefix: Log prefix ('train' or 'val').
        """
        if not self.is_main_process:
            return

        images = worst_batch_data['images']  # [B, C, D, H, W] on CPU
        loss = worst_batch_data['loss']

        # Move to device for consistent processing
        images = images.to(self.device)

        # Decode if in latent space
        if isinstance(self.space, LatentSpace):
            with torch.no_grad():
                images = self.space.decode(images)

        # Extract center slices
        center_idx = images.shape[2] // 2
        center_slices = images[:, :, center_idx, :, :]  # [B, C, H, W]

        # Normalize to [0, 1]
        center_slices = torch.clamp(center_slices, 0, 1)

        # Create grid
        grid = make_grid(center_slices, nrow=2, normalize=False)
        self.writer.add_image(f'worst_batch/{prefix}_loss_{loss:.4f}', grid, epoch)

        # Also log the loss value
        self.writer.add_scalar(f'worst_batch/{prefix}_loss', loss, epoch)

    @torch.no_grad()
    def _visualize_denoising_trajectory(
        self,
        epoch: int,
        num_steps: int = 5,
    ) -> None:
        """Visualize intermediate denoising steps.

        Shows the progression from noise to clean sample at multiple timesteps.

        Args:
            epoch: Current epoch number.
            num_steps: Number of intermediate steps to visualize.
        """
        if not self.is_main_process:
            return

        self.model.eval()

        # Generate one sample with trajectory
        if isinstance(self.space, LatentSpace):
            latent_shape = (1, self.space.latent_channels,
                           self.volume_depth // 8, self.volume_height // 8, self.volume_width // 8)
            noise = torch.randn(latent_shape, device=self.device)
        else:
            noise = torch.randn(1, 1, self.volume_depth, self.volume_height, self.volume_width,
                               device=self.device)

        # Generate with trajectory capture
        if self.use_size_bin_embedding:
            # Size bin mode: use zero conditioning for visualization
            zero_size_bins = torch.zeros(1, self.size_bin_num_bins, device=self.device)
            trajectory = self._generate_trajectory_with_size_bins(noise, zero_size_bins, num_steps=25, capture_every=5)
        else:
            model_input = noise
            if self.mode.is_conditional:
                cond_shape = list(noise.shape)
                cond_shape[1] = 1
                conditioning = torch.zeros(cond_shape, device=self.device)
                model_input = torch.cat([noise, conditioning], dim=1)
            trajectory = self._generate_trajectory(model_input, num_steps=25, capture_every=5)

        # Decode trajectory if in latent space
        if isinstance(self.space, LatentSpace):
            trajectory = [self.space.decode(t) for t in trajectory]

        # Extract center slices from trajectory
        slices = []
        for i, vol in enumerate(trajectory):
            center_idx = vol.shape[2] // 2
            center_slice = vol[0, :, center_idx, :, :]  # [C, H, W]
            center_slice = torch.clamp(center_slice, 0, 1)
            slices.append(center_slice)

        # Stack and create grid: each column is a timestep
        if slices:
            slices_tensor = torch.stack(slices, dim=0)  # [T, C, H, W]
            grid = make_grid(slices_tensor, nrow=len(slices), normalize=False)
            self.writer.add_image('denoising_trajectory/center_slice', grid, epoch)

    @torch.no_grad()
    def _generate_trajectory(
        self,
        model_input: torch.Tensor,
        num_steps: int = 25,
        capture_every: int = 5,
    ) -> List[torch.Tensor]:
        """Generate samples while capturing intermediate states.

        Args:
            model_input: Starting noisy tensor (may include conditioning).
            num_steps: Total denoising steps.
            capture_every: Capture state every N steps.

        Returns:
            List of intermediate tensors.
        """
        # Extract noise from model_input (first channels)
        if self.mode.is_conditional and not self.use_size_bin_embedding:
            in_ch = 1 if not isinstance(self.space, LatentSpace) else self.space.latent_channels
            x = model_input[:, :in_ch].clone()
            conditioning = model_input[:, in_ch:]
        else:
            x = model_input.clone()
            conditioning = None

        trajectory = [x.clone()]
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = 1.0 - i * dt
            t_tensor = torch.full((x.shape[0],), t, device=x.device)

            # Forward pass
            if conditioning is not None:
                combined = torch.cat([x, conditioning], dim=1)
                v_pred = self.model(combined, t_tensor)
            else:
                v_pred = self.model(x, t_tensor)

            # Euler step
            x = x - dt * v_pred

            # Capture at intervals
            if (i + 1) % capture_every == 0:
                trajectory.append(x.clone())

        return trajectory

    @torch.no_grad()
    def _generate_trajectory_with_size_bins(
        self,
        noise: torch.Tensor,
        size_bins: torch.Tensor,
        num_steps: int = 25,
        capture_every: int = 5,
    ) -> List[torch.Tensor]:
        """Generate samples with size_bin conditioning while capturing trajectory.

        Args:
            noise: Starting noise tensor [B, C, D, H, W].
            size_bins: Size bin conditioning [B, num_bins].
            num_steps: Total denoising steps.
            capture_every: Capture state every N steps.

        Returns:
            List of intermediate tensors.
        """
        x = noise.clone()
        trajectory = [x.clone()]
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = 1.0 - i * dt
            t_tensor = torch.full((x.shape[0],), t, device=x.device)

            # Forward with size_bins
            v_pred = self.model(x, t_tensor, size_bins=size_bins)

            # Euler step
            x = x - dt * v_pred

            # Capture at intervals
            if (i + 1) % capture_every == 0:
                trajectory.append(x.clone())

        return trajectory

    @torch.no_grad()
    def _generate_with_size_bins(
        self,
        noise: torch.Tensor,
        size_bins: torch.Tensor,
        num_steps: int = 25,
    ) -> torch.Tensor:
        """Generate samples with size_bin conditioning.

        Uses the RFlow sampling loop with size_bins passed to the model.

        Args:
            noise: Starting noise tensor [B, C, D, H, W].
            size_bins: Size bin conditioning [B, num_bins].
            num_steps: Number of denoising steps.

        Returns:
            Generated samples [B, C, D, H, W].
        """
        x = noise.clone()
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = 1.0 - i * dt
            t_tensor = torch.full((x.shape[0],), t, device=x.device)

            # Forward with size_bins
            v_pred = self.model(x, t_tensor, size_bins=size_bins)

            # Euler step
            x = x - dt * v_pred

        return x

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

    # =========================================================================
    # Abstract method implementations (required by BaseTrainer)
    # =========================================================================

    def _get_trainer_type(self) -> str:
        """Return trainer type string for metadata."""
        return 'diffusion_3d'

    def train_step(self, batch: Any) -> TrainingStepResult:
        """Execute single training step (required by BaseTrainer).

        Delegates to internal _train_step implementation.

        Args:
            batch: Batch from dataloader.

        Returns:
            TrainingStepResult with loss values.
        """
        return self._train_step(batch)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Tuple[float, float, float]:
        """Train the model for one epoch (required by BaseTrainer).

        Args:
            train_loader: DataLoader for training data.
            epoch: Current epoch number.

        Returns:
            Tuple of (avg_loss, mse_loss, perceptual_loss).
            Note: perceptual_loss is 0.0 for diffusion (no perceptual loss).
        """
        avg_loss = self._train_epoch(train_loader, epoch)
        # Return (total, mse, perceptual) - diffusion has no perceptual loss
        return avg_loss, avg_loss, 0.0

    def compute_validation_losses(
        self,
        epoch: int,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
        """Compute losses and metrics on validation set (required by BaseTrainer).

        Args:
            epoch: Current epoch number.

        Returns:
            Tuple of (losses_dict, sample_outputs).
            losses_dict contains 'mse' and optionally other metrics.
            sample_outputs is None for 3D diffusion (handled separately).
        """
        if self.val_loader is None:
            return {'mse': 0.0}, None

        val_loss = self._validate(self.val_loader, epoch)
        return {'mse': val_loss}, None

    def evaluate_test_set(
        self,
        test_loader: DataLoader,
        checkpoint_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate diffusion model on test set.

        Runs inference on the entire test set and computes metrics:
        - MSE (prediction error)
        - For seg modes: Dice, IoU
        - For image modes: PSNR, MS-SSIM

        Results are saved to test_results_{checkpoint_name}.json and logged to TensorBoard.

        Args:
            test_loader: DataLoader for test set.
            checkpoint_name: Name of checkpoint to load ("best", "latest", or None
                for current model state).

        Returns:
            Dict with test metrics.
        """
        if not self.is_main_process:
            return {}

        # Load checkpoint if specified
        if checkpoint_name is not None:
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{checkpoint_name}.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded {checkpoint_name} checkpoint for test evaluation")

                # Load EMA state if available
                if self.ema is not None and 'ema_state_dict' in checkpoint:
                    self.ema.load_state_dict(checkpoint['ema_state_dict'])
                    logger.info("Loaded EMA state from checkpoint")
            else:
                logger.warning(f"Checkpoint {checkpoint_path} not found, using current model state")
                checkpoint_name = "current"

        label = checkpoint_name or "current"
        logger.info("=" * 60)
        logger.info(f"EVALUATING ON TEST SET ({label.upper()} MODEL)")
        logger.info("=" * 60)

        # Use EMA model if available
        if self.ema is not None:
            model_to_use = self.ema.ema_model
            logger.info("Using EMA model for evaluation")
        else:
            model_to_use = self.model
        model_to_use.eval()

        # Accumulators
        total_mse = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_dice = 0.0
        total_iou = 0.0
        n_batches = 0
        n_samples = 0

        # Regional metrics tracker
        regional_tracker = None
        if self.log_regional_losses and not self.is_seg_mode:
            regional_tracker = RegionalMetricsTracker3D(
                volume_size=(self.volume_height, self.volume_width, self.volume_depth),
                fov_mm=self.cfg.get('fov_mm', 240.0),
                loss_fn='l1',
                device=self.device,
            )

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test evaluation", disable=not self.is_main_process):
                prepared = self.mode.prepare_batch(batch, self.device)
                images = prepared['images']
                labels = prepared.get('labels')
                size_bins = prepared.get('size_bins')
                batch_size = images.shape[0]

                # Sample noise and timesteps
                noise = torch.randn_like(images)
                timesteps = self.strategy.sample_timesteps(images)
                noisy_images = self.strategy.add_noise(images, noise, timesteps)

                # Forward pass
                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                    if self.use_controlnet and labels is not None:
                        prediction = model_to_use(
                            x=noisy_images,
                            timesteps=timesteps,
                            controlnet_cond=labels,
                        )
                    elif self.use_size_bin_embedding:
                        prediction = model_to_use(noisy_images, timesteps, size_bins=size_bins)
                    else:
                        if labels is not None:
                            if isinstance(self.space, LatentSpace):
                                labels_enc = self.space.encode(labels)
                            else:
                                labels_enc = labels
                            model_input = torch.cat([noisy_images, labels_enc], dim=1)
                        else:
                            model_input = noisy_images
                        prediction = model_to_use(x=model_input, timesteps=timesteps)

                    mse_loss, predicted_clean = self.strategy.compute_loss(
                        prediction, images, noise, noisy_images, timesteps
                    )

                # Compute metrics
                total_mse += mse_loss.item()

                # Decode if in latent space
                if isinstance(self.space, LatentSpace):
                    images_pixel = self.space.decode(images)
                    predicted_pixel = self.space.decode(predicted_clean)
                else:
                    images_pixel = images
                    predicted_pixel = predicted_clean

                if self.is_seg_mode:
                    dice = compute_dice(predicted_pixel, images_pixel, apply_sigmoid=False)
                    iou = compute_iou(predicted_pixel, images_pixel, apply_sigmoid=False)
                    total_dice += dice
                    total_iou += iou
                else:
                    psnr = compute_psnr(predicted_pixel, images_pixel)
                    ssim = self.ssim_metric(predicted_pixel, images_pixel).mean().item()
                    total_psnr += psnr
                    total_ssim += ssim

                    # Regional metrics
                    if regional_tracker is not None and labels is not None:
                        regional_tracker.update(predicted_pixel, images_pixel, labels)

                n_batches += 1
                n_samples += batch_size

        model_to_use.train()

        # Handle empty test set
        if n_batches == 0:
            logger.warning(f"Test set ({label}) is empty, skipping evaluation")
            return {}

        # Compute averages
        metrics = {
            'mse': total_mse / n_batches,
            'n_samples': n_samples,
        }

        if self.is_seg_mode:
            metrics['dice'] = total_dice / n_batches
            metrics['iou'] = total_iou / n_batches
        else:
            metrics['psnr'] = total_psnr / n_batches
            metrics['msssim'] = total_ssim / n_batches

        # Log results
        logger.info(f"Test Results - {label} ({n_samples} samples):")
        logger.info(f"  MSE:     {metrics['mse']:.6f}")
        if self.is_seg_mode:
            logger.info(f"  Dice:    {metrics['dice']:.4f}")
            logger.info(f"  IoU:     {metrics['iou']:.4f}")
        else:
            logger.info(f"  PSNR:    {metrics['psnr']:.2f} dB")
            logger.info(f"  MS-SSIM: {metrics['msssim']:.4f}")

        # Save results to JSON
        results_path = os.path.join(self.save_dir, f'test_results_{label}.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Test results saved to: {results_path}")

        # Log to TensorBoard using unified system
        tb_prefix = f'test_{label}'
        if self._unified_metrics is not None:
            # Format metrics for unified logging
            test_metrics = {'MSE': metrics['mse']}
            if self.is_seg_mode:
                test_metrics['Dice'] = metrics['dice']
                test_metrics['IoU'] = metrics['iou']
            else:
                test_metrics['PSNR'] = metrics['psnr']
                test_metrics['MS-SSIM'] = metrics['msssim']

            self._unified_metrics.log_test(test_metrics, prefix=tb_prefix)

            # Regional metrics (already logged via log_to_tensorboard)
            if regional_tracker is not None:
                regional_tracker.log_to_tensorboard(self.writer, 0, prefix=f'{tb_prefix}_regional')

        return metrics
