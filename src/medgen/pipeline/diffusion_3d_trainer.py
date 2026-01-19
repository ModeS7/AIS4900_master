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
import random
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
from tqdm import tqdm


@dataclass
class TrainStepResult:
    """Extended result for 3D training step with grad_norm."""
    total_loss: float
    mse_loss: float
    grad_norm: float
    timesteps: Optional[torch.Tensor] = None  # For timestep bin tracking
    perceptual_loss: float = 0.0  # 2.5D perceptual loss (center slice)

from monai.networks.nets import DiffusionModelUNet
from monai.metrics import SSIMMetric

from medgen.core import setup_distributed, create_warmup_cosine_scheduler
from .base_trainer import BaseTrainer
from .results import TrainingStepResult
from medgen.diffusion import (
    ConditionalSingleMode, SegmentationMode, SegmentationConditionedMode, TrainingMode,
    DDPMStrategy, RFlowStrategy, DiffusionStrategy,
    DiffusionSpace, PixelSpace, LatentSpace,
)
from .utils import (
    get_vram_usage,
    log_vram_to_tensorboard,
    save_full_checkpoint,
    EpochTimeEstimator,
)
from medgen.metrics import FLOPsTracker
from medgen.metrics import (
    compute_psnr,
    compute_msssim,
    compute_lpips_3d,
    compute_dice,
    compute_iou,
    # Unified metrics system
    UnifiedMetrics,
    # Regional tracking (still needed for conditional init)
    RegionalMetricsTracker3D,
)
from medgen.models import (
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
            use_discrete_timesteps=cfg.strategy.get('use_discrete_timesteps', True),
            sample_method=cfg.strategy.get('sample_method', 'logit-normal'),
            use_timestep_transform=cfg.strategy.get('use_timestep_transform', True),
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

        # Generation metrics (KID, CMMD) - uses same GenerationMetrics as 2D trainer
        # Features are extracted slice-wise (2.5D) and can be shared with 2D
        self._gen_metrics: Optional['GenerationMetrics'] = None
        gen_cfg = cfg.training.get('generation_metrics', {})
        if gen_cfg.get('enabled', False):
            from medgen.metrics.generation import GenerationMetricsConfig
            # Use training batch_size * depth for 3D slice-wise extraction
            feature_batch_size = gen_cfg.get('feature_batch_size', None)
            if feature_batch_size is None:
                feature_batch_size = max(32, cfg.training.batch_size * 16)  # Reasonable default for slices
            # Use absolute cache_dir from paths config
            gen_cache_dir = gen_cfg.get('cache_dir', None)
            if gen_cache_dir is None:
                base_cache = getattr(cfg.paths, 'cache_dir', '.cache')
                gen_cache_dir = f"{base_cache}/generation_features"
            # 3D volumes are much larger than 2D images - cap sample counts to avoid OOM
            # Even with config values from default.yaml (100/500/1000), use sensible 3D limits
            samples_per_epoch_3d = min(gen_cfg.get('samples_per_epoch', 1), 2)  # Max 2 volumes/epoch
            samples_extended_3d = min(gen_cfg.get('samples_extended', 4), 4)    # Max 4 volumes
            samples_test_3d = min(gen_cfg.get('samples_test', 10), 10)          # Max 10 volumes

            self._gen_metrics_config = GenerationMetricsConfig(
                enabled=True,
                samples_per_epoch=samples_per_epoch_3d,
                samples_extended=samples_extended_3d,
                samples_test=samples_test_3d,
                steps_per_epoch=gen_cfg.get('steps_per_epoch', 10),
                steps_extended=gen_cfg.get('steps_extended', 25),
                steps_test=gen_cfg.get('steps_test', 50),
                cache_dir=gen_cache_dir,
                feature_batch_size=feature_batch_size,
            )
        else:
            self._gen_metrics_config = None
        # Note: uses self.figure_interval from base_trainer for consistency

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
        self.is_seg_mode = self.mode_name == 'seg'

        # Perceptual loss weight (disabled for seg mode - binary masks don't work with VGG features)
        # Uses 2.5D approach: compute perceptual loss on center slice for efficiency
        self.perceptual_weight: float = 0.0 if self.is_seg_mode else cfg.training.get('perceptual_weight', 0.0)

        # Size bin embedding for seg_conditioned mode (matches 2D trainer)
        self.use_size_bin_embedding = (self.mode_name == 'seg_conditioned')
        if self.use_size_bin_embedding:
            size_bin_cfg = cfg.mode.get('size_bins', {})
            bin_edges = list(size_bin_cfg.get('edges', [0, 3, 6, 10, 15, 20, 30]))
            # Default: len(edges) bins (6 bounded + 1 overflow for >= last edge)
            self.size_bin_num_bins = size_bin_cfg.get('num_bins', len(bin_edges))
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

        # Verbosity (controls tqdm progress bars - disabled on cluster to keep .err files clean)
        self.verbose = cfg.training.get('verbose', True)
        self.num_intermediate_steps = log_cfg.get('num_intermediate_steps', 5)
        self.log_regional_losses = log_cfg.get('regional_losses', True)
        self.log_timestep_region_losses = log_cfg.get('timestep_region_losses', True)
        self.log_flops = log_cfg.get('flops', True)
        self.log_lpips = log_cfg.get('lpips', True) and not self.is_seg_mode  # Slice-wise LPIPS

        # Memory profiling (detailed VRAM logging at key points)
        self.memory_profiling: bool = cfg.training.get('memory_profiling', False)
        if self.memory_profiling and self.is_main_process:
            logger.info("Memory profiling ENABLED - detailed VRAM logging at key points")

        # FLOPs tracker (measured at start of training)
        self._flops_tracker = FLOPsTracker()

        # Limit batches per epoch (for fast debugging)
        self.limit_train_batches: Optional[int] = cfg.training.get('limit_train_batches', None)
        if self.limit_train_batches is not None and self.is_main_process:
            logger.info(f"Limiting training to {self.limit_train_batches} batches per epoch")

        # Min-SNR weighting (reduces impact of high-noise timesteps)
        # NOTE: Min-SNR is DDPM-specific (uses alpha_bar from noise schedule).
        # For RFlow, the SNR concept doesn't apply - disable with warning.
        self.use_min_snr: bool = cfg.training.get('use_min_snr', False)
        self.min_snr_gamma: float = cfg.training.get('min_snr_gamma', 5.0)
        if self.use_min_snr and self.strategy_name == 'rflow':
            import warnings
            warnings.warn(
                "Min-SNR weighting is DDPM-specific and has no theoretical basis for RFlow. "
                "The SNR formula (alpha_bar / (1 - alpha_bar)) is tied to DDPM's noise schedule. "
                "Disabling Min-SNR for RFlow training.",
                UserWarning,
                stacklevel=2,
            )
            self.use_min_snr = False
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
            from medgen.augmentation import ScoreAugTransform3D
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

            # Mode intensity scaling is not supported in 3D (requires mode_id/mode embedding)
            if score_aug_cfg.get('mode_intensity_scaling', False):
                logger.warning(
                    "mode_intensity_scaling is not supported in 3D diffusion "
                    "(requires mode_id from multi-modality mode). Ignoring."
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
                from medgen.augmentation import SDATransform3D
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
            from medgen.losses import RegionalWeightComputer3D
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

        # Augmented diffusion (DC-AE 1.5 - channel masking for latent space training)
        aug_diff_cfg = cfg.training.get('augmented_diffusion', {})
        self.augmented_diffusion_enabled: bool = aug_diff_cfg.get('enabled', False)
        self.aug_diff_min_channels: int = aug_diff_cfg.get('min_channels', 16)
        self.aug_diff_channel_step: int = aug_diff_cfg.get('channel_step', 4)
        self._aug_diff_channel_steps: Optional[List[int]] = None

        # Self-conditioning (consistency loss via two forward passes)
        self_cond_cfg = cfg.training.get('self_conditioning', {})
        self.self_conditioning_enabled: bool = self_cond_cfg.get('enabled', False)
        self.self_conditioning_prob: float = self_cond_cfg.get('prob', 0.5)
        self.self_conditioning_weight: float = self_cond_cfg.get('consistency_weight', 0.1)

        # Feature perturbation (noise on intermediate UNet layers)
        feat_cfg = cfg.training.get('feature_perturbation', {})
        self.feature_perturbation_enabled: bool = feat_cfg.get('enabled', False)
        self.feature_perturbation_std: float = feat_cfg.get('std', 0.1)
        self.feature_perturbation_layers: List[str] = feat_cfg.get('layers', ['mid'])
        self._perturbation_hooks: List[Any] = []

        # Conditioning dropout (train with dropped conditioning for classifier-free guidance)
        # For 3D, we explicitly drop conditioning since all volumes have tumors (unlike 2D slices)
        cond_dropout_cfg = cfg.training.get('conditioning_dropout', {})
        self.conditioning_dropout_prob: float = cond_dropout_cfg.get('prob', 0.15)
        if self.conditioning_dropout_prob > 0 and self.mode.is_conditional and self.is_main_process:
            logger.info(f"Conditioning dropout enabled: prob={self.conditioning_dropout_prob}")

        # Cached training samples for visualization (real conditioning instead of zeros)
        # Uses training data to keep validation/test datasets properly separated
        self._cached_train_batch: Optional[Dict[str, torch.Tensor]] = None

        # Timestep bins for per-bin loss tracking
        self.num_timestep_bins = 10

        # Unified metrics system (initialized in train() when writer is available)
        # Note: figure_interval is set in BaseTrainer from cfg.training.figure_interval
        self._unified_metrics: Optional[UnifiedMetrics] = None
        self._last_val_metrics: Optional[Dict[str, float]] = None

        if self._gen_metrics_config is not None and self.is_main_process:
            logger.info(
                f"Generation metrics enabled: {self._gen_metrics_config.samples_per_epoch} volumes/epoch "
                f"({self._gen_metrics_config.steps_per_epoch} steps), "
                f"{self._gen_metrics_config.samples_extended} volumes/extended "
                f"({self._gen_metrics_config.steps_extended} steps)"
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
            # Segmentation conditioned mode with size bin embedding
            size_bin_config = dict(self.cfg.mode.get('size_bins', {})) if 'size_bins' in self.cfg.mode else None
            return SegmentationConditionedMode(size_bin_config)
        elif name == 'bravo':
            return ConditionalSingleMode()
        else:
            # Default to conditional single for 3D (most common)
            return ConditionalSingleMode()

    # ─────────────────────────────────────────────────────────────────────────
    # Generation Metrics (KID, CMMD) - Uses shared GenerationMetrics from 2D
    # ─────────────────────────────────────────────────────────────────────────

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

        Generates in batches of self.batch_size (from training.batch_size) to avoid OOM.
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
        # Use training.batch_size for generation batching (typically 1 for 3D)
        batch_size = self.batch_size
        all_samples = []

        # Determine base shape (without batch dimension)
        if isinstance(self.space, LatentSpace):
            latent_depth = self.volume_depth // self.space.scale_factor
            latent_height = self.volume_height // self.space.scale_factor
            latent_width = self.volume_width // self.space.scale_factor
            base_shape = (self.space.latent_channels, latent_depth, latent_height, latent_width)
        else:
            base_shape = (1, self.volume_depth, self.volume_height, self.volume_width)

        # Generate in batches to avoid OOM
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            current_batch_size = end_idx - start_idx

            # Create noise for this batch
            noise_shape = (current_batch_size,) + base_shape
            noise = torch.randn(noise_shape, device=self.device)

            # Handle conditioning based on mode
            if self.mode.is_conditional and self._fixed_conditioning_volumes is not None:
                # Use fixed conditioning (cycle if needed)
                n_cond = self._fixed_conditioning_volumes.shape[0]
                indices = [i % n_cond for i in range(start_idx, end_idx)]
                conditioning = self._fixed_conditioning_volumes[indices]  # Pixel-space

                if self.use_controlnet:
                    # ControlNet mode: wrap model with pixel-space conditioning
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

            # Generate samples for this batch
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

            # Move to CPU to free GPU memory
            all_samples.append(torch.clamp(samples, 0, 1).cpu())

            # Clean up batch tensors
            del noise, model_input, samples
            if self.mode.is_conditional and self._fixed_conditioning_volumes is not None:
                del conditioning

        # Concatenate all batches and move back to device
        result = torch.cat(all_samples, dim=0).to(self.device)
        del all_samples

        # Threshold seg mode output at 0.5 to get binary masks
        if self.is_seg_mode:
            result = (result > 0.5).float()

        return result

    @torch.no_grad()
    def _compute_generation_metrics_3d(
        self,
        model: nn.Module,
        epoch: int,
        extended: bool = False,
    ) -> Dict[str, float]:
        """Compute KID and CMMD using 2.5D slice-wise approach.

        Generates 3D samples, then uses GenerationMetrics to extract slice
        features and compute metrics against cached reference features.

        Args:
            model: Model to use for generation (can be EMA model).
            epoch: Current epoch number.
            extended: If True, use more samples and steps for thorough evaluation.

        Returns:
            Dictionary with KID/CMMD metrics (same key format as 2D).
        """
        if self._gen_metrics is None:
            return {}

        # Select samples and steps based on mode
        if extended:
            num_samples = self._gen_metrics_config.samples_extended
            num_steps = self._gen_metrics_config.steps_extended
        else:
            num_samples = self._gen_metrics_config.samples_per_epoch
            num_steps = self._gen_metrics_config.steps_per_epoch

        # Generate 3D samples
        samples = self._generate_samples_3d(model, num_samples, num_steps)

        # Use GenerationMetrics to compute metrics (handles 3D slice-wise)
        results = self._gen_metrics.compute_metrics_from_samples(samples, extended=extended)

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

        # Handle size bin embedding for seg mode
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
            from medgen.augmentation import ScoreAugModelWrapper3D
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

        # Setup perceptual loss function (2.5D: compute on center slice)
        # Uses RadImageNet-pretrained ResNet50 for medical image features
        self.perceptual_loss_fn = None
        if self.perceptual_weight > 0:
            from medgen.losses import PerceptualLoss
            cache_dir = getattr(self.cfg.paths, 'cache_dir', None)
            self.perceptual_loss_fn = PerceptualLoss(
                spatial_dims=2,
                network_type="radimagenet_resnet50",
                cache_dir=cache_dir,
                pretrained=True,
                device=self.device,
            )
            logger.info(f"Perceptual loss enabled (2.5D center slice) with weight={self.perceptual_weight}")

        # Setup EMA (on the appropriate model)
        ema_target = self.controlnet if (self.use_controlnet and self.controlnet_freeze_unet) else self.model
        if self.use_ema:
            self.ema = EMA(
                ema_target,
                beta=self.ema_decay,
                update_every=1,
            )
            logger.info(f"EMA enabled with decay={self.ema_decay}")

        # Setup feature perturbation hooks if enabled
        self._setup_feature_perturbation()

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

    def _log_memory(self, label: str, reset_peak: bool = False) -> None:
        """Log detailed VRAM usage at a checkpoint (only if memory_profiling enabled).

        Args:
            label: Description of the checkpoint (e.g., "after_model_setup").
            reset_peak: If True, reset peak memory stats after logging.
        """
        if not self.memory_profiling or not self.is_main_process:
            return

        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        peak = torch.cuda.max_memory_allocated(self.device) / 1e9

        logger.info(
            f"[MEMORY] {label}: "
            f"allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, peak={peak:.2f}GB"
        )

        if reset_peak:
            torch.cuda.reset_peak_memory_stats(self.device)

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

    def _cache_training_samples(self, train_loader: DataLoader) -> None:
        """Cache first training batch for deterministic visualization.

        Uses training data (not validation) to keep datasets properly separated.
        Cached once at training start for reproducibility across epochs.

        Args:
            train_loader: Training dataloader.
        """
        if self._cached_train_batch is not None:
            return  # Already cached

        try:
            batch = next(iter(train_loader))
            prepared = self.mode.prepare_batch(batch, self.device)
            images = prepared['images']
            labels = prepared.get('labels')
            size_bins = prepared.get('size_bins')

            self._cached_train_batch = {
                'images': images.detach().clone(),
                'labels': labels.detach().clone() if labels is not None else None,
                'size_bins': size_bins.detach().clone() if size_bins is not None else None,
            }

            if self.is_main_process:
                logger.info(f"Cached {images.shape[0]} training samples for visualization")
        except Exception as e:
            if self.is_main_process:
                logger.warning(f"Could not cache training samples: {e}")

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

    def _get_aug_diff_channel_steps(self, num_channels: int) -> List[int]:
        """Get list of channel counts for augmented diffusion masking.

        Returns [min_channels, min+step, min+2*step, ..., num_channels].
        Paper uses [16, 20, 24, ..., c] with step=4, min=16.

        Args:
            num_channels: Total number of latent channels.

        Returns:
            List of valid channel counts to sample from during training.
        """
        if self._aug_diff_channel_steps is None:
            steps = list(range(
                self.aug_diff_min_channels,
                num_channels + 1,
                self.aug_diff_channel_step
            ))
            # Ensure max channels is always included
            if not steps or steps[-1] != num_channels:
                steps.append(num_channels)
            self._aug_diff_channel_steps = steps
        return self._aug_diff_channel_steps

    def _create_aug_diff_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create channel mask for augmented diffusion training.

        From DC-AE 1.5 paper (Eq. 2):
        - Sample random channel count c' from [min_channels, ..., num_channels]
        - Create mask: [1,...,1 (c' times), 0,...,0]

        Args:
            tensor: Input tensor [B, C, D, H, W] to get shape from.

        Returns:
            Mask tensor [1, C, 1, 1, 1] for broadcasting.
        """
        C = tensor.shape[1]
        steps = self._get_aug_diff_channel_steps(C)
        c_prime = random.choice(steps)

        mask = torch.zeros(1, C, 1, 1, 1, device=tensor.device, dtype=tensor.dtype)
        mask[:, :c_prime, :, :, :] = 1.0
        return mask

    def _compute_self_conditioning_loss(
        self,
        model_input: torch.Tensor,
        timesteps: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Compute self-conditioning consistency loss.

        With probability `prob`, runs model a second time and computes
        consistency loss between the two predictions.

        Args:
            model_input: Current model input tensor.
            timesteps: Current timesteps.
            prediction: Current prediction from main forward pass.

        Returns:
            Consistency loss (0 if disabled or skipped this batch).
        """
        if not self.self_conditioning_enabled:
            return torch.tensor(0.0, device=model_input.device)

        # With probability (1-prob), skip self-conditioning
        if random.random() >= self.self_conditioning_prob:
            return torch.tensor(0.0, device=model_input.device)

        # Get second prediction (detached first prediction as reference)
        with torch.no_grad():
            prediction_ref = self.model(x=model_input, timesteps=timesteps)
            prediction_ref = prediction_ref.detach()

        # Consistency loss: predictions should be similar
        consistency_loss = F.mse_loss(prediction.float(), prediction_ref.float())
        return consistency_loss

    def _setup_feature_perturbation(self) -> None:
        """Setup forward hooks for feature perturbation."""
        self._perturbation_hooks = []

        if not self.feature_perturbation_enabled:
            return

        def make_hook(noise_std):
            def hook(module, input, output):
                if self.model.training:
                    noise = torch.randn_like(output) * noise_std
                    return output + noise
                return output
            return hook

        # Register hooks on specified layers
        # UNet structure: down_blocks, mid_block, up_blocks
        if hasattr(self.model_raw, 'mid_block') and 'mid' in self.feature_perturbation_layers:
            handle = self.model_raw.mid_block.register_forward_hook(make_hook(self.feature_perturbation_std))
            self._perturbation_hooks.append(handle)

        if hasattr(self.model_raw, 'down_blocks') and 'encoder' in self.feature_perturbation_layers:
            for block in self.model_raw.down_blocks:
                handle = block.register_forward_hook(make_hook(self.feature_perturbation_std))
                self._perturbation_hooks.append(handle)

        if hasattr(self.model_raw, 'up_blocks') and 'decoder' in self.feature_perturbation_layers:
            for block in self.model_raw.up_blocks:
                handle = block.register_forward_hook(make_hook(self.feature_perturbation_std))
                self._perturbation_hooks.append(handle)

        if self._perturbation_hooks and self.is_main_process:
            logger.info(f"Feature perturbation enabled: {len(self._perturbation_hooks)} hooks on {self.feature_perturbation_layers}")

    def _remove_feature_perturbation_hooks(self) -> None:
        """Remove feature perturbation hooks."""
        for handle in getattr(self, '_perturbation_hooks', []):
            handle.remove()
        self._perturbation_hooks = []

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

        # Detect if input is discrete (int) or continuous (float)
        is_discrete = timesteps.dtype in (torch.int32, torch.int64, torch.long)
        # Normalize to [0, 1], add jitter, clamp, scale back
        t_normalized = timesteps.float() / self.num_timesteps
        t_jittered = t_normalized + torch.randn_like(t_normalized) * std
        t_jittered = t_jittered.clamp(0.0, 1.0)
        t_scaled = t_jittered * self.num_timesteps
        # Preserve dtype: int for DDPM, float for RFlow
        if is_discrete:
            return t_scaled.long()
        else:
            return t_scaled

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
            modality=self.mode_name if self.mode_name != 'seg' else None,
            device=self.device,
            enable_regional=self.log_regional_losses and not self.is_seg_mode,
            num_timestep_bins=self.num_timestep_bins,
            fov_mm=self.cfg.paths.get('fov_mm', 240.0),
            volume_size=(self.volume_height, self.volume_width, self.volume_depth),
        )

        # Measure FLOPs at start of training
        self._measure_model_flops(train_loader)

        # Memory profiling checkpoint: after model setup
        self._log_memory("after_model_setup", reset_peak=True)

        # Initialize generation metrics (cache reference features, set conditioning)
        # Uses same GenerationMetrics as 2D - features cached to disk and shared
        if self._gen_metrics_config is not None and self.is_main_process:
            self._log_memory("before_gen_metrics_init")
            logger.info("Initializing 3D generation metrics...")
            from medgen.metrics.generation import GenerationMetrics
            self._gen_metrics = GenerationMetrics(
                self._gen_metrics_config,
                self.device,
                self.save_dir,
                space=self.space,
                mode_name=self.mode_name,
            )
            # Set fixed conditioning volumes for generation
            self._set_fixed_conditioning_3d(train_loader, num_volumes=self._gen_metrics_config.samples_extended)
            # Cache reference features (shared with 2D, uses content-based key)
            if val_loader is not None:
                import hashlib
                data_dir = str(self.cfg.paths.data_dir)
                cache_key = f"{data_dir}_{self.mode_name}_{self.volume_height}x{self.volume_depth}_3d"
                cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
                cache_id = f"{self.mode_name}_{self.volume_height}x{self.volume_depth}_{cache_hash}"
                self._gen_metrics.cache_reference_features(train_loader, val_loader, experiment_id=cache_id)
            self._log_memory("after_gen_metrics_init_and_cache")

        # Memory profiling checkpoint: before training loop
        self._log_memory("before_training_loop", reset_peak=True)

        # Cache first training batch for deterministic visualization
        # Uses training data (not validation) to keep datasets separate
        self._cache_training_samples(train_loader)

        # Time estimator for ETA calculation (excludes first epoch warmup)
        time_estimator = EpochTimeEstimator(self.cfg.training.epochs)

        for epoch in range(self.cfg.training.epochs):
            self._current_epoch = epoch
            epoch_start = time.time()

            # Memory profiling: before training epoch (only first 3 epochs to reduce log spam)
            if epoch < 3:
                self._log_memory(f"epoch_{epoch}_before_train")

            # Training epoch
            train_loss = self._train_epoch(train_loader, epoch)

            # Memory profiling: after training epoch
            if epoch < 3:
                self._log_memory(f"epoch_{epoch}_after_train")

            # Step LR scheduler once per epoch (not per step)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Validation (every epoch by default)
            val_loss = None
            val_interval = self.cfg.training.get('val_interval', 1)
            if val_loader is not None and (epoch + 1) % val_interval == 0:
                # Sync CUDA and clear cache before validation to prevent worker issues
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                if epoch < 3:
                    self._log_memory(f"epoch_{epoch}_before_val")

                val_loss = self._validate(val_loader, epoch)

                if epoch < 3:
                    self._log_memory(f"epoch_{epoch}_after_val")

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
            if self._gen_metrics is not None and self.is_main_process:
                # Clear fragmented memory before generation (prevents OOM from reserved-but-unused memory)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                if epoch < 3:
                    self._log_memory(f"epoch_{epoch}_before_gen_metrics")

                try:
                    # Use EMA model if available (matches 2D)
                    model_to_use = self.ema.ema_model if self.ema is not None else self.model
                    model_to_use.eval()

                    # Quick metrics every epoch (1 volume)
                    gen_results = self._compute_generation_metrics_3d(model_to_use, epoch, extended=False)
                    self._unified_metrics.log_generation(epoch, gen_results)

                    # Extended metrics at figure_interval (4 volumes)
                    if (epoch + 1) % self.figure_interval == 0:
                        ext_results = self._compute_generation_metrics_3d(model_to_use, epoch, extended=True)
                        self._unified_metrics.log_generation(epoch, ext_results)

                    model_to_use.train()

                    if epoch < 3:
                        self._log_memory(f"epoch_{epoch}_after_gen_metrics")
                except Exception as e:
                    logger.warning(f"Generation metrics computation failed: {e}")
                finally:
                    # Always clean up after generation metrics to prevent memory buildup
                    torch.cuda.empty_cache()

            # Log epoch summary using unified system
            epoch_time = time.time() - epoch_start
            self._unified_metrics.log_console_summary(
                epoch=epoch,
                total_epochs=self.cfg.training.epochs,
                elapsed_time=epoch_time,
                time_estimator=time_estimator,
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

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not self.verbose)
        for batch_idx, batch in enumerate(pbar):
            # Limit batches per epoch (for fast debugging)
            if self.limit_train_batches is not None and batch_idx >= self.limit_train_batches:
                break

            result = self._train_step(batch)

            # Accumulate losses using unified system
            self._unified_metrics.update_loss('MSE', result.mse_loss)
            if self.perceptual_weight > 0:
                self._unified_metrics.update_loss('Total', result.total_loss)
                self._unified_metrics.update_loss('Perceptual', result.perceptual_loss)
            self._unified_metrics.update_grad_norm(result.grad_norm)

            # Update progress bar
            pbar.set_postfix({'loss': f'{result.total_loss:.4f}'})

            self._global_step += 1

        # Log training metrics using unified system
        if self.is_main_process:
            self._unified_metrics.update_lr(self.optimizer.param_groups[0]['lr'])
            self._unified_metrics.update_vram()
            self._unified_metrics.log_training(epoch)

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
        size_bins = prepared.get('size_bins')  # [B, num_bins] - for seg mode

        # Conditioning dropout: randomly drop conditioning for classifier-free guidance training
        # Unlike 2D (where negative slices naturally provide unconditional samples),
        # 3D volumes always have tumors so we must explicitly drop conditioning
        if self.conditioning_dropout_prob > 0 and random.random() < self.conditioning_dropout_prob:
            if labels is not None:
                labels = torch.zeros_like(labels)
            if size_bins is not None:
                size_bins = torch.zeros_like(size_bins)

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

        # DC-AE 1.5: Augmented Diffusion Training - apply channel masking
        # Only active for latent diffusion (LatentSpace)
        aug_diff_mask = None
        if self.augmented_diffusion_enabled and isinstance(self.space, LatentSpace):
            aug_diff_mask = self._create_aug_diff_mask(noise)
            noise = noise * aug_diff_mask
            noisy_images = noisy_images * aug_diff_mask

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

            # DC-AE 1.5: Mask prediction for augmented diffusion training
            # Paper Eq. 2: ||ε·mask - ε_θ(x_t·mask, t)·mask||²
            if aug_diff_mask is not None:
                prediction = prediction * aug_diff_mask

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

            # 2.5D Perceptual loss on center slice
            p_loss = torch.tensor(0.0, device=self.device)
            if self.perceptual_weight > 0 and self.perceptual_loss_fn is not None:
                # Get predicted_clean from prediction
                # For RFlow: x_0 = x_t - t * v
                # For DDPM: x_0 = (x_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
                _, predicted_clean = self.strategy.compute_loss(
                    prediction, images, noise, noisy_images, timesteps
                )
                # Extract center slice (2.5D approach)
                center_idx = images.shape[2] // 2  # [B, C, D, H, W] -> D dimension
                pred_slice = predicted_clean[:, :, center_idx, :, :].float()  # [B, C, H, W]
                img_slice = images[:, :, center_idx, :, :].float()
                p_loss = self.perceptual_loss_fn(pred_slice, img_slice)
                total_loss = total_loss + self.perceptual_weight * p_loss

            # Self-conditioning consistency loss (only for standard/omega modes)
            if self.self_conditioning_enabled and not self.use_controlnet and not self.use_size_bin_embedding:
                # model_input is set in omega or standard branch
                consistency_loss = self._compute_self_conditioning_loss(
                    model_input, timesteps, prediction
                )
                total_loss = total_loss + self.self_conditioning_weight * consistency_loss

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

        # NOTE: LR scheduler step moved to train() - once per epoch, not per step

        # EMA update
        if self.ema is not None:
            self.ema.update()

        # Return extended result with grad_norm and timesteps
        p_loss_val = p_loss.item() if isinstance(p_loss, torch.Tensor) else p_loss
        return TrainStepResult(
            total_loss=total_loss.item(),  # Total loss (MSE + SDA + Perceptual if enabled)
            mse_loss=mse_loss.item(),
            grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            timesteps=timesteps.detach(),  # Keep for timestep bin tracking
            perceptual_loss=p_loss_val,
        )

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader, epoch: int) -> float:
        """Run validation.

        For seg modes (seg, seg): computes Dice/IoU
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

        # Worst batch tracking - simple scalar approach (matches 2D trainer)
        worst_loss = 0.0
        worst_batch_data: Optional[Dict[str, Any]] = None
        min_batch_size = self.batch_size  # Don't track small last batches

        for batch in tqdm(val_loader, desc="Validation", disable=not self.verbose):
            prepared = self.mode.prepare_batch(batch, self.device)
            images = prepared['images']
            labels = prepared.get('labels')
            size_bins = prepared.get('size_bins')  # For seg mode
            current_batch_size = images.shape[0]

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
            # Only log Total when it differs from MSE (perceptual weight > 0)
            if self.perceptual_weight > 0:
                self._unified_metrics.update_loss('Total', batch_loss, phase='val')

            # Track timestep losses using unified system
            if self.log_timestep_losses:
                # Use average timestep for the batch, normalized to 0.0-1.0
                avg_t = timesteps.float().mean().item()
                t_normalized = avg_t / self.num_timesteps
                self._unified_metrics.update_timestep_loss(t_normalized, batch_loss)

            # Decode if in latent space for quality metrics and visualization
            if isinstance(self.space, LatentSpace):
                images_pixel = self.space.decode(images)
                predicted_pixel = self.space.decode(predicted_clean)
            else:
                images_pixel = images
                predicted_pixel = predicted_clean

            # Track worst batch AFTER decode (so visualization shows pixel-space images)
            if self.log_worst_batch and batch_loss > worst_loss and current_batch_size >= min_batch_size:
                worst_loss = batch_loss
                worst_batch_data = {
                    'original': images_pixel.cpu(),
                    'generated': predicted_pixel.cpu(),
                    'mask': labels.cpu() if labels is not None else None,
                    'timesteps': timesteps.cpu(),
                    'loss': batch_loss,
                }

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

                # Timestep-region tracking (for heatmap) - split by tumor/background
                # Adapted for 3D: uses center slice for efficiency
                if self.log_timestep_region_losses and labels is not None:
                    batch_size = predicted_pixel.shape[0]
                    center_idx = predicted_pixel.shape[2] // 2
                    # Extract center slices [B, C, H, W]
                    pred_slice = predicted_pixel[:, :, center_idx, :, :]
                    img_slice = images_pixel[:, :, center_idx, :, :]
                    mask_slice = labels[:, :, center_idx, :, :]
                    # Compute error map [B, H, W]
                    error_map = ((pred_slice - img_slice) ** 2).mean(dim=1)
                    mask_binary = mask_slice[:, 0] > 0.5  # [B, H, W]
                    for i in range(batch_size):
                        t_norm = timesteps[i].item() / self.num_timesteps
                        sample_error = error_map[i]  # [H, W]
                        sample_mask = mask_binary[i]  # [H, W]
                        tumor_px = sample_mask.sum().item()
                        bg_px = (~sample_mask).sum().item()
                        tumor_loss = (sample_error * sample_mask.float()).sum().item() if tumor_px > 0 else 0.0
                        bg_loss = (sample_error * (~sample_mask).float()).sum().item() if bg_px > 0 else 0.0
                        self._unified_metrics.update_timestep_region_loss(
                            t_norm, tumor_loss, bg_loss, int(tumor_px), int(bg_px)
                        )

            num_batches += 1

        # Get validation metrics for return value and epoch summary
        val_metrics = self._unified_metrics.get_validation_metrics()
        self._last_val_metrics = val_metrics
        avg_loss = val_metrics.get('MSE', 0.0)

        # Log to TensorBoard
        if self.is_main_process:
            # Log validation metrics using unified system
            self._unified_metrics.log_validation(epoch)

            # Log worst batch visualization at figure_interval (uses unified metrics)
            if self.log_worst_batch and worst_batch_data is not None and (epoch + 1) % self.figure_interval == 0:
                self._unified_metrics.log_worst_batch(
                    original=worst_batch_data['original'].to(self.device),
                    reconstructed=worst_batch_data['generated'].to(self.device),
                    loss=worst_batch_data['loss'],
                    epoch=epoch,
                    phase='val',
                    mask=worst_batch_data.get('mask'),
                    timesteps=worst_batch_data.get('timesteps'),
                )

            # Log timestep-region heatmap at figure_interval
            if (epoch + 1) % self.figure_interval == 0 and self.log_timestep_region_losses:
                self._unified_metrics.log_timestep_region_heatmap(epoch)

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

        Uses REAL conditioning from cached TRAINING samples instead of zeros.
        This matches the 2D trainer approach and ensures the model gets proper
        conditioning for generation (model was trained with real masks, not zeros).
        Uses training data to keep validation/test datasets properly separated.

        Args:
            epoch: Current epoch number.
        """
        if not self.is_main_process:
            return

        self.model.eval()

        # Use cached training batch for real conditioning (instead of zeros!)
        # This is critical: model trained with real masks cannot generate with zeros
        if self._cached_train_batch is not None:
            cached_images = self._cached_train_batch['images']
            cached_labels = self._cached_train_batch.get('labels')
            cached_size_bins = self._cached_train_batch.get('size_bins')
            batch_size = min(4, cached_images.shape[0])

            # Generate noise matching the cached batch shape
            if isinstance(self.space, LatentSpace):
                # Get latent shape from actual encoding
                with torch.no_grad():
                    encoded = self.space.encode(cached_images[:batch_size])
                noise = torch.randn_like(encoded)
            else:
                noise = torch.randn_like(cached_images[:batch_size])

            # Use real conditioning from cached batch
            if self.use_size_bin_embedding:
                model_input = noise
                size_bins = cached_size_bins[:batch_size] if cached_size_bins is not None else None
            elif self.mode.is_conditional and cached_labels is not None:
                labels = cached_labels[:batch_size]
                if isinstance(self.space, LatentSpace):
                    labels_encoded = self.space.encode(labels)
                else:
                    labels_encoded = labels
                model_input = torch.cat([noise, labels_encoded], dim=1)
            else:
                model_input = noise
        else:
            # Fallback: no cached batch yet - cannot generate properly without real conditioning
            if self.mode.is_conditional:
                logger.warning("Cannot visualize samples: no cached training batch for conditioning")
                return
            # Unconditional mode can proceed
            batch_size = 4
            if isinstance(self.space, LatentSpace):
                latent_shape = (batch_size, self.space.latent_channels,
                               self.volume_depth // 8, self.volume_height // 8, self.volume_width // 8)
                noise = torch.randn(latent_shape, device=self.device)
            else:
                noise = torch.randn(batch_size, 1, self.volume_depth, self.volume_height, self.volume_width,
                                   device=self.device)

            if self.use_size_bin_embedding:
                model_input = noise
                # Use uniform distribution instead of zeros
                size_bins = torch.ones(batch_size, self.size_bin_num_bins, device=self.device) / self.size_bin_num_bins
            else:
                model_input = noise

        # Generate samples
        if self.use_size_bin_embedding:
            samples = self._generate_with_size_bins(noise, size_bins, num_steps=25)
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

        # Log using unified metrics (handles 3D center slice extraction)
        self._unified_metrics.log_generated_samples(samples, epoch, tag='Generated_Samples', nrow=2)

    @torch.no_grad()
    def _visualize_denoising_trajectory(
        self,
        epoch: int,
        num_steps: int = 5,
    ) -> None:
        """Visualize intermediate denoising steps.

        Shows the progression from noise to clean sample at multiple timesteps.
        Uses REAL conditioning from cached training samples (model cannot generate
        properly with zeros if trained with real conditioning).

        Args:
            epoch: Current epoch number.
            num_steps: Number of intermediate steps to visualize.
        """
        if not self.is_main_process:
            return

        self.model.eval()

        # Use cached training batch for real conditioning (critical for proper generation)
        if self._cached_train_batch is not None:
            cached_images = self._cached_train_batch['images']
            cached_labels = self._cached_train_batch.get('labels')
            cached_size_bins = self._cached_train_batch.get('size_bins')

            # Generate noise matching first cached sample
            if isinstance(self.space, LatentSpace):
                with torch.no_grad():
                    encoded = self.space.encode(cached_images[:1])
                noise = torch.randn_like(encoded)
            else:
                noise = torch.randn_like(cached_images[:1])

            # Generate with real conditioning from cached batch
            if self.use_size_bin_embedding:
                size_bins = cached_size_bins[:1] if cached_size_bins is not None else None
                if size_bins is not None:
                    trajectory = self._generate_trajectory_with_size_bins(
                        noise, size_bins, num_steps=25, capture_every=5
                    )
                else:
                    # Uniform size bins as fallback (better than zeros)
                    uniform_bins = torch.ones(1, self.size_bin_num_bins, device=self.device) / self.size_bin_num_bins
                    trajectory = self._generate_trajectory_with_size_bins(
                        noise, uniform_bins, num_steps=25, capture_every=5
                    )
            elif self.mode.is_conditional and cached_labels is not None:
                labels = cached_labels[:1]
                if isinstance(self.space, LatentSpace):
                    labels_encoded = self.space.encode(labels)
                else:
                    labels_encoded = labels
                model_input = torch.cat([noise, labels_encoded], dim=1)
                trajectory = self._generate_trajectory(model_input, num_steps=25, capture_every=5)
            else:
                trajectory = self._generate_trajectory(noise, num_steps=25, capture_every=5)
        else:
            # Fallback: no cached batch yet (shouldn't happen after first training step)
            logger.warning("No cached training batch for denoising trajectory - using unconditional")
            if isinstance(self.space, LatentSpace):
                latent_shape = (1, self.space.latent_channels,
                               self.volume_depth // 8, self.volume_height // 8, self.volume_width // 8)
                noise = torch.randn(latent_shape, device=self.device)
            else:
                noise = torch.randn(1, 1, self.volume_depth, self.volume_height, self.volume_width,
                                   device=self.device)
            trajectory = self._generate_trajectory(noise, num_steps=25, capture_every=5)

        # Decode trajectory if in latent space
        if isinstance(self.space, LatentSpace):
            trajectory = [self.space.decode(t) for t in trajectory]

        # Log using unified metrics (handles 3D center slice extraction)
        self._unified_metrics.log_denoising_trajectory(trajectory, epoch, tag='denoising_trajectory')

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
        num_train_timesteps = self.scheduler.num_train_timesteps

        for i in range(num_steps):
            t = 1.0 - i * dt
            # Scale to training range [0, num_train_timesteps] for correct embeddings
            t_scaled = t * num_train_timesteps
            t_tensor = torch.full((x.shape[0],), t_scaled, device=x.device)

            # Forward pass
            if conditioning is not None:
                combined = torch.cat([x, conditioning], dim=1)
                v_pred = self.model(combined, t_tensor)
            else:
                v_pred = self.model(x, t_tensor)

            # Euler step: x_{t-dt} = x_t + dt * v_pred (MONAI convention: v points toward clean)
            x = x + dt * v_pred

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
        num_train_timesteps = self.scheduler.num_train_timesteps

        for i in range(num_steps):
            t = 1.0 - i * dt
            # Scale to training range [0, num_train_timesteps] for correct embeddings
            t_scaled = t * num_train_timesteps
            t_tensor = torch.full((x.shape[0],), t_scaled, device=x.device)

            # Forward with size_bins
            v_pred = self.model(x, t_tensor, size_bins=size_bins)

            # Euler step: x_{t-dt} = x_t + dt * v_pred (MONAI convention: v points toward clean)
            x = x + dt * v_pred

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
        num_train_timesteps = self.scheduler.num_train_timesteps

        for i in range(num_steps):
            t = 1.0 - i * dt
            # Scale to training range [0, num_train_timesteps] for correct embeddings
            t_scaled = t * num_train_timesteps
            t_tensor = torch.full((x.shape[0],), t_scaled, device=x.device)

            # Forward with size_bins
            v_pred = self.model(x, t_tensor, size_bins=size_bins)

            # Euler step: x_{t-dt} = x_t + dt * v_pred (MONAI convention: v points toward clean)
            x = x + dt * v_pred

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

        # Worst batch tracking (matches 2D trainer)
        worst_batch_loss = 0.0
        worst_batch_data: Optional[Dict[str, Any]] = None

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
            for batch in tqdm(test_loader, desc="Test evaluation", disable=not self.verbose):
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
                batch_loss = mse_loss.item()
                total_mse += batch_loss

                # Decode if in latent space for metrics and visualization
                if isinstance(self.space, LatentSpace):
                    images_pixel = self.space.decode(images)
                    predicted_pixel = self.space.decode(predicted_clean)
                else:
                    images_pixel = images
                    predicted_pixel = predicted_clean

                # Track worst batch AFTER decode (matches 2D trainer, filter small last batches)
                if batch_loss > worst_batch_loss and batch_size >= self.batch_size:
                    worst_batch_loss = batch_loss
                    worst_batch_data = {
                        'original': images_pixel.cpu(),
                        'generated': predicted_pixel.cpu(),
                        'mask': labels.cpu() if labels is not None else None,
                        'timesteps': timesteps.cpu(),
                        'loss': batch_loss,
                    }

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

        # Visualize worst batch from test set (uses unified metrics)
        if worst_batch_data is not None:
            self._unified_metrics.log_worst_batch(
                original=worst_batch_data['original'].to(self.device),
                reconstructed=worst_batch_data['generated'].to(self.device),
                loss=worst_batch_data['loss'],
                epoch=0,  # Test uses epoch 0
                phase='test',
                mask=worst_batch_data.get('mask'),
                timesteps=worst_batch_data.get('timesteps'),
            )

        return metrics
