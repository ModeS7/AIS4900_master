"""
Diffusion model trainer module.

This module provides the DiffusionTrainer class which inherits from BaseTrainer
and implements diffusion-specific functionality:
- Strategy pattern (DDPM, Rectified Flow)
- Mode pattern (seg, bravo, dual, multi)
- Timestep-based noise training
- SAM optimizer support
- ScoreAug transforms
"""
import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from ema_pytorch import EMA
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from monai.networks.nets import DiffusionModelUNet

from medgen.core import ModeType, setup_distributed, create_warmup_cosine_scheduler, create_warmup_constant_scheduler, wrap_model_for_training
from .base_trainer import BaseTrainer
from .optimizers import SAM
from .results import TrainingStepResult
from medgen.models import create_diffusion_model, get_model_type, is_transformer_model
from medgen.losses import PerceptualLoss
from medgen.diffusion import (
    ConditionalDualMode,
    ConditionalSingleMode,
    MultiModalityMode,
    SegmentationConditionedMode,
    SegmentationMode,
    TrainingMode,
    DDPMStrategy, RFlowStrategy, DiffusionStrategy,
    DiffusionSpace, PixelSpace,
)
from medgen.evaluation import ValidationVisualizer
from .utils import (
    get_vram_usage,
    log_vram_to_tensorboard,
    log_epoch_summary,
    save_full_checkpoint,
    create_epoch_iterator,
)
from medgen.metrics import (
    MetricsTracker,
    create_reconstruction_figure,
    RegionalMetricsTracker,
    compute_msssim,
    compute_psnr,
    compute_lpips,
    compute_dice,
    compute_iou,
    # Unified metrics system
    UnifiedMetrics,
)
from medgen.metrics import FLOPsTracker
from .regional_weighting import RegionalWeightComputer, create_regional_weight_computer
from medgen.models import (
    create_controlnet_for_unet,
    freeze_unet_for_controlnet,
    ControlNetConditionedUNet,
    ControlNetGenerationWrapper,
)

logger = logging.getLogger(__name__)


class DiffusionTrainer(BaseTrainer):
    """Unified diffusion model trainer composing strategy and mode.

    Inherits from BaseTrainer and adds:
    - Strategy pattern for noise prediction (DDPM, RFlow)
    - Mode pattern for input/output formats (seg, bravo, dual, multi)
    - SAM optimizer support
    - ScoreAug transforms for data augmentation
    - Compiled forward paths for performance

    Args:
        cfg: Hydra configuration object containing all settings.
        space: Optional DiffusionSpace for pixel/latent space operations.
            Defaults to PixelSpace (identity, backward compatible).

    Example:
        >>> trainer = DiffusionTrainer(cfg)
        >>> trainer.setup_model(train_dataset)
        >>> trainer.train(train_loader, train_dataset)
    """

    def __init__(self, cfg: DictConfig, space: Optional[DiffusionSpace] = None) -> None:
        # Initialize base trainer (distributed setup, TensorBoard, trackers)
        super().__init__(cfg)

        self.space = space if space is not None else PixelSpace()

        # ─────────────────────────────────────────────────────────────────────
        # Diffusion-specific config
        # ─────────────────────────────────────────────────────────────────────
        self.strategy_name: str = cfg.strategy.name
        self.mode_name: str = cfg.mode.name
        self.image_size: int = cfg.model.image_size
        self.num_timesteps: int = cfg.strategy.num_train_timesteps
        self.eta_min: float = cfg.training.get('eta_min', 1e-6)

        # Perceptual weight (disabled for seg modes - binary masks don't work with VGG features)
        is_seg_mode = self.mode_name in ('seg', 'seg_conditioned')
        self.perceptual_weight: float = 0.0 if is_seg_mode else cfg.training.perceptual_weight

        # Min-SNR weighting
        self.use_min_snr: bool = cfg.training.use_min_snr
        self.min_snr_gamma: float = cfg.training.min_snr_gamma

        # FP32 loss computation (set False to reproduce pre-Jan-7-2026 BF16 behavior)
        self.use_fp32_loss: bool = cfg.training.get('use_fp32_loss', True)
        logger.info(f"[DEBUG] use_fp32_loss = {self.use_fp32_loss} (from config override)")

        # SAM (Sharpness-Aware Minimization)
        # DEPRECATED: SAM/ASAM requires 2x compute cost with minimal benefit for diffusion models.
        # Consider using gradient_noise, curriculum, or timestep_jitter instead.
        sam_cfg = cfg.training.get('sam', {})
        self.use_sam: bool = sam_cfg.get('enabled', False)
        self.sam_rho: float = sam_cfg.get('rho', 0.05)
        self.sam_adaptive: bool = sam_cfg.get('adaptive', False)
        if self.use_sam:
            import warnings
            warnings.warn(
                "SAM/ASAM optimizer is DEPRECATED and will be removed in a future version. "
                "SAM requires 2x compute cost with minimal benefit for diffusion models. "
                "Consider using gradient_noise, curriculum, or timestep_jitter instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Optimizer settings
        optimizer_cfg = cfg.training.get('optimizer', {})
        self.weight_decay: float = optimizer_cfg.get('weight_decay', 0.0)

        # EMA (from config, not in base trainer)
        self.use_ema: bool = cfg.training.use_ema
        self.ema_decay: float = cfg.training.ema.decay
        self.ema: Optional[EMA] = None

        # Initialize strategy and mode
        self.strategy = self._create_strategy(self.strategy_name)
        self.mode = self._create_mode(self.mode_name)
        self.scheduler = self.strategy.setup_scheduler(self.num_timesteps, self.image_size)

        # Initialize metrics tracker
        self.metrics = MetricsTracker(
            cfg=cfg,
            device=self.device,
            writer=self.writer,
            save_dir=self.save_dir,
            is_main_process=self.is_main_process,
            is_conditional=self.mode.is_conditional,
        )
        self.metrics.set_scheduler(self.scheduler)

        # Disable LPIPS for seg modes (binary masks don't work with VGG features)
        if is_seg_mode:
            self.metrics.log_lpips = False

        # Initialize unified metrics system for consistent logging
        # Note: UnifiedMetrics is initialized in train() when writer is fully set up
        self._unified_metrics: Optional[UnifiedMetrics] = None

        # Initialize model components (set during setup_model)
        self.perceptual_loss_fn: Optional[nn.Module] = None

        # Validation loader (set in train())
        self.val_loader: Optional[DataLoader] = None

        # Visualization helper (initialized in setup_model after strategy is ready)
        self.visualizer: Optional[ValidationVisualizer] = None

        # Global step counter (for gradient noise decay, etc.)
        self._global_step: int = 0
        self._current_epoch: int = 0  # Set in train_epoch

        # ScoreAug initialization (applies transforms to noisy data)
        self.score_aug = None
        self.use_omega_conditioning = False
        self.use_mode_intensity_scaling = False
        self._apply_mode_intensity_scale = None  # Function reference (lazy import)
        score_aug_cfg = cfg.training.get('score_aug', {})
        if score_aug_cfg.get('enabled', False):
            from medgen.augmentation import ScoreAugTransform
            self.score_aug = ScoreAugTransform(
                rotation=score_aug_cfg.get('rotation', True),
                flip=score_aug_cfg.get('flip', True),
                translation=score_aug_cfg.get('translation', False),
                cutout=score_aug_cfg.get('cutout', False),
                brightness=score_aug_cfg.get('brightness', False),
                brightness_range=score_aug_cfg.get('brightness_range', 1.2),
                compose=score_aug_cfg.get('compose', False),
                compose_prob=score_aug_cfg.get('compose_prob', 0.5),
            )
            self.use_omega_conditioning = score_aug_cfg.get('use_omega_conditioning', False)

            # Mode intensity scaling: scales input by modality-specific factor
            # Forces model to use mode conditioning (similar to how rotation requires omega)
            self.use_mode_intensity_scaling = score_aug_cfg.get('mode_intensity_scaling', False)
            if self.use_mode_intensity_scaling:
                from medgen.augmentation import apply_mode_intensity_scale
                self._apply_mode_intensity_scale = apply_mode_intensity_scale

            # Validate: rotation/flip require omega conditioning per ScoreAug paper
            # Gaussian noise is rotation-invariant, allowing model to "cheat" without conditioning
            has_spatial_transforms = (
                score_aug_cfg.get('rotation', True) or score_aug_cfg.get('flip', True)
            )
            if has_spatial_transforms and not self.use_omega_conditioning:
                raise ValueError(
                    "ScoreAug rotation/flip require omega conditioning (per ScoreAug paper). "
                    "Gaussian noise is rotation-invariant, allowing the model to detect "
                    "rotation from noise patterns and 'cheat' by inverting before denoising. "
                    "Fix: Set training.score_aug.use_omega_conditioning=true"
                )

            # Validate: mode_intensity_scaling requires omega conditioning + mode embedding
            if self.use_mode_intensity_scaling and not self.use_omega_conditioning:
                raise ValueError(
                    "Mode intensity scaling requires omega conditioning. "
                    "Fix: Set training.score_aug.use_omega_conditioning=true"
                )

            if self.is_main_process:
                transforms = []
                if score_aug_cfg.get('rotation', True):
                    transforms.append('rotation')
                if score_aug_cfg.get('flip', True):
                    transforms.append('flip')
                if score_aug_cfg.get('translation', False):
                    transforms.append('translation')
                if score_aug_cfg.get('cutout', False):
                    transforms.append('cutout')
                if score_aug_cfg.get('brightness', False):
                    transforms.append(f"brightness({score_aug_cfg.get('brightness_range', 1.2)})")
                n_options = len(transforms) + 1
                logger.info(
                    f"ScoreAug enabled: transforms=[{', '.join(transforms)}], "
                    f"each with 1/{n_options} prob (uniform), "
                    f"omega_conditioning={self.use_omega_conditioning}, "
                    f"mode_intensity_scaling={self.use_mode_intensity_scaling}"
                )

        # SDA (Shifted Data Augmentation) initialization
        # Unlike ScoreAug (transforms noisy data), SDA transforms CLEAN data
        # and uses a shifted noise level to prevent leakage
        self.sda = None
        self.sda_weight = 1.0
        sda_cfg = cfg.training.get('sda', {})
        if sda_cfg.get('enabled', False):
            from medgen.augmentation import SDATransform
            self.sda = SDATransform(
                rotation=sda_cfg.get('rotation', True),
                flip=sda_cfg.get('flip', True),
                noise_shift=sda_cfg.get('noise_shift', 0.1),
                prob=sda_cfg.get('prob', 0.5),
            )
            self.sda_weight = sda_cfg.get('weight', 1.0)

            if self.is_main_process:
                transforms = []
                if sda_cfg.get('rotation', True):
                    transforms.append('rotation')
                if sda_cfg.get('flip', True):
                    transforms.append('flip')
                logger.info(
                    f"SDA enabled: transforms=[{', '.join(transforms)}], "
                    f"noise_shift={sda_cfg.get('noise_shift', 0.1)}, "
                    f"prob={sda_cfg.get('prob', 0.5)}, weight={self.sda_weight}"
                )

            # Validate: SDA and ScoreAug shouldn't both be enabled
            # (they do similar things in different ways)
            if self.score_aug is not None:
                logger.warning(
                    "Both SDA and ScoreAug are enabled. This is unusual - "
                    "SDA transforms clean data, ScoreAug transforms noisy data. "
                    "Consider using only one for clearer experiments."
                )

        # Mode embedding for multi-modality training
        self.use_mode_embedding = cfg.mode.get('use_mode_embedding', False)
        self.mode_embedding_strategy = cfg.mode.get('mode_embedding_strategy', 'full')
        self.mode_embedding_dropout = cfg.mode.get('mode_embedding_dropout', 0.2)
        self.late_mode_start_level = cfg.mode.get('late_mode_start_level', 2)

        if self.use_mode_embedding and self.is_main_process:
            logger.info(
                f"Mode embedding enabled: strategy={self.mode_embedding_strategy}, "
                f"dropout={self.mode_embedding_dropout}, late_start_level={self.late_mode_start_level}"
            )

        # Size bin embedding for seg_conditioned mode
        self.use_size_bin_embedding = (self.mode_name == 'seg_conditioned')
        if self.use_size_bin_embedding:
            size_bin_cfg = cfg.mode.get('size_bins', {})
            self.size_bin_num_bins = size_bin_cfg.get('num_bins', 6)
            self.size_bin_max_count = size_bin_cfg.get('max_count_per_bin', 10)
            self.size_bin_embed_dim = size_bin_cfg.get('embedding_dim', 32)
            if self.is_main_process:
                logger.info(
                    f"Size bin embedding enabled: num_bins={self.size_bin_num_bins}, "
                    f"max_count={self.size_bin_max_count}, embed_dim={self.size_bin_embed_dim}"
                )

        # DC-AE 1.5: Augmented Diffusion Training (channel masking for latent diffusion)
        aug_diff_cfg = cfg.training.get('augmented_diffusion', {})
        self.augmented_diffusion_enabled: bool = aug_diff_cfg.get('enabled', False)
        self.aug_diff_min_channels: int = aug_diff_cfg.get('min_channels', 16)
        self.aug_diff_channel_step: int = aug_diff_cfg.get('channel_step', 4)
        self._aug_diff_channel_steps: Optional[List[int]] = None  # Computed lazily

        if self.augmented_diffusion_enabled and self.is_main_process:
            # Only effective for latent diffusion
            if self.space.scale_factor > 1:
                logger.info(
                    f"DC-AE 1.5 Augmented Diffusion Training enabled: "
                    f"min_channels={self.aug_diff_min_channels}, step={self.aug_diff_channel_step}"
                )
            else:
                logger.warning(
                    "Augmented Diffusion Training enabled but using pixel space. "
                    "This has no effect - only applies to latent diffusion."
                )

        # Log gradient noise configuration
        grad_noise_cfg = cfg.training.get('gradient_noise', {})
        if grad_noise_cfg.get('enabled', False) and self.is_main_process:
            logger.info(
                f"Gradient noise injection enabled: "
                f"sigma={grad_noise_cfg.get('sigma', 0.01)}, decay={grad_noise_cfg.get('decay', 0.55)}"
            )

        # Log curriculum timestep configuration
        curriculum_cfg = cfg.training.get('curriculum', {})
        if curriculum_cfg.get('enabled', False) and self.is_main_process:
            logger.info(
                f"Curriculum timestep scheduling enabled: "
                f"warmup_epochs={curriculum_cfg.get('warmup_epochs', 50)}, "
                f"range [{curriculum_cfg.get('min_t_start', 0.0)}-{curriculum_cfg.get('max_t_start', 0.3)}] -> "
                f"[{curriculum_cfg.get('min_t_end', 0.0)}-{curriculum_cfg.get('max_t_end', 1.0)}]"
            )

        # Log "clean" regularization techniques
        jitter_cfg = cfg.training.get('timestep_jitter', {})
        if jitter_cfg.get('enabled', False) and self.is_main_process:
            logger.info(f"Timestep jitter enabled: std={jitter_cfg.get('std', 0.05)}")

        self_cond_cfg = cfg.training.get('self_conditioning', {})
        if self_cond_cfg.get('enabled', False) and self.is_main_process:
            logger.info(f"Self-conditioning enabled: prob={self_cond_cfg.get('prob', 0.5)}")

        feat_cfg = cfg.training.get('feature_perturbation', {})
        if feat_cfg.get('enabled', False) and self.is_main_process:
            logger.info(
                f"Feature perturbation enabled: std={feat_cfg.get('std', 0.1)}, "
                f"layers={feat_cfg.get('layers', ['mid'])}"
            )

        noise_aug_cfg = cfg.training.get('noise_augmentation', {})
        if noise_aug_cfg.get('enabled', False) and self.is_main_process:
            logger.info(f"Noise augmentation enabled: std={noise_aug_cfg.get('std', 0.1)}")

        # Region-weighted loss (per-pixel weighting by tumor size)
        # Only applies to conditional modes (bravo, dual, multi) where seg mask is available
        self.regional_weight_computer: Optional[RegionalWeightComputer] = None
        rw_cfg = cfg.training.get('regional_weighting', {})
        if rw_cfg.get('enabled', False):
            if self.mode.is_conditional:
                self.regional_weight_computer = create_regional_weight_computer(cfg)
                if self.is_main_process:
                    weights = rw_cfg.get('weights', {})
                    logger.info(
                        f"Region-weighted loss enabled: "
                        f"tiny={weights.get('tiny', 2.5)}, small={weights.get('small', 1.8)}, "
                        f"medium={weights.get('medium', 1.4)}, large={weights.get('large', 1.2)}, "
                        f"bg={rw_cfg.get('background_weight', 1.0)}"
                    )
            else:
                if self.is_main_process:
                    logger.warning(
                        "Regional weighting enabled but mode is not conditional (seg mode). "
                        "Skipping - regional weighting requires segmentation mask as conditioning."
                    )

        # Generation quality metrics (KID, CMMD) for overfitting detection
        self._gen_metrics: Optional['GenerationMetrics'] = None
        gen_cfg = cfg.training.get('generation_metrics', {})
        if gen_cfg.get('enabled', False):
            from medgen.pipeline.metrics.generation import GenerationMetricsConfig, GenerationMetrics
            # Use training batch_size by default for torch.compile consistency
            feature_batch_size = gen_cfg.get('feature_batch_size', None)
            if feature_batch_size is None:
                feature_batch_size = cfg.training.batch_size
            self._gen_metrics_config = GenerationMetricsConfig(
                enabled=True,
                samples_per_epoch=gen_cfg.get('samples_per_epoch', 100),
                samples_extended=gen_cfg.get('samples_extended', 500),
                samples_test=gen_cfg.get('samples_test', 1000),
                steps_per_epoch=gen_cfg.get('steps_per_epoch', 10),
                steps_extended=gen_cfg.get('steps_extended', 25),
                steps_test=gen_cfg.get('steps_test', 50),
                cache_dir=gen_cfg.get('cache_dir', '.cache/generation_features'),
                feature_batch_size=feature_batch_size,
            )
            if self.is_main_process:
                logger.info(
                    f"Generation metrics enabled: {self._gen_metrics_config.samples_per_epoch} samples/epoch "
                    f"({self._gen_metrics_config.steps_per_epoch} steps), "
                    f"{self._gen_metrics_config.samples_extended} samples/extended "
                    f"({self._gen_metrics_config.steps_extended} steps)"
                )
        else:
            self._gen_metrics_config = None

        # ControlNet configuration (for pixel-resolution conditioning in latent diffusion)
        controlnet_cfg = cfg.get('controlnet', {})
        self.use_controlnet: bool = controlnet_cfg.get('enabled', False)
        self.controlnet_freeze_unet: bool = controlnet_cfg.get('freeze_unet', True)
        self.controlnet_scale: float = controlnet_cfg.get('conditioning_scale', 1.0)
        self.controlnet: Optional[nn.Module] = None

        if self.use_controlnet and self.is_main_process:
            logger.info(
                f"ControlNet enabled: freeze_unet={self.controlnet_freeze_unet}, "
                f"conditioning_scale={self.controlnet_scale}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # DC-AE 1.5: Augmented Diffusion Training Methods
    # ─────────────────────────────────────────────────────────────────────────

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
            tensor: Input tensor [B, C, H, W] to get shape from.

        Returns:
            Mask tensor [1, C, 1, 1] for broadcasting.
        """
        C = tensor.shape[1]
        steps = self._get_aug_diff_channel_steps(C)
        c_prime = random.choice(steps)

        mask = torch.zeros(1, C, 1, 1, device=tensor.device, dtype=tensor.dtype)
        mask[:, :c_prime, :, :] = 1.0
        return mask

    def _create_fallback_save_dir(self) -> str:
        """Create fallback save directory for diffusion trainer."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = self.cfg.training.get('name', '')
        # Use cfg directly since instance attributes may not be set yet
        strategy_name = self.cfg.strategy.name
        mode_name = self.cfg.mode.name
        image_size = self.cfg.model.image_size
        run_name = f"{exp_name}{strategy_name}_{image_size}_{timestamp}"
        return os.path.join(self.cfg.paths.model_dir, 'diffusion_2d', mode_name, run_name)

    def _create_strategy(self, strategy: str) -> DiffusionStrategy:
        """Create a diffusion strategy instance."""
        strategies: Dict[str, type] = {
            'ddpm': DDPMStrategy,
            'rflow': RFlowStrategy
        }
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")
        return strategies[strategy]()

    def _create_mode(self, mode: str) -> TrainingMode:
        """Create a training mode instance."""
        modes: Dict[str, type] = {
            'seg': SegmentationMode,
            'seg_conditioned': SegmentationConditionedMode,
            'bravo': ConditionalSingleMode,
            'dual': ConditionalDualMode,
            'multi': MultiModalityMode,
        }
        if mode not in modes:
            raise ValueError(f"Unknown mode: {mode}. Choose from {list(modes.keys())}")

        if mode == ModeType.DUAL or mode == 'dual':
            image_keys = list(self.cfg.mode.image_keys) if 'image_keys' in self.cfg.mode else None
            return ConditionalDualMode(image_keys)

        if mode == 'multi':
            image_keys = list(self.cfg.mode.image_keys) if 'image_keys' in self.cfg.mode else None
            return MultiModalityMode(image_keys)

        if mode == 'seg_conditioned':
            size_bin_config = dict(self.cfg.mode.get('size_bins', {})) if 'size_bins' in self.cfg.mode else None
            return SegmentationConditionedMode(size_bin_config)

        return modes[mode]()

    def setup_model(self, train_dataset: Dataset) -> None:
        """Initialize model, optimizer, and loss functions.

        Args:
            train_dataset: Training dataset for model config extraction.
        """
        model_cfg = self.mode.get_model_config()

        # Adjust channels for latent space
        in_channels = self.space.get_latent_channels(model_cfg['in_channels'])
        out_channels = self.space.get_latent_channels(model_cfg['out_channels'])

        # For ControlNet: conditioning goes through ControlNet, not concatenation
        # UNet in_channels = out_channels (no +1 for conditioning)
        if self.use_controlnet:
            in_channels = out_channels
            if self.is_main_process:
                logger.info(f"ControlNet mode: UNet in_channels={in_channels} (no conditioning concatenation)")

        if self.is_main_process and self.space.scale_factor > 1:
            logger.info(f"Latent space: {model_cfg['in_channels']} -> {in_channels} channels, "
                       f"scale factor {self.space.scale_factor}x")

        # Get model type and check if transformer-based
        self.model_type = get_model_type(self.cfg)
        self.is_transformer = is_transformer_model(self.cfg)

        # Create raw model via factory
        if self.is_transformer:
            raw_model = create_diffusion_model(self.cfg, self.device, in_channels, out_channels)

            if self.use_omega_conditioning or self.use_mode_embedding:
                raise ValueError(
                    "Omega/mode conditioning wrappers are not yet supported for transformer models. "
                    "Either use model=default (UNet) or disable omega_conditioning/mode_embedding."
                )
        else:
            channels = tuple(self.cfg.model.channels)
            attention_levels = tuple(self.cfg.model.attention_levels)
            num_res_blocks = self.cfg.model.num_res_blocks
            num_head_channels = self.cfg.model.num_head_channels

            raw_model = DiffusionModelUNet(
                spatial_dims=self.cfg.model.get('spatial_dims', 2),
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                attention_levels=attention_levels,
                num_res_blocks=num_res_blocks,
                num_head_channels=num_head_channels
            ).to(self.device)

        # Determine if DDPOptimizer should be disabled for large models
        disable_ddp_opt = self.cfg.training.get('disable_ddp_optimizer', False)
        if self.mode_name == ModeType.DUAL and self.image_size >= 256:
            disable_ddp_opt = True

        use_compile = self.cfg.training.get('use_compile', True)

        # Handle embedding wrappers (UNet only)
        if not self.is_transformer:
            channels = tuple(self.cfg.model.channels)
            time_embed_dim = 4 * channels[0]
        else:
            time_embed_dim = None

        # Handle embedding wrappers: omega, mode, or both
        if not self.is_transformer and (self.use_omega_conditioning or self.use_mode_embedding):
            from medgen.data import create_conditioning_wrapper
            wrapper, wrapper_name = create_conditioning_wrapper(
                model=raw_model,
                use_omega=self.use_omega_conditioning,
                use_mode=self.use_mode_embedding,
                embed_dim=time_embed_dim,
                mode_strategy=self.mode_embedding_strategy,
                mode_dropout_prob=self.mode_embedding_dropout,
                late_mode_start_level=self.late_mode_start_level,
            )
            wrapper = wrapper.to(self.device)

            if self.is_main_process:
                logger.info(f"Conditioning: {wrapper_name} wrapper applied (embed_dim={time_embed_dim})")

            if use_compile:
                wrapper.model = torch.compile(wrapper.model, mode="default")
                if self.is_main_process:
                    logger.info(f"Single-GPU: Compiled inner UNet ({wrapper_name} wrapper uncompiled)")

            self.model = wrapper
            self.model_raw = wrapper

        else:
            self.model, self.model_raw = wrap_model_for_training(
                raw_model,
                use_multi_gpu=self.use_multi_gpu,
                local_rank=self.local_rank if self.use_multi_gpu else 0,
                use_compile=use_compile,
                compile_mode="default",
                disable_ddp_optimizer=disable_ddp_opt,
                is_main_process=self.is_main_process,
            )

        # Block DDP with embedding wrappers - embeddings won't sync across GPUs
        if self.use_multi_gpu and (self.use_omega_conditioning or self.use_mode_embedding):
            raise ValueError(
                "DDP is not compatible with embedding wrappers (ScoreAug, ModeEmbed). "
                "Embeddings would NOT be synchronized across GPUs, causing silent training corruption. "
                "Either disable DDP (use single GPU) or disable omega_conditioning/mode_embedding."
            )

        # Handle size bin embedding for seg_conditioned mode
        if self.use_size_bin_embedding:
            from medgen.data import SizeBinModelWrapper

            if self.is_transformer:
                raise ValueError(
                    "Size bin embedding is not supported with transformer models. "
                    "Use model=default (UNet) for seg_conditioned mode."
                )

            # Get time_embed_dim from model
            if time_embed_dim is None:
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

            # Block DDP with size bin embedding
            if self.use_multi_gpu:
                raise ValueError(
                    "DDP is not compatible with size bin embedding. "
                    "Embeddings would NOT be synchronized across GPUs. "
                    "Use single GPU for seg_conditioned mode."
                )

        # Setup ControlNet for pixel-resolution conditioning in latent diffusion
        if self.use_controlnet:
            if self.is_transformer:
                raise ValueError("ControlNet is not supported with transformer models. Use model=default (UNet).")

            # Determine latent channels for ControlNet
            latent_channels = out_channels  # Same as UNet in_channels after ControlNet adjustment

            # Create ControlNet matching UNet architecture
            self.controlnet = create_controlnet_for_unet(
                unet=self.model_raw,
                cfg=self.cfg,
                device=self.device,
                spatial_dims=2,
                latent_channels=latent_channels,
            )

            # Enable gradient checkpointing if requested
            controlnet_cfg = self.cfg.get('controlnet', {})
            if controlnet_cfg.get('gradient_checkpointing', False):
                if hasattr(self.controlnet, 'enable_gradient_checkpointing'):
                    self.controlnet.enable_gradient_checkpointing()
                    if self.is_main_process:
                        logger.info("ControlNet gradient checkpointing enabled")

            # Freeze UNet if configured (Stage 2 training)
            if self.controlnet_freeze_unet:
                freeze_unet_for_controlnet(self.model_raw)

            # Create combined model wrapper
            self.model = ControlNetConditionedUNet(
                unet=self.model_raw,
                controlnet=self.controlnet,
                conditioning_scale=self.controlnet_scale,
            )

            if self.is_main_process:
                num_params = sum(p.numel() for p in self.controlnet.parameters())
                logger.info(f"ControlNet created: {num_params:,} parameters")

        # Setup perceptual loss (skip for seg modes where perceptual_weight=0)
        # This saves ~200MB GPU memory from loading ResNet50
        cache_dir = getattr(self.cfg.paths, 'cache_dir', None)
        if self.perceptual_weight > 0:
            self.perceptual_loss_fn = PerceptualLoss(
                spatial_dims=2,
                network_type="radimagenet_resnet50",
                cache_dir=cache_dir,
                pretrained=True,
                device=self.device,
                use_compile=use_compile,
            )
        else:
            self.perceptual_loss_fn = None
            if self.is_main_process:
                logger.info("Perceptual loss disabled (perceptual_weight=0), skipping ResNet50 loading")

        # Compile fused forward pass setup
        # Compiled forward doesn't support mode_id parameter, so disable when using
        # mode embedding or omega conditioning (wrappers need extra kwargs)
        compile_fused = self.cfg.training.get('compile_fused_forward', True)
        if self.use_multi_gpu:
            compile_fused = False
        elif self.space.scale_factor > 1:
            compile_fused = False
        elif self.use_min_snr:
            compile_fused = False
        elif self.regional_weight_computer is not None:
            compile_fused = False
        elif self.score_aug is not None:
            compile_fused = False
        elif self.sda is not None:
            compile_fused = False
        elif self.use_sam:
            compile_fused = False
        elif self.use_mode_embedding:
            compile_fused = False
        elif self.use_omega_conditioning:
            compile_fused = False
        elif self.augmented_diffusion_enabled:
            compile_fused = False
        elif self.use_controlnet:
            compile_fused = False
        elif self.use_size_bin_embedding:
            compile_fused = False

        self._setup_compiled_forward(compile_fused)

        # Setup optimizer (with optional SAM wrapper)
        # Determine which parameters to train
        if self.use_controlnet:
            if self.controlnet_freeze_unet:
                # Stage 2: Only train ControlNet
                train_params = list(self.controlnet.parameters())
                if self.is_main_process:
                    trainable = sum(p.numel() for p in train_params if p.requires_grad)
                    logger.info(f"Training only ControlNet ({trainable:,} trainable params, UNet frozen)")
            else:
                # Joint training: Both UNet and ControlNet
                train_params = list(self.model_raw.parameters()) + list(self.controlnet.parameters())
                if self.is_main_process:
                    trainable = sum(p.numel() for p in train_params if p.requires_grad)
                    logger.info(f"Joint training: UNet + ControlNet ({trainable:,} trainable params)")
        else:
            train_params = self.model_raw.parameters()

        if self.use_sam:
            self.optimizer = SAM(
                train_params,
                base_optimizer=AdamW,
                rho=self.sam_rho,
                adaptive=self.sam_adaptive,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            if self.is_main_process:
                logger.info(f"Using SAM optimizer (rho={self.sam_rho}, adaptive={self.sam_adaptive})")
        else:
            self.optimizer = AdamW(
                train_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        if self.is_main_process and self.weight_decay > 0:
            logger.info(f"Using weight decay: {self.weight_decay}")

        # Learning rate scheduler (cosine or constant)
        scheduler_type = self.cfg.training.get('scheduler', 'cosine')
        if scheduler_type == 'constant':
            self.lr_scheduler = create_warmup_constant_scheduler(
                self.optimizer,
                warmup_epochs=self.warmup_epochs,
                total_epochs=self.n_epochs,
            )
            if self.is_main_process:
                logger.info("Using constant LR scheduler (warmup then constant)")
        else:
            self.lr_scheduler = create_warmup_cosine_scheduler(
                self.optimizer,
                warmup_epochs=self.warmup_epochs,
                total_epochs=self.n_epochs,
                eta_min=self.eta_min,
            )

        # Create EMA wrapper if enabled
        self._setup_ema()

        # Initialize visualization helper
        self.visualizer = ValidationVisualizer(
            cfg=self.cfg,
            strategy=self.strategy,
            mode=self.mode,
            metrics=self.metrics,
            writer=self.writer,
            save_dir=self.save_dir,
            device=self.device,
            is_main_process=self.is_main_process,
            space=self.space,
            use_controlnet=self.use_controlnet,
            controlnet=self.controlnet,
        )

        # Initialize generation metrics if enabled
        # Skip for seg/seg_conditioned modes - generation metrics use image feature extractors
        # which don't make sense for binary masks, and seg_conditioned uses size_bins conditioning
        if self._gen_metrics_config is not None and self._gen_metrics_config.enabled:
            if self.mode_name in ('seg', 'seg_conditioned'):
                if self.is_main_process:
                    logger.info(f"{self.mode_name} mode: generation metrics disabled (binary masks)")
                self._gen_metrics = None
            else:
                from medgen.pipeline.metrics.generation import GenerationMetrics
                self._gen_metrics = GenerationMetrics(
                    self._gen_metrics_config,
                    self.device,
                    self.save_dir,
                    space=self.space,
                )
                if self.is_main_process:
                    logger.info("Generation metrics initialized (caching happens at training start)")

        # Save metadata
        if self.is_main_process:
            self._save_metadata()

        # Initialize metrics accumulators
        self.metrics.init_accumulators()

        # Setup feature perturbation hooks if enabled
        self._setup_feature_perturbation()

    def _setup_ema(self) -> None:
        """Setup EMA wrapper if enabled."""
        if self.use_ema:
            self.ema = EMA(
                self.model_raw,
                beta=self.ema_decay,
                update_after_step=self.cfg.training.ema.update_after_step,
                update_every=self.cfg.training.ema.update_every,
            )
            if self.is_main_process:
                logger.info(f"EMA enabled with decay={self.ema_decay}")

    def _update_ema(self) -> None:
        """Update EMA model weights."""
        if self.ema is not None:
            self.ema.update()

    def _add_gradient_noise(self, step: int) -> None:
        """Add Gaussian noise to gradients for regularization.

        Noise decays over training as: sigma / (1 + step)^decay
        Reference: "Adding Gradient Noise Improves Learning" (Neelakantan et al., 2015)

        Args:
            step: Current global training step.
        """
        grad_noise_cfg = self.cfg.training.get('gradient_noise', {})
        if not grad_noise_cfg.get('enabled', False):
            return

        sigma = grad_noise_cfg.get('sigma', 0.01)
        decay = grad_noise_cfg.get('decay', 0.55)

        # Decay noise over training
        noise_std = sigma / (1 + step) ** decay

        for param in self.model_raw.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_std
                param.grad.add_(noise)

    def _get_curriculum_range(self, epoch: int) -> Optional[Tuple[float, float]]:
        """Get timestep range for curriculum learning.

        Linearly interpolates from start range to end range over warmup_epochs.

        Args:
            epoch: Current training epoch.

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

        Increases noise-level diversity without changing output distribution.

        Args:
            timesteps: Original timesteps tensor.

        Returns:
            Jittered timesteps (clamped to valid range).
        """
        jitter_cfg = self.cfg.training.get('timestep_jitter', {})
        if not jitter_cfg.get('enabled', False):
            return timesteps

        std = jitter_cfg.get('std', 0.05)
        # Convert to float, add jitter, convert back
        t_float = timesteps.float() / self.num_timesteps
        jitter = torch.randn_like(t_float) * std
        t_jittered = (t_float + jitter).clamp(0.0, 1.0)
        return (t_jittered * self.num_timesteps).long()

    def _apply_noise_augmentation(
        self,
        noise: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Add perturbation to noise vector for regularization.

        Increases noise diversity without affecting what model learns to output.

        Args:
            noise: Original noise tensor or dict of tensors.

        Returns:
            Perturbed noise (renormalized to maintain variance).
        """
        noise_aug_cfg = self.cfg.training.get('noise_augmentation', {})
        if not noise_aug_cfg.get('enabled', False):
            return noise

        std = noise_aug_cfg.get('std', 0.1)

        if isinstance(noise, dict):
            perturbed = {}
            for k, v in noise.items():
                perturbation = torch.randn_like(v) * std
                # Add perturbation and renormalize to maintain unit variance
                perturbed_v = v + perturbation
                perturbed[k] = perturbed_v / perturbed_v.std() * v.std()
            return perturbed
        else:
            perturbation = torch.randn_like(noise) * std
            perturbed = noise + perturbation
            return perturbed / perturbed.std() * noise.std()

    def _setup_feature_perturbation(self) -> None:
        """Setup forward hooks for feature perturbation."""
        self._feature_hooks = []
        feat_cfg = self.cfg.training.get('feature_perturbation', {})

        if not feat_cfg.get('enabled', False):
            return

        std = feat_cfg.get('std', 0.1)
        layers = feat_cfg.get('layers', ['mid'])

        def make_hook(noise_std):
            def hook(module, input, output):
                if self.model.training:
                    noise = torch.randn_like(output) * noise_std
                    return output + noise
                return output
            return hook

        # Register hooks on specified layers
        # UNet structure: down_blocks, mid_block, up_blocks
        if hasattr(self.model_raw, 'mid_block') and 'mid' in layers:
            handle = self.model_raw.mid_block.register_forward_hook(make_hook(std))
            self._feature_hooks.append(handle)

        if hasattr(self.model_raw, 'down_blocks') and 'encoder' in layers:
            for block in self.model_raw.down_blocks:
                handle = block.register_forward_hook(make_hook(std))
                self._feature_hooks.append(handle)

        if hasattr(self.model_raw, 'up_blocks') and 'decoder' in layers:
            for block in self.model_raw.up_blocks:
                handle = block.register_forward_hook(make_hook(std))
                self._feature_hooks.append(handle)

    def _remove_feature_perturbation_hooks(self) -> None:
        """Remove feature perturbation hooks."""
        for handle in getattr(self, '_feature_hooks', []):
            handle.remove()
        self._feature_hooks = []

    def _compute_self_conditioning_loss(
        self,
        model_input: torch.Tensor,
        timesteps: torch.Tensor,
        prediction: torch.Tensor,
        mode_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute self-conditioning consistency loss.

        With probability `prob`, runs model a second time and computes
        consistency loss between the two predictions.

        Args:
            model_input: Current model input tensor.
            timesteps: Current timesteps.
            prediction: Current prediction from main forward pass.
            mode_id: Optional mode ID for multi-modality.

        Returns:
            Consistency loss (0 if disabled or skipped this batch).
        """
        self_cond_cfg = self.cfg.training.get('self_conditioning', {})
        if not self_cond_cfg.get('enabled', False):
            return torch.tensor(0.0, device=model_input.device)

        prob = self_cond_cfg.get('prob', 0.5)

        # With probability (1-prob), skip self-conditioning
        if random.random() >= prob:
            return torch.tensor(0.0, device=model_input.device)

        # Get second prediction (detached first prediction as reference)
        with torch.no_grad():
            if self.use_mode_embedding:
                prediction_ref = self.model(model_input, timesteps, mode_id=mode_id)
            else:
                prediction_ref = self.model(x=model_input, timesteps=timesteps)
            prediction_ref = prediction_ref.detach()

        # Consistency loss: predictions should be similar
        consistency_loss = F.mse_loss(prediction.float(), prediction_ref.float())

        return consistency_loss

    def _compute_min_snr_weighted_mse(
        self,
        prediction: torch.Tensor,
        images: Union[torch.Tensor, Dict[str, torch.Tensor]],
        noise: Union[torch.Tensor, Dict[str, torch.Tensor]],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss with Min-SNR weighting.

        Applies per-sample SNR-based weights to the MSE loss to prevent
        high-noise timesteps from dominating training.

        Args:
            prediction: Model prediction (noise or velocity).
            images: Original clean images.
            noise: Added noise.
            timesteps: Diffusion timesteps for each sample.

        Returns:
            Weighted MSE loss scalar.
        """
        snr_weights = self.metrics.compute_snr_weights(timesteps)

        # Cast to FP32 for MSE computation (BF16 underflow causes ~15-20% lower loss)
        if isinstance(images, dict):
            keys = list(images.keys())
            if self.strategy_name == 'rflow':
                target_0 = images[keys[0]] - noise[keys[0]]
                target_1 = images[keys[1]] - noise[keys[1]]
            else:
                target_0, target_1 = noise[keys[0]], noise[keys[1]]
            pred_0, pred_1 = prediction[:, 0:1, :, :], prediction[:, 1:2, :, :]
            mse_0 = ((pred_0.float() - target_0.float()) ** 2).mean(dim=(1, 2, 3))
            mse_1 = ((pred_1.float() - target_1.float()) ** 2).mean(dim=(1, 2, 3))
            mse_per_sample = (mse_0 + mse_1) / 2
        else:
            target = images - noise if self.strategy_name == 'rflow' else noise
            mse_per_sample = ((prediction.float() - target.float()) ** 2).mean(dim=(1, 2, 3))

        return (mse_per_sample * snr_weights).mean()

    def _compute_region_weighted_mse(
        self,
        prediction: torch.Tensor,
        images: Union[torch.Tensor, Dict[str, torch.Tensor]],
        noise: Union[torch.Tensor, Dict[str, torch.Tensor]],
        seg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss with per-pixel regional weighting.

        Applies higher weights to small tumor regions based on RANO-BM
        size classification using Feret diameter.

        Args:
            prediction: Model prediction (noise or velocity).
            images: Original clean images.
            noise: Added noise.
            seg_mask: Binary segmentation mask [B, 1, H, W].

        Returns:
            Region-weighted MSE loss scalar.
        """
        # Compute weight map from segmentation mask
        weight_map = self.regional_weight_computer(seg_mask)  # [B, 1, H, W]

        # Compute per-pixel MSE
        if isinstance(images, dict):
            keys = list(images.keys())
            if self.strategy_name == 'rflow':
                target_0 = images[keys[0]] - noise[keys[0]]
                target_1 = images[keys[1]] - noise[keys[1]]
            else:
                target_0, target_1 = noise[keys[0]], noise[keys[1]]
            pred_0, pred_1 = prediction[:, 0:1, :, :], prediction[:, 1:2, :, :]

            # Per-pixel MSE with weights
            mse_0 = (pred_0.float() - target_0.float()) ** 2  # [B, 1, H, W]
            mse_1 = (pred_1.float() - target_1.float()) ** 2  # [B, 1, H, W]

            # Apply regional weights
            weighted_mse_0 = (mse_0 * weight_map).mean()
            weighted_mse_1 = (mse_1 * weight_map).mean()
            return (weighted_mse_0 + weighted_mse_1) / 2
        else:
            target = images - noise if self.strategy_name == 'rflow' else noise
            mse = (prediction.float() - target.float()) ** 2  # [B, C, H, W]

            # Apply regional weights (broadcast over channels)
            weighted_mse = (mse * weight_map).mean()
            return weighted_mse

    def _setup_compiled_forward(self, enabled: bool) -> None:
        """Setup compiled forward functions for fused model + loss computation."""
        self._use_compiled_forward = enabled

        if not enabled:
            self._compiled_forward_single = None
            self._compiled_forward_dual = None
            return

        # Capture use_fp32_loss for closure
        use_fp32 = self.use_fp32_loss

        # Define and compile forward functions
        # When use_fp32=False, reproduces pre-Jan-7-2026 BF16 behavior
        def _forward_single(
            model: nn.Module,
            perceptual_fn: nn.Module,
            model_input: torch.Tensor,
            timesteps: torch.Tensor,
            images: torch.Tensor,
            noise: torch.Tensor,
            noisy_images: torch.Tensor,
            perceptual_weight: float,
            strategy_name: str,
            num_train_timesteps: int,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            prediction = model(model_input, timesteps)

            if strategy_name == 'rflow':
                t_normalized = timesteps.float() / float(num_train_timesteps)
                t_expanded = t_normalized.view(-1, 1, 1, 1)
                predicted_clean = torch.clamp(noisy_images + t_expanded * prediction, 0, 1)
            else:
                predicted_clean = torch.clamp(noisy_images - prediction, 0, 1)

            if strategy_name == 'rflow':
                target = images - noise
                if use_fp32:
                    # FP32: accurate gradients (recommended)
                    mse_loss = ((prediction.float() - target.float()) ** 2).mean()
                else:
                    # BF16: reproduces old behavior (suboptimal gradients)
                    mse_loss = ((prediction - target) ** 2).mean()
            else:
                if use_fp32:
                    mse_loss = ((prediction.float() - noise.float()) ** 2).mean()
                else:
                    mse_loss = ((prediction - noise) ** 2).mean()

            # Perceptual loss always uses FP32 (pretrained networks need it)
            p_loss = perceptual_fn(predicted_clean.float(), images.float()) if perceptual_weight > 0 else torch.tensor(0.0, device=images.device)
            total_loss = mse_loss + perceptual_weight * p_loss
            return total_loss, mse_loss, p_loss, predicted_clean

        def _forward_dual(
            model: nn.Module,
            perceptual_fn: nn.Module,
            model_input: torch.Tensor,
            timesteps: torch.Tensor,
            images_0: torch.Tensor,
            images_1: torch.Tensor,
            noise_0: torch.Tensor,
            noise_1: torch.Tensor,
            noisy_0: torch.Tensor,
            noisy_1: torch.Tensor,
            perceptual_weight: float,
            strategy_name: str,
            num_train_timesteps: int,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            prediction = model(model_input, timesteps)
            pred_0 = prediction[:, 0:1, :, :]
            pred_1 = prediction[:, 1:2, :, :]

            if strategy_name == 'rflow':
                t_normalized = timesteps.float() / float(num_train_timesteps)
                t_expanded = t_normalized.view(-1, 1, 1, 1)
                clean_0 = torch.clamp(noisy_0 + t_expanded * pred_0, 0, 1)
                clean_1 = torch.clamp(noisy_1 + t_expanded * pred_1, 0, 1)
                target_0 = images_0 - noise_0
                target_1 = images_1 - noise_1
                if use_fp32:
                    # FP32: accurate gradients (recommended)
                    mse_loss = (((pred_0.float() - target_0.float()) ** 2).mean() + ((pred_1.float() - target_1.float()) ** 2).mean()) / 2
                else:
                    # BF16: reproduces old behavior (suboptimal gradients)
                    mse_loss = (((pred_0 - target_0) ** 2).mean() + ((pred_1 - target_1) ** 2).mean()) / 2
            else:
                clean_0 = torch.clamp(noisy_0 - pred_0, 0, 1)
                clean_1 = torch.clamp(noisy_1 - pred_1, 0, 1)
                if use_fp32:
                    mse_loss = (((pred_0.float() - noise_0.float()) ** 2).mean() + ((pred_1.float() - noise_1.float()) ** 2).mean()) / 2
                else:
                    mse_loss = (((pred_0 - noise_0) ** 2).mean() + ((pred_1 - noise_1) ** 2).mean()) / 2

            if perceptual_weight > 0:
                # Perceptual loss always uses FP32 (pretrained networks need it)
                p_loss = (perceptual_fn(clean_0.float(), images_0.float()) + perceptual_fn(clean_1.float(), images_1.float())) / 2
            else:
                p_loss = torch.tensor(0.0, device=images_0.device)

            total_loss = mse_loss + perceptual_weight * p_loss
            return total_loss, mse_loss, p_loss, clean_0, clean_1

        self._compiled_forward_single = torch.compile(
            _forward_single, mode="reduce-overhead", fullgraph=True
        )
        self._compiled_forward_dual = torch.compile(
            _forward_dual, mode="reduce-overhead", fullgraph=True
        )

        if self.is_main_process:
            precision = "FP32" if use_fp32 else "BF16 (legacy)"
            logger.info(f"Compiled fused forward passes (CUDA graphs enabled, MSE precision: {precision})")

    def _get_trainer_type(self) -> str:
        """Return trainer type for metadata."""
        return 'diffusion'

    def _get_metadata_extra(self) -> Dict[str, Any]:
        """Return diffusion-specific metadata."""
        return {
            'strategy': self.strategy_name,
            'mode': self.mode_name,
            'image_size': self.image_size,
            'num_timesteps': self.num_timesteps,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'use_sam': self.use_sam,
            'use_ema': self.use_ema,
            'created_at': datetime.now().isoformat(),
        }

    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for checkpoint."""
        model_cfg = self.mode.get_model_config()
        return {
            'model_type': self.model_type,
            'in_channels': model_cfg['in_channels'],
            'out_channels': model_cfg['out_channels'],
            'strategy': self.strategy_name,
            'mode': self.mode_name,
        }

    def train_step(self, batch: Any) -> TrainingStepResult:
        """Execute single training step.

        Args:
            batch: Input batch from dataloader.

        Returns:
            TrainingStepResult with total, MSE, and perceptual losses.
        """
        self.optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=torch.bfloat16):
            prepared = self.mode.prepare_batch(batch, self.device)
            images = prepared['images']
            labels = prepared.get('labels')
            mode_id = prepared.get('mode_id')  # For multi-modality mode
            size_bins = prepared.get('size_bins')  # For seg_conditioned mode

            # Store pixel-space labels for ControlNet (before any encoding)
            controlnet_cond = labels if self.use_controlnet else None

            # Encode to diffusion space (identity for PixelSpace)
            images = self.space.encode_batch(images)
            # For ControlNet: keep labels in pixel space (conditioning through ControlNet)
            # For concatenation: encode labels to latent space
            if labels is not None and not self.use_controlnet:
                labels = self.space.encode(labels)

            labels_dict = {'labels': labels}

            if isinstance(images, dict):
                noise = {key: torch.randn_like(img).to(self.device) for key, img in images.items()}
            else:
                noise = torch.randn_like(images).to(self.device)

            # Apply noise augmentation (perturb noise vector for diversity)
            noise = self._apply_noise_augmentation(noise)

            # Sample timesteps (with optional curriculum learning)
            curriculum_range = self._get_curriculum_range(self._current_epoch)
            timesteps = self.strategy.sample_timesteps(images, curriculum_range)

            # Apply timestep jitter (adds noise-level diversity)
            timesteps = self._apply_timestep_jitter(timesteps)

            noisy_images = self.strategy.add_noise(images, noise, timesteps)

            # DC-AE 1.5: Augmented Diffusion Training - apply channel masking
            # Only active for latent diffusion (scale_factor > 1)
            aug_diff_mask = None
            if self.augmented_diffusion_enabled and self.space.scale_factor > 1:
                if isinstance(noise, dict):
                    # Dual mode: apply same mask to both modalities
                    keys = list(noise.keys())
                    aug_diff_mask = self._create_aug_diff_mask(noise[keys[0]])
                    noise = {k: v * aug_diff_mask for k, v in noise.items()}
                    noisy_images = {k: v * aug_diff_mask for k, v in noisy_images.items()}
                else:
                    aug_diff_mask = self._create_aug_diff_mask(noise)
                    noise = noise * aug_diff_mask
                    noisy_images = noisy_images * aug_diff_mask

            # For ControlNet: use only noisy images (no concatenation)
            # For standard: concatenate noisy images with labels
            if self.use_controlnet:
                model_input = noisy_images  # Labels passed via controlnet_cond
            else:
                model_input = self.mode.format_model_input(noisy_images, labels_dict)

            if self._use_compiled_forward and self.mode_name == ModeType.DUAL:
                # Note: compiled forward is disabled when use_min_snr=True
                keys = list(images.keys())
                total_loss, mse_loss, p_loss, clean_0, clean_1 = self._compiled_forward_dual(
                    self.model,
                    self.perceptual_loss_fn,
                    model_input,
                    timesteps,
                    images[keys[0]],
                    images[keys[1]],
                    noise[keys[0]],
                    noise[keys[1]],
                    noisy_images[keys[0]],
                    noisy_images[keys[1]],
                    self.perceptual_weight,
                    self.strategy_name,
                    self.num_timesteps,
                )
                predicted_clean = {keys[0]: clean_0, keys[1]: clean_1}

            elif self._use_compiled_forward and self.mode_name in (ModeType.SEG, ModeType.BRAVO):
                # Note: compiled forward is disabled when use_min_snr=True
                total_loss, mse_loss, p_loss, predicted_clean = self._compiled_forward_single(
                    self.model,
                    self.perceptual_loss_fn,
                    model_input,
                    timesteps,
                    images,
                    noise,
                    noisy_images,
                    self.perceptual_weight,
                    self.strategy_name,
                    self.num_timesteps,
                )

            else:
                # ScoreAug path: transform noisy input and target together
                if self.score_aug is not None:
                    # Compute velocity target BEFORE ScoreAug
                    if self.strategy_name == 'rflow':
                        if isinstance(images, dict):
                            velocity_target = {k: images[k] - noise[k] for k in images.keys()}
                        else:
                            velocity_target = images - noise
                    else:
                        # DDPM predicts noise
                        velocity_target = noise

                    # For dual mode, stack velocity targets for joint transform
                    if isinstance(velocity_target, dict):
                        keys = list(velocity_target.keys())
                        stacked_target = torch.cat([velocity_target[k] for k in keys], dim=1)
                        aug_input, aug_target, omega = self.score_aug(model_input, stacked_target)
                        # Unstack back to dict
                        aug_velocity = {
                            keys[0]: aug_target[:, 0:1],
                            keys[1]: aug_target[:, 1:2],
                        }
                    else:
                        aug_input, aug_velocity, omega = self.score_aug(model_input, velocity_target)

                    # Apply mode intensity scaling if enabled (after ScoreAug, before model)
                    # This scales the input by a modality-specific factor, forcing the model
                    # to use mode conditioning to correctly predict the unscaled target
                    if self.use_mode_intensity_scaling and mode_id is not None:
                        aug_input, _ = self._apply_mode_intensity_scale(aug_input, mode_id)

                    # Get prediction from augmented input
                    if self.use_omega_conditioning and self.use_mode_embedding:
                        # Model is CombinedModelWrapper, pass both omega and mode_id
                        prediction = self.model(aug_input, timesteps, omega=omega, mode_id=mode_id)
                    elif self.use_omega_conditioning:
                        # Model is ScoreAugModelWrapper, pass omega and mode_id for conditioning
                        prediction = self.model(aug_input, timesteps, omega=omega, mode_id=mode_id)
                    elif self.use_mode_embedding:
                        # Model is ModeEmbedModelWrapper, pass mode_id for conditioning
                        prediction = self.model(aug_input, timesteps, mode_id=mode_id)
                    else:
                        prediction = self.strategy.predict_noise_or_velocity(self.model, aug_input, timesteps)

                    # Compute MSE loss with augmented target
                    if isinstance(aug_velocity, dict):
                        keys = list(aug_velocity.keys())
                        pred_0 = prediction[:, 0:1, :, :]
                        pred_1 = prediction[:, 1:2, :, :]
                        mse_loss = (((pred_0 - aug_velocity[keys[0]]) ** 2).mean() +
                                    ((pred_1 - aug_velocity[keys[1]]) ** 2).mean()) / 2
                    else:
                        mse_loss = ((prediction - aug_velocity) ** 2).mean()

                    # Compute predicted_clean in augmented space, then inverse transform
                    if self.perceptual_weight > 0:
                        # Reconstruct from augmented noisy images
                        if isinstance(noisy_images, dict):
                            keys = list(noisy_images.keys())
                            # Apply same transform to noisy_images for reconstruction
                            stacked_noisy = torch.cat([noisy_images[k] for k in keys], dim=1)
                            aug_noisy = self.score_aug.apply_omega(stacked_noisy, omega)
                            aug_noisy_dict = {keys[0]: aug_noisy[:, 0:1], keys[1]: aug_noisy[:, 1:2]}

                            if self.strategy_name == 'rflow':
                                t_norm = timesteps.float() / float(self.num_timesteps)
                                t_exp = t_norm.view(-1, 1, 1, 1)
                                aug_clean = {k: torch.clamp(aug_noisy_dict[k] + t_exp * prediction[:, i:i+1], 0, 1)
                                             for i, k in enumerate(keys)}
                            else:
                                aug_clean = {k: torch.clamp(aug_noisy_dict[k] - prediction[:, i:i+1], 0, 1)
                                             for i, k in enumerate(keys)}

                            # Inverse transform to original space
                            inv_clean = {k: self.score_aug.inverse_apply_omega(v, omega) for k, v in aug_clean.items()}
                            if any(v is None for v in inv_clean.values()):
                                # Non-invertible transform (rotation/flip), skip perceptual loss
                                if self.perceptual_weight > 0:
                                    logger.debug("Perceptual loss skipped: non-invertible ScoreAug transform applied")
                                p_loss = torch.tensor(0.0, device=self.device)
                                predicted_clean = aug_clean  # Use augmented for metrics
                            else:
                                predicted_clean = inv_clean
                                p_loss = self.perceptual_loss_fn(predicted_clean.float(), images.float())
                        else:
                            # Single channel mode
                            aug_noisy = self.score_aug.apply_omega(noisy_images, omega)
                            if self.strategy_name == 'rflow':
                                t_norm = timesteps.float() / float(self.num_timesteps)
                                t_exp = t_norm.view(-1, 1, 1, 1)
                                aug_clean = torch.clamp(aug_noisy + t_exp * prediction, 0, 1)
                            else:
                                aug_clean = torch.clamp(aug_noisy - prediction, 0, 1)

                            inv_clean = self.score_aug.inverse_apply_omega(aug_clean, omega)
                            if inv_clean is None:
                                # Non-invertible transform (rotation/flip), skip perceptual loss
                                if self.perceptual_weight > 0:
                                    logger.debug("Perceptual loss skipped: non-invertible ScoreAug transform applied")
                                p_loss = torch.tensor(0.0, device=self.device)
                                predicted_clean = aug_clean
                            else:
                                predicted_clean = inv_clean
                                p_loss = self.perceptual_loss_fn(predicted_clean.float(), images.float())
                    else:
                        p_loss = torch.tensor(0.0, device=self.device)
                        predicted_clean = images  # Placeholder for metrics

                    total_loss = mse_loss + self.perceptual_weight * p_loss

                else:
                    # Standard path (no ScoreAug)
                    if self.use_controlnet and controlnet_cond is not None:
                        # ControlNet path: pass pixel-space conditioning
                        prediction = self.model(
                            x=model_input,
                            timesteps=timesteps,
                            controlnet_cond=controlnet_cond,
                        )
                    elif self.use_mode_embedding:
                        # Model is ModeEmbedModelWrapper, pass mode_id for conditioning
                        prediction = self.model(model_input, timesteps, mode_id=mode_id)
                    elif self.use_size_bin_embedding:
                        # Model is SizeBinModelWrapper, pass size_bins for conditioning
                        prediction = self.model(model_input, timesteps, size_bins=size_bins)
                    else:
                        prediction = self.strategy.predict_noise_or_velocity(self.model, model_input, timesteps)

                    # DC-AE 1.5: Mask prediction for augmented diffusion training
                    # Paper Eq. 2: ||ε·mask - ε_θ(x_t·mask, t)·mask||²
                    if aug_diff_mask is not None:
                        if isinstance(prediction, dict):
                            prediction = {k: v * aug_diff_mask for k, v in prediction.items()}
                        else:
                            prediction = prediction * aug_diff_mask

                    mse_loss, predicted_clean = self.strategy.compute_loss(prediction, images, noise, noisy_images, timesteps)

                    if self.use_min_snr:
                        mse_loss = self._compute_min_snr_weighted_mse(
                            prediction, images, noise, timesteps
                        )

                    # Apply regional weighting (per-pixel weights by tumor size)
                    if self.regional_weight_computer is not None and labels is not None:
                        mse_loss = self._compute_region_weighted_mse(
                            prediction, images, noise, labels
                        )

                    # Compute perceptual loss (decode for latent space)
                    if self.perceptual_weight > 0:
                        if self.space.scale_factor > 1:
                            # Decode to pixel space for perceptual loss
                            pred_decoded = self.space.decode_batch(predicted_clean)
                            images_decoded = self.space.decode_batch(images)
                        else:
                            pred_decoded = predicted_clean
                            images_decoded = images

                        # Wrapper handles both tensor and dict inputs
                        # Cast to FP32 for perceptual loss stability
                        p_loss = self.perceptual_loss_fn(pred_decoded.float(), images_decoded.float())
                    else:
                        p_loss = torch.tensor(0.0, device=self.device)

                    total_loss = mse_loss + self.perceptual_weight * p_loss

                    # Self-conditioning consistency loss
                    self_cond_cfg = self.cfg.training.get('self_conditioning', {})
                    if self_cond_cfg.get('enabled', False):
                        consistency_weight = self_cond_cfg.get('consistency_weight', 0.1)
                        consistency_loss = self._compute_self_conditioning_loss(
                            model_input, timesteps, prediction, mode_id
                        )
                        total_loss = total_loss + consistency_weight * consistency_loss

            # SDA (Shifted Data Augmentation) path
            # Unlike ScoreAug, SDA transforms CLEAN data before noise addition
            # and uses shifted timesteps to prevent leakage
            sda_loss = torch.tensor(0.0, device=self.device)
            if self.sda is not None and self.score_aug is None:
                # Apply SDA to clean images
                if isinstance(images, dict):
                    # Dual mode - apply same transform to both
                    keys = list(images.keys())
                    stacked_images = torch.cat([images[k] for k in keys], dim=1)
                    aug_stacked, sda_info = self.sda(stacked_images)

                    if sda_info is not None:
                        # Unstack augmented images
                        aug_images_dict = {
                            keys[0]: aug_stacked[:, 0:1],
                            keys[1]: aug_stacked[:, 1:2],
                        }

                        # Shift timesteps for augmented path
                        shifted_timesteps = self.sda.shift_timesteps(timesteps)

                        # Add noise at SHIFTED timesteps
                        aug_noisy_dict = {
                            k: self.strategy.add_noise(aug_images_dict[k], noise[k], shifted_timesteps)
                            for k in keys
                        }

                        # Format input and get prediction
                        aug_labels_dict = {'labels': labels}
                        aug_model_input = self.mode.format_model_input(aug_noisy_dict, aug_labels_dict)

                        if self.use_mode_embedding:
                            aug_prediction = self.model(aug_model_input, shifted_timesteps, mode_id=mode_id)
                        else:
                            aug_prediction = self.strategy.predict_noise_or_velocity(
                                self.model, aug_model_input, shifted_timesteps
                            )

                        # Compute augmented target (transform original target)
                        if self.strategy_name == 'rflow':
                            aug_velocity = {
                                k: self.sda.apply_to_target(images[k] - noise[k], sda_info)
                                for k in keys
                            }
                            aug_mse = sum(
                                ((aug_prediction[:, i:i+1] - aug_velocity[k]) ** 2).mean()
                                for i, k in enumerate(keys)
                            ) / len(keys)
                        else:
                            aug_noise = {k: self.sda.apply_to_target(noise[k], sda_info) for k in keys}
                            aug_mse = sum(
                                ((aug_prediction[:, i:i+1] - aug_noise[k]) ** 2).mean()
                                for i, k in enumerate(keys)
                            ) / len(keys)

                        sda_loss = aug_mse
                else:
                    # Single channel mode
                    aug_images, sda_info = self.sda(images)

                    if sda_info is not None:
                        # Shift timesteps for augmented path
                        shifted_timesteps = self.sda.shift_timesteps(timesteps)

                        # Add noise at SHIFTED timesteps
                        aug_noisy = self.strategy.add_noise(aug_images, noise, shifted_timesteps)

                        # Format input and get prediction
                        aug_labels_dict = {'labels': labels}
                        aug_model_input = self.mode.format_model_input(aug_noisy, aug_labels_dict)

                        if self.use_mode_embedding:
                            aug_prediction = self.model(aug_model_input, shifted_timesteps, mode_id=mode_id)
                        else:
                            aug_prediction = self.strategy.predict_noise_or_velocity(
                                self.model, aug_model_input, shifted_timesteps
                            )

                        # Compute augmented target (transform original target)
                        if self.strategy_name == 'rflow':
                            velocity_target = images - noise
                            aug_velocity = self.sda.apply_to_target(velocity_target, sda_info)
                            aug_mse = ((aug_prediction - aug_velocity) ** 2).mean()
                        else:
                            aug_noise = self.sda.apply_to_target(noise, sda_info)
                            aug_mse = ((aug_prediction - aug_noise) ** 2).mean()

                        sda_loss = aug_mse

                # Add SDA loss to total
                total_loss = total_loss + self.sda_weight * sda_loss

        if self.use_sam:
            # SAM requires two forward-backward passes
            # Save values before second pass (CUDA graphs may overwrite tensors)
            total_loss_val = total_loss.item()
            mse_loss_val = mse_loss.item()
            p_loss_val = p_loss.item()
            # Clone predicted_clean for metrics tracking (may be dict for dual mode)
            if isinstance(predicted_clean, dict):
                predicted_clean_saved = {k: v.detach().clone() for k, v in predicted_clean.items()}
            else:
                predicted_clean_saved = predicted_clean.detach().clone()

            # First pass: compute gradient and perturb weights
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model_raw.parameters(), max_norm=self.cfg.training.gradient_clip_norm
            )
            self.optimizer.first_step(zero_grad=True)

            # Second pass: compute gradient at perturbed point
            # Need to recompute forward pass with same batch data
            with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                if self._use_compiled_forward and self.mode_name == ModeType.DUAL:
                    keys = list(images.keys())
                    total_loss_2, _, _, _, _ = self._compiled_forward_dual(
                        self.model, self.perceptual_loss_fn, model_input, timesteps,
                        images[keys[0]], images[keys[1]], noise[keys[0]], noise[keys[1]],
                        noisy_images[keys[0]], noisy_images[keys[1]],
                        self.perceptual_weight, self.strategy_name, self.num_timesteps,
                    )
                elif self._use_compiled_forward and self.mode_name in (ModeType.SEG, ModeType.BRAVO):
                    total_loss_2, _, _, _ = self._compiled_forward_single(
                        self.model, self.perceptual_loss_fn, model_input, timesteps,
                        images, noise, noisy_images,
                        self.perceptual_weight, self.strategy_name, self.num_timesteps,
                    )
                elif self.score_aug is not None:
                    # ScoreAug path - recompute with same augmentation (omega)
                    # Skip perceptual loss in SAM second pass (augmented space, minor contribution)
                    if self.use_omega_conditioning and self.use_mode_embedding:
                        prediction_2 = self.model(aug_input, timesteps, omega=omega, mode_id=mode_id)
                    elif self.use_omega_conditioning:
                        prediction_2 = self.model(aug_input, timesteps, omega=omega, mode_id=mode_id)
                    elif self.use_mode_embedding:
                        prediction_2 = self.model(aug_input, timesteps, mode_id=mode_id)
                    else:
                        prediction_2 = self.strategy.predict_noise_or_velocity(self.model, aug_input, timesteps)
                    if isinstance(aug_velocity, dict):
                        keys = list(aug_velocity.keys())
                        mse_loss_2 = (((prediction_2[:, 0:1] - aug_velocity[keys[0]]) ** 2).mean() +
                                      ((prediction_2[:, 1:2] - aug_velocity[keys[1]]) ** 2).mean()) / 2
                    else:
                        mse_loss_2 = ((prediction_2 - aug_velocity) ** 2).mean()
                    total_loss_2 = mse_loss_2  # MSE only, perceptual loss not recomputed
                else:
                    # Standard path - recompute
                    if self.use_mode_embedding:
                        prediction_2 = self.model(model_input, timesteps, mode_id=mode_id)
                    else:
                        prediction_2 = self.strategy.predict_noise_or_velocity(self.model, model_input, timesteps)
                    mse_loss_2, predicted_clean_2 = self.strategy.compute_loss(prediction_2, images, noise, noisy_images, timesteps)
                    if self.use_min_snr:
                        mse_loss_2 = self._compute_min_snr_weighted_mse(prediction_2, images, noise, timesteps)
                    if self.perceptual_weight > 0:
                        if self.space.scale_factor > 1:
                            pred_decoded_2 = self.space.decode_batch(predicted_clean_2)
                            images_decoded_2 = self.space.decode_batch(images)
                        else:
                            pred_decoded_2, images_decoded_2 = predicted_clean_2, images
                        p_loss_2 = self.perceptual_loss_fn(pred_decoded_2.float(), images_decoded_2.float())
                    else:
                        p_loss_2 = torch.tensor(0.0, device=self.device)
                    total_loss_2 = mse_loss_2 + self.perceptual_weight * p_loss_2

            total_loss_2.backward()
            self.optimizer.second_step(zero_grad=True)

            if self.use_ema:
                self._update_ema()

            # Track metrics using MetricsTracker (use saved predicted_clean)
            mask = labels_dict.get('labels')
            with torch.no_grad():
                self.metrics.track_step(
                    timesteps=timesteps,
                    predicted_clean=predicted_clean_saved,
                    images=images,
                    mask=mask,
                    grad_norm=grad_norm,
                )

            return TrainingStepResult(
                total_loss=total_loss_val,
                reconstruction_loss=0.0,  # Not applicable for diffusion
                perceptual_loss=p_loss_val,
                mse_loss=mse_loss_val,
            )
        else:
            # Standard optimizer step
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model_raw.parameters(), max_norm=self.cfg.training.gradient_clip_norm
            )

            # Add gradient noise (for regularization, decays over training)
            self._add_gradient_noise(self._global_step)

            self.optimizer.step()
            self._global_step += 1

            if self.use_ema:
                self._update_ema()

            # Track metrics using MetricsTracker
            mask = labels_dict.get('labels')
            with torch.no_grad():
                self.metrics.track_step(
                    timesteps=timesteps,
                    predicted_clean=predicted_clean,
                    images=images,
                    mask=mask,
                    grad_norm=grad_norm,
                )

            return TrainingStepResult(
                total_loss=total_loss.item(),
                reconstruction_loss=0.0,  # Not applicable for diffusion
                perceptual_loss=p_loss.item(),
                mse_loss=mse_loss.item(),
            )

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Tuple[float, float, float]:
        """Train the model for one epoch.

        Args:
            data_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Tuple of (avg_loss, avg_mse_loss, avg_perceptual_loss).
        """
        self._current_epoch = epoch
        self.model.train()
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_perceptual_loss = 0

        epoch_iter = create_epoch_iterator(
            data_loader, epoch, self.is_cluster, self.is_main_process,
            limit_batches=self.limit_train_batches
        )

        for step, batch in enumerate(epoch_iter):
            result = self.train_step(batch)

            # Step profiler to mark training step boundary
            self._profiler_step()

            epoch_loss += result.total_loss
            epoch_mse_loss += result.mse_loss
            epoch_perceptual_loss += result.perceptual_loss

            if hasattr(epoch_iter, 'set_postfix'):
                epoch_iter.set_postfix(loss=f"{epoch_loss / (step + 1):.6f}")

            if epoch == 1 and step == 0 and self.is_main_process:
                logger.info(get_vram_usage(self.device))

        n_batches = self.limit_train_batches if self.limit_train_batches else len(data_loader)
        avg_loss = epoch_loss / n_batches
        avg_mse = epoch_mse_loss / n_batches
        avg_perceptual = epoch_perceptual_loss / n_batches

        # Log training losses using unified system
        if self.is_main_process and not self.use_multi_gpu:
            self._unified_metrics.update_loss('Total', avg_loss)
            self._unified_metrics.update_loss('MSE', avg_mse)
            self._unified_metrics.update_loss('Perceptual', avg_perceptual)
            self._unified_metrics.update_lr(self.lr_scheduler.get_last_lr()[0])
            self._unified_metrics.update_vram()
            self._unified_metrics.log_training(epoch)
            self._unified_metrics.reset_training()

        return avg_loss, avg_mse, avg_perceptual

    def compute_validation_losses(self, epoch: int) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
        """Compute losses and metrics on validation set.

        Args:
            epoch: Current epoch number (for TensorBoard logging).

        Returns:
            Tuple of (metrics dict, worst_batch_data or None).
            Metrics dict contains: mse, perceptual, total, msssim, psnr.
            Worst batch data contains: original, generated, mask, timesteps, loss.
        """
        if self.val_loader is None:
            return {}, None

        # Save random state - validation uses torch.randn_like() which would otherwise
        # shift the global RNG and cause training to diverge across epochs
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None

        model_to_use = self.ema.ema_model if self.ema is not None else self.model_raw
        model_to_use.eval()

        total_mse = 0.0
        total_perc = 0.0
        total_loss = 0.0
        total_msssim = 0.0
        total_psnr = 0.0
        total_lpips = 0.0
        total_dice = 0.0
        total_iou = 0.0
        n_batches = 0

        # Check if seg mode (use Dice/IoU instead of perceptual metrics)
        is_seg_mode = self.mode_name in ('seg', 'seg_conditioned')

        # Per-channel metrics for dual/multi modes
        per_channel_metrics: Dict[str, Dict[str, float]] = {}

        # Track worst validation batch (only from full-sized batches)
        worst_loss = 0.0
        worst_batch_data: Optional[Dict[str, Any]] = None
        min_batch_size = self.batch_size  # Don't track small last batches

        # Initialize regional tracker for validation (if enabled)
        regional_tracker = None
        if self.metrics.log_regional_losses:
            regional_tracker = RegionalMetricsTracker(
                image_size=self.image_size,
                fov_mm=self.cfg.paths.get('fov_mm', 240.0),
                loss_fn='mse',
                device=self.device,
            )

        # Initialize timestep loss tracking for validation
        num_timestep_bins = 10
        timestep_loss_sum = torch.zeros(num_timestep_bins, device=self.device)
        timestep_loss_count = torch.zeros(num_timestep_bins, device=self.device, dtype=torch.long)

        # Mark CUDA graph step boundary to prevent tensor caching issues
        torch.compiler.cudagraph_mark_step_begin()

        with torch.no_grad():
            for batch in self.val_loader:
                prepared = self.mode.prepare_batch(batch, self.device)
                images = prepared['images']
                labels = prepared.get('labels')
                mode_id = prepared.get('mode_id')  # For multi-modality mode

                # Get current batch size
                if isinstance(images, dict):
                    first_key = list(images.keys())[0]
                    current_batch_size = images[first_key].shape[0]
                else:
                    current_batch_size = images.shape[0]

                # Encode to diffusion space (identity for PixelSpace)
                images = self.space.encode_batch(images)
                if labels is not None:
                    labels = self.space.encode(labels)

                labels_dict = {'labels': labels}

                # Sample timesteps and noise
                if isinstance(images, dict):
                    noise = {key: torch.randn_like(img).to(self.device) for key, img in images.items()}
                else:
                    noise = torch.randn_like(images).to(self.device)

                timesteps = self.strategy.sample_timesteps(images)
                noisy_images = self.strategy.add_noise(images, noise, timesteps)
                model_input = self.mode.format_model_input(noisy_images, labels_dict)

                # Apply mode intensity scaling for validation consistency
                if self.use_mode_intensity_scaling and mode_id is not None:
                    model_input, _ = self._apply_mode_intensity_scale(model_input, mode_id)

                # Predict and compute loss
                if self.use_mode_embedding and mode_id is not None:
                    prediction = model_to_use(model_input, timesteps, mode_id=mode_id)
                else:
                    prediction = self.strategy.predict_noise_or_velocity(model_to_use, model_input, timesteps)
                mse_loss, predicted_clean = self.strategy.compute_loss(prediction, images, noise, noisy_images, timesteps)

                # Compute perceptual loss
                if self.perceptual_weight > 0:
                    if self.space.scale_factor > 1:
                        pred_decoded = self.space.decode_batch(predicted_clean)
                        images_decoded = self.space.decode_batch(images)
                    else:
                        pred_decoded = predicted_clean
                        images_decoded = images
                    p_loss = self.perceptual_loss_fn(pred_decoded.float(), images_decoded.float())
                else:
                    p_loss = torch.tensor(0.0, device=self.device)

                loss = mse_loss + self.perceptual_weight * p_loss
                loss_val = loss.item()

                total_mse += mse_loss.item()
                total_perc += p_loss.item()
                total_loss += loss_val

                # Track worst batch (only from full-sized batches)
                if loss_val > worst_loss and current_batch_size >= min_batch_size:
                    worst_loss = loss_val
                    if isinstance(images, dict):
                        worst_batch_data = {
                            'original': {k: v.cpu() for k, v in images.items()},
                            'generated': {k: v.cpu() for k, v in predicted_clean.items()},
                            'mask': labels.cpu() if labels is not None else None,
                            'timesteps': timesteps.cpu(),
                            'loss': loss_val,
                        }
                    else:
                        worst_batch_data = {
                            'original': images.cpu(),
                            'generated': predicted_clean.cpu(),
                            'mask': labels.cpu() if labels is not None else None,
                            'timesteps': timesteps.cpu(),
                            'loss': loss_val,
                        }

                # Quality metrics (decode to pixel space for latent diffusion)
                if self.space.scale_factor > 1:
                    metrics_pred = self.space.decode_batch(predicted_clean)
                    metrics_gt = self.space.decode_batch(images)
                else:
                    metrics_pred = predicted_clean
                    metrics_gt = images

                if isinstance(metrics_pred, dict):
                    # Dual/multi mode: compute per-channel AND average metrics
                    keys = list(metrics_pred.keys())
                    channel_msssim = {}
                    channel_psnr = {}
                    channel_lpips = {}

                    for key in keys:
                        channel_msssim[key] = compute_msssim(metrics_pred[key], metrics_gt[key])
                        channel_psnr[key] = compute_psnr(metrics_pred[key], metrics_gt[key])
                        if self.metrics.log_lpips:
                            channel_lpips[key] = compute_lpips(metrics_pred[key], metrics_gt[key], self.device)

                        # Accumulate per-channel metrics
                        if key not in per_channel_metrics:
                            per_channel_metrics[key] = {'msssim': 0.0, 'psnr': 0.0, 'lpips': 0.0, 'count': 0}
                        per_channel_metrics[key]['msssim'] += channel_msssim[key]
                        per_channel_metrics[key]['psnr'] += channel_psnr[key]
                        if self.metrics.log_lpips:
                            per_channel_metrics[key]['lpips'] += channel_lpips[key]
                        per_channel_metrics[key]['count'] += 1

                    # Average across channels for combined metrics
                    msssim_val = sum(channel_msssim.values()) / len(keys)
                    psnr_val = sum(channel_psnr.values()) / len(keys)
                    lpips_val = sum(channel_lpips.values()) / len(keys) if self.metrics.log_lpips else 0.0
                else:
                    msssim_val = compute_msssim(metrics_pred, metrics_gt)
                    psnr_val = compute_psnr(metrics_pred, metrics_gt)
                    if self.metrics.log_lpips:
                        lpips_val = compute_lpips(metrics_pred, metrics_gt, self.device)
                    else:
                        lpips_val = 0.0

                total_msssim += msssim_val
                total_psnr += psnr_val
                total_lpips += lpips_val

                # Compute Dice/IoU for seg modes
                # For diffusion output: predicted_clean is already in [0, 1] range (not logits)
                if is_seg_mode:
                    dice_val = compute_dice(metrics_pred, metrics_gt, apply_sigmoid=False)
                    iou_val = compute_iou(metrics_pred, metrics_gt, apply_sigmoid=False)
                    total_dice += dice_val
                    total_iou += iou_val

                n_batches += 1

                # Regional tracking (tumor vs background)
                if regional_tracker is not None and labels is not None:
                    regional_tracker.update(predicted_clean, images, labels)

                # Timestep loss tracking (per-bin reconstruction MSE)
                if self.metrics.log_timestep_losses:
                    if isinstance(predicted_clean, dict):
                        pred = torch.cat(list(predicted_clean.values()), dim=1)
                        img = torch.cat(list(images.values()), dim=1)
                    else:
                        pred, img = predicted_clean, images
                    mse_per_sample = ((pred - img) ** 2).mean(dim=(1, 2, 3))
                    bin_size = self.num_timesteps // num_timestep_bins
                    bin_indices = (timesteps // bin_size).clamp(max=num_timestep_bins - 1).long()
                    timestep_loss_sum.scatter_add_(0, bin_indices, mse_per_sample)
                    timestep_loss_count.scatter_add_(0, bin_indices, torch.ones_like(bin_indices))

        model_to_use.train()

        # Handle empty validation set
        if n_batches == 0:
            logger.warning("Validation set is empty, skipping metrics")
            return {}, None

        metrics = {
            'mse': total_mse / n_batches,
            'perceptual': total_perc / n_batches,
            'total': total_loss / n_batches,
            'msssim': total_msssim / n_batches,
            'psnr': total_psnr / n_batches,
        }
        if self.metrics.log_lpips:
            metrics['lpips'] = total_lpips / n_batches

        # Add Dice/IoU for seg modes
        if is_seg_mode:
            metrics['dice'] = total_dice / n_batches
            metrics['iou'] = total_iou / n_batches

        # Log to TensorBoard using unified system
        if self.writer is not None and self._unified_metrics is not None:
            # Update unified metrics with validation values
            self._unified_metrics.update_loss('MSE', metrics['mse'], phase='val')
            self._unified_metrics.update_loss('Perceptual', metrics['perceptual'], phase='val')
            self._unified_metrics.update_loss('Total', metrics['total'], phase='val')

            # Compute 3D MS-SSIM first so we can include it in metrics
            msssim_3d = None
            if self.log_msssim:
                msssim_3d = self._compute_volume_3d_msssim(epoch, data_split='val')
                if msssim_3d is not None:
                    metrics['msssim_3d'] = msssim_3d

            # Update quality metrics using unified system
            # Note: For 2D trainer, we manually update with computed values since
            # the validation loop already computed them batch-wise
            self._unified_metrics._val_psnr_sum = metrics['psnr']
            self._unified_metrics._val_psnr_count = 1
            self._unified_metrics._val_msssim_sum = metrics['msssim']
            self._unified_metrics._val_msssim_count = 1
            if 'lpips' in metrics:
                self._unified_metrics._val_lpips_sum = metrics['lpips']
                self._unified_metrics._val_lpips_count = 1
            if msssim_3d is not None:
                self._unified_metrics._val_msssim_3d_sum = msssim_3d
                self._unified_metrics._val_msssim_3d_count = 1
            if 'dice' in metrics:
                self._unified_metrics._val_dice_sum = metrics['dice']
                self._unified_metrics._val_dice_count = 1
            if 'iou' in metrics:
                self._unified_metrics._val_iou_sum = metrics['iou']
                self._unified_metrics._val_iou_count = 1

            # Log validation metrics
            self._unified_metrics.log_validation(epoch)

            # Log per-channel metrics for dual/multi modes
            for channel_key, channel_data in per_channel_metrics.items():
                count = channel_data['count']
                if count > 0:
                    avg_psnr = channel_data['psnr'] / count
                    avg_msssim = channel_data['msssim'] / count
                    suffix = f'_{channel_key}'
                    self.writer.add_scalar(f'Validation/PSNR{suffix}', avg_psnr, epoch)
                    self.writer.add_scalar(f'Validation/MS-SSIM{suffix}', avg_msssim, epoch)
                    if self.metrics.log_lpips:
                        avg_lpips = channel_data['lpips'] / count
                        self.writer.add_scalar(f'Validation/LPIPS{suffix}', avg_lpips, epoch)

            # Log regional metrics (tumor vs background) with modality suffix
            if regional_tracker is not None:
                is_single_modality = self.mode_name not in ('multi_modality', 'dual', 'multi')
                if is_single_modality:
                    regional_tracker.log_to_tensorboard(self.writer, epoch, prefix=f'regional_{self.mode_name}')
                else:
                    regional_tracker.log_to_tensorboard(self.writer, epoch, prefix='regional')

            # Log per-timestep validation losses using unified system
            if self.metrics.log_timestep_losses:
                counts = timestep_loss_count.cpu()
                sums = timestep_loss_sum.cpu()
                bin_size = self.num_timesteps // num_timestep_bins
                for i in range(num_timestep_bins):
                    if counts[i] > 0:
                        avg_loss = (sums[i] / counts[i]).item()
                        t_normalized = (i + 0.5) / num_timestep_bins  # Center of bin
                        self._unified_metrics.update_timestep_loss(t_normalized, avg_loss)

            # Compute generation quality metrics (KID, CMMD)
            if self._gen_metrics is not None:
                try:
                    model_to_use = self.ema.ema_model if self.ema is not None else self.model_raw
                    model_to_use.eval()

                    # Quick metrics every epoch
                    gen_results = self._gen_metrics.compute_epoch_metrics(
                        model_to_use, self.strategy, self.mode
                    )
                    self._unified_metrics.log_generation(epoch, gen_results)

                    # Extended metrics at figure_interval
                    if (epoch + 1) % self.figure_interval == 0:
                        extended_results = self._gen_metrics.compute_extended_metrics(
                            model_to_use, self.strategy, self.mode
                        )
                        self._unified_metrics.log_generation(epoch, extended_results)

                    model_to_use.train()
                except Exception as e:
                    logger.warning(f"Generation metrics computation failed: {e}")

            # Reset validation metrics for next epoch
            self._unified_metrics.reset_validation()

        # Restore random state to not affect subsequent training epochs
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, self.device)

        return metrics, worst_batch_data

    def _save_checkpoint(self, epoch: int, name: str) -> None:
        """Save checkpoint using standardized format."""
        model_config = self._get_model_config()
        save_full_checkpoint(
            model=self.model_raw,
            optimizer=self.optimizer,
            epoch=epoch,
            save_dir=self.save_dir,
            filename=f"checkpoint_{name}",
            model_config=model_config,
            scheduler=self.lr_scheduler,
            ema=self.ema,
        )

    def train(
        self,
        train_loader: DataLoader,
        train_dataset: Dataset,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """Main training loop.

        Args:
            train_loader: Training data loader.
            train_dataset: Training dataset (for sample generation).
            val_loader: Optional validation dataloader.
        """
        total_start = time.time()
        self.val_loader = val_loader

        # Initialize unified metrics system
        is_seg_mode = self.mode_name in ('seg', 'seg_conditioned')
        self._unified_metrics = UnifiedMetrics(
            writer=self.writer,
            mode=self.mode_name,
            spatial_dims=2,
            modality=self.mode_name if self.mode_name not in ('multi', 'dual') else None,
            device=self.device,
            enable_regional=self.metrics.log_regional_losses,
            num_timestep_bins=10,
            image_size=self.image_size,
            fov_mm=self.cfg.paths.get('fov_mm', 240.0),
        )

        # Measure FLOPs
        self._measure_model_flops(train_loader)

        if self.is_main_process and self.mode_name in ('seg', 'seg_conditioned'):
            logger.info(f"{self.mode_name} mode: perceptual loss and LPIPS disabled (binary masks)")

        # Initialize generation metrics (cache reference features)
        if self._gen_metrics is not None and self.is_main_process:
            # Determine seg channel index based on mode
            seg_channel_idx = 1 if self.mode_name in ('bravo', 'multi', 'multi_modality') else 2
            self._gen_metrics.set_fixed_conditioning(
                train_dataset,
                num_masks=self._gen_metrics_config.samples_extended,  # Use extended count for conditioning
                seg_channel_idx=seg_channel_idx,
            )
            # Cache reference features from train and val loaders
            if val_loader is not None:
                self._gen_metrics.cache_reference_features(
                    train_loader,
                    val_loader,
                    experiment_id=str(self.save_dir.name) if hasattr(self.save_dir, 'name') else str(self.save_dir),
                )

        avg_loss = float('inf')
        avg_mse = float('inf')

        try:
            for epoch in range(self.n_epochs):
                epoch_start = time.time()

                if self.use_multi_gpu and hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)

                avg_loss, avg_mse, avg_perceptual = self.train_epoch(train_loader, epoch)

                if self.use_multi_gpu:
                    loss_tensor = torch.tensor([avg_loss, avg_mse, avg_perceptual], device=self.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    avg_loss, avg_mse, avg_perceptual = (loss_tensor / self.world_size).cpu().numpy()

                epoch_time = time.time() - epoch_start
                self.lr_scheduler.step()

                if self.is_main_process:
                    log_epoch_summary(epoch, self.n_epochs, (avg_loss, avg_mse, avg_perceptual), epoch_time)

                    # Log training losses using unified system (DDP path)
                    if self.use_multi_gpu:
                        self._unified_metrics.update_loss('Total', avg_loss)
                        self._unified_metrics.update_loss('MSE', avg_mse)
                        self._unified_metrics.update_loss('Perceptual', avg_perceptual)
                        self._unified_metrics.update_lr(self.lr_scheduler.get_last_lr()[0])
                        self._unified_metrics.update_vram()
                        self._unified_metrics.log_training(epoch)
                        self._unified_metrics.reset_training()

                    val_metrics, worst_val_data = self.compute_validation_losses(epoch)
                    log_figures = (epoch + 1) % self.figure_interval == 0

                    self.metrics.log_epoch(epoch, log_all=True, is_figure_epoch=log_figures)
                    self._flops_tracker.log_epoch(self.writer, epoch)
                    log_vram_to_tensorboard(self.writer, self.device, epoch)

                    if log_figures and worst_val_data is not None:
                        self.visualizer.log_worst_batch(epoch, worst_val_data)

                    self._compute_per_modality_validation(epoch)

                    if log_figures or (epoch + 1) == self.n_epochs:
                        model_to_use = self.ema.ema_model if self.ema is not None else self.model_raw
                        self.visualizer.generate_samples(model_to_use, train_dataset, epoch)

                    self._save_checkpoint(epoch, "latest")

                    loss_for_selection = val_metrics.get('total', avg_loss)
                    if loss_for_selection < self.best_loss:
                        self.best_loss = loss_for_selection
                        self._save_checkpoint(epoch, "best")
                        loss_type = "val" if val_metrics else "train"
                        logger.info(f"New best model saved ({loss_type} loss: {loss_for_selection:.6f})")

        finally:
            total_time = time.time() - total_start

            if self.is_main_process:
                logger.info(f"Training completed! Total time: {total_time:.1f}s ({total_time / 3600:.1f}h)")
                self._update_metadata_final(avg_loss, avg_mse, total_time)

            if self.use_multi_gpu:
                try:
                    dist.destroy_process_group()
                except Exception as e:
                    logger.warning(f"Error destroying process group: {e}")

    def _measure_model_flops(self, train_loader: DataLoader) -> None:
        """Measure model FLOPs using batch_size=1 to avoid OOM during torch.compile."""
        if not self.metrics.log_flops:
            return

        try:
            batch = next(iter(train_loader))
            prepared = self.mode.prepare_batch(batch, self.device)
            images = prepared['images']
            labels = prepared.get('labels')

            # Slice to batch_size=1 to avoid OOM during torch.compile tracing
            # torch.compile compiles for specific shapes; using full batch can cause
            # excessive memory during the compilation graph creation
            if isinstance(images, dict):
                images = {key: img[:1] for key, img in images.items()}
                noise = {key: torch.randn_like(img).to(self.device) for key, img in images.items()}
            else:
                images = images[:1]
                noise = torch.randn_like(images).to(self.device)

            if labels is not None:
                labels = labels[:1]
            labels_dict = {'labels': labels}

            timesteps = self.strategy.sample_timesteps(images)
            noisy_images = self.strategy.add_noise(images, noise, timesteps)
            model_input = self.mode.format_model_input(noisy_images, labels_dict)

            self._flops_tracker.measure(
                self.model_raw,
                model_input[:1] if isinstance(model_input, torch.Tensor) else model_input,
                steps_per_epoch=len(train_loader),
                timesteps=timesteps[:1] if isinstance(timesteps, torch.Tensor) else timesteps,
                is_main_process=self.is_main_process,
            )
        except Exception as e:
            if self.is_main_process:
                logger.warning(f"Could not measure FLOPs: {e}")

    def _compute_per_modality_validation(self, epoch: int) -> None:
        """Compute and log validation metrics for each modality separately.

        For multi-modality training, this logs PSNR, LPIPS, MS-SSIM and regional
        metrics for each modality (bravo, t1_pre, t1_gd) to compare with
        single-modality experiments.

        Args:
            epoch: Current epoch number.
        """
        if not hasattr(self, 'per_modality_val_loaders') or not self.per_modality_val_loaders:
            return

        model_to_use = self.ema.ema_model if self.ema is not None else self.model_raw
        model_to_use.eval()

        for modality, loader in self.per_modality_val_loaders.items():
            total_psnr = 0.0
            total_lpips = 0.0
            total_msssim = 0.0
            n_batches = 0

            # Initialize regional tracker for this modality
            regional_tracker = None
            if self.metrics.log_regional_losses:
                regional_tracker = RegionalMetricsTracker(
                    image_size=self.image_size,
                    fov_mm=self.cfg.paths.get('fov_mm', 240.0),
                    loss_fn='mse',
                    device=self.device,
                )

            with torch.no_grad():
                for batch in loader:
                    prepared = self.mode.prepare_batch(batch, self.device)
                    images = prepared['images']
                    labels = prepared.get('labels')

                    # Encode to diffusion space (identity for PixelSpace)
                    images = self.space.encode_batch(images)
                    if labels is not None:
                        labels = self.space.encode(labels)

                    labels_dict = {'labels': labels}

                    # Sample timesteps and noise
                    noise = torch.randn_like(images).to(self.device)
                    timesteps = self.strategy.sample_timesteps(images)
                    noisy_images = self.strategy.add_noise(images, noise, timesteps)
                    model_input = self.mode.format_model_input(noisy_images, labels_dict)

                    # Predict
                    prediction = self.strategy.predict_noise_or_velocity(model_to_use, model_input, timesteps)
                    _, predicted_clean = self.strategy.compute_loss(prediction, images, noise, noisy_images, timesteps)

                    # Compute metrics
                    total_psnr += compute_psnr(predicted_clean, images)
                    if self.metrics.log_lpips:
                        total_lpips += compute_lpips(predicted_clean, images, device=self.device)
                    total_msssim += compute_msssim(predicted_clean, images)

                    # Regional tracking (tumor vs background)
                    if regional_tracker is not None and labels is not None:
                        regional_tracker.update(predicted_clean, images, labels)

                    n_batches += 1

            # Compute averages and log per-modality metrics
            if n_batches > 0 and self.writer is not None:
                avg_psnr = total_psnr / n_batches
                avg_msssim = total_msssim / n_batches

                # Log per-modality quality metrics
                self.writer.add_scalar(f'Validation/PSNR_{modality}', avg_psnr, epoch)
                self.writer.add_scalar(f'Validation/MS-SSIM_{modality}', avg_msssim, epoch)

                if self.metrics.log_lpips:
                    avg_lpips = total_lpips / n_batches
                    self.writer.add_scalar(f'Validation/LPIPS_{modality}', avg_lpips, epoch)

                # Compute 3D MS-SSIM for this modality
                if self.log_msssim:
                    msssim_3d = self._compute_volume_3d_msssim(
                        epoch, data_split='val', modality_override=modality
                    )
                    if msssim_3d is not None:
                        self.writer.add_scalar(f'Validation/MS-SSIM-3D_{modality}', msssim_3d, epoch)

                # Log regional metrics for this modality
                if regional_tracker is not None:
                    regional_tracker.log_to_tensorboard(
                        self.writer, epoch, prefix=f'regional_{modality}'
                    )

    def _compute_volume_3d_msssim(
        self,
        epoch: int,
        data_split: str = 'val',
        modality_override: Optional[str] = None,
    ) -> Optional[float]:
        """Compute 3D MS-SSIM by reconstructing full volumes slice-by-slice.

        For diffusion models, this:
        1. Loads full 3D volumes
        2. Processes each slice: add noise at mid-range timestep → denoise → get predicted clean
        3. Stacks slices back into a volume
        4. Computes 3D MS-SSIM between reconstructed and original volumes

        This measures cross-slice consistency of the diffusion model's denoising quality.

        Args:
            epoch: Current epoch number.
            data_split: Which data split to use ('val' or 'test_new').
            modality_override: Optional specific modality to compute for
                (e.g., 'bravo', 't1_pre'). If None, uses mode from config.

        Returns:
            Average 3D MS-SSIM across all volumes, or None if unavailable.
        """
        if not self.log_msssim:
            return None

        # Import here to avoid circular imports
        from medgen.data.loaders.vae import create_vae_volume_validation_dataloader

        # Determine modality - use override if provided, else from config
        if modality_override is not None:
            modality = modality_override
        else:
            # Use out_channels to determine volume channels (excludes conditioning)
            mode_name = self.cfg.mode.get('name', 'bravo')
            out_channels = self.cfg.mode.get('out_channels', 1)
            modality = 'dual' if out_channels > 1 else mode_name

        # Skip for multi_modality mode - volume loader doesn't support mixed modalities
        # and computing volume metrics on mixed slices doesn't make sense
        if modality == 'multi_modality':
            return None

        # Create volume dataloader
        result = create_vae_volume_validation_dataloader(self.cfg, modality, data_split)
        if result is None:
            return None

        volume_loader, _ = result

        model_to_use = self.ema.ema_model if self.ema is not None else self.model_raw
        model_to_use.eval()

        total_msssim_3d = 0.0
        n_volumes = 0
        slice_batch_size = self.batch_size

        # Use mid-range timestep for reconstruction quality measurement
        mid_timestep = self.num_timesteps // 2

        # Save random state - this method generates random noise which would otherwise
        # shift the global RNG and cause training to diverge across epochs
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None

        with torch.inference_mode():
            for batch in volume_loader:
                # batch['image'] is [1, C, H, W, D] (batch_size=1 for volumes)
                volume = batch['image'].to(self.device, non_blocking=True)
                volume = volume.squeeze(0)  # [C, H, W, D]

                n_channels_vol, height, width, depth = volume.shape

                # Pre-allocate output tensor
                all_recon = torch.empty(
                    (depth, n_channels_vol, height, width),
                    dtype=torch.bfloat16,
                    device=self.device
                )

                # Process slices in batches
                for start_idx in range(0, depth, slice_batch_size):
                    end_idx = min(start_idx + slice_batch_size, depth)
                    current_batch_size = end_idx - start_idx

                    # [C, H, W, D] -> [B, C, H, W]
                    slice_tensor = volume[:, :, :, start_idx:end_idx].permute(3, 0, 1, 2)

                    # Encode to diffusion space (identity for PixelSpace)
                    slice_encoded = self.space.encode_batch(slice_tensor)

                    # Add noise at mid-range timestep
                    noise = torch.randn_like(slice_encoded)
                    timesteps = torch.full(
                        (current_batch_size,),
                        mid_timestep,
                        device=self.device,
                        dtype=torch.long
                    )
                    noisy_slices = self.strategy.add_noise(slice_encoded, noise, timesteps)

                    with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                        # For conditional modes, use zeros as conditioning (no tumor)
                        # This measures pure denoising ability without semantic guidance
                        if self.mode.is_conditional:
                            dummy_labels = torch.zeros_like(slice_encoded[:, :1])  # Single channel
                            model_input = self.mode.format_model_input(noisy_slices, {'labels': dummy_labels})
                        else:
                            model_input = self.mode.format_model_input(noisy_slices, {'labels': None})

                        # Predict noise/velocity
                        prediction = self.strategy.predict_noise_or_velocity(
                            model_to_use, model_input, timesteps
                        )

                        # Get predicted clean images
                        _, predicted_clean = self.strategy.compute_loss(
                            prediction, slice_encoded, noise, noisy_slices, timesteps
                        )

                    # Decode from diffusion space if needed
                    if self.space.scale_factor > 1:
                        predicted_clean = self.space.decode_batch(predicted_clean)

                    all_recon[start_idx:end_idx] = predicted_clean

                # Reshape for 3D MS-SSIM: [D, C, H, W] -> [1, C, D, H, W]
                recon_3d = all_recon.permute(1, 0, 2, 3).unsqueeze(0)
                volume_3d = volume.permute(0, 3, 1, 2).unsqueeze(0)

                # Compute 3D MS-SSIM
                msssim_3d = compute_msssim(recon_3d.float(), volume_3d.float(), spatial_dims=3)
                total_msssim_3d += msssim_3d
                n_volumes += 1

        model_to_use.train()

        # Restore random state to not affect subsequent training epochs
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, self.device)

        if n_volumes == 0:
            return None

        return total_msssim_3d / n_volumes

    def _update_metadata_final(self, final_loss: float, final_mse: float, total_time: float) -> None:
        """Update metadata with final training stats."""
        metadata_path = os.path.join(self.save_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata['final_loss'] = final_loss
            metadata['final_mse'] = final_mse
            metadata['total_time_seconds'] = total_time
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    def evaluate_test_set(
        self,
        test_loader: DataLoader,
        checkpoint_name: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate diffusion model on test set.

        Runs inference on the entire test set and computes metrics:
        - MSE (prediction error)
        - MS-SSIM (Multi-Scale Structural Similarity)
        - PSNR (Peak Signal-to-Noise Ratio)

        Results are saved to test_results_{checkpoint_name}.json and logged to TensorBoard.

        Args:
            test_loader: DataLoader for test set.
            checkpoint_name: Name of checkpoint to load ("best", "latest", or None
                for current model state).

        Returns:
            Dict with test metrics: 'mse', 'msssim', 'psnr', 'n_samples'.
        """
        if not self.is_main_process:
            return {}

        # Load checkpoint if specified
        if checkpoint_name is not None:
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{checkpoint_name}.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model_raw.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded {checkpoint_name} checkpoint for test evaluation")

                # Load EMA state if available and EMA is configured
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
            model_to_use = self.model_raw
        model_to_use.eval()

        # Accumulators for metrics
        total_mse = 0.0
        total_msssim = 0.0
        total_psnr = 0.0
        total_lpips = 0.0
        n_batches = 0
        n_samples = 0

        # Track worst batch by loss
        worst_batch_loss = 0.0
        worst_batch_data: Optional[Dict[str, Any]] = None

        # Timestep bin accumulators (10 bins: 0-99, 100-199, ..., 900-999)
        num_timestep_bins = 10
        timestep_loss_sum = torch.zeros(num_timestep_bins, device=self.device)
        timestep_loss_count = torch.zeros(num_timestep_bins, device=self.device, dtype=torch.long)

        # Initialize regional tracker for test (if enabled)
        regional_tracker = None
        if self.metrics.log_regional_losses:
            regional_tracker = RegionalMetricsTracker(
                image_size=self.image_size,
                fov_mm=self.cfg.paths.get('fov_mm', 240.0),
                loss_fn='mse',
                device=self.device,
            )

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test evaluation", ncols=100, disable=self.is_cluster):
                prepared = self.mode.prepare_batch(batch, self.device)
                images = prepared['images']
                labels = prepared.get('labels')
                mode_id = prepared.get('mode_id')
                batch_size = images[list(images.keys())[0]].shape[0] if isinstance(images, dict) else images.shape[0]

                # Encode to diffusion space
                images = self.space.encode_batch(images)
                if labels is not None:
                    labels = self.space.encode(labels)

                labels_dict = {'labels': labels}

                # Sample timesteps and noise
                if isinstance(images, dict):
                    noise = {key: torch.randn_like(img).to(self.device) for key, img in images.items()}
                else:
                    noise = torch.randn_like(images).to(self.device)

                timesteps = self.strategy.sample_timesteps(images)
                noisy_images = self.strategy.add_noise(images, noise, timesteps)
                model_input = self.mode.format_model_input(noisy_images, labels_dict)

                # Apply mode intensity scaling for test consistency
                if self.use_mode_intensity_scaling and mode_id is not None:
                    model_input, _ = self._apply_mode_intensity_scale(model_input, mode_id)

                with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    if self.use_mode_embedding and mode_id is not None:
                        prediction = model_to_use(model_input, timesteps, mode_id=mode_id)
                    else:
                        prediction = self.strategy.predict_noise_or_velocity(model_to_use, model_input, timesteps)
                    mse_loss, predicted_clean = self.strategy.compute_loss(prediction, images, noise, noisy_images, timesteps)

                # Compute metrics
                loss_val = mse_loss.item()
                total_mse += loss_val

                # Track per-timestep-bin losses
                with torch.no_grad():
                    if isinstance(predicted_clean, dict):
                        keys = list(predicted_clean.keys())
                        mse_per_sample = (
                            (predicted_clean[keys[0]] - images[keys[0]]).pow(2).mean(dim=(1, 2, 3)) +
                            (predicted_clean[keys[1]] - images[keys[1]]).pow(2).mean(dim=(1, 2, 3))
                        ) / 2
                    else:
                        mse_per_sample = (predicted_clean - images).pow(2).mean(dim=(1, 2, 3))
                    bin_size = self.num_timesteps // num_timestep_bins
                    bin_indices = (timesteps // bin_size).clamp(max=num_timestep_bins - 1).long()
                    timestep_loss_sum.scatter_add_(0, bin_indices, mse_per_sample)
                    timestep_loss_count.scatter_add_(0, bin_indices, torch.ones_like(bin_indices))

                # Decode to pixel space for metrics
                if self.space.scale_factor > 1:
                    metrics_pred = self.space.decode_batch(predicted_clean)
                    metrics_gt = self.space.decode_batch(images)
                else:
                    metrics_pred = predicted_clean
                    metrics_gt = images

                if isinstance(metrics_pred, dict):
                    keys = list(metrics_pred.keys())
                    msssim_val = (compute_msssim(metrics_pred[keys[0]], metrics_gt[keys[0]]) +
                                  compute_msssim(metrics_pred[keys[1]], metrics_gt[keys[1]])) / 2
                    psnr_val = (compute_psnr(metrics_pred[keys[0]], metrics_gt[keys[0]]) +
                                compute_psnr(metrics_pred[keys[1]], metrics_gt[keys[1]])) / 2
                    if self.metrics.log_lpips:
                        lpips_val = (compute_lpips(metrics_pred[keys[0]], metrics_gt[keys[0]], self.device) +
                                     compute_lpips(metrics_pred[keys[1]], metrics_gt[keys[1]], self.device)) / 2
                    else:
                        lpips_val = 0.0
                else:
                    msssim_val = compute_msssim(metrics_pred, metrics_gt)
                    psnr_val = compute_psnr(metrics_pred, metrics_gt)
                    if self.metrics.log_lpips:
                        lpips_val = compute_lpips(metrics_pred, metrics_gt, self.device)
                    else:
                        lpips_val = 0.0

                # Regional metrics tracking (tumor vs background)
                if regional_tracker is not None and labels is not None:
                    # Decode labels to pixel space if needed
                    labels_pixel = self.space.decode(labels) if self.space.scale_factor > 1 else labels
                    regional_tracker.update(metrics_pred, metrics_gt, labels_pixel)

                # Track worst batch
                if loss_val > worst_batch_loss and batch_size >= self.batch_size:
                    worst_batch_loss = loss_val
                    if isinstance(images, dict):
                        worst_batch_data = {
                            'original': {k: v.cpu() for k, v in images.items()},
                            'generated': {k: v.cpu() for k, v in predicted_clean.items()},
                            'timesteps': timesteps.cpu(),
                            'loss': loss_val,
                        }
                    else:
                        worst_batch_data = {
                            'original': images.cpu(),
                            'generated': predicted_clean.cpu(),
                            'timesteps': timesteps.cpu(),
                            'loss': loss_val,
                        }

                total_msssim += msssim_val
                total_psnr += psnr_val
                total_lpips += lpips_val
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
            'msssim': total_msssim / n_batches,
            'psnr': total_psnr / n_batches,
            'n_samples': n_samples,
        }
        if self.metrics.log_lpips:
            metrics['lpips'] = total_lpips / n_batches

        # Log results
        logger.info(f"Test Results - {label} ({n_samples} samples):")
        logger.info(f"  MSE:     {metrics['mse']:.6f}")
        logger.info(f"  MS-SSIM: {metrics['msssim']:.4f}")
        logger.info(f"  PSNR:    {metrics['psnr']:.2f} dB")
        if 'lpips' in metrics:
            logger.info(f"  LPIPS:   {metrics['lpips']:.4f}")

        # Save results to JSON
        results_path = os.path.join(self.save_dir, f'test_results_{label}.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Test results saved to: {results_path}")

        # Log to TensorBoard using unified system
        tb_prefix = f'test_{label}'
        if self.writer is not None and self._unified_metrics is not None:
            # Build test metrics dict for unified logging
            test_metrics = {
                'PSNR': metrics['psnr'],
                'MS-SSIM': metrics['msssim'],
                'MSE': metrics['mse'],
            }
            if 'lpips' in metrics:
                test_metrics['LPIPS'] = metrics['lpips']

            self._unified_metrics.log_test(test_metrics, prefix=tb_prefix)

            # Log timestep bin losses using unified system
            bin_size = self.num_timesteps // num_timestep_bins
            counts = timestep_loss_count.cpu()
            sums = timestep_loss_sum.cpu()
            timestep_bins = {}
            for bin_idx in range(num_timestep_bins):
                bin_start = bin_idx * bin_size
                bin_end = (bin_idx + 1) * bin_size - 1
                count = counts[bin_idx].item()
                if count > 0:
                    avg_loss = (sums[bin_idx] / count).item()
                    timestep_bins[f'{bin_start:04d}-{bin_end:04d}'] = avg_loss
            if timestep_bins:
                # Log timestep losses directly since log_test doesn't handle this prefix
                for bin_name, loss in timestep_bins.items():
                    self.writer.add_scalar(f'{tb_prefix}/Timestep/{bin_name}', loss, 0)

            # Log regional metrics with modality suffix for single-modality modes
            is_single_modality = self.mode_name not in ('multi_modality', 'dual', 'multi')
            if regional_tracker is not None:
                regional_prefix = f'{tb_prefix}_regional_{self.mode_name}' if is_single_modality else f'{tb_prefix}_regional'
                regional_tracker.log_to_tensorboard(self.writer, 0, prefix=regional_prefix)

            # Compute and log volume 3D MS-SSIM
            if self.log_msssim:
                msssim_3d = self._compute_volume_3d_msssim(0, data_split='test_new')
                if msssim_3d is not None:
                    metrics['msssim_3d'] = msssim_3d
                    suffix = f'_{self.mode_name}' if is_single_modality else ''
                    self.writer.add_scalar(f'{tb_prefix}/MS-SSIM-3D{suffix}', msssim_3d, 0)

            # Create visualization of worst batch
            if worst_batch_data is not None:
                fig = self._create_test_reconstruction_figure(
                    worst_batch_data['original'],
                    worst_batch_data['generated'],
                    metrics,
                    label,
                    worst_batch_data['timesteps'],
                )
                self.writer.add_figure(f'{tb_prefix}/worst_batch', fig, 0)
                plt.close(fig)

                # Also save as image file
                fig_path = os.path.join(self.save_dir, f'test_worst_batch_{label}.png')
                fig = self._create_test_reconstruction_figure(
                    worst_batch_data['original'],
                    worst_batch_data['generated'],
                    metrics,
                    label,
                    worst_batch_data['timesteps'],
                )
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Test worst batch saved to: {fig_path}")

            # Compute generation quality metrics (FID, KID, CMMD) if enabled
            if self._gen_metrics is not None:
                try:
                    logger.info("Computing generation metrics (FID, KID, CMMD)...")
                    test_gen_results = self._gen_metrics.compute_test_metrics(
                        model_to_use, self.strategy, self.mode, test_loader
                    )
                    # Log to TensorBoard
                    for key, value in test_gen_results.items():
                        self.writer.add_scalar(f'{tb_prefix}/{key}', value, 0)
                        metrics[f'gen_{key.lower()}'] = value
                    # Log to console
                    if 'FID' in test_gen_results:
                        logger.info(f"  FID:     {test_gen_results['FID']:.4f}")
                    if 'KID_mean' in test_gen_results:
                        logger.info(f"  KID:     {test_gen_results['KID_mean']:.4f} +/- {test_gen_results.get('KID_std', 0):.4f}")
                    if 'CMMD' in test_gen_results:
                        logger.info(f"  CMMD:    {test_gen_results['CMMD']:.4f}")
                except Exception as e:
                    logger.warning(f"Generation metrics computation failed: {e}")

        return metrics

    def _create_test_reconstruction_figure(
        self,
        original: torch.Tensor,
        predicted: torch.Tensor,
        metrics: Dict[str, float],
        label: str,
        timesteps: Optional[torch.Tensor] = None,
    ) -> plt.Figure:
        """Create side-by-side test evaluation figure.

        Uses shared create_reconstruction_figure for consistent visualization.

        Args:
            original: Original images [B, C, H, W] (CPU).
            predicted: Predicted clean images [B, C, H, W] (CPU).
            metrics: Dict with test metrics (mse, msssim, psnr, optionally lpips).
            label: Checkpoint label (best, latest, current).
            timesteps: Optional timesteps for each sample.

        Returns:
            Matplotlib figure.
        """
        title = f"Worst Test Batch ({label})"
        display_metrics = {
            'MS-SSIM': metrics['msssim'],
            'PSNR': metrics['psnr'],
        }
        if 'lpips' in metrics:
            display_metrics['LPIPS'] = metrics['lpips']
        return create_reconstruction_figure(
            original=original,
            generated=predicted,
            title=title,
            max_samples=8,
            metrics=display_metrics,
            timesteps=timesteps,
        )

    def close_writer(self) -> None:
        """Close TensorBoard writer. Call after all logging is complete."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
