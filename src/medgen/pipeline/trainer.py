"""
Diffusion model trainer module (2D).

This module provides the DiffusionTrainer class which inherits from DiffusionTrainerBase
and implements 2D-specific diffusion training functionality:
- Strategy pattern (DDPM, Rectified Flow) - from base
- Mode pattern (seg, bravo, dual, multi, seg_conditioned)
- Timestep-based noise training
- ScoreAug v2 transforms
- Compiled forward paths for performance
"""
import logging
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import matplotlib

if TYPE_CHECKING:
    from medgen.metrics.generation import GenerationMetrics
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from ema_pytorch import EMA
from monai.networks.nets import DiffusionModelUNet
from omegaconf import DictConfig
from torch import nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from medgen.core import (
    ModeType,
    create_plateau_scheduler,
    create_warmup_constant_scheduler,
    create_warmup_cosine_scheduler,
    wrap_model_for_training,
)
from medgen.diffusion import (
    ConditionalDualMode,
    ConditionalSingleMode,
    DDPMStrategy,
    DiffusionSpace,
    DiffusionStrategy,
    LatentSegConditionedMode,
    MultiModalityMode,
    RFlowStrategy,
    SegmentationConditionedInputMode,
    SegmentationConditionedMode,
    SegmentationMode,
    TrainingMode,
)
from medgen.evaluation import ValidationVisualizer
from medgen.losses import PerceptualLoss, create_regional_weight_computer
from medgen.metrics import UnifiedMetrics
from medgen.models import (
    ControlNetConditionedUNet,
    create_controlnet_for_unet,
    create_diffusion_model,
    freeze_unet_for_controlnet,
    get_model_type,
    is_transformer_model,
)

from .diffusion_trainer_base import DiffusionTrainerBase
from .results import TrainingStepResult
from .utils import (
    EpochTimeEstimator,
    create_epoch_iterator,
    get_vram_usage,
    log_epoch_summary,
    save_full_checkpoint,
)

logger = logging.getLogger(__name__)


class DiffusionTrainer(DiffusionTrainerBase):
    """Unified 2D/3D diffusion model trainer.

    Supports both 2D and 3D diffusion training via the spatial_dims parameter.
    Inherits from DiffusionTrainerBase and adds dimension-specific functionality:
    - 2D: [B, C, H, W] tensors, full image perceptual loss
    - 3D: [B, C, D, H, W] tensors, 2.5D perceptual loss (center slice)

    Args:
        cfg: Hydra configuration object containing all settings.
        spatial_dims: Number of spatial dimensions (2 or 3). Defaults to 2.
        space: Optional DiffusionSpace for pixel/latent space operations.
            Defaults to PixelSpace (identity, backward compatible).

    Attributes:
        model: The diffusion model for training/inference. Conforms to DiffusionModel
            protocol but may be wrapped with ScoreAug, ModeEmbed, or SizeBin
            conditioning wrappers. Use for forward passes.
        model_raw: The unwrapped model (nn.Module) for accessing parameters,
            state_dict, and other module operations. Use for optimizer setup,
            gradient clipping, and checkpoint saving.

    Example:
        >>> # 2D training (default)
        >>> trainer = DiffusionTrainer(cfg)
        >>> # 3D training
        >>> trainer = DiffusionTrainer(cfg, spatial_dims=3)
        >>> # Or use convenience constructors
        >>> trainer = DiffusionTrainer.create_3d(cfg)
    """

    @property
    def spatial_dims(self) -> int:
        """Return spatial dimensions (2 or 3)."""
        return self._spatial_dims

    def __init__(
        self,
        cfg: DictConfig,
        spatial_dims: int = 2,
        space: DiffusionSpace | None = None,
    ) -> None:
        # Validate spatial_dims
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        self._spatial_dims = spatial_dims

        # Initialize base class (handles common diffusion config)
        super().__init__(cfg, space)

        # ─────────────────────────────────────────────────────────────────────
        # Dimension-specific size config
        # ─────────────────────────────────────────────────────────────────────
        if spatial_dims == 2:
            self.image_size: int = cfg.model.image_size
        else:
            # 3D volume dimensions (with defaults for robustness)
            self.volume_height: int = cfg.volume.get('height', 256)
            self.volume_width: int = cfg.volume.get('width', 256)
            self.volume_depth: int = cfg.volume.get('depth', 160)
            # For compatibility with 2D code that uses image_size
            self.image_size: int = cfg.volume.get('height', 256)

        self.eta_min: float = cfg.training.get('eta_min', 1e-6)

        # Perceptual weight (disabled for seg modes - binary masks don't work with VGG features)
        is_seg_mode = self.mode_name in ('seg', 'seg_conditioned')
        self.perceptual_weight: float = 0.0 if is_seg_mode else cfg.training.get('perceptual_weight', 0.0)

        # FP32 loss computation (set False to reproduce pre-Jan-7-2026 BF16 behavior)
        self.use_fp32_loss: bool = cfg.training.get('use_fp32_loss', True)
        if self.is_main_process:
            logger.debug(f"use_fp32_loss = {self.use_fp32_loss}")

        # Optimizer settings
        optimizer_cfg = cfg.training.get('optimizer', {})
        self.weight_decay: float = optimizer_cfg.get('weight_decay', 0.0)

        # Initialize mode (2D-specific) and scheduler
        # Note: strategy is already created in base class
        self.mode = self._create_mode(self.mode_name)
        # Setup scheduler with dimension-appropriate parameters
        scheduler_kwargs = {
            'num_timesteps': self.num_timesteps,
            'image_size': self.image_size,
            'use_discrete_timesteps': cfg.strategy.get('use_discrete_timesteps', True),
            'sample_method': cfg.strategy.get('sample_method', 'logit-normal'),
            'use_timestep_transform': cfg.strategy.get('use_timestep_transform', True),
        }
        if spatial_dims == 3:
            scheduler_kwargs['depth_size'] = self.volume_depth
            scheduler_kwargs['spatial_dims'] = 3
        self.scheduler = self.strategy.setup_scheduler(**scheduler_kwargs)

        # ─────────────────────────────────────────────────────────────────────
        # 3D-specific memory optimizations (mandatory for 3D)
        # ─────────────────────────────────────────────────────────────────────
        if spatial_dims == 3:
            self.use_amp: bool = True  # Always use AMP for 3D
            self.use_gradient_checkpointing: bool = cfg.training.get('gradient_checkpointing', True)
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        else:
            self.use_amp: bool = cfg.training.get('use_amp', False)
            self.use_gradient_checkpointing: bool = cfg.training.get('gradient_checkpointing', False)
            self.scaler = None

        # Logging config
        logging_cfg = cfg.training.get('logging', {})
        self.log_msssim: bool = logging_cfg.get('msssim', True)
        self.log_psnr: bool = logging_cfg.get('psnr', True)
        # Disable LPIPS for seg modes (binary masks don't work with VGG features)
        self.log_lpips: bool = False if is_seg_mode else logging_cfg.get('lpips', False)
        self.log_timestep_region_losses: bool = logging_cfg.get('timestep_region_losses', True)
        self.log_worst_batch: bool = logging_cfg.get('worst_batch', True)
        self.log_intermediate_steps: bool = logging_cfg.get('intermediate_steps', True)
        self.num_intermediate_steps: int = logging_cfg.get('num_intermediate_steps', 5)

        # Initialize unified metrics system for consistent logging
        # Note: UnifiedMetrics is initialized in train() when writer is fully set up
        self._unified_metrics: UnifiedMetrics | None = None

        # Model components
        self.perceptual_loss_fn: nn.Module | None = None
        self.visualizer: ValidationVisualizer | None = None

        # Cached training samples for deterministic visualization
        # (Uses training data to keep validation/test datasets properly separated)
        self._cached_train_batch: dict[str, torch.Tensor] | None = None

        # Cached volume loaders for 3D MS-SSIM (avoid recreating datasets every epoch)
        self._volume_loaders_cache: dict[str, DataLoader] = {}

        # ─────────────────────────────────────────────────────────────────────
        # Modular initialization (uses helper methods for cleaner code)
        # ─────────────────────────────────────────────────────────────────────

        # ScoreAug initialization (applies transforms to noisy data)
        self._setup_score_aug()

        # SDA (Shifted Data Augmentation) initialization
        self._setup_sda()

        # Mode embedding and size bin embedding
        self._setup_conditional_embeddings()

        # DC-AE 1.5: Augmented Diffusion Training
        self._setup_augmented_diffusion()

        # Log training tricks configuration
        self._log_training_tricks_config()

        # Region-weighted loss
        self._setup_regional_weighting()

        # Generation quality metrics (KID, CMMD) for overfitting detection
        self._gen_metrics: GenerationMetrics | None = None
        gen_cfg = cfg.training.get('generation_metrics', {})
        if gen_cfg.get('enabled', False):
            from medgen.metrics.generation import GenerationMetricsConfig
            # Use training batch_size by default for torch.compile consistency
            feature_batch_size = gen_cfg.get('feature_batch_size', None)
            if feature_batch_size is None:
                if spatial_dims == 3:
                    # 3D: use larger batch for slice-wise extraction
                    feature_batch_size = max(32, cfg.training.get('batch_size', 16) * 16)
                else:
                    feature_batch_size = cfg.training.get('batch_size', 16)
            # Use absolute cache_dir from paths config, fallback to relative
            gen_cache_dir = gen_cfg.get('cache_dir', None)
            if gen_cache_dir is None:
                base_cache = getattr(cfg.paths, 'cache_dir', '.cache')
                gen_cache_dir = f"{base_cache}/generation_features"

            # 3D volumes are much larger - cap sample counts to avoid OOM
            if spatial_dims == 3:
                samples_per_epoch = min(gen_cfg.get('samples_per_epoch', 1), 2)
                samples_extended = min(gen_cfg.get('samples_extended', 4), 4)
                samples_test = min(gen_cfg.get('samples_test', 10), 10)
            else:
                samples_per_epoch = gen_cfg.get('samples_per_epoch', 100)
                samples_extended = gen_cfg.get('samples_extended', 500)
                samples_test = gen_cfg.get('samples_test', 1000)

            # Get original_depth for 3D (used to exclude padded slices from metrics)
            original_depth = None
            if spatial_dims == 3:
                original_depth = cfg.volume.get('original_depth', None)

            # Get size bin config for seg_conditioned mode
            size_bin_edges = None
            size_bin_fov_mm = 240.0
            if self.mode_name == 'seg_conditioned':
                size_bin_cfg = cfg.mode.get('size_bins', {})
                size_bin_edges = list(size_bin_cfg.get('edges', [0, 3, 6, 10, 15, 20, 30]))
                size_bin_fov_mm = float(size_bin_cfg.get('fov_mm', 240.0))

            self._gen_metrics_config = GenerationMetricsConfig(
                enabled=True,
                samples_per_epoch=samples_per_epoch,
                samples_extended=samples_extended,
                samples_test=samples_test,
                steps_per_epoch=gen_cfg.get('steps_per_epoch', 10),
                steps_extended=gen_cfg.get('steps_extended', 25),
                steps_test=gen_cfg.get('steps_test', 50),
                cache_dir=gen_cache_dir,
                feature_batch_size=feature_batch_size,
                original_depth=original_depth,
                size_bin_edges=size_bin_edges,
                size_bin_fov_mm=size_bin_fov_mm,
            )
            if self.is_main_process:
                sample_type = "volumes" if spatial_dims == 3 else "samples"
                logger.info(
                    f"Generation metrics enabled: {self._gen_metrics_config.samples_per_epoch} {sample_type}/epoch "
                    f"({self._gen_metrics_config.steps_per_epoch} steps), "
                    f"{self._gen_metrics_config.samples_extended} {sample_type}/extended "
                    f"({self._gen_metrics_config.steps_extended} steps)"
                )
        else:
            self._gen_metrics_config = None

        # ControlNet configuration
        self._setup_controlnet()

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience Constructors
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def create_2d(cls, cfg: DictConfig, **kwargs) -> 'DiffusionTrainer':
        """Create 2D diffusion trainer.

        Args:
            cfg: Hydra configuration object.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            DiffusionTrainer with spatial_dims=2.
        """
        return cls(cfg, spatial_dims=2, **kwargs)

    @classmethod
    def create_3d(cls, cfg: DictConfig, **kwargs) -> 'DiffusionTrainer':
        """Create 3D diffusion trainer.

        Args:
            cfg: Hydra configuration object.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            DiffusionTrainer with spatial_dims=3.
        """
        return cls(cfg, spatial_dims=3, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization Helpers (modular setup methods)
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_score_aug(self) -> None:
        """Initialize ScoreAug transform and related settings.

        Sets up:
        - self.score_aug: ScoreAugTransform instance or None
        - self.use_omega_conditioning: bool
        - self.use_mode_intensity_scaling: bool
        - self._apply_mode_intensity_scale: function reference or None

        Validates ScoreAug configuration constraints (rotation/flip require omega).
        """
        from .diffusion_config import ScoreAugConfig, validate_score_aug_config

        score_aug_cfg = ScoreAugConfig.from_hydra(self.cfg)

        self.score_aug = None
        self.use_omega_conditioning = False
        self.use_mode_intensity_scaling = False
        self._apply_mode_intensity_scale = None

        if not score_aug_cfg.enabled:
            return

        # Validate configuration constraints
        validate_score_aug_config(score_aug_cfg, self.spatial_dims, self.is_main_process)

        # Create ScoreAugTransform
        from medgen.augmentation import ScoreAugTransform
        self.score_aug = ScoreAugTransform(
            spatial_dims=self.spatial_dims,
            rotation=score_aug_cfg.rotation,
            flip=score_aug_cfg.flip,
            translation=score_aug_cfg.translation,
            cutout=score_aug_cfg.cutout,
            compose=score_aug_cfg.compose,
            compose_prob=score_aug_cfg.compose_prob,
            v2_mode=score_aug_cfg.v2_mode,
            nondestructive_prob=score_aug_cfg.nondestructive_prob,
            destructive_prob=score_aug_cfg.destructive_prob,
            cutout_vs_pattern=score_aug_cfg.cutout_vs_pattern,
            patterns_checkerboard=score_aug_cfg.patterns_checkerboard,
            patterns_grid_dropout=score_aug_cfg.patterns_grid_dropout,
            patterns_coarse_dropout=score_aug_cfg.patterns_coarse_dropout,
            patterns_patch_dropout=score_aug_cfg.patterns_patch_dropout,
        )

        self.use_omega_conditioning = score_aug_cfg.use_omega_conditioning

        # Mode intensity scaling (2D only)
        self.use_mode_intensity_scaling = score_aug_cfg.use_mode_intensity_scaling
        if self.use_mode_intensity_scaling:
            if self.spatial_dims == 3:
                if self.is_main_process:
                    logger.warning(
                        "mode_intensity_scaling is not supported in 3D diffusion. Ignoring."
                    )
                self.use_mode_intensity_scaling = False
            else:
                from medgen.augmentation import apply_mode_intensity_scale
                self._apply_mode_intensity_scale = apply_mode_intensity_scale

        # Log configuration
        if self.is_main_process:
            transforms = []
            if score_aug_cfg.rotation:
                transforms.append('rotation')
            if score_aug_cfg.flip:
                transforms.append('flip')
            if score_aug_cfg.translation:
                transforms.append('translation')
            if score_aug_cfg.cutout:
                transforms.append('cutout')
            if score_aug_cfg.brightness and self.spatial_dims == 2:
                transforms.append(f"brightness({score_aug_cfg.brightness_range})")
            n_options = len(transforms) + 1
            logger.info(
                f"ScoreAug {self.spatial_dims}D enabled: transforms=[{', '.join(transforms)}], "
                f"each with 1/{n_options} prob (uniform), "
                f"omega_conditioning={self.use_omega_conditioning}, "
                f"mode_intensity_scaling={self.use_mode_intensity_scaling}"
            )

    def _setup_sda(self) -> None:
        """Initialize SDA (Shifted Data Augmentation) transform.

        Sets up:
        - self.sda: SDATransform instance or None
        - self.sda_weight: float

        SDA is mutually exclusive with ScoreAug.
        """
        from .diffusion_config import SDAConfig

        sda_cfg = SDAConfig.from_hydra(self.cfg)

        self.sda = None
        self.sda_weight = 1.0

        if not sda_cfg.enabled:
            return

        # SDA and ScoreAug are mutually exclusive
        if self.score_aug is not None:
            if self.is_main_process:
                logger.warning("SDA and ScoreAug are mutually exclusive. Disabling SDA.")
            return

        from medgen.augmentation import SDATransform
        self.sda = SDATransform(
            rotation=sda_cfg.rotation,
            flip=sda_cfg.flip,
            noise_shift=sda_cfg.noise_shift,
            prob=sda_cfg.prob,
        )
        self.sda_weight = sda_cfg.weight

        if self.is_main_process:
            transforms = []
            if sda_cfg.rotation:
                transforms.append('rotation')
            if sda_cfg.flip:
                transforms.append('flip')
            logger.info(
                f"SDA {self.spatial_dims}D enabled: transforms=[{', '.join(transforms)}], "
                f"noise_shift={sda_cfg.noise_shift}, prob={sda_cfg.prob}, weight={self.sda_weight}"
            )

    def _setup_conditional_embeddings(self) -> None:
        """Initialize mode embedding and size bin embedding settings.

        Sets up:
        - self.use_mode_embedding: bool
        - self.mode_embedding_strategy: str
        - self.mode_embedding_dropout: float
        - self.late_mode_start_level: int
        - self.use_size_bin_embedding: bool
        - self.size_bin_num_bins: int
        - self.size_bin_max_count: int
        - self.size_bin_embed_dim: int
        """
        from .diffusion_config import ModeEmbeddingConfig, SizeBinConfig

        mode_cfg = ModeEmbeddingConfig.from_hydra(self.cfg)
        size_bin_cfg = SizeBinConfig.from_hydra(self.cfg, self.mode_name)

        # Mode embedding
        self.use_mode_embedding = mode_cfg.enabled
        self.mode_embedding_strategy = mode_cfg.strategy
        self.mode_embedding_dropout = mode_cfg.dropout
        self.late_mode_start_level = mode_cfg.late_start_level

        if self.use_mode_embedding and self.is_main_process:
            logger.info(
                f"Mode embedding enabled: strategy={self.mode_embedding_strategy}, "
                f"dropout={self.mode_embedding_dropout}, late_start_level={self.late_mode_start_level}"
            )

        # Size bin embedding
        self.use_size_bin_embedding = size_bin_cfg.enabled
        if self.use_size_bin_embedding:
            self.size_bin_num_bins = size_bin_cfg.num_bins
            self.size_bin_max_count = size_bin_cfg.max_count
            self.size_bin_embed_dim = size_bin_cfg.embed_dim
            if self.is_main_process:
                logger.info(
                    f"Size bin embedding enabled: num_bins={self.size_bin_num_bins}, "
                    f"max_count={self.size_bin_max_count}, embed_dim={self.size_bin_embed_dim}"
                )

    def _setup_augmented_diffusion(self) -> None:
        """Initialize DC-AE 1.5 augmented diffusion training settings.

        Sets up:
        - self.augmented_diffusion_enabled: bool
        - self.aug_diff_min_channels: int
        - self.aug_diff_channel_step: int
        - self._aug_diff_channel_steps: list[int] | None
        """
        from .diffusion_config import AugmentedDiffusionConfig

        aug_cfg = AugmentedDiffusionConfig.from_hydra(self.cfg)

        self.augmented_diffusion_enabled = aug_cfg.enabled
        self.aug_diff_min_channels = aug_cfg.min_channels
        self.aug_diff_channel_step = aug_cfg.channel_step
        self._aug_diff_channel_steps = None  # Computed lazily

        if self.augmented_diffusion_enabled and self.is_main_process:
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

    def _setup_regional_weighting(self) -> None:
        """Initialize region-weighted loss computer.

        Sets up:
        - self.regional_weight_computer: RegionalWeightComputer | None
        """
        from .diffusion_config import RegionalWeightingConfig

        rw_cfg = RegionalWeightingConfig.from_hydra(self.cfg)

        self.regional_weight_computer = None

        if not rw_cfg.enabled:
            return

        if self.mode.is_conditional:
            self.regional_weight_computer = create_regional_weight_computer(self.cfg)
            if self.is_main_process:
                logger.info(
                    f"Region-weighted loss enabled: "
                    f"tiny={rw_cfg.tiny_weight}, small={rw_cfg.small_weight}, "
                    f"medium={rw_cfg.medium_weight}, large={rw_cfg.large_weight}, "
                    f"bg={rw_cfg.background_weight}"
                )
        else:
            if self.is_main_process:
                logger.warning(
                    "Regional weighting enabled but mode is not conditional (seg mode). "
                    "Skipping - regional weighting requires segmentation mask as conditioning."
                )

    def _setup_controlnet(self) -> None:
        """Initialize ControlNet configuration.

        Sets up:
        - self.use_controlnet: bool
        - self.controlnet_freeze_unet: bool
        - self.controlnet_scale: float
        - self.controlnet_cfg_dropout_prob: float
        - self.controlnet: nn.Module | None
        - self.controlnet_stage1: bool
        """
        from .diffusion_config import ControlNetConfig

        cn_cfg = ControlNetConfig.from_hydra(self.cfg)

        self.use_controlnet = cn_cfg.enabled
        self.controlnet_freeze_unet = cn_cfg.freeze_unet
        self.controlnet_scale = cn_cfg.conditioning_scale
        self.controlnet_cfg_dropout_prob = cn_cfg.cfg_dropout_prob
        self.controlnet = None
        self.controlnet_stage1 = cn_cfg.stage1

        if self.controlnet_stage1 and self.is_main_process:
            logger.info(
                "ControlNet Stage 1: Training unconditional UNet (in_channels=out_channels). "
                "Use this checkpoint for Stage 2 with controlnet.enabled=true"
            )

        if self.use_controlnet and self.is_main_process:
            logger.info(
                f"ControlNet Stage 2: freeze_unet={self.controlnet_freeze_unet}, "
                f"conditioning_scale={self.controlnet_scale}"
            )

    def _log_training_tricks_config(self) -> None:
        """Log configuration for various training tricks."""
        if not self.is_main_process:
            return

        # Gradient noise
        grad_noise_cfg = self.cfg.training.get('gradient_noise', {})
        if grad_noise_cfg.get('enabled', False):
            logger.info(
                f"Gradient noise injection enabled: "
                f"sigma={grad_noise_cfg.get('sigma', 0.01)}, decay={grad_noise_cfg.get('decay', 0.55)}"
            )

        # Curriculum
        curriculum_cfg = self.cfg.training.get('curriculum', {})
        if curriculum_cfg.get('enabled', False):
            logger.info(
                f"Curriculum timestep scheduling enabled: "
                f"warmup_epochs={curriculum_cfg.get('warmup_epochs', 50)}, "
                f"range [{curriculum_cfg.get('min_t_start', 0.0)}-{curriculum_cfg.get('max_t_start', 0.3)}] -> "
                f"[{curriculum_cfg.get('min_t_end', 0.0)}-{curriculum_cfg.get('max_t_end', 1.0)}]"
            )

        # Timestep jitter
        jitter_cfg = self.cfg.training.get('timestep_jitter', {})
        if jitter_cfg.get('enabled', False):
            logger.info(f"Timestep jitter enabled: std={jitter_cfg.get('std', 0.05)}")

        # Self-conditioning
        self_cond_cfg = self.cfg.training.get('self_conditioning', {})
        if self_cond_cfg.get('enabled', False):
            logger.info(f"Self-conditioning enabled: prob={self_cond_cfg.get('prob', 0.5)}")

        # Feature perturbation
        feat_cfg = self.cfg.training.get('feature_perturbation', {})
        if feat_cfg.get('enabled', False):
            logger.info(
                f"Feature perturbation enabled: std={feat_cfg.get('std', 0.1)}, "
                f"layers={feat_cfg.get('layers', ['mid'])}"
            )

        # Noise augmentation
        noise_aug_cfg = self.cfg.training.get('noise_augmentation', {})
        if noise_aug_cfg.get('enabled', False):
            logger.info(f"Noise augmentation enabled: std={noise_aug_cfg.get('std', 0.1)}")

    # ─────────────────────────────────────────────────────────────────────────
    # DC-AE 1.5: Augmented Diffusion Training Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _get_aug_diff_channel_steps(self, num_channels: int) -> list[int]:
        """Get list of channel counts for augmented diffusion masking."""
        from .training_tricks import get_aug_diff_channel_steps
        return get_aug_diff_channel_steps(self, num_channels)

    def _create_aug_diff_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create channel mask for augmented diffusion training."""
        from .training_tricks import create_aug_diff_mask
        return create_aug_diff_mask(self, tensor)

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
        strategies: dict[str, type] = {
            'ddpm': DDPMStrategy,
            'rflow': RFlowStrategy
        }
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")
        return strategies[strategy]()

    def _create_mode(self, mode: str) -> TrainingMode:
        """Create a training mode instance."""
        modes: dict[str, type] = {
            'seg': SegmentationMode,
            'seg_conditioned': SegmentationConditionedMode,
            'seg_conditioned_input': SegmentationConditionedInputMode,
            'bravo': ConditionalSingleMode,
            'bravo_seg_cond': LatentSegConditionedMode,
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

        if mode == 'seg_conditioned_input':
            size_bin_config = dict(self.cfg.mode.get('size_bins', {})) if 'size_bins' in self.cfg.mode else None
            return SegmentationConditionedInputMode(size_bin_config)

        if mode == 'bravo_seg_cond':
            latent_channels = self.cfg.mode.get('latent_channels', 4)
            return LatentSegConditionedMode(latent_channels)

        return modes[mode]()

    def _clear_caches(self) -> None:
        """Clear internal caches. Call between training runs or when switching datasets.

        Clears:
        - _cached_train_batch: Cached training samples for visualization
        - _volume_loaders_cache: Cached DataLoaders for 3D MS-SSIM
        """
        self._cached_train_batch = None
        self._volume_loaders_cache.clear()
        if self.is_main_process:
            logger.debug("Cleared trainer caches")

    def setup_model(self, train_dataset: Dataset) -> None:
        """Initialize model, optimizer, and loss functions.

        Args:
            train_dataset: Training dataset for model config extraction.
        """
        model_cfg = self.mode.get_model_config()

        # Adjust channels for latent space
        in_channels = self.space.get_latent_channels(model_cfg['in_channels'])
        out_channels = self.space.get_latent_channels(model_cfg['out_channels'])

        # For ControlNet Stage 1 or Stage 2: conditioning goes through ControlNet, not concatenation
        # UNet in_channels = out_channels (no +1 for conditioning)
        if self.use_controlnet or self.controlnet_stage1:
            in_channels = out_channels
            if self.is_main_process:
                stage = "Stage 1 (prep)" if self.controlnet_stage1 else "Stage 2"
                logger.info(f"ControlNet {stage}: UNet in_channels={in_channels} (no conditioning concatenation)")

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
            channels = tuple(self.cfg.model.get('channels', [128, 256, 256, 512]))
            attention_levels = tuple(self.cfg.model.get('attention_levels', [False, False, True, True]))
            num_res_blocks = self.cfg.model.get('num_res_blocks', 2)
            num_head_channels = self.cfg.model.get('num_head_channels', 0)

            raw_model = DiffusionModelUNet(
                spatial_dims=self.cfg.model.get('spatial_dims', 2),
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                attention_levels=attention_levels,
                num_res_blocks=num_res_blocks,
                num_head_channels=num_head_channels,
                norm_num_groups=self.cfg.model.get('norm_num_groups', 32),
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
                spatial_dims=self.spatial_dims,
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
        if self.use_multi_gpu or self.space.scale_factor > 1 or self.use_min_snr or self.regional_weight_computer is not None or self.score_aug is not None or self.sda is not None or self.use_mode_embedding or self.use_omega_conditioning or self.augmented_diffusion_enabled or self.use_controlnet or self.use_size_bin_embedding or self.mode_name not in (ModeType.SEG, ModeType.BRAVO, ModeType.DUAL):
            compile_fused = False

        self._setup_compiled_forward(compile_fused)

        # Setup optimizer
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

        self.optimizer = AdamW(
            train_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.is_main_process and self.weight_decay > 0:
            logger.info(f"Using weight decay: {self.weight_decay}")

        # Learning rate scheduler (cosine, constant, or plateau)
        self.scheduler_type = self.cfg.training.get('scheduler', 'cosine')
        if self.scheduler_type == 'constant':
            self.lr_scheduler = create_warmup_constant_scheduler(
                self.optimizer,
                warmup_epochs=self.warmup_epochs,
                total_epochs=self.n_epochs,
            )
            if self.is_main_process:
                logger.info("Using constant LR scheduler (warmup then constant)")
        elif self.scheduler_type == 'plateau':
            plateau_cfg = self.cfg.training.get('plateau', {})
            self.lr_scheduler = create_plateau_scheduler(
                self.optimizer,
                mode='min',
                factor=plateau_cfg.get('factor', 0.5),
                patience=plateau_cfg.get('patience', 10),
                min_lr=plateau_cfg.get('min_lr', 1e-6),
            )
            if self.is_main_process:
                logger.info(
                    f"Using ReduceLROnPlateau scheduler "
                    f"(factor={plateau_cfg.get('factor', 0.5)}, patience={plateau_cfg.get('patience', 10)})"
                )
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
            writer=self.writer,
            save_dir=self.save_dir,
            device=self.device,
            is_main_process=self.is_main_process,
            space=self.space,
            use_controlnet=self.use_controlnet,
            controlnet=self.controlnet,
        )

        # Initialize generation metrics if enabled
        if self._gen_metrics_config is not None and self._gen_metrics_config.enabled:
            from medgen.metrics.generation import GenerationMetrics
            self._gen_metrics = GenerationMetrics(
                self._gen_metrics_config,
                self.device,
                self.save_dir,
                space=self.space,
                mode_name=self.mode_name,
            )
            if self.is_main_process:
                logger.info("Generation metrics initialized (caching happens at training start)")

        # Save metadata
        if self.is_main_process:
            self._save_metadata()

        # Setup feature perturbation hooks if enabled
        self._setup_feature_perturbation()

    def _setup_ema(self) -> None:
        """Setup EMA wrapper if enabled."""
        if self.use_ema:
            self.ema = EMA(
                self.model_raw,
                beta=self.ema_decay,
                update_after_step=self.cfg.training.ema.get('update_after_step', 100),
                update_every=self.cfg.training.ema.get('update_every', 10),
            )
            if self.is_main_process:
                logger.info(f"EMA enabled with decay={self.ema_decay}")

    def _update_ema(self) -> None:
        """Update EMA model weights."""
        if self.ema is not None:
            self.ema.update()

    def _add_gradient_noise(self, step: int) -> None:
        """Add Gaussian noise to gradients for regularization."""
        from .training_tricks import add_gradient_noise
        add_gradient_noise(self, step)

    def _get_curriculum_range(self, epoch: int) -> tuple[float, float] | None:
        """Get timestep range for curriculum learning."""
        from .training_tricks import get_curriculum_range
        return get_curriculum_range(self, epoch)

    def _apply_timestep_jitter(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to timesteps for regularization."""
        from .training_tricks import apply_timestep_jitter
        return apply_timestep_jitter(self, timesteps)

    def _apply_noise_augmentation(
        self,
        noise: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Add perturbation to noise vector for regularization."""
        from .training_tricks import apply_noise_augmentation
        return apply_noise_augmentation(self, noise)

    def _apply_conditioning_dropout(
        self,
        conditioning: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor | None:
        """Apply per-sample CFG dropout to conditioning tensor."""
        from .training_tricks import apply_conditioning_dropout
        return apply_conditioning_dropout(self, conditioning, batch_size)

    def _setup_feature_perturbation(self) -> None:
        """Setup forward hooks for feature perturbation."""
        from .training_tricks import setup_feature_perturbation
        setup_feature_perturbation(self)

    def _remove_feature_perturbation_hooks(self) -> None:
        """Remove feature perturbation hooks."""
        from .training_tricks import remove_feature_perturbation_hooks
        remove_feature_perturbation_hooks(self)

    def _compute_self_conditioning_loss(
        self,
        model_input: torch.Tensor,
        timesteps: torch.Tensor,
        prediction: torch.Tensor,
        mode_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute self-conditioning consistency loss."""
        from .losses import compute_self_conditioning_loss
        return compute_self_conditioning_loss(self, model_input, timesteps, prediction, mode_id)

    def _compute_min_snr_weighted_mse(
        self,
        prediction: torch.Tensor,
        images: torch.Tensor | dict[str, torch.Tensor],
        noise: torch.Tensor | dict[str, torch.Tensor],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss with Min-SNR weighting."""
        from .losses import compute_min_snr_weighted_mse
        return compute_min_snr_weighted_mse(self, prediction, images, noise, timesteps)

    def _compute_region_weighted_mse(
        self,
        prediction: torch.Tensor,
        images: torch.Tensor | dict[str, torch.Tensor],
        noise: torch.Tensor | dict[str, torch.Tensor],
        seg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss with per-pixel regional weighting."""
        from .losses import compute_region_weighted_mse
        return compute_region_weighted_mse(self, prediction, images, noise, seg_mask)

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
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        from .profiling import get_trainer_type
        return get_trainer_type()

    def _get_metadata_extra(self) -> dict[str, Any]:
        """Return diffusion-specific metadata."""
        from .profiling import get_metadata_extra
        return get_metadata_extra(self)

    def _get_model_config(self) -> dict[str, Any]:
        """Get model configuration for checkpoint."""
        from .profiling import get_model_config
        return get_model_config(self)

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
            bin_maps = prepared.get('bin_maps')  # For seg_conditioned_input mode
            is_latent = prepared.get('is_latent', False)  # Latent dataloader flag
            labels_is_latent = prepared.get('labels_is_latent', False)  # Labels already encoded

            # CFG dropout for size_bins/bin_maps is handled in dataloader (cfg_dropout_prob)
            # No trainer-level dropout needed - dataloader applies per-sample dropout

            # Store pixel-space labels for ControlNet (before any encoding)
            controlnet_cond = labels if self.use_controlnet else None

            # Apply CFG dropout to ControlNet conditioning (enables CFG at inference)
            if controlnet_cond is not None:
                batch_size = images.shape[0] if not isinstance(images, dict) else next(iter(images.values())).shape[0]
                controlnet_cond = self._apply_conditioning_dropout(controlnet_cond, batch_size)

            # Encode to diffusion space (identity for PixelSpace)
            # Skip encoding if data is already in latent space (from latent dataloader)
            if not is_latent:
                images = self.space.encode_batch(images)
            # For ControlNet: keep labels in pixel space (conditioning through ControlNet)
            # For concatenation: encode labels to latent space
            # Skip if labels are already encoded (bravo_seg_cond mode)
            if labels is not None and not self.use_controlnet and not labels_is_latent:
                labels = self.space.encode(labels)

            labels_dict = {'labels': labels, 'bin_maps': bin_maps}

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

            # For ControlNet (Stage 1 or 2): use only noisy images (no concatenation)
            # For standard: concatenate noisy images with labels
            if self.use_controlnet or self.controlnet_stage1:
                model_input = noisy_images  # Stage 1: unconditional, Stage 2: via controlnet_cond
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
                    from .training_tricks import compute_scoreaug_loss
                    mse_loss, p_loss, predicted_clean = compute_scoreaug_loss(
                        self, model_input, timesteps, images, noise, noisy_images, mode_id
                    )
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
                    # For 3D: use 2.5D approach (center slice) for efficiency
                    if self.perceptual_weight > 0:
                        if self.space.scale_factor > 1:
                            # Decode to pixel space for perceptual loss
                            pred_decoded = self.space.decode_batch(predicted_clean)
                            images_decoded = self.space.decode_batch(images)
                        else:
                            pred_decoded = predicted_clean
                            images_decoded = images

                        # For 3D: extract center slice (2.5D perceptual loss)
                        if self.spatial_dims == 3:
                            pred_decoded = self._extract_center_slice(pred_decoded)
                            images_decoded = self._extract_center_slice(images_decoded)

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
            if self.sda is not None and self.score_aug is None:
                from .training_tricks import compute_sda_loss
                sda_loss = compute_sda_loss(
                    self, images, noise, timesteps, labels, mode_id
                )
                total_loss = total_loss + self.sda_weight * sda_loss

        # Optimizer step (with gradient scaler for 3D AMP)
        if self.scaler is not None:
            # 3D path: use gradient scaler
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model_raw.parameters(), max_norm=self.cfg.training.get('gradient_clip_norm', 1.0)
            )
            self._add_gradient_noise(self._global_step)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 2D path: standard backward
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model_raw.parameters(), max_norm=self.cfg.training.get('gradient_clip_norm', 1.0)
            )
            self._add_gradient_noise(self._global_step)
            self.optimizer.step()

        self._global_step += 1

        if self.use_ema:
            self._update_ema()

        # Track gradient norm
        if self.log_grad_norm and grad_norm is not None and self._unified_metrics is not None:
            grad_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            self._unified_metrics.update_grad_norm(grad_val)

        return TrainingStepResult(
            total_loss=total_loss.item(),
            reconstruction_loss=0.0,  # Not applicable for diffusion
            perceptual_loss=p_loss.item(),
            mse_loss=mse_loss.item(),
        )

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> tuple[float, float, float]:
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
            data_loader, epoch, not self.verbose, self.is_main_process,
            limit_batches=self.limit_train_batches
        )

        for step, batch in enumerate(epoch_iter):
            # Cache first batch for deterministic visualization (for 3D, uses real conditioning)
            if self._cached_train_batch is None and self.spatial_dims == 3:
                prepared = self.mode.prepare_batch(batch, self.device)
                self._cached_train_batch = {
                    'images': prepared['images'].detach().clone(),
                    'labels': prepared.get('labels').detach().clone() if prepared.get('labels') is not None else None,
                    'size_bins': prepared.get('size_bins').detach().clone() if prepared.get('size_bins') is not None else None,
                    'bin_maps': prepared.get('bin_maps').detach().clone() if prepared.get('bin_maps') is not None else None,
                    'is_latent': prepared.get('is_latent', False),
                    'labels_is_latent': prepared.get('labels_is_latent', False),
                }

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
            self._unified_metrics.update_loss('MSE', avg_mse)
            if self.perceptual_weight > 0:
                self._unified_metrics.update_loss('Total', avg_loss)
                self._unified_metrics.update_loss('Perceptual', avg_perceptual)
            self._unified_metrics.update_lr(self.lr_scheduler.get_last_lr()[0])
            self._unified_metrics.update_vram()
            self._unified_metrics.log_training(epoch)
            self._unified_metrics.reset_training()

        return avg_loss, avg_mse, avg_perceptual

    def compute_validation_losses(self, epoch: int) -> tuple[dict[str, float], dict[str, Any] | None]:
        """Compute losses and metrics on validation set."""
        from .validation import compute_validation_losses
        return compute_validation_losses(self, epoch)

    # ─────────────────────────────────────────────────────────────────────────
    # Sample Generation and Visualization (unified for 2D/3D)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _visualize_samples(
        self,
        model: nn.Module,
        epoch: int,
        train_dataset: Dataset | None = None,
    ) -> None:
        """Generate and visualize samples."""
        from .visualization import visualize_samples
        visualize_samples(self, model, epoch, train_dataset)

    @torch.no_grad()
    def _visualize_samples_3d(self, model: nn.Module, epoch: int) -> None:
        """Generate and visualize 3D samples (center slices)."""
        from .visualization import visualize_samples_3d
        visualize_samples_3d(self, model, epoch)

    @torch.no_grad()
    def _visualize_denoising_trajectory(self, model: nn.Module, epoch: int, num_steps: int = 5) -> None:
        """Visualize intermediate denoising steps."""
        from .visualization import visualize_denoising_trajectory
        visualize_denoising_trajectory(self, model, epoch, num_steps)

    @torch.no_grad()
    def _visualize_denoising_trajectory_3d(
        self,
        model: nn.Module,
        epoch: int,
        num_steps: int = 5,
    ) -> None:
        """Visualize intermediate denoising steps for 3D volumes."""
        from .visualization import visualize_denoising_trajectory_3d
        visualize_denoising_trajectory_3d(self, model, epoch, num_steps)

    @torch.no_grad()
    def _generate_trajectory_3d(
        self,
        model: nn.Module,
        model_input: torch.Tensor,
        num_steps: int = 25,
        capture_every: int = 5,
    ) -> list[torch.Tensor]:
        """Generate samples while capturing intermediate states (3D)."""
        from .visualization import generate_trajectory_3d
        return generate_trajectory_3d(self, model, model_input, num_steps, capture_every)

    @torch.no_grad()
    def _generate_with_size_bins_3d(
        self,
        noise: torch.Tensor,
        size_bins: torch.Tensor,
        num_steps: int = 25,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate 3D samples with size bin conditioning."""
        from .visualization import generate_with_size_bins_3d
        return generate_with_size_bins_3d(self, noise, size_bins, num_steps, cfg_scale)

    @torch.no_grad()
    def _generate_trajectory_with_size_bins_3d(
        self,
        noise: torch.Tensor,
        size_bins: torch.Tensor,
        num_steps: int = 25,
        capture_every: int = 5,
    ) -> list[torch.Tensor]:
        """Generate 3D samples with size bins while capturing trajectory."""
        from .visualization import generate_trajectory_with_size_bins_3d
        return generate_trajectory_with_size_bins_3d(self, noise, size_bins, num_steps, capture_every)

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
        val_loader: DataLoader | None = None,
        pixel_train_loader: DataLoader | None = None,
        pixel_val_loader: DataLoader | None = None,
    ) -> None:
        """Main training loop.

        Args:
            train_loader: Training data loader (latent or pixel space).
            train_dataset: Training dataset (for sample generation).
            val_loader: Optional validation dataloader (latent or pixel space).
            pixel_train_loader: Optional pixel-space train loader for reference features.
                Only needed when train_loader is latent space.
            pixel_val_loader: Optional pixel-space val loader for reference features.
                Only needed when val_loader is latent space.
        """
        # Clear caches from any previous run
        self._clear_caches()

        total_start = time.time()
        self.val_loader = val_loader

        # Initialize unified metrics system
        # Build volume_size for 3D regional tracking
        volume_size = None
        if self.spatial_dims == 3:
            volume_size = (self.volume_height, self.volume_width, self.volume_depth)

        # Determine modality for metric suffixes
        # seg_conditioned modes: no suffix (distinguish by TensorBoard run color)
        # bravo_seg_cond: use 'bravo' as the target modality
        if self.mode_name in ('multi', 'dual'):
            metric_modality = None
        elif self.mode_name.startswith('seg_conditioned'):
            metric_modality = None  # No suffix for seg_conditioned modes
        elif self.mode_name == 'bravo_seg_cond':
            metric_modality = 'bravo'
        else:
            metric_modality = self.mode_name

        self._unified_metrics = UnifiedMetrics(
            writer=self.writer,
            mode=self.mode_name,
            spatial_dims=self.spatial_dims,
            modality=metric_modality,
            device=self.device,
            enable_regional=self.log_regional_losses,
            num_timestep_bins=10,
            image_size=self.image_size,
            volume_size=volume_size,
            fov_mm=self.cfg.paths.get('fov_mm', 240.0),
            # Logging config flags
            log_grad_norm=self.log_grad_norm,
            log_timestep_losses=self.log_timestep_losses,
            log_regional_losses=self.log_regional_losses,
            log_msssim=self.log_msssim,
            log_psnr=self.log_psnr,
            log_lpips=self.log_lpips,
            log_flops=self.log_flops,
            # SNR weight config
            strategy_name=self.strategy_name,
            num_train_timesteps=self.num_timesteps,
            use_min_snr=self.use_min_snr,
            min_snr_gamma=self.min_snr_gamma,
        )
        self._unified_metrics.set_scheduler(self.scheduler)

        # Measure FLOPs
        self._measure_model_flops(train_loader)

        if self.is_main_process and self.mode_name in ('seg', 'seg_conditioned'):
            logger.info(f"{self.mode_name} mode: perceptual loss and LPIPS disabled (binary masks)")

        # Initialize generation metrics (cache reference features)
        if self._gen_metrics is not None and self.is_main_process:
            # Determine seg channel index based on mode
            # seg/seg_conditioned: data is just [seg] or (seg, size_bins), so index 0
            # bravo, multi, multi_modality: data is [image, seg], so index 1
            # dual: data is [t1_pre, t1_gd, seg], so index 2
            if self.mode_name in ('seg', 'seg_conditioned'):
                seg_channel_idx = 0
            elif self.mode_name in ('bravo', 'multi', 'multi_modality'):
                seg_channel_idx = 1
            else:
                seg_channel_idx = 2
            self._gen_metrics.set_fixed_conditioning(
                train_dataset,
                num_masks=self._gen_metrics_config.samples_extended,  # Use extended count for conditioning
                seg_channel_idx=seg_channel_idx,
            )
            # Cache reference features from pixel-space loaders
            # For latent diffusion: use separate pixel loaders for feature extraction
            # For pixel-space diffusion: use train/val loaders directly
            # Use content-based cache key so all experiments with same data share cache
            ref_train_loader = pixel_train_loader if pixel_train_loader is not None else train_loader
            ref_val_loader = pixel_val_loader if pixel_val_loader is not None else val_loader
            if ref_val_loader is not None:
                import hashlib
                data_dir = str(self.cfg.paths.data_dir)
                cache_key = f"{data_dir}_{self.mode_name}_{self.image_size}"
                cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
                cache_id = f"{self.mode_name}_{self.image_size}_{cache_hash}"
                self._gen_metrics.cache_reference_features(
                    ref_train_loader,
                    ref_val_loader,
                    experiment_id=cache_id,
                )

        avg_loss = float('inf')
        avg_mse = float('inf')
        time_estimator = EpochTimeEstimator(self.n_epochs)

        try:
            for epoch in range(self.n_epochs):
                epoch_start = time.time()

                if self.use_multi_gpu and hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)

                avg_loss, avg_mse, avg_perceptual = self.train_epoch(train_loader, epoch)

                if self.use_multi_gpu:
                    # Synchronize all ranks before collective operation to prevent deadlock
                    dist.barrier()
                    loss_tensor = torch.tensor([avg_loss, avg_mse, avg_perceptual], device=self.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    avg_loss, avg_mse, avg_perceptual = (loss_tensor / self.world_size).cpu().numpy()

                # Step non-plateau schedulers here (plateau needs val_loss, stepped later)
                if self.scheduler_type != 'plateau':
                    self.lr_scheduler.step()

                if self.is_main_process:
                    # Log training losses using unified system (DDP path)
                    if self.use_multi_gpu:
                        self._unified_metrics.update_loss('MSE', avg_mse)
                        if self.perceptual_weight > 0:
                            self._unified_metrics.update_loss('Total', avg_loss)
                            self._unified_metrics.update_loss('Perceptual', avg_perceptual)
                        self._unified_metrics.update_lr(self.lr_scheduler.get_last_lr()[0])
                        self._unified_metrics.update_vram()
                        self._unified_metrics.log_training(epoch)
                        self._unified_metrics.reset_training()

                    val_metrics, worst_val_data = self.compute_validation_losses(epoch)
                    log_figures = (epoch + 1) % self.figure_interval == 0

                    self._unified_metrics.log_flops_from_tracker(self._flops_tracker, epoch)
                    self._unified_metrics.log_vram(epoch)

                    if log_figures and worst_val_data is not None:
                        # Get latent-space data
                        original = worst_val_data['original']
                        generated = worst_val_data['generated']

                        # Log latent space visualization before decoding (for latent diffusion)
                        if self.space.scale_factor > 1:
                            self._unified_metrics.log_latent_samples(
                                generated.to(self.device), epoch, tag='val/worst_batch_latent'
                            )
                            # Decode from latent space
                            original = self.space.decode(original.to(self.device))
                            generated = self.space.decode(generated.to(self.device))

                        self._unified_metrics.log_worst_batch(
                            original=original,
                            reconstructed=generated,
                            loss=worst_val_data['loss'],
                            epoch=epoch,
                            phase='val',
                            mask=worst_val_data.get('mask'),
                            timesteps=worst_val_data.get('timesteps'),
                        )

                    self._compute_per_modality_validation(epoch)

                    if log_figures or (epoch + 1) == self.n_epochs:
                        model_to_use = self.ema.ema_model if self.ema is not None else self.model_raw
                        self._visualize_samples(model_to_use, epoch, train_dataset)

                    self._save_checkpoint(epoch, "latest")

                    loss_for_selection = val_metrics.get('total', avg_loss)

                    # Step plateau scheduler with validation loss
                    if self.scheduler_type == 'plateau':
                        self.lr_scheduler.step(loss_for_selection)

                    if loss_for_selection < self.best_loss:
                        self.best_loss = loss_for_selection
                        self._save_checkpoint(epoch, "best")
                        loss_type = "val" if val_metrics else "train"
                        logger.info(f"New best model saved ({loss_type} loss: {loss_for_selection:.6f})")

                    # Log epoch summary with FULL epoch time (training + validation + viz + checkpointing)
                    # This gives accurate ETA by including all epoch overhead, not just training
                    epoch_time = time.time() - epoch_start
                    log_epoch_summary(epoch, self.n_epochs, (avg_loss, avg_mse, avg_perceptual), epoch_time, time_estimator)

        finally:
            total_time = time.time() - total_start

            if self.is_main_process:
                logger.info(f"Training completed! Total time: {total_time:.1f}s ({total_time / 3600:.1f}h)")
                self._update_metadata_final(avg_loss, avg_mse, total_time)

            if self.use_multi_gpu:
                try:
                    dist.destroy_process_group()
                except RuntimeError as e:
                    # Expected during abnormal shutdown (e.g., process already terminated)
                    logger.debug(f"Process group cleanup (rank={self.rank}): {e}")

    def _measure_model_flops(self, train_loader: DataLoader) -> None:
        """Measure model FLOPs using batch_size=1 to avoid OOM during torch.compile."""
        from .profiling import measure_model_flops
        measure_model_flops(self, train_loader)

    def _compute_per_modality_validation(self, epoch: int) -> None:
        """Compute and log validation metrics for each modality separately."""
        from .validation import compute_per_modality_validation
        compute_per_modality_validation(self, epoch)

    def _compute_volume_3d_msssim(
        self,
        epoch: int,
        data_split: str = 'val',
        modality_override: str | None = None,
    ) -> float | None:
        """Compute 3D MS-SSIM by reconstructing full volumes."""
        from .evaluation import compute_volume_3d_msssim
        return compute_volume_3d_msssim(self, epoch, data_split, modality_override)

    def _compute_volume_3d_msssim_native(
        self,
        epoch: int,
        data_split: str = 'val',
        modality_override: str | None = None,
    ) -> float | None:
        """Compute 3D MS-SSIM for 3D diffusion models (native volume processing)."""
        from .evaluation import compute_volume_3d_msssim_native
        return compute_volume_3d_msssim_native(self, epoch, data_split, modality_override)

    def _update_metadata_final(self, final_loss: float, final_mse: float, total_time: float) -> None:
        """Update metadata with final training stats."""
        from .profiling import update_metadata_final
        update_metadata_final(self, final_loss, final_mse, total_time)

    def evaluate_test_set(
        self,
        test_loader: DataLoader,
        checkpoint_name: str | None = None
    ) -> dict[str, float]:
        """Evaluate diffusion model on test set."""
        from .evaluation import evaluate_test_set
        return evaluate_test_set(self, test_loader, checkpoint_name)

    def _create_test_reconstruction_figure(
        self,
        original: torch.Tensor,
        predicted: torch.Tensor,
        metrics: dict[str, float],
        label: str,
        timesteps: torch.Tensor | None = None,
    ) -> plt.Figure:
        """Create side-by-side test evaluation figure."""
        from .evaluation import create_test_reconstruction_figure
        return create_test_reconstruction_figure(original, predicted, metrics, label, timesteps)

    def close_writer(self) -> None:
        """Close TensorBoard writer. Call after all logging is complete."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
