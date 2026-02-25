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
from typing import TYPE_CHECKING, Any

import matplotlib

if TYPE_CHECKING:
    from medgen.metrics.generation import GenerationMetrics
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from ema_pytorch import EMA
from omegaconf import DictConfig
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

from medgen.core import (
    ModeType,
)
from medgen.diffusion import (
    ConditionalDualMode,
    ConditionalSingleMode,
    DDPMStrategy,
    DiffusionSpace,
    DiffusionStrategy,
    LatentSegConditionedMode,
    LatentSpace,
    MultiModalityMode,
    RFlowStrategy,
    SegmentationConditionedInputMode,
    SegmentationConditionedMode,
    SegmentationMode,
    TrainingMode,
)
from medgen.evaluation import ValidationVisualizer
from medgen.metrics import UnifiedMetrics

from .base_config import ModeConfig, StrategyConfig
from .diffusion_config import DiffusionTrainerConfig
from .diffusion_trainer_base import DiffusionTrainerBase
from .results import BatchType, TrainingStepResult
from .utils import (
    create_epoch_iterator,
    get_vram_usage,
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
        # Extract typed config (single-source defaults)
        # ─────────────────────────────────────────────────────────────────────
        dc = DiffusionTrainerConfig.from_hydra(cfg, spatial_dims=spatial_dims)
        self._diffusion_config = dc
        mc = ModeConfig.from_hydra(cfg)
        self._mode_config = mc
        sc = StrategyConfig.from_hydra(cfg)

        # ─────────────────────────────────────────────────────────────────────
        # Dimension-specific size config (from typed config)
        # ─────────────────────────────────────────────────────────────────────
        self.image_size: int = dc.image_size
        if spatial_dims == 3:
            self.volume_height: int = dc.volume_height
            self.volume_width: int = dc.volume_width
            self.volume_depth: int = dc.volume_depth

        self.eta_min: float = dc.eta_min

        # Perceptual weight (disabled for seg modes - binary masks don't work with VGG features)
        self.perceptual_weight: float = dc.perceptual_weight

        # FP32 loss computation (set False to reproduce pre-Jan-7-2026 BF16 behavior)
        self.use_fp32_loss: bool = dc.use_fp32_loss
        if self.is_main_process:
            logger.debug(f"use_fp32_loss = {self.use_fp32_loss}")

        # Optimizer settings (from typed config)
        self.weight_decay: float = dc.weight_decay

        # Initialize mode and scheduler
        self.mode = self._create_mode(self.mode_name)
        # Setup scheduler with dimension-appropriate parameters
        scheduler_kwargs = {
            'num_timesteps': self.num_timesteps,
            'image_size': self.image_size,
            'use_discrete_timesteps': sc.use_discrete_timesteps,
            'sample_method': sc.sample_method,
            'use_timestep_transform': sc.use_timestep_transform,
            'prediction_type': sc.prediction_type,
            'schedule': sc.schedule,
        }
        if spatial_dims == 3:
            scheduler_kwargs['depth_size'] = self.volume_depth
            scheduler_kwargs['spatial_dims'] = 3
        # Disable scheduler clipping for wavelet/latent/rescaled space:
        # MONAI schedulers clip predicted x₀ to [-1, 1] by default, which
        # destroys wavelet coefficients and latent representations.
        if self.space.needs_decode:
            scheduler_kwargs['clip_sample'] = False
        self.scheduler = self.strategy.setup_scheduler(**scheduler_kwargs)

        # ODE solver config (RFlow only, used during generation)
        if hasattr(self.strategy, 'ode_solver'):
            self.strategy.ode_solver = sc.ode_solver
            self.strategy.ode_atol = sc.ode_atol
            self.strategy.ode_rtol = sc.ode_rtol

        # ─────────────────────────────────────────────────────────────────────
        # 3D-specific memory optimizations (from typed config)
        # ─────────────────────────────────────────────────────────────────────
        self.use_amp: bool = dc.use_amp
        self.use_gradient_checkpointing: bool = dc.use_gradient_checkpointing
        if spatial_dims == 3:
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        else:
            self.scaler = None

        # Logging config (from typed LoggingConfig)
        lc = dc.logging
        self.log_msssim: bool = lc.msssim
        self.log_psnr: bool = lc.psnr
        self.log_lpips: bool = lc.lpips
        self.log_timestep_region_losses: bool = lc.timestep_region_losses
        self.log_worst_batch: bool = lc.worst_batch
        self.log_intermediate_steps: bool = lc.intermediate_steps
        self.num_intermediate_steps: int = lc.num_intermediate_steps

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
        from medgen.metrics.generation import GenerationMetricsConfig
        gen_metrics_cfg = GenerationMetricsConfig.from_hydra(cfg, spatial_dims)
        if gen_metrics_cfg.enabled:
            self._gen_metrics_config = gen_metrics_cfg
            if self.is_main_process:
                sample_type = "volumes" if spatial_dims == 3 else "samples"
                logger.info(
                    f"Generation metrics enabled: {gen_metrics_cfg.samples_per_epoch} {sample_type}/epoch "
                    f"({gen_metrics_cfg.steps_per_epoch} steps), "
                    f"{gen_metrics_cfg.samples_extended} {sample_type}/extended "
                    f"({gen_metrics_cfg.steps_extended} steps)"
                )
        else:
            self._gen_metrics_config = None

        # ControlNet configuration
        self._setup_controlnet()

        # Pixel-space loaders for latent diffusion reference features
        # (set by script before train() call)
        self.pixel_train_loader: DataLoader | None = None
        self.pixel_val_loader: DataLoader | None = None

        # Hook state attributes
        self._last_worst_val_data: dict[str, Any] | None = None
        self._last_avg_losses: dict[str, float] = {}

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
        """Initialize ScoreAug transform and related settings."""
        from .diffusion_init_helpers import setup_score_aug
        setup_score_aug(self)

    def _setup_sda(self) -> None:
        """Initialize SDA (Shifted Data Augmentation) transform."""
        from .diffusion_init_helpers import setup_sda
        setup_sda(self)

    def _setup_conditional_embeddings(self) -> None:
        """Initialize mode embedding and size bin embedding settings."""
        from .diffusion_init_helpers import setup_conditional_embeddings
        setup_conditional_embeddings(self)

    def _setup_augmented_diffusion(self) -> None:
        """Initialize DC-AE 1.5 augmented diffusion training settings."""
        from .diffusion_init_helpers import setup_augmented_diffusion
        setup_augmented_diffusion(self)

    def _setup_regional_weighting(self) -> None:
        """Initialize region-weighted loss computer."""
        from .diffusion_init_helpers import setup_regional_weighting
        setup_regional_weighting(self)

    def _setup_controlnet(self) -> None:
        """Initialize ControlNet configuration."""
        from .diffusion_init_helpers import setup_controlnet
        setup_controlnet(self)

    def _log_training_tricks_config(self) -> None:
        """Log configuration for various training tricks."""
        from .diffusion_init_helpers import log_training_tricks_config
        log_training_tricks_config(self)

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
        from .diffusion_model_setup import create_fallback_save_dir
        return create_fallback_save_dir(self)

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

        mc = self._mode_config

        if mode == ModeType.DUAL or mode == 'dual':
            image_keys = mc.image_keys if mc.image_keys else None
            return ConditionalDualMode(image_keys)

        if mode == 'multi':
            image_keys = mc.image_keys if mc.image_keys else None
            return MultiModalityMode(image_keys)

        if mode == 'seg_conditioned':
            return SegmentationConditionedMode(mc.size_bins)

        if mode == 'seg_conditioned_input':
            return SegmentationConditionedInputMode(mc.size_bins)

        if mode == 'bravo_seg_cond':
            latent_channels = mc.latent_channels if mc.latent_channels is not None else 4
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
        """Initialize model, optimizer, and loss functions."""
        from .diffusion_model_setup import setup_model
        setup_model(self, train_dataset)
        self._setup_checkpoint_manager()

    def _setup_ema(self) -> None:
        """Setup EMA wrapper if enabled."""
        if self.use_ema:
            ema_cfg = self.cfg.training.get('ema', {})
            self.ema = EMA(
                self.model_raw,
                beta=self.ema_decay,
                update_after_step=int(ema_cfg.get('update_after_step', 100)),
                update_every=int(ema_cfg.get('update_every', 10)),
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
        from .diffusion_model_setup import setup_compiled_forward
        setup_compiled_forward(self, enabled)

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

    def train_step(self, batch: BatchType) -> TrainingStepResult:
        """Execute single training step.

        Supports gradient accumulation: gradients are accumulated over
        `gradient_accumulation_steps` micro-batches before optimizer.step().

        Args:
            batch: Input batch from dataloader.

        Returns:
            TrainingStepResult with total, MSE, and perceptual losses.
        """
        accum_steps = self.gradient_accumulation_steps
        is_accum_start = (self._accum_step % accum_steps == 0)
        is_accum_end = (self._accum_step % accum_steps == accum_steps - 1)

        if is_accum_start:
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
            # Only active for learned latent spaces (VAE/VQ-VAE/DC-AE), not wavelet/S2D
            aug_diff_mask = None
            if self.augmented_diffusion_enabled and isinstance(self.space, LatentSpace):
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

                    if self.rflow_snr_gamma > 0:
                        from .losses import compute_rflow_snr_weighted_mse
                        target = self.strategy.compute_target(images, noise)
                        mse_loss = compute_rflow_snr_weighted_mse(
                            prediction, target, timesteps,
                            self.num_timesteps, self.rflow_snr_gamma,
                        )

                    if self.velocity_loss_type == 'pseudo_huber':
                        from .losses import compute_pseudo_huber_loss
                        target = self.strategy.compute_target(images, noise)
                        mse_loss = compute_pseudo_huber_loss(prediction, target)
                    elif self.velocity_loss_type == 'lpips_huber':
                        from .losses import compute_lpips_huber_loss
                        target = self.strategy.compute_target(images, noise)
                        mse_loss = compute_lpips_huber_loss(
                            prediction, target, timesteps, self.num_timesteps,
                        )

                    # Apply regional weighting (per-pixel weights by tumor size)
                    if self.regional_weight_computer is not None:
                        # For seg_conditioned: images IS the seg mask (labels=None)
                        seg_for_weighting = labels if labels is not None else images
                        if seg_for_weighting is not None and not isinstance(seg_for_weighting, dict):
                            mse_loss = self._compute_region_weighted_mse(
                                prediction, images, noise, seg_for_weighting
                            )

                    # Compute perceptual loss (decode for latent space)
                    # For 3D: use 2.5D approach (center slice) for efficiency
                    if self.perceptual_weight > 0:
                        if self.space.needs_decode:
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
                    tt = self._training_tricks
                    if tt.self_cond.enabled:
                        consistency_loss = self._compute_self_conditioning_loss(
                            model_input, timesteps, prediction, mode_id
                        )
                        total_loss = total_loss + tt.self_cond.consistency_weight * consistency_loss

            # Auxiliary bin prediction loss (forces model to encode bin info)
            aux_bin_loss = torch.tensor(0.0, device=self.device)
            self._aux_bin_level_losses = None  # Reset per-level tracking
            if self.use_size_bin_embedding and getattr(self, 'size_bin_aux_loss_weight', 0) > 0 and size_bins is not None:
                # Skip CFG-dropped samples (all-zeros = unconditional)
                mask = size_bins.sum(dim=1) > 0  # [B] — True for conditioned samples

                bin_pred_multi = getattr(self, '_bin_prediction_multi', None)
                bin_pred_head = getattr(self, '_bin_prediction_head', None)

                if bin_pred_multi is not None and mask.any():
                    # Multi-level path: compute per-level MSE, average them
                    predictions = bin_pred_multi.predict_all()
                    if predictions:
                        target = size_bins[mask].float()
                        level_losses = {}
                        for level_name, pred in predictions.items():
                            level_losses[level_name] = nn.functional.mse_loss(pred[mask], target)
                        aux_bin_loss = torch.stack(list(level_losses.values())).mean()
                        total_loss = total_loss + self.size_bin_aux_loss_weight * aux_bin_loss
                        # Store for per-level TensorBoard logging
                        self._aux_bin_level_losses = {k: v.item() for k, v in level_losses.items()}
                    bin_pred_multi.clear_all_caches()

                elif bin_pred_head is not None and mask.any():
                    # Single-head path (bottleneck only, exp2d behavior)
                    bin_pred = bin_pred_head.predict()  # [B, num_bins]
                    if bin_pred is not None:
                        target = size_bins[mask].float()
                        pred = bin_pred[mask]
                        aux_bin_loss = nn.functional.mse_loss(pred, target)
                        total_loss = total_loss + self.size_bin_aux_loss_weight * aux_bin_loss
                    bin_pred_head.clear_cache()

            # SDA (Shifted Data Augmentation) path
            # Unlike ScoreAug, SDA transforms CLEAN data before noise addition
            # and uses shifted timesteps to prevent leakage
            if self.sda is not None and self.score_aug is None:
                from .training_tricks import compute_sda_loss
                sda_loss = compute_sda_loss(
                    self, images, noise, timesteps, labels, mode_id
                )
                total_loss = total_loss + self.sda_weight * sda_loss

        # Scale loss for gradient accumulation (so accumulated gradients average correctly)
        scaled_loss = total_loss / accum_steps if accum_steps > 1 else total_loss

        # Backward pass (always runs, accumulates gradients)
        grad_norm = None
        if self.scaler is not None:
            # 3D path: use gradient scaler
            self.scaler.scale(scaled_loss).backward()
        else:
            # 2D path: standard backward
            scaled_loss.backward()

        # Optimizer step only at accumulation boundary
        if is_accum_end:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model_raw.parameters(), max_norm=self.gradient_clip_norm
                )
                grad_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                if self._grad_skip_detector.should_skip(grad_val):
                    self.optimizer.zero_grad()
                else:
                    self._add_gradient_noise(self._global_step)
                    self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model_raw.parameters(), max_norm=self.gradient_clip_norm
                )
                grad_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                if self._grad_skip_detector.should_skip(grad_val):
                    self.optimizer.zero_grad()
                else:
                    self._add_gradient_noise(self._global_step)
                    self.optimizer.step()

            self._global_step += 1

            if self.use_ema:
                self._update_ema()

        self._accum_step += 1

        # Track gradient norm (only when optimizer stepped)
        if is_accum_end and self.log_grad_norm and grad_norm is not None and self._unified_metrics is not None:
            grad_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            self._unified_metrics.update_grad_norm(grad_val)

        return TrainingStepResult(
            total_loss=total_loss.item(),
            reconstruction_loss=0.0,  # Not applicable for diffusion
            perceptual_loss=p_loss.item(),
            mse_loss=mse_loss.item(),
            aux_bin_loss=aux_bin_loss.item(),
        )

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> dict[str, float]:
        """Train the model for one epoch.

        Args:
            data_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dict with keys 'loss', 'mse', 'perceptual' containing averaged epoch losses.
        """
        self._current_epoch = epoch
        self.model.train()

        # Release cached memory before training to prevent OOM from fragmentation
        # (3D generation metrics load/unload feature extractors, leaving fragmented pools)
        if self.spatial_dims == 3:
            torch.cuda.empty_cache()

        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_perceptual_loss = 0
        epoch_aux_bin_loss = 0
        epoch_aux_bin_level_losses: dict[str, float] = {}  # Per-level accumulation

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
            epoch_aux_bin_loss += result.aux_bin_loss
            # Accumulate per-level aux bin losses (set by train_step when multilevel)
            level_losses = getattr(self, '_aux_bin_level_losses', None)
            if level_losses:
                for level_name, loss_val in level_losses.items():
                    epoch_aux_bin_level_losses[level_name] = (
                        epoch_aux_bin_level_losses.get(level_name, 0.0) + loss_val
                    )

            if hasattr(epoch_iter, 'set_postfix'):
                epoch_iter.set_postfix(loss=f"{epoch_loss / (step + 1):.6f}")

            if epoch == 1 and step == 0 and self.is_main_process:
                logger.info(get_vram_usage(self.device))

            # Break mid-epoch on SIGTERM so checkpoint can be saved before SIGKILL
            if self._sigterm_received:
                logger.warning(f"SIGTERM: breaking out of epoch {epoch} at step {step + 1}/{len(data_loader)}")
                break

        # Flush partial gradient accumulation window at epoch end
        accum_steps = self.gradient_accumulation_steps
        if accum_steps > 1 and self._accum_step % accum_steps != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model_raw.parameters(), max_norm=self.gradient_clip_norm
                )
                grad_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                if not self._grad_skip_detector.should_skip(grad_val):
                    self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model_raw.parameters(), max_norm=self.gradient_clip_norm
                )
                grad_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                if not self._grad_skip_detector.should_skip(grad_val):
                    self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._global_step += 1
            if self.use_ema:
                self._update_ema()
            self._accum_step = 0

        n_batches = min(step + 1, self.limit_train_batches or len(data_loader))
        avg_loss = epoch_loss / n_batches
        avg_mse = epoch_mse_loss / n_batches
        avg_perceptual = epoch_perceptual_loss / n_batches
        avg_aux_bin = epoch_aux_bin_loss / n_batches

        # Sync losses across DDP ranks before logging
        if self.use_multi_gpu:
            dist.barrier()
            loss_tensor = torch.tensor([avg_loss, avg_mse, avg_perceptual], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss, avg_mse, avg_perceptual = (loss_tensor / self.world_size).cpu().tolist()

        # Log training losses (always uses synced values for multi-GPU)
        if self.is_main_process:
            self._unified_metrics.update_loss('MSE', avg_mse)
            if self.perceptual_weight > 0:
                self._unified_metrics.update_loss('Total', avg_loss)
                self._unified_metrics.update_loss('Perceptual', avg_perceptual)
            if avg_aux_bin > 0:
                self._unified_metrics.update_loss('AuxBin', avg_aux_bin)
            # Per-level aux bin losses (multilevel mode)
            if epoch_aux_bin_level_losses:
                for level_name, total_level_loss in epoch_aux_bin_level_losses.items():
                    avg_level_loss = total_level_loss / n_batches
                    self._unified_metrics.update_loss(f'AuxBin/{level_name}', avg_level_loss)
            self._unified_metrics.update_lr(self.lr_scheduler.get_last_lr()[0])
            self._unified_metrics.update_vram()
            self._unified_metrics.log_training(epoch)
            self._unified_metrics.reset_training()

        return {'loss': avg_loss, 'mse': avg_mse, 'perceptual': avg_perceptual}

    def compute_validation_losses(self, epoch: int, log_figures: bool = True) -> dict[str, float]:
        """Compute losses and metrics on validation set."""
        from .validation import compute_validation_losses
        val_metrics, worst_val_data = compute_validation_losses(self, epoch)
        self._last_worst_val_data = worst_val_data if log_figures else None
        return val_metrics

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
    def _generate_trajectory(
        self,
        model: nn.Module,
        model_input: torch.Tensor,
        num_steps: int = 25,
        capture_every: int = 5,
    ) -> list[torch.Tensor]:
        """Generate samples while capturing intermediate states."""
        from .visualization import generate_trajectory
        return generate_trajectory(
            model, model_input,
            num_train_timesteps=self.scheduler.num_train_timesteps,
            is_conditional=self.mode.is_conditional,
            latent_channels=self.space.latent_channels,
            scale_factor=self.space.scale_factor,
            num_steps=num_steps,
            capture_every=capture_every,
        )

    @torch.no_grad()
    def _generate_with_size_bins(
        self,
        model: nn.Module,
        noise: torch.Tensor,
        size_bins: torch.Tensor,
        num_steps: int = 25,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate samples with size bin conditioning."""
        from .visualization import generate_with_size_bins
        return generate_with_size_bins(
            model, noise, size_bins,
            num_train_timesteps=self.scheduler.num_train_timesteps,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
        )

    @torch.no_grad()
    def _generate_trajectory_with_size_bins(
        self,
        model: nn.Module,
        noise: torch.Tensor,
        size_bins: torch.Tensor,
        num_steps: int = 25,
        capture_every: int = 5,
    ) -> list[torch.Tensor]:
        """Generate samples with size bins while capturing trajectory."""
        from .visualization import generate_trajectory_with_size_bins
        return generate_trajectory_with_size_bins(
            model, noise, size_bins,
            num_train_timesteps=self.scheduler.num_train_timesteps,
            num_steps=num_steps,
            capture_every=capture_every,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Hook Overrides (customize BaseTrainer.train() behavior)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_training_start(self) -> None:
        """Initialize metrics, generation metrics, and logging."""
        super()._on_training_start()
        self._clear_caches()

        # Enable EDM preconditioning on strategy if configured
        if self._strategy_config.sigma_data > 0 and self.strategy_name == 'rflow':
            out_ch = self._mode_config.out_channels
            self.strategy.set_preconditioning(self._strategy_config.sigma_data, out_ch)

        # Initialize unified metrics system
        volume_size = None
        if self.spatial_dims == 3:
            volume_size = (self.volume_height, self.volume_width, self.volume_depth)

        # Determine modality for metric suffixes
        # Multi/dual handle per-channel suffixes internally, so no global suffix
        if self.mode_name in ('multi', 'dual'):
            metric_modality = None
        elif self.mode_name == 'bravo_seg_cond':
            metric_modality = 'bravo'
        else:
            metric_modality = self.mode_name  # e.g. 'bravo', 'seg'

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
            fov_mm=self._paths_config.fov_mm,
            log_grad_norm=self.log_grad_norm,
            log_timestep_losses=self.log_timestep_losses,
            log_regional_losses=self.log_regional_losses,
            log_msssim=self.log_msssim,
            log_psnr=self.log_psnr,
            log_lpips=self.log_lpips,
            log_flops=self.log_flops,
            strategy_name=self.strategy_name,
            num_train_timesteps=self.num_timesteps,
            use_min_snr=self.use_min_snr,
            min_snr_gamma=self.min_snr_gamma,
        )
        self._unified_metrics.set_scheduler(self.scheduler)

        if self.is_main_process and self.mode_name in ('seg', 'seg_conditioned'):
            logger.info(f"{self.mode_name} mode: perceptual loss and LPIPS disabled (binary masks)")

        # Initialize generation metrics (cache reference features)
        if self._gen_metrics is not None and self.is_main_process:
            if self.mode_name in ('seg', 'seg_conditioned'):
                seg_channel_idx = 0
            elif self.mode_name in ('bravo', 'multi', 'multi_modality'):
                seg_channel_idx = 1
            else:
                seg_channel_idx = 2
            self._gen_metrics.set_fixed_conditioning(
                self.train_dataset,
                num_masks=self._gen_metrics_config.samples_extended,
                seg_channel_idx=seg_channel_idx,
            )
            ref_train_loader = self.pixel_train_loader if self.pixel_train_loader is not None else self._train_loader
            ref_val_loader = self.pixel_val_loader if self.pixel_val_loader is not None else self.val_loader
            if ref_val_loader is not None:
                import hashlib
                data_dir = str(self._paths_config.data_dir)
                cache_key = f"{data_dir}_{self.mode_name}_{self.image_size}"
                cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
                cache_id = f"{self.mode_name}_{self.image_size}_{cache_hash}"
                self._gen_metrics.cache_reference_features(
                    ref_train_loader,
                    ref_val_loader,
                    experiment_id=cache_id,
                )

        if self.is_main_process:
            n_batches = len(self._train_loader) if hasattr(self, '_train_loader') else 0
            accum = self.gradient_accumulation_steps
            effective_bs = self.batch_size * accum
            accum_info = f", grad_accum={accum} (effective_bs={effective_bs})" if accum > 1 else ""
            logger.info(
                f"Starting training: {self.n_epochs} epochs, "
                f"{n_batches} batches/epoch, batch_size={self.batch_size}{accum_info}"
            )

    def _step_scheduler(self, epoch: int, val_metrics: dict[str, float]) -> None:
        """Step scheduler, handling plateau type specially."""
        if self.lr_scheduler is None:
            return
        if self.scheduler_type == 'plateau':
            loss = val_metrics.get('total', val_metrics.get('loss', 0))
            self.lr_scheduler.step(loss)
        else:
            self.lr_scheduler.step()

    def _on_epoch_end(self, epoch: int, avg_losses: dict[str, float], val_metrics: dict[str, float]) -> None:
        super()._on_epoch_end(epoch, avg_losses, val_metrics)

        # Store for _on_training_end metadata finalization
        self._last_avg_losses = avg_losses

        # Worst-batch visualization
        log_figures = (epoch + 1) % self.figure_interval == 0
        worst_val_data = getattr(self, '_last_worst_val_data', None)
        if log_figures and worst_val_data is not None:
            original = worst_val_data['original']
            generated = worst_val_data['generated']
            if self.space.needs_decode:
                self._unified_metrics.log_latent_samples(
                    generated.to(self.device), epoch, tag='val/worst_batch_latent'
                )
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

        # Per-modality validation
        self._compute_per_modality_validation(epoch)

        # Sample visualization
        if log_figures or (epoch + 1) == self.n_epochs:
            model_to_use = self.ema.ema_model if self.ema is not None else self.model_raw
            self._visualize_samples(model_to_use, epoch, self.train_dataset)

    def _on_training_end(self, total_time: float) -> None:
        if self.is_main_process:
            last_losses = getattr(self, '_last_avg_losses', {})
            avg_loss = last_losses.get('loss', float('inf'))
            avg_mse = last_losses.get('mse', float('inf'))
            self._update_metadata_final(avg_loss, avg_mse, total_time)
        super()._on_training_end(total_time)

    def _get_best_metric_name(self) -> str:
        return 'total'

    def _get_checkpoint_extra_state(self) -> dict | None:
        return {'best_loss': self.best_loss}

    def _log_epoch_summary(self, epoch, total_epochs, avg_losses, val_metrics, elapsed_time):
        from .utils import log_epoch_summary
        log_epoch_summary(
            epoch, total_epochs,
            (avg_losses.get('loss', 0), avg_losses.get('mse', 0), avg_losses.get('perceptual', 0)),
            elapsed_time, self._time_estimator,
        )

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
            extra_state={'best_loss': self.best_loss},
        )

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load training state from a checkpoint for resume.

        Restores model weights, optimizer, scheduler, EMA, and best_loss.
        Handles missing keys gracefully for older checkpoints.

        Args:
            checkpoint_path: Path to checkpoint file (.pt).

        Returns:
            start_epoch: The epoch to resume from (saved epoch + 1).
        """
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model_raw.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Restored LR scheduler state")

        if 'ema_state_dict' in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
            logger.info("Restored EMA state")

        # Restore best metric for checkpoint tracking.
        # Prefer best_metric (CM's authoritative value) over best_loss, because
        # self.best_loss is never updated when CM is active — it stays at inf.
        if 'best_metric' in checkpoint:
            self.best_loss = checkpoint['best_metric']
            logger.info(f"Restored best_loss (from best_metric): {self.best_loss:.6f}")
        elif 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']
            logger.info(f"Restored best_loss: {self.best_loss:.6f}")

        # Sync to CheckpointManager to prevent best model regression on resume
        if self.best_loss < float('inf') and self.checkpoint_manager is not None:
            self.checkpoint_manager.set_best_metric(self.best_loss)

        saved_epoch = checkpoint['epoch']
        start_epoch = saved_epoch + 1
        logger.info(f"Resuming from epoch {start_epoch} (checkpoint saved at epoch {saved_epoch})")
        return start_epoch

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

