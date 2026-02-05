"""
Compression trainer module for VAE, VQ-VAE, DC-AE (2D and 3D).

This module provides:
- BaseCompressionTrainer: Unified base class for all compression model trainers
  supporting both 2D images and 3D volumes via spatial_dims parameter

All compression trainers share:
- GAN training (optional discriminator)
- Perceptual loss (2D or 2.5D for 3D)
- EMA support
- Gradient norm tracking for G and D
- Worst batch tracking and visualization
- Regional metrics (tumor vs background)
"""
import logging
import os
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medgen.evaluation import ValidationRunner

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from ema_pytorch import EMA
from monai.losses import PatchAdversarialLoss
from monai.networks.nets import PatchDiscriminator
from omegaconf import DictConfig
from torch import nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from medgen.core import create_warmup_cosine_scheduler, wrap_model_for_training
from medgen.core.defaults import COMPRESSION_DEFAULTS
from medgen.core.dict_utils import get_with_fallbacks
from medgen.losses import LPIPSLoss, PerceptualLoss
from medgen.metrics import (
    GradientNormTracker,
    RegionalMetricsTracker,
    SimpleLossAccumulator,
    UnifiedMetrics,
    compute_lpips,
    compute_lpips_3d,
    compute_msssim,
    compute_msssim_2d_slicewise,
    compute_psnr,
    create_worst_batch_figure,
)

from .base_trainer import BaseTrainer
from .results import TrainingStepResult
from .utils import save_full_checkpoint

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Batch Preparation Helpers
# =============================================================================

def _tensor_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move tensor to device with async transfer and MetaTensor handling.

    Args:
        tensor: Input tensor (may be MONAI MetaTensor).
        device: Target device.

    Returns:
        Tensor on target device.
    """
    if hasattr(tensor, 'as_tensor'):
        tensor = tensor.as_tensor()
    return tensor.to(device, non_blocking=True)


class BaseCompressionTrainer(BaseTrainer):
    """Unified base trainer for compression models (VAE, VQ-VAE, DC-AE).

    Supports both 2D images and 3D volumes via the spatial_dims parameter.

    Extends BaseTrainer with:
    - GAN training infrastructure (discriminator, adversarial loss)
    - Perceptual loss (2D or 2.5D for 3D volumes)
    - EMA support for generator
    - Dual gradient tracking (G and D)
    - Worst batch visualization
    - Per-modality validation
    - Gradient checkpointing support (3D)

    Subclasses must implement:
    - setup_model(): Create model, discriminator, optimizers
    - _forward_for_validation(): Model-specific forward pass
    - _get_model_config(): Return config dict for checkpoint
    - train_step(): Model-specific training step

    Args:
        cfg: Hydra configuration object.
        spatial_dims: Spatial dimensions (2 for 2D images, 3 for 3D volumes).
    """

    def __init__(self, cfg: DictConfig, spatial_dims: int = 2) -> None:
        """Initialize compression trainer.

        Args:
            cfg: Hydra configuration object.
            spatial_dims: Spatial dimensions (2 or 3).
        """
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        self._spatial_dims = spatial_dims

        super().__init__(cfg)

        # ─────────────────────────────────────────────────────────────────────
        # Dimension-specific size config
        # ─────────────────────────────────────────────────────────────────────
        if spatial_dims == 3:
            self.volume_depth: int = cfg.volume.get('depth', 160)
            self.volume_height: int = cfg.volume.get('height', 256)
            self.volume_width: int = cfg.volume.get('width', 256)
            # 3D-specific: 2.5D perceptual loss config
            self.use_2_5d_perceptual: bool = self._get_2_5d_perceptual(cfg)
            self.perceptual_slice_fraction: float = self._get_perceptual_slice_fraction(cfg)
            # 3D-specific: gradient checkpointing
            self.gradient_checkpointing: bool = cfg.training.get('gradient_checkpointing', True)
        else:
            self.image_size: int = cfg.model.image_size

        # ─────────────────────────────────────────────────────────────────────
        # Extract compression-specific config
        # ─────────────────────────────────────────────────────────────────────
        self.disc_lr: float = self._get_disc_lr(cfg)
        self.perceptual_weight: float = self._get_perceptual_weight(cfg)
        self.perceptual_loss_type: str = self._get_perceptual_loss_type(cfg)
        self.adv_weight: float = self._get_adv_weight(cfg)

        # GAN config
        self.disable_gan: bool = self._get_disable_gan(cfg)
        self.disc_num_layers: int = self._get_disc_num_layers(cfg)
        self.disc_num_channels: int = self._get_disc_num_channels(cfg)

        # EMA config
        self.use_ema: bool = cfg.training.get('use_ema', True)
        self.ema_decay: float = cfg.training.ema.get('decay', 0.9999)

        # torch.compile option
        self.use_compile: bool = cfg.training.get('use_compile', True)
        self.compile_mode: str = "default"  # Subclasses can override

        # Precision config
        precision_cfg = cfg.training.get('precision', {})
        self.pure_weights: bool = precision_cfg.get('pure_weights', False)
        dtype_str = precision_cfg.get('dtype', 'bf16')
        self.weight_dtype: torch.dtype = {
            'bf16': torch.bfloat16,
            'fp16': torch.float16,
            'fp32': torch.float32,
        }.get(dtype_str, torch.bfloat16)

        # Constant LR option (for progressive training)
        progressive_cfg = cfg.get('progressive', {})
        self.use_constant_lr: bool = progressive_cfg.get('use_constant_lr', False)

        # Optimizer betas (important for GAN training stability)
        # Default: (0.9, 0.999), Phase 3 DC-AE: (0.5, 0.9) per paper
        optimizer_cfg = cfg.training.get('optimizer', {})
        betas_list = optimizer_cfg.get('betas', [0.9, 0.999])
        self.optimizer_betas: tuple[float, float] = tuple(betas_list)

        # ─────────────────────────────────────────────────────────────────────
        # Initialize compression-specific trackers
        # ─────────────────────────────────────────────────────────────────────
        self._grad_norm_tracker_d = GradientNormTracker()

        # ─────────────────────────────────────────────────────────────────────
        # Model placeholders (set in setup_model)
        # ─────────────────────────────────────────────────────────────────────
        self.discriminator: nn.Module | None = None
        self.discriminator_raw: nn.Module | None = None
        self.optimizer_d: AdamW | None = None
        self.lr_scheduler_d: LRScheduler | None = None
        self.perceptual_loss_fn: nn.Module | None = None  # PerceptualLoss or LPIPSLoss
        self.adv_loss_fn: PatchAdversarialLoss | None = None
        self.ema: EMA | None = None

        # Per-modality validation loaders (set in train())
        self.per_modality_val_loaders: dict[str, DataLoader] | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Config extraction methods (can be overridden by subclasses)
    # ─────────────────────────────────────────────────────────────────────────

    # List of config sections to check for trainer-specific settings
    # Subclasses should override _CONFIG_SECTION_2D and _CONFIG_SECTION_3D
    _CONFIG_SECTIONS = ('vae', 'vqvae', 'dcae', 'vae_3d', 'vqvae_3d')
    _CONFIG_SECTION_2D: str = 'vae'  # Override in subclass
    _CONFIG_SECTION_3D: str = 'vae_3d'  # Override in subclass

    # Default config values - subclasses can override these class attributes
    _DEFAULT_DISC_LR_2D: float = COMPRESSION_DEFAULTS.disc_lr_2d
    _DEFAULT_DISC_LR_3D: float = COMPRESSION_DEFAULTS.disc_lr_3d
    _DEFAULT_PERCEPTUAL_WEIGHT_2D: float = COMPRESSION_DEFAULTS.perceptual_weight_2d
    _DEFAULT_PERCEPTUAL_WEIGHT_3D: float = COMPRESSION_DEFAULTS.perceptual_weight_3d
    _DEFAULT_ADV_WEIGHT_2D: float = COMPRESSION_DEFAULTS.adv_weight_2d
    _DEFAULT_ADV_WEIGHT_3D: float = COMPRESSION_DEFAULTS.adv_weight_3d

    def _get_config_section(self) -> str:
        """Get the config section name for this trainer's spatial_dims."""
        return self._CONFIG_SECTION_3D if self.spatial_dims == 3 else self._CONFIG_SECTION_2D

    def _get_config_value(self, cfg: DictConfig, key: str, default: Any) -> Any:
        """Get a value from any trainer config section.

        Searches through vae, vqvae, dcae, vae_3d, vqvae_3d sections
        for the specified key.

        Args:
            cfg: Hydra configuration object.
            key: Config key to look for.
            default: Default value if key not found.

        Returns:
            Value from config or default.
        """
        for section in self._CONFIG_SECTIONS:
            if section in cfg:
                return cfg[section].get(key, default)
        return default

    def _get_config_value_dimensional(
        self, cfg: DictConfig, key: str, default_2d: Any, default_3d: Any
    ) -> Any:
        """Get config value with dimension-specific defaults.

        Searches the trainer's config section first (e.g., 'vae' or 'vae_3d'),
        then falls back to dimension-specific default.

        Args:
            cfg: Hydra configuration.
            key: Config key to look for.
            default_2d: Default for 2D (spatial_dims=2).
            default_3d: Default for 3D (spatial_dims=3).

        Returns:
            Config value or appropriate default.
        """
        section = self._get_config_section()
        default = default_3d if self.spatial_dims == 3 else default_2d
        if section in cfg:
            return cfg[section].get(key, default)
        return default

    def _get_disc_lr(self, cfg: DictConfig) -> float:
        """Get discriminator learning rate from config."""
        return self._get_config_value_dimensional(
            cfg, 'disc_lr', self._DEFAULT_DISC_LR_2D, self._DEFAULT_DISC_LR_3D
        )

    def _get_perceptual_weight(self, cfg: DictConfig) -> float:
        """Get perceptual loss weight from config."""
        return self._get_config_value_dimensional(
            cfg, 'perceptual_weight', self._DEFAULT_PERCEPTUAL_WEIGHT_2D, self._DEFAULT_PERCEPTUAL_WEIGHT_3D
        )

    def _get_adv_weight(self, cfg: DictConfig) -> float:
        """Get adversarial loss weight from config."""
        return self._get_config_value_dimensional(
            cfg, 'adv_weight', self._DEFAULT_ADV_WEIGHT_2D, self._DEFAULT_ADV_WEIGHT_3D
        )

    def _get_disable_gan(self, cfg: DictConfig) -> bool:
        """Get disable_gan flag from config.

        Subclasses may override this for special logic (e.g., DC-AE phase-based).
        """
        # Check progressive config first (for staged training)
        progressive_cfg = cfg.get('progressive', {})
        if progressive_cfg.get('disable_gan', False):
            return True
        return self._get_config_value_dimensional(cfg, 'disable_gan', False, False)

    def _get_disc_num_layers(self, cfg: DictConfig) -> int:
        """Get discriminator number of layers from config."""
        return self._get_config_value(cfg, 'disc_num_layers', 3)

    def _get_disc_num_channels(self, cfg: DictConfig) -> int:
        """Get discriminator number of channels from config."""
        return self._get_config_value(cfg, 'disc_num_channels', 64)

    def _get_perceptual_loss_type(self, cfg: DictConfig) -> str:
        """Get perceptual loss type from config.

        Options:
            - 'radimagenet': MONAI's RadImageNet ResNet50 (default)
            - 'lpips': LPIPS library with VGG backbone (DC-AE paper uses this)

        Returns:
            Loss type string.
        """
        return self._get_config_value(cfg, 'perceptual_loss_type', 'radimagenet')

    def _get_2_5d_perceptual(self, cfg: DictConfig) -> bool:
        """Get 2.5D perceptual loss flag (3D only).

        When enabled, perceptual loss is computed on sampled 2D slices
        rather than full 3D volumes.

        Returns:
            True if 2.5D perceptual loss is enabled.
        """
        for section in ['vae_3d', 'vqvae_3d', 'dcae_3d']:
            if section in cfg:
                return cfg[section].get('use_2_5d_perceptual', True)
        return True

    def _get_perceptual_slice_fraction(self, cfg: DictConfig) -> float:
        """Get fraction of slices to sample for 2.5D perceptual loss.

        Returns:
            Fraction of depth slices to sample (0.0-1.0).
        """
        for section in ['vae_3d', 'vqvae_3d', 'dcae_3d']:
            if section in cfg:
                return cfg[section].get('perceptual_slice_fraction', 0.25)
        return 0.25

    # ─────────────────────────────────────────────────────────────────────────
    # Dimension Properties and Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def spatial_dims(self) -> int:
        """Return spatial dimensions (2 for images, 3 for volumes)."""
        return self._spatial_dims

    def _get_spatial_shape(self) -> tuple[int, ...]:
        """Get spatial shape based on dimensionality.

        Returns:
            (H, W) for 2D or (D, H, W) for 3D.
        """
        if self.spatial_dims == 2:
            return (self.image_size, self.image_size)
        return (self.volume_depth, self.volume_height, self.volume_width)

    def _extract_center_slice(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract center slice from 3D tensor for 2D visualization.

        Args:
            tensor: Input tensor [B, C, D, H, W] for 3D or [B, C, H, W] for 2D.

        Returns:
            2D tensor [B, C, H, W].
        """
        if self.spatial_dims == 2:
            return tensor
        center_idx = tensor.shape[2] // 2
        return tensor[:, :, center_idx, :, :]

    def _create_lpips_fn(self):
        """Create LPIPS function appropriate for spatial dimensions.

        Returns:
            compute_lpips for 2D, compute_lpips_3d for 3D.
        """
        if self.spatial_dims == 2:
            return compute_lpips
        return compute_lpips_3d

    def _create_worst_batch_figure_fn(self):
        """Create worst batch figure function appropriate for spatial dimensions.

        Returns:
            create_worst_batch_figure for 2D, create_worst_batch_figure_3d for 3D.
        """
        if self.spatial_dims == 2:
            return create_worst_batch_figure
        from medgen.metrics import create_worst_batch_figure_3d
        return create_worst_batch_figure_3d

    # ─────────────────────────────────────────────────────────────────────────
    # Unified Metrics System
    # ─────────────────────────────────────────────────────────────────────────

    def _init_unified_metrics(self, trainer_type: str) -> None:
        """Initialize unified metrics system.

        Called by subclass __init__ after model-specific config is set.
        Creates UnifiedMetrics for TensorBoard logging.

        Args:
            trainer_type: One of 'vae', 'vqvae', 'dcae'.
        """
        spatial_dims = getattr(self, 'spatial_dims', 2)
        mode_name = getattr(self, 'mode_name', 'multi_modality')

        # Determine modality for TensorBoard suffix
        # seg_conditioned modes: no suffix (distinguish by TensorBoard run color)
        is_single_modality = mode_name not in ('multi_modality', 'dual', 'multi')
        is_seg_conditioned = mode_name.startswith('seg_conditioned')
        modality = None if is_seg_conditioned else (mode_name if is_single_modality else None)

        # Get image size and fov_mm for regional tracker
        image_size = getattr(self.cfg.model, 'image_size', 256)
        fov_mm = self.cfg.paths.get('fov_mm', 240.0)

        self._unified_metrics = UnifiedMetrics(
            writer=self.writer,
            mode=mode_name,
            spatial_dims=spatial_dims,
            modality=modality,
            device=self.device,
            enable_regional=self.log_regional_losses,
            enable_codebook=(trainer_type == 'vqvae'),
            image_size=image_size,
            fov_mm=fov_mm,
        )

        # Create loss accumulator for epoch-level loss tracking
        # This is used by subclass train_epoch() methods
        self._loss_accumulator = SimpleLossAccumulator()

    def _log_training_metrics_unified(self, epoch: int, avg_losses: dict[str, float]) -> None:
        """Log training metrics using unified system.

        Args:
            epoch: Current epoch number.
            avg_losses: Dictionary of averaged losses from training.
        """
        if not hasattr(self, '_unified_metrics') or self._unified_metrics is None:
            return

        seg_mode = getattr(self, 'seg_mode', False)
        if seg_mode:
            # Use dedicated seg training logging
            self._unified_metrics.log_seg_training(avg_losses, epoch)
        else:
            # Standard loss logging
            for key, value in avg_losses.items():
                self._unified_metrics.update_loss(key, value, phase='train')
            self._unified_metrics.log_training(epoch)
            self._unified_metrics.reset_training()

    def _log_validation_metrics_unified(
        self,
        epoch: int,
        metrics: dict[str, float],
        prefix: str = '',
    ) -> None:
        """Log validation metrics using unified system.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of validation metrics.
            prefix: Optional prefix for per-modality logging.
        """
        if hasattr(self, '_unified_metrics') and self._unified_metrics is not None:
            # Update metric accumulators
            if 'psnr' in metrics:
                self._unified_metrics._val_psnr_sum = metrics['psnr']
                self._unified_metrics._val_psnr_count = 1
            if 'msssim' in metrics:
                self._unified_metrics._val_msssim_sum = metrics['msssim']
                self._unified_metrics._val_msssim_count = 1
            if 'lpips' in metrics:
                self._unified_metrics._val_lpips_sum = metrics['lpips']
                self._unified_metrics._val_lpips_count = 1
            if 'msssim_3d' in metrics:
                self._unified_metrics._val_msssim_3d_sum = metrics['msssim_3d']
                self._unified_metrics._val_msssim_3d_count = 1
            if 'dice_score' in metrics:
                self._unified_metrics._val_dice_sum = metrics['dice_score']
                self._unified_metrics._val_dice_count = 1
            if 'iou' in metrics:
                self._unified_metrics._val_iou_sum = metrics['iou']
                self._unified_metrics._val_iou_count = 1

            # Log validation losses
            if 'val_loss' in metrics:
                self._unified_metrics.update_loss('Total', metrics['val_loss'], phase='val')

            self._unified_metrics.log_validation(epoch)
            self._unified_metrics.reset_validation()

    def _log_epoch_summary_unified(
        self,
        epoch: int,
        avg_losses: dict[str, float],
        val_metrics: dict[str, float] | None,
        elapsed_time: float,
    ) -> None:
        """Log epoch summary using unified system.

        Args:
            epoch: Current epoch number.
            avg_losses: Dictionary of averaged training losses.
            val_metrics: Dictionary of validation metrics (may be None).
            elapsed_time: Time taken for epoch in seconds.
        """
        if hasattr(self, '_unified_metrics') and self._unified_metrics is not None:
            self._unified_metrics.log_console_summary(epoch, self.n_epochs, elapsed_time)

    # ─────────────────────────────────────────────────────────────────────────
    # Model setup helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _create_discriminator(
        self,
        n_channels: int,
        spatial_dims: int = 2,
    ) -> nn.Module:
        """Create PatchDiscriminator.

        Args:
            n_channels: Number of input channels.
            spatial_dims: Spatial dimensions (2 for 2D, 3 for 3D).

        Returns:
            PatchDiscriminator model on device.
        """
        return PatchDiscriminator(
            spatial_dims=spatial_dims,
            in_channels=n_channels,
            channels=self.disc_num_channels,
            num_layers_d=self.disc_num_layers,
        ).to(self.device)

    def _create_perceptual_loss(self, spatial_dims: int = 2) -> nn.Module:
        """Create perceptual loss function based on config.

        Uses self.perceptual_loss_type to choose:
        - 'radimagenet': MONAI RadImageNet ResNet50 (medical imaging features)
        - 'lpips': LPIPS library with VGG backbone (DC-AE paper)

        Args:
            spatial_dims: Spatial dimensions (2 for 2D, 3 for 3D).
                         Note: LPIPS only supports 2D.

        Returns:
            PerceptualLoss or LPIPSLoss instance.
        """
        loss_type = self.perceptual_loss_type.lower()

        if loss_type == 'lpips':
            if spatial_dims != 2:
                logger.warning(
                    f"LPIPS only supports 2D images. Got spatial_dims={spatial_dims}. "
                    "Falling back to RadImageNet perceptual loss."
                )
                loss_type = 'radimagenet'
            else:
                logger.info("Using LPIPS loss with VGG backbone (DC-AE paper setting)")
                return LPIPSLoss(
                    net='vgg',
                    device=self.device,
                    use_compile=self.use_compile,
                )

        # Default: RadImageNet perceptual loss
        logger.info("Using RadImageNet ResNet50 perceptual loss")
        cache_dir = getattr(self.cfg.paths, 'cache_dir', None)
        return PerceptualLoss(
            spatial_dims=spatial_dims,
            network_type="radimagenet_resnet50",
            cache_dir=cache_dir,
            pretrained=True,
            device=self.device,
            use_compile=self.use_compile,
        )

    def _create_adversarial_loss(self) -> None:
        """Create adversarial loss function if GAN is enabled.

        Sets self.adv_loss_fn to PatchAdversarialLoss with least_squares criterion.
        Does nothing if self.disable_gan is True.
        """
        if self.disable_gan:
            return

        from monai.losses import PatchAdversarialLoss
        self.adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

    def _setup_optimizers(self, n_channels: int) -> None:
        """Setup optimizers and schedulers for generator and discriminator.

        Args:
            n_channels: Number of input channels (for discriminator).
        """
        # Generator optimizer with configurable betas
        # Default: (0.9, 0.999), Phase 3 DC-AE: (0.5, 0.9) per paper
        self.optimizer = AdamW(
            self.model_raw.parameters(),
            lr=self.learning_rate,
            betas=self.optimizer_betas,
        )

        # Discriminator optimizer (only if GAN enabled)
        # Uses same betas as generator for consistent GAN training dynamics
        if not self.disable_gan and self.discriminator_raw is not None:
            self.optimizer_d = AdamW(
                self.discriminator_raw.parameters(),
                lr=self.disc_lr,
                betas=self.optimizer_betas,
            )

        if self.is_main_process:
            logger.info(f"Optimizer betas: {self.optimizer_betas}")

        # LR schedulers (only if not using constant LR)
        if not self.use_constant_lr:
            self.lr_scheduler = create_warmup_cosine_scheduler(
                self.optimizer,
                warmup_epochs=self.warmup_epochs,
                total_epochs=self.n_epochs,
            )
            if not self.disable_gan and self.optimizer_d is not None:
                self.lr_scheduler_d = create_warmup_cosine_scheduler(
                    self.optimizer_d,
                    warmup_epochs=self.warmup_epochs,
                    total_epochs=self.n_epochs,
                )
        else:
            if self.is_main_process:
                logger.info(f"Using constant LR: {self.learning_rate} (scheduler disabled)")

    def _setup_ema(self) -> None:
        """Setup EMA for generator if enabled."""
        if self.use_ema and self.model_raw is not None:
            self.ema = EMA(
                self.model_raw,
                beta=self.ema_decay,
                update_after_step=self.cfg.training.ema.get('update_after_step', 100),
                update_every=self.cfg.training.ema.get('update_every', 10),
            )
            if self.is_main_process:
                logger.info(f"EMA enabled with decay={self.ema_decay}")

    def _wrap_models(
        self,
        raw_model: nn.Module,
        raw_disc: nn.Module | None = None,
    ) -> None:
        """Wrap models with DDP and/or torch.compile.

        Args:
            raw_model: Raw generator model.
            raw_disc: Raw discriminator model (optional).
        """
        # Convert weights to target dtype if pure_weights is enabled
        if self.pure_weights and self.weight_dtype != torch.float32:
            raw_model = raw_model.to(self.weight_dtype)
            if raw_disc is not None:
                raw_disc = raw_disc.to(self.weight_dtype)
            if self.is_main_process:
                logger.info(f"Converted model weights to {self.weight_dtype}")

        # Wrap generator
        self.model, self.model_raw = wrap_model_for_training(
            raw_model,
            use_multi_gpu=self.use_multi_gpu,
            local_rank=self.local_rank if self.use_multi_gpu else 0,
            use_compile=self.use_compile,
            compile_mode=self.compile_mode,
            is_main_process=self.is_main_process,
        )

        # Wrap discriminator if provided
        if raw_disc is not None:
            self.discriminator, self.discriminator_raw = wrap_model_for_training(
                raw_disc,
                use_multi_gpu=self.use_multi_gpu,
                local_rank=self.local_rank if self.use_multi_gpu else 0,
                use_compile=self.use_compile,
                compile_mode=self.compile_mode,
                is_main_process=False,  # Suppress duplicate logging
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Training helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _prepare_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare batch for compression training (2D or 3D)."""
        from .compression_training import prepare_batch
        return prepare_batch(self, batch)

    def _train_discriminator_step(
        self,
        images: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        """Train discriminator on real vs fake images."""
        from .compression_training import train_discriminator_step
        return train_discriminator_step(self, images, reconstruction)

    def _compute_adversarial_loss(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Compute adversarial loss for generator."""
        from .compression_training import compute_adversarial_loss
        return compute_adversarial_loss(self, reconstruction)

    def _compute_perceptual_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss (2D or 2.5D for 3D)."""
        from .compression_training import compute_perceptual_loss
        return compute_perceptual_loss(self, reconstruction, target)

    def _compute_2_5d_perceptual_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss on sampled 2D slices from 3D volumes."""
        from .compression_training import compute_2_5d_perceptual_loss
        return compute_2_5d_perceptual_loss(self, reconstruction, target)

    def _compute_kl_loss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss for VAE training."""
        from .compression_training import compute_kl_loss
        return compute_kl_loss(mean, logvar)

    def _update_ema(self) -> None:
        """Update EMA if enabled."""
        if self.ema is not None:
            self.ema.update()

    def _get_model_for_eval(self) -> nn.Module:
        """Get model for evaluation (EMA or regular).

        Returns:
            Model to use for evaluation.
        """
        if self.ema is not None:
            return self.ema.ema_model
        return self.model_raw

    # ─────────────────────────────────────────────────────────────────────────
    # Train step template (shared across VAE, VQ-VAE, DC-AE)
    # ─────────────────────────────────────────────────────────────────────────

    def train_step(self, batch: Any) -> TrainingStepResult:
        """Template train step for compression trainers."""
        from .compression_training import compression_train_step
        return compression_train_step(self, batch)

    def _forward_for_training(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Model-specific forward pass for training.

        Subclasses MUST override this method to implement their forward pass logic.

        Returns (reconstruction, regularization_loss):
        - VAE: compute kl_loss from mean/logvar, return (recon, kl_weight * kl_loss)
        - VQ-VAE: model returns vq_loss, return (recon, vq_loss)
        - DC-AE: no regularization, return (recon, 0.0)

        Args:
            images: Input images.

        Returns:
            Tuple of (reconstruction, regularization_loss).
        """
        raise NotImplementedError("Subclasses must implement _forward_for_training()")

    def _get_reconstruction_loss_weight(self) -> float:
        """Get weight for L1 reconstruction loss.

        Override in subclass if L1 loss should be weighted (e.g., DC-AE).

        Returns:
            Weight for L1 loss (default: 1.0).
        """
        return 1.0

    def _use_discriminator_before_generator(self) -> bool:
        """Whether to train discriminator before generator step.

        VAE/VQ-VAE: True for 2D (discriminator sees detached reconstruction)
        DC-AE: Always False (discriminator after generator)

        Override in subclass to change timing.

        Returns:
            True if D step should come before G step.
        """
        return self.spatial_dims == 2

    def _track_seg_breakdown(self, seg_breakdown: dict[str, torch.Tensor]) -> None:
        """Track segmentation loss breakdown for epoch averaging.

        Override in subclass to accumulate breakdown for seg_mode.

        Args:
            seg_breakdown: Dictionary with 'bce', 'dice', 'boundary' loss tensors.
        """
        if hasattr(self, '_epoch_seg_breakdown'):
            for key in seg_breakdown:
                # Call .item() here to convert tensor to float for accumulation
                val = seg_breakdown[key]
                self._epoch_seg_breakdown[key] += val.item() if isinstance(val, torch.Tensor) else val

    # ─────────────────────────────────────────────────────────────────────────
    # Training epoch template
    # ─────────────────────────────────────────────────────────────────────────

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> dict[str, float]:
        """Train for one epoch using template method pattern."""
        from .compression_training import compression_train_epoch
        return compression_train_epoch(self, data_loader, epoch)

    def _get_loss_key(self) -> str | None:
        """Return loss dictionary key for regularization term.

        Override in subclass:
        - VAE: return 'kl'
        - VQ-VAE: return 'vq'
        - DC-AE: return None (no regularization)
        """
        return None

    def _get_postfix_metrics(
        self, avg_so_far: dict[str, float], current_losses: dict[str, float]
    ) -> dict[str, str]:
        """Return metrics dict for progress bar postfix.

        Override in subclass for custom progress bar display.

        Args:
            avg_so_far: Running average of losses for epoch.
            current_losses: Current batch losses (for breakdown display).

        Returns:
            Dict mapping metric name to formatted string value.
        """
        if not self.disable_gan:
            return {
                'G': f"{avg_so_far.get('gen', 0):.4f}",
                'D': f"{avg_so_far.get('disc', 0):.4f}",
                'L1': f"{avg_so_far.get('recon', 0):.4f}",
            }
        return {
            'G': f"{avg_so_far.get('gen', 0):.4f}",
            'L1': f"{avg_so_far.get('recon', 0):.4f}",
        }

    def _on_train_epoch_start(self, epoch: int) -> None:
        """Hook called before epoch training starts.

        Override in subclass for initialization (e.g., seg breakdown tracking).
        """
        pass

    def _on_train_epoch_end(self, epoch: int, avg_losses: dict[str, float]) -> None:
        """Hook called after epoch training ends.

        Override in subclass for post-processing (e.g., seg breakdown averaging).
        """
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Gradient norm tracking
    # ─────────────────────────────────────────────────────────────────────────

    def _on_epoch_start(self, epoch: int) -> None:
        """Reset gradient trackers at start of epoch."""
        super()._on_epoch_start(epoch)
        self._grad_norm_tracker.reset()
        self._grad_norm_tracker_d.reset()

    def _log_grad_norms(self, epoch: int) -> None:
        """Log gradient norm statistics to TensorBoard."""
        if not self.log_grad_norm or self.writer is None:
            return
        # Use _g suffix only when discriminator exists for clarity
        gen_prefix = 'training/grad_norm_g' if not self.disable_gan else 'training/grad_norm'
        self._unified_metrics.log_grad_norm_from_tracker(self._grad_norm_tracker, epoch, prefix=gen_prefix)
        if not self.disable_gan and self.discriminator is not None:
            self._unified_metrics.log_grad_norm_from_tracker(self._grad_norm_tracker_d, epoch, prefix='training/grad_norm_d')

    # ─────────────────────────────────────────────────────────────────────────
    # Epoch hooks
    # ─────────────────────────────────────────────────────────────────────────

    def _on_epoch_end(
        self,
        epoch: int,
        avg_losses: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Handle end of epoch: step schedulers, log metrics."""
        # Step schedulers (only if not using constant LR)
        if not self.use_constant_lr:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if not self.disable_gan and self.lr_scheduler_d is not None:
                self.lr_scheduler_d.step()

        # Log learning rates
        self._log_learning_rate(epoch, self.lr_scheduler, "LR/Generator")
        if not self.disable_gan and self.lr_scheduler_d is not None:
            self._log_learning_rate(epoch, self.lr_scheduler_d, "LR/Discriminator")

        # Log gradient norms
        self._log_grad_norms(epoch)

        # Log VRAM and FLOPs
        self._log_vram(epoch)
        self._log_flops(epoch)

        # Per-modality validation (multi_modality mode, skip for seg_mode)
        if self.per_modality_val_loaders and not getattr(self, 'seg_mode', False):
            self._compute_per_modality_validation(epoch)

        # Per-channel validation (dual mode, skip for seg_mode)
        if not getattr(self, 'seg_mode', False):
            self._compute_per_channel_validation(epoch)

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _create_regional_tracker(self):
        """Create regional metrics tracker (2D or 3D)."""
        from .compression_validation import create_regional_tracker
        return create_regional_tracker(self)

    def _create_worst_batch_figure(
        self,
        worst_batch_data: dict[str, Any],
    ) -> plt.Figure:
        """Create worst batch figure for TensorBoard (2D or 3D)."""
        from .compression_validation import create_worst_batch_figure
        return create_worst_batch_figure(self, worst_batch_data)

    def _create_validation_runner(self) -> 'ValidationRunner':
        """Create ValidationRunner for this trainer (2D or 3D)."""
        from .compression_validation import create_validation_runner
        return create_validation_runner(self)

    def compute_validation_losses(
        self,
        epoch: int,
        log_figures: bool = True,
    ) -> dict[str, float]:
        """Compute validation losses using ValidationRunner (2D or 3D)."""
        from .compression_validation import compute_validation_losses
        return compute_validation_losses(self, epoch, log_figures)

    def _capture_worst_batch(
        self,
        images: torch.Tensor,
        reconstruction: torch.Tensor,
        loss: float,
        l1_loss: torch.Tensor,
        p_loss: torch.Tensor,
        reg_loss: torch.Tensor,
    ) -> dict[str, Any]:
        """Capture worst batch data for visualization (2D or 3D)."""
        from .compression_validation import capture_worst_batch
        return capture_worst_batch(self, images, reconstruction, loss, l1_loss, p_loss, reg_loss)

    def _log_validation_metrics_core(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Log validation metrics with modality suffix handling."""
        from .compression_validation import log_validation_metrics_core
        log_validation_metrics_core(self, epoch, metrics)

    def _log_validation_metrics(
        self,
        epoch: int,
        metrics: dict[str, float],
        worst_batch_data: dict[str, Any] | None,
        regional_tracker: RegionalMetricsTracker | None,
        log_figures: bool,
    ) -> None:
        """Log validation metrics to TensorBoard using unified system."""
        from .compression_validation import log_validation_metrics
        log_validation_metrics(self, epoch, metrics, worst_batch_data, regional_tracker, log_figures)

    def _compute_per_modality_validation(self, epoch: int) -> None:
        """Compute per-modality validation metrics (2D or 3D)."""
        from .compression_validation import compute_per_modality_validation
        compute_per_modality_validation(self, epoch)

    def _compute_per_channel_validation(self, epoch: int) -> None:
        """Compute per-channel validation metrics for dual mode."""
        from .compression_validation import compute_per_channel_validation
        compute_per_channel_validation(self, epoch)

    def _compute_volume_3d_msssim(
        self,
        epoch: int,
        data_split: str = 'val',
        modality_override: str | None = None,
    ) -> float | None:
        """Compute 3D MS-SSIM by reconstructing full volumes slice-by-slice."""
        from .compression_validation import compute_volume_3d_msssim
        return compute_volume_3d_msssim(self, epoch, data_split, modality_override)

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_checkpoint_manager(self) -> None:
        """Setup checkpoint manager with GAN components."""
        from .compression_checkpointing import setup_checkpoint_manager
        setup_checkpoint_manager(self)

    def _get_checkpoint_extra_state(self) -> dict[str, Any] | None:
        """Return extra state for compression trainer checkpoints."""
        from .compression_checkpointing import get_checkpoint_extra_state
        return get_checkpoint_extra_state(self)

    def _save_checkpoint(self, epoch: int, name: str) -> None:
        """Save checkpoint with standardized format."""
        from .compression_checkpointing import save_checkpoint
        save_checkpoint(self, epoch, name)

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> int:
        """Load checkpoint to resume training."""
        from .compression_checkpointing import load_checkpoint
        return load_checkpoint(self, checkpoint_path, load_optimizer)

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: dict[str, float],
        val_metrics: dict[str, float],
        elapsed_time: float,
    ) -> None:
        """Log epoch completion summary."""
        from .compression_checkpointing import log_epoch_summary
        log_epoch_summary(self, epoch, total_epochs, avg_losses, val_metrics, elapsed_time)

    # ─────────────────────────────────────────────────────────────────────────
    # Pretrained weights loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_pretrained_weights(
        self,
        raw_model: nn.Module,
        raw_disc: nn.Module | None,
        checkpoint_path: str,
        model_name: str = "model",
    ) -> None:
        """Load pretrained weights from checkpoint."""
        from .compression_checkpointing import load_pretrained_weights
        load_pretrained_weights(self, raw_model, raw_disc, checkpoint_path, model_name)

    def _load_pretrained_weights_base(
        self,
        base_model: nn.Module,
        checkpoint_path: str,
        model_name: str = "model",
    ) -> None:
        """Load pretrained weights into base model (before checkpointing wrapper)."""
        from .compression_checkpointing import load_pretrained_weights_base
        load_pretrained_weights_base(self, base_model, checkpoint_path, model_name)

    # ─────────────────────────────────────────────────────────────────────────
    # Abstract methods
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def _forward_for_validation(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass for validation.

        Returns (reconstruction, regularization_loss):
        - VAE: (recon, kl_loss)
        - VQ-VAE: (recon, vq_loss)
        - DC-AE: (recon, 0.0)

        Args:
            model: Model to use for inference.
            images: Input images.

        Returns:
            Tuple of (reconstruction, regularization_loss).
        """
        ...

    @abstractmethod
    def _get_model_config(self) -> dict[str, Any]:
        """Get model configuration for checkpoint.

        Returns:
            Dictionary of model configuration.
        """
        ...

    @abstractmethod
    def _test_forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Perform forward pass for test evaluation.

        Args:
            model: Model to use for inference.
            images: Input images.

        Returns:
            Reconstructed images tensor.
        """
        ...

    # ─────────────────────────────────────────────────────────────────────────
    # Test evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def _create_test_evaluator(self):
        """Create test evaluator for this trainer (2D or 3D)."""
        from .compression_checkpointing import create_test_evaluator
        return create_test_evaluator(self)

    def evaluate_test_set(
        self,
        test_loader: DataLoader,
        checkpoint_name: str | None = None,
    ) -> dict[str, float]:
        """Evaluate compression model on test set."""
        from .compression_checkpointing import evaluate_test_set
        return evaluate_test_set(self, test_loader, checkpoint_name)
