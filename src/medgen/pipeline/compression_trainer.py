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
        self.ema_decay: float = cfg.training.ema.get('decay', 0.999)

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
        """Prepare batch for compression training (2D or 3D).

        Handles multiple batch formats:
        - Tuple of (images, mask)
        - Dict with image keys (2D) or 'image'/'images' key (3D)
        - Single tensor

        Args:
            batch: Input batch.

        Returns:
            Tuple of (images, mask).
        """
        # 3D-specific handling
        if self.spatial_dims == 3:
            if isinstance(batch, dict):
                images = get_with_fallbacks(batch, 'image', 'images')
                mask = get_with_fallbacks(batch, 'seg', 'mask')
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
                mask = batch[1] if len(batch) > 1 else None
            else:
                images = batch
                mask = None

            images = _tensor_to_device(images, self.device)
            mask = _tensor_to_device(mask, self.device) if mask is not None else None
            return images, mask

        # 2D handling
        # Handle tuple of (image, seg)
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            images, mask = batch
            return _tensor_to_device(images, self.device), _tensor_to_device(mask, self.device)

        # Handle dict batches
        if isinstance(batch, dict):
            image_keys = self.cfg.mode.get('image_keys', ['t1_pre', 't1_gd'])
            tensors = [_tensor_to_device(batch[k], self.device) for k in image_keys if k in batch]
            images = torch.cat(tensors, dim=1)
            mask = _tensor_to_device(batch['seg'], self.device) if 'seg' in batch else None
            return images, mask

        # Handle tensor input
        tensor = _tensor_to_device(batch, self.device)

        # Check if seg is stacked as last channel
        n_image_channels = self.cfg.mode.get('in_channels', 2)
        if tensor.shape[1] > n_image_channels:
            images = tensor[:, :n_image_channels, :, :]
            mask = tensor[:, n_image_channels:n_image_channels + 1, :, :]
            return images, mask

        return tensor, None

    def _train_discriminator_step(
        self,
        images: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        """Train discriminator on real vs fake images.

        Args:
            images: Real images [B, C, H, W].
            reconstruction: Generated images [B, C, H, W].

        Returns:
            Discriminator loss.
        """
        if self.disable_gan or self.discriminator is None:
            return torch.tensor(0.0, device=self.device)

        self.optimizer_d.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            # Real images -> discriminator should output 1
            logits_real = self.discriminator(images.contiguous())
            # Fake images -> discriminator should output 0
            # Detach to prevent gradient flow through generator (saves memory)
            logits_fake = self.discriminator(reconstruction.detach().contiguous())

            d_loss = 0.5 * (
                self.adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                + self.adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
            )

        d_loss.backward()

        # Gradient clipping and tracking
        grad_clip = self.cfg.training.get('gradient_clip_norm', 1.0)
        grad_norm_d = 0.0
        if grad_clip > 0:
            grad_norm_d = torch.nn.utils.clip_grad_norm_(
                self.discriminator_raw.parameters(), max_norm=grad_clip
            ).item()

        self.optimizer_d.step()

        # Track discriminator gradient norm
        if self.log_grad_norm:
            self._grad_norm_tracker_d.update(grad_norm_d)

        return d_loss

    def _compute_adversarial_loss(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Compute adversarial loss for generator.

        Args:
            reconstruction: Generated images [B, C, H, W].

        Returns:
            Adversarial loss.
        """
        if self.disable_gan or self.discriminator is None:
            return torch.tensor(0.0, device=self.device)

        logits_fake = self.discriminator(reconstruction.contiguous())
        return self.adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)

    def _compute_perceptual_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss (2D or 2.5D for 3D).

        For 3D volumes with use_2_5d_perceptual=True, computes loss on
        sampled 2D slices. Otherwise computes standard perceptual loss.

        Args:
            reconstruction: Generated images/volumes.
            target: Target images/volumes.

        Returns:
            Perceptual loss value.
        """
        if self.perceptual_loss_fn is None:
            return torch.tensor(0.0, device=self.device)

        # 3D with 2.5D perceptual loss
        if self.spatial_dims == 3 and getattr(self, 'use_2_5d_perceptual', False):
            return self._compute_2_5d_perceptual_loss(reconstruction, target)

        return self.perceptual_loss_fn(reconstruction, target)

    def _compute_2_5d_perceptual_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss on sampled 2D slices from 3D volumes.

        Args:
            reconstruction: Reconstructed volume [B, C, D, H, W].
            target: Target volume [B, C, D, H, W].

        Returns:
            Perceptual loss averaged over sampled slices.
        """
        if self.perceptual_loss_fn is None:
            return torch.tensor(0.0, device=self.device)

        depth = reconstruction.shape[2]
        slice_fraction = getattr(self, 'perceptual_slice_fraction', 0.25)
        n_slices = max(1, int(depth * slice_fraction))

        # Sample slice indices
        indices = torch.randperm(depth)[:n_slices].to(self.device)

        total_loss = 0.0
        for idx in indices:
            recon_slice = reconstruction[:, :, idx, :, :]
            target_slice = target[:, :, idx, :, :]
            total_loss += self.perceptual_loss_fn(recon_slice, target_slice)

        return total_loss / n_slices

    def _compute_kl_loss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss for VAE training.

        Works for both 2D [B, C, H, W] and 3D [B, C, D, H, W] tensors by
        summing over all spatial dimensions, then averaging over batch.

        Args:
            mean: Mean of latent distribution.
            logvar: Log variance of latent distribution.

        Returns:
            KL divergence loss (scalar).
        """
        # Sum over all spatial dimensions (everything except batch dim 0)
        spatial_dims = list(range(1, mean.dim()))
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=spatial_dims)
        return kl.mean()

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
        """Template train step for compression trainers.

        Implements the common training step pattern shared by VAE, VQ-VAE, and DC-AE.
        Subclasses customize via hook methods:
        - _forward_for_training(): Model-specific forward pass returning (reconstruction, reg_loss)
        - _get_reconstruction_loss_weight(): Return L1 weight (1.0 for VAE/VQ-VAE, configurable for DC-AE)
        - _use_discriminator_before_generator(): Whether to run D step before G (VAE/VQ-VAE 2D: True)
        - _track_seg_breakdown(): Track seg loss breakdown for epoch averaging

        Args:
            batch: Input batch.

        Returns:
            TrainingStepResult with all loss components.
        """
        images, mask = self._prepare_batch(batch)
        grad_clip = self.cfg.training.get('gradient_clip_norm', 1.0)

        d_loss = torch.tensor(0.0, device=self.device)
        adv_loss = torch.tensor(0.0, device=self.device)

        # ==================== Discriminator Step (before generator, if applicable) ====================
        if self._use_discriminator_before_generator() and not self.disable_gan:
            with torch.no_grad():
                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    reconstruction_for_d, _ = self._forward_for_training(images)
            d_loss = self._train_discriminator_step(images, reconstruction_for_d)

        # ==================== Generator Step ====================
        self.optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            # Model-specific forward pass
            reconstruction, reg_loss = self._forward_for_training(images)

            # Compute reconstruction loss (L1 or seg-specific)
            seg_mode = getattr(self, 'seg_mode', False)
            seg_loss_fn = getattr(self, 'seg_loss_fn', None)

            if seg_mode and seg_loss_fn is not None:
                seg_loss, seg_breakdown = seg_loss_fn(reconstruction, images)
                l1_loss = seg_loss
                p_loss = torch.tensor(0.0, device=self.device)
                self._track_seg_breakdown(seg_breakdown)
            else:
                l1_loss = torch.nn.functional.l1_loss(reconstruction.float(), images.float())
                p_loss = self._compute_perceptual_loss(reconstruction.float(), images.float())

            # Adversarial loss
            if not self.disable_gan:
                adv_loss = self._compute_adversarial_loss(reconstruction)

            # Total generator loss with configurable weights
            l1_weight = self._get_reconstruction_loss_weight()
            g_loss = (
                l1_weight * l1_loss
                + self.perceptual_weight * p_loss
                + reg_loss
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

        # ==================== Discriminator Step (after generator, if applicable) ====================
        if not self._use_discriminator_before_generator() and not self.disable_gan:
            d_loss = self._train_discriminator_step(images, reconstruction.detach())

        return TrainingStepResult(
            total_loss=g_loss.item(),
            reconstruction_loss=l1_loss.item(),
            perceptual_loss=p_loss.item() if isinstance(p_loss, torch.Tensor) else p_loss,
            regularization_loss=reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            adversarial_loss=adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
            discriminator_loss=d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss,
        )

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

    def _track_seg_breakdown(self, seg_breakdown: dict[str, float]) -> None:
        """Track segmentation loss breakdown for epoch averaging.

        Override in subclass to accumulate breakdown for seg_mode.

        Args:
            seg_breakdown: Dictionary with 'bce', 'dice', 'boundary' losses.
        """
        if hasattr(self, '_epoch_seg_breakdown'):
            for key in seg_breakdown:
                self._epoch_seg_breakdown[key] += seg_breakdown[key]

    # ─────────────────────────────────────────────────────────────────────────
    # Training epoch template
    # ─────────────────────────────────────────────────────────────────────────

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> dict[str, float]:
        """Train for one epoch using template method pattern.

        Subclasses customize via hook methods:
        - _get_loss_key(): Return loss dictionary key for regularization ('kl', 'vq', None)
        - _get_postfix_metrics(): Return metrics dict for progress bar
        - _on_train_epoch_start(): Optional setup (e.g., seg breakdown tracking)
        - _on_train_epoch_end(): Optional teardown (e.g., seg breakdown averaging)

        Args:
            data_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dict with average losses.
        """
        import itertools

        from tqdm import tqdm

        from .utils import create_epoch_iterator, get_vram_usage

        self.model.train()
        if not self.disable_gan and self.discriminator is not None:
            self.discriminator.train()

        self._loss_accumulator.reset()
        self._on_train_epoch_start(epoch)

        # Create epoch iterator (handles 2D vs 3D differences)
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
            losses = result.to_legacy_dict(self._get_loss_key())

            # Step profiler to mark training step boundary
            self._profiler_step()

            # Accumulate with unified system
            self._loss_accumulator.update(losses)

            if hasattr(epoch_iter, 'set_postfix'):
                avg_so_far = self._loss_accumulator.compute()
                epoch_iter.set_postfix(self._get_postfix_metrics(avg_so_far, losses))

            if epoch == 1 and step == 0 and self.is_main_process:
                logger.info(get_vram_usage(self.device))

        # Compute average losses using unified system
        avg_losses = self._loss_accumulator.compute()

        # Track batch count for seg breakdown averaging
        self._last_epoch_batch_count = self._loss_accumulator._count

        # Call subclass hook for post-epoch processing
        self._on_train_epoch_end(epoch, avg_losses)

        # Log training metrics using unified system
        self._log_training_metrics_unified(epoch, avg_losses)

        return avg_losses

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
        """Create regional metrics tracker (2D or 3D).

        Returns:
            RegionalMetricsTracker (2D) or RegionalMetricsTracker3D (3D).
        """
        if self.spatial_dims == 3:
            from medgen.metrics import RegionalMetricsTracker3D
            return RegionalMetricsTracker3D(
                volume_size=(self.volume_height, self.volume_width, self.volume_depth),
                fov_mm=self.cfg.paths.get('fov_mm', 240.0),
                loss_fn='l1',
                device=self.device,
            )
        return RegionalMetricsTracker(
            image_size=self.cfg.model.image_size,
            fov_mm=self.cfg.paths.get('fov_mm', 240.0),
            loss_fn='l1',
            device=self.device,
        )

    def _create_worst_batch_figure(
        self,
        worst_batch_data: dict[str, Any],
    ) -> plt.Figure:
        """Create worst batch figure for TensorBoard (2D or 3D).

        Args:
            worst_batch_data: Dict with 'original', 'generated', 'loss', 'loss_breakdown'.

        Returns:
            Matplotlib figure.
        """
        figure_fn = self._create_worst_batch_figure_fn()
        return figure_fn(
            original=worst_batch_data['original'],
            generated=worst_batch_data['generated'],
            loss=worst_batch_data['loss'],
            loss_breakdown=worst_batch_data.get('loss_breakdown'),
        )

    def _create_validation_runner(self) -> 'ValidationRunner':
        """Create ValidationRunner for this trainer (2D or 3D).

        Factory method that creates a ValidationRunner with trainer-specific
        configuration and callbacks.

        Returns:
            Configured ValidationRunner instance.
        """
        from medgen.evaluation import ValidationConfig, ValidationRunner

        config = ValidationConfig(
            log_msssim=self.log_msssim,
            log_psnr=self.log_psnr,
            log_lpips=self.log_lpips,
            log_regional_losses=self.log_regional_losses,
            weight_dtype=self.weight_dtype,
            use_compile=self.use_compile,
            spatial_dims=self.spatial_dims,
        )

        regional_factory = None
        if self.log_regional_losses:
            regional_factory = self._create_regional_tracker

        return ValidationRunner(
            config=config,
            device=self.device,
            forward_fn=self._forward_for_validation,
            perceptual_loss_fn=self._compute_perceptual_loss,
            regional_tracker_factory=regional_factory,
            prepare_batch_fn=self._prepare_batch,
        )

    def compute_validation_losses(
        self,
        epoch: int,
        log_figures: bool = True,
    ) -> dict[str, float]:
        """Compute validation losses using ValidationRunner (2D or 3D).

        Args:
            epoch: Current epoch number.
            log_figures: Whether to log figures (worst_batch).

        Returns:
            Dictionary of validation metrics.
        """
        if self.val_loader is None:
            return {}

        # Get model for evaluation
        model_to_use = self._get_model_for_eval()
        model_to_use.eval()

        # Run validation using extracted runner
        runner = self._create_validation_runner()
        result = runner.run(
            val_loader=self.val_loader,
            model=model_to_use,
            perceptual_weight=self.perceptual_weight,
            log_figures=log_figures,
        )

        model_to_use.train()

        # Compute 3D MS-SSIM on full volumes (2D trainers only, skip for seg_mode)
        # 3D trainers compute their own volumetric MS-SSIM in the runner
        # Must be done BEFORE logging so the metric gets logged with modality suffix
        if self.spatial_dims == 2 and not getattr(self, 'seg_mode', False):
            msssim_3d = self._compute_volume_3d_msssim(epoch, data_split='val')
            if msssim_3d is not None:
                result.metrics['msssim_3d'] = msssim_3d

        # Log to TensorBoard
        self._log_validation_metrics(
            epoch, result.metrics, result.worst_batch_data,
            result.regional_tracker, log_figures
        )

        return result.metrics

    def _capture_worst_batch(
        self,
        images: torch.Tensor,
        reconstruction: torch.Tensor,
        loss: float,
        l1_loss: torch.Tensor,
        p_loss: torch.Tensor,
        reg_loss: torch.Tensor,
    ) -> dict[str, Any]:
        """Capture worst batch data for visualization (2D or 3D).

        Args:
            images: Original images/volumes.
            reconstruction: Reconstructed images/volumes.
            loss: Total loss value.
            l1_loss: L1 loss tensor.
            p_loss: Perceptual loss tensor.
            reg_loss: Regularization loss tensor.

        Returns:
            Dictionary with worst batch data.
        """
        # 3D: Simple capture without dict conversion
        if self.spatial_dims == 3:
            return {
                'original': images.cpu(),
                'generated': reconstruction.float().cpu(),
                'loss': loss,
                'loss_breakdown': {
                    'L1': l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss,
                    'Perc': p_loss.item() if isinstance(p_loss, torch.Tensor) else p_loss,
                    'Reg': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
                },
            }

        # 2D: Handle dual mode with dict conversion
        n_channels = self.cfg.mode.get('in_channels', 1)

        # Convert to dict format for dual mode
        if n_channels == 2:
            image_keys = self.cfg.mode.get('image_keys', ['t1_pre', 't1_gd'])
            orig_dict = {
                image_keys[0]: images[:, 0:1].cpu(),
                image_keys[1]: images[:, 1:2].cpu(),
            }
            gen_dict = {
                image_keys[0]: reconstruction[:, 0:1].float().cpu(),
                image_keys[1]: reconstruction[:, 1:2].float().cpu(),
            }
        else:
            orig_dict = images.cpu()
            gen_dict = reconstruction.float().cpu()

        return {
            'original': orig_dict,
            'generated': gen_dict,
            'loss': loss,
            'loss_breakdown': {
                'L1': l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss,
                'Perc': p_loss.item() if isinstance(p_loss, torch.Tensor) else p_loss,
                'Reg': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            },
        }

    def _log_validation_metrics_core(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Log validation metrics with modality suffix handling.

        This method handles the core metrics logging with proper modality suffixes.
        Uses UnifiedMetrics for consistent TensorBoard paths.
        Subclasses should call this for metrics logging, then add their own
        worst batch and regional tracker handling.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of validation metrics.
        """
        if self.writer is None:
            return

        # Check for seg_mode - use dedicated seg validation logging
        seg_mode = getattr(self, 'seg_mode', False)
        if seg_mode:
            # Use dedicated seg validation logging for consistent paths
            self._unified_metrics.log_seg_validation(metrics, epoch)
            return

        # Get mode name for modality suffix
        mode_name = self.cfg.mode.get('name', 'bravo')
        n_channels = self.cfg.mode.get('in_channels', 1)
        is_multi_modality = mode_name == 'multi_modality'
        is_dual = n_channels == 2 and mode_name == 'dual'
        is_seg_conditioned = mode_name.startswith('seg_conditioned')

        # For single-modality modes (not multi_modality or dual), use modality suffix
        # Multi-modality and dual modes are handled by their respective per-modality loops
        # seg_conditioned modes: no suffix needed (can distinguish by TensorBoard run color)
        if not is_multi_modality and not is_dual:
            modality_metrics = {
                'psnr': metrics.get('psnr'),
                'msssim': metrics.get('msssim'),
                'lpips': metrics.get('lpips'),
                'msssim_3d': metrics.get('msssim_3d'),
                'dice': metrics.get('dice_score'),
                'iou': metrics.get('iou'),
            }
            # No suffix for seg_conditioned modes (distinguish by TensorBoard run color)
            modality = '' if is_seg_conditioned else mode_name
            self._unified_metrics.log_per_modality_validation(modality_metrics, modality, epoch)

            # Also log losses without suffix (gen, l1, perc, reg)
            loss_metrics = {k: v for k, v in metrics.items()
                          if k in ('gen', 'l1', 'perc', 'reg', 'bce', 'dice', 'boundary')}
            if loss_metrics:
                self._log_validation_metrics_unified(epoch, loss_metrics)
        else:
            # Multi-modality/dual: log aggregate metrics, per-modality handled separately
            self._log_validation_metrics_unified(epoch, metrics)

    def _log_validation_metrics(
        self,
        epoch: int,
        metrics: dict[str, float],
        worst_batch_data: dict[str, Any] | None,
        regional_tracker: RegionalMetricsTracker | None,
        log_figures: bool,
    ) -> None:
        """Log validation metrics to TensorBoard using unified system.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of validation metrics.
            worst_batch_data: Worst batch data for visualization.
            regional_tracker: Regional metrics tracker.
            log_figures: Whether to log figures.
        """
        if self.writer is None:
            return

        # Log metrics with modality suffix handling
        self._log_validation_metrics_core(epoch, metrics)

        # Log worst batch figure (uses unified metrics)
        if log_figures and worst_batch_data is not None:
            self._unified_metrics.log_worst_batch(
                original=worst_batch_data['original'],
                reconstructed=worst_batch_data['generated'],
                loss=worst_batch_data['loss'],
                epoch=epoch,
                phase='val',
            )

        # Log regional metrics with modality suffix for single-modality modes
        if regional_tracker is not None:
            mode_name = self.cfg.mode.get('name', 'bravo')
            is_multi_modality = mode_name == 'multi_modality'
            is_dual = self.cfg.mode.get('in_channels', 1) == 2 and mode_name == 'dual'
            is_seg_conditioned = mode_name.startswith('seg_conditioned')
            # No suffix for multi_modality, dual, or seg_conditioned modes
            if is_multi_modality or is_dual or is_seg_conditioned:
                modality_override = None
            else:
                modality_override = mode_name
            self._unified_metrics.log_validation_regional(regional_tracker, epoch, modality_override=modality_override)

    def _compute_per_modality_validation(self, epoch: int) -> None:
        """Compute per-modality validation metrics (2D or 3D).

        For 3D volumes:
        - Uses compute_lpips_3d (slice-by-slice)
        - Computes both volumetric MS-SSIM-3D and slice-wise MS-SSIM-2D

        Args:
            epoch: Current epoch number.
        """
        if not self.per_modality_val_loaders:
            return

        model_to_use = self._get_model_for_eval()
        model_to_use.eval()

        # Get appropriate LPIPS function for dimensionality
        lpips_fn = self._create_lpips_fn()

        for modality, loader in self.per_modality_val_loaders.items():
            total_psnr = 0.0
            total_lpips = 0.0
            total_msssim = 0.0
            total_msssim_3d = 0.0  # Volumetric 3D MS-SSIM (only for 3D)
            n_batches = 0

            # Regional tracker for this modality
            regional_tracker = None
            if self.log_regional_losses:
                regional_tracker = self._create_regional_tracker()

            with torch.inference_mode():
                for batch in loader:
                    images, mask = self._prepare_batch(batch)

                    with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                        reconstruction, _ = self._forward_for_validation(model_to_use, images)

                    # Compute metrics
                    if self.log_psnr:
                        total_psnr += compute_psnr(reconstruction, images)
                    if self.log_lpips:
                        total_lpips += lpips_fn(
                            reconstruction.float(), images.float(), device=self.device
                        )
                    if self.log_msssim:
                        if self.spatial_dims == 3:
                            # 3D: Compute both volumetric and slice-wise
                            total_msssim_3d += compute_msssim(
                                reconstruction.float(), images.float(), spatial_dims=3
                            )
                            total_msssim += compute_msssim_2d_slicewise(
                                reconstruction.float(), images.float()
                            )
                        else:
                            total_msssim += compute_msssim(reconstruction, images)

                    # Regional tracking
                    if regional_tracker is not None and mask is not None:
                        regional_tracker.update(reconstruction.float(), images.float(), mask)

                    n_batches += 1

            # Log metrics using unified system
            if n_batches > 0 and self.writer is not None:
                # Compute 3D MS-SSIM for 2D trainers (full volume reconstruction)
                msssim_3d = None
                if self.spatial_dims == 2:
                    if self.log_msssim and not getattr(self, 'seg_mode', False):
                        msssim_3d = self._compute_volume_3d_msssim(
                            epoch, data_split='val', modality_override=modality
                        )
                else:
                    # 3D trainers already computed volumetric MS-SSIM above
                    msssim_3d = total_msssim_3d / n_batches if self.log_msssim else None

                # Build metrics dict for unified logging
                modality_metrics = {
                    'psnr': total_psnr / n_batches if self.log_psnr else None,
                    'msssim': total_msssim / n_batches if self.log_msssim else None,
                    'lpips': total_lpips / n_batches if self.log_lpips else None,
                    'msssim_3d': msssim_3d,
                }
                self._unified_metrics.log_per_modality_validation(modality_metrics, modality, epoch)

                # Log regional metrics using unified system
                if regional_tracker is not None:
                    self._unified_metrics.log_validation_regional(regional_tracker, epoch, modality_override=modality)

        model_to_use.train()

    def _compute_per_channel_validation(self, epoch: int) -> None:
        """Compute per-channel validation metrics for dual mode.

        For dual mode (2 channels), computes metrics separately for each channel
        (e.g., t1_pre, t1_gd) and logs them to TensorBoard.

        Args:
            epoch: Current epoch number.
        """
        n_channels = self.cfg.mode.get('in_channels', 1)
        if n_channels != 2 or self.val_loader is None:
            return

        image_keys = self.cfg.mode.get('image_keys', ['t1_pre', 't1_gd'])
        model_to_use = self._get_model_for_eval()
        model_to_use.eval()

        # Per-channel accumulators
        channel_metrics = {key: {'psnr': 0.0, 'lpips': 0.0, 'msssim': 0.0} for key in image_keys}
        n_batches = 0

        with torch.inference_mode():
            for batch in self.val_loader:
                images, _ = self._prepare_batch(batch)

                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    reconstruction, _ = self._forward_for_validation(model_to_use, images)

                # Compute per-channel metrics
                for i, key in enumerate(image_keys):
                    img_ch = images[:, i:i+1]
                    rec_ch = reconstruction[:, i:i+1]

                    if self.log_psnr:
                        channel_metrics[key]['psnr'] += compute_psnr(rec_ch, img_ch)
                    if self.log_lpips:
                        channel_metrics[key]['lpips'] += compute_lpips(rec_ch, img_ch, device=self.device)
                    if self.log_msssim:
                        channel_metrics[key]['msssim'] += compute_msssim(rec_ch, img_ch)

                n_batches += 1

        model_to_use.train()

        # Log per-channel metrics using unified system
        if n_batches > 0 and self.writer is not None:
            # Build per-channel data for unified logging
            per_channel_data = {}
            for key in image_keys:
                # Compute 3D MS-SSIM for this channel if needed
                msssim_3d = None
                if self.log_msssim and not getattr(self, 'seg_mode', False):
                    msssim_3d = self._compute_volume_3d_msssim(
                        epoch, data_split='val', modality_override=key
                    )

                per_channel_data[key] = {
                    'psnr': channel_metrics[key]['psnr'] if self.log_psnr else 0,
                    'msssim': channel_metrics[key]['msssim'] if self.log_msssim else 0,
                    'lpips': channel_metrics[key]['lpips'] if self.log_lpips else 0,
                    'count': n_batches,
                }

                # Log 3D MS-SSIM separately per channel (not part of standard per-channel)
                if msssim_3d is not None:
                    self._unified_metrics.log_per_modality_validation(
                        {'msssim_3d': msssim_3d}, key, epoch
                    )

            self._unified_metrics.log_per_channel_validation(per_channel_data, epoch)

    def _compute_volume_3d_msssim(
        self,
        epoch: int,
        data_split: str = 'val',
        modality_override: str | None = None,
    ) -> float | None:
        """Compute 3D MS-SSIM by reconstructing full volumes slice-by-slice.

        For 2D trainers, this loads full 3D volumes, processes each slice through
        the model, stacks reconstructed slices back into a volume, and computes
        3D MS-SSIM. This shows how well 2D models maintain cross-slice consistency.

        Optimizations applied:
        - inference_mode instead of no_grad (faster)
        - Tensor slicing instead of loop for batch extraction
        - Non-blocking GPU transfers
        - Pre-allocated output tensor

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
            mode_name = self.cfg.mode.get('name', 'bravo')
            n_channels = self.cfg.mode.get('in_channels', 1)
            # Use subdir for file loading (e.g., 'seg' instead of 'seg_conditioned')
            subdir = self.cfg.mode.get('subdir', mode_name)
            modality = 'dual' if n_channels > 1 else subdir

        # Create volume dataloader
        result = create_vae_volume_validation_dataloader(self.cfg, modality, data_split)
        if result is None:
            return None

        volume_loader, _ = result

        model_to_use = self._get_model_for_eval()
        model_to_use.eval()

        total_msssim_3d = 0.0
        n_volumes = 0
        slice_batch_size = self.cfg.training.batch_size  # Reuse training batch size

        with torch.inference_mode():  # Faster than no_grad
            for batch in volume_loader:
                # batch['image'] is [1, C, H, W, D] (batch_size=1 for volumes)
                # Non-blocking transfer to GPU
                volume = batch['image'].to(self.device, non_blocking=True)  # [1, C, H, W, D]
                volume = volume.squeeze(0)  # [C, H, W, D]

                n_channels_vol, height, width, depth = volume.shape

                # Pre-allocate output tensor on GPU
                all_recon = torch.empty(
                    (depth, n_channels_vol, height, width),
                    dtype=self.weight_dtype,
                    device=self.device
                )

                # Process slices in batches using tensor slicing (no Python loop for extraction)
                for start_idx in range(0, depth, slice_batch_size):
                    end_idx = min(start_idx + slice_batch_size, depth)

                    # Direct tensor slicing: [C, H, W, D] -> [B, C, H, W]
                    # Transpose to get slices along last dim, then slice
                    slice_tensor = volume[:, :, :, start_idx:end_idx].permute(3, 0, 1, 2)

                    with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                        # Forward through model
                        recon, _ = self._forward_for_validation(model_to_use, slice_tensor)

                    # Write directly to pre-allocated tensor
                    all_recon[start_idx:end_idx] = recon

                # Reshape: [D, C, H, W] -> [1, C, D, H, W]
                recon_3d = all_recon.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, D, H, W]
                volume_3d = volume.permute(0, 3, 1, 2).unsqueeze(0)  # [1, C, D, H, W]

                # Compute 3D MS-SSIM
                msssim_3d = compute_msssim(recon_3d.float(), volume_3d.float(), spatial_dims=3)
                total_msssim_3d += msssim_3d
                n_volumes += 1

        model_to_use.train()

        if n_volumes == 0:
            return None

        avg_msssim_3d = total_msssim_3d / n_volumes

        # Note: Logging is handled by caller through _log_validation_metrics_core
        # which applies proper modality suffix for single-modality modes
        return avg_msssim_3d

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_checkpoint_manager(self) -> None:
        """Setup checkpoint manager with GAN components.

        Overrides BaseTrainer to add discriminator, optimizer_d, scheduler_d.
        """
        if not self.is_main_process:
            return

        from .checkpoint_manager import CheckpointManager

        self.checkpoint_manager = CheckpointManager(
            save_dir=self.save_dir,
            model=self.model_raw,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler if not self.use_constant_lr else None,
            ema=self.ema if self.use_ema else None,
            config=self._get_model_config(),
            # GAN components
            discriminator=self.discriminator_raw if not self.disable_gan else None,
            optimizer_d=self.optimizer_d if not self.disable_gan else None,
            scheduler_d=self.lr_scheduler_d if not self.disable_gan and not self.use_constant_lr else None,
            metric_name=self._get_best_metric_name(),
            keep_last_n=self.cfg.training.get('keep_last_n_checkpoints', 0),
            device=self.device,
        )

    def _get_checkpoint_extra_state(self) -> dict[str, Any] | None:
        """Return extra state for compression trainer checkpoints.

        Includes discriminator config and training flags.
        """
        extra_state = {
            'disable_gan': self.disable_gan,
            'use_constant_lr': self.use_constant_lr,
        }

        # Add discriminator config if GAN is enabled
        if not self.disable_gan and self.discriminator_raw is not None:
            extra_state['disc_config'] = {
                'in_channels': self.cfg.mode.get('in_channels', 1),
                'channels': self.disc_num_channels,
                'num_layers_d': self.disc_num_layers,
            }

        return extra_state

    def _save_checkpoint(self, epoch: int, name: str) -> None:
        """Save checkpoint with standardized format.

        Args:
            epoch: Current epoch number.
            name: Checkpoint name ("latest" or "best").
        """
        if not self.is_main_process:
            return

        model_config = self._get_model_config()

        # Build extra state
        extra_state = {
            'disable_gan': self.disable_gan,
            'use_constant_lr': self.use_constant_lr,
        }

        # Add discriminator state if GAN is enabled
        if not self.disable_gan and self.discriminator_raw is not None:
            extra_state['discriminator_state_dict'] = self.discriminator_raw.state_dict()
            extra_state['disc_config'] = {
                'in_channels': self.cfg.mode.get('in_channels', 1),
                'channels': self.disc_num_channels,
                'num_layers_d': self.disc_num_layers,
            }
            if self.optimizer_d is not None:
                extra_state['optimizer_d_state_dict'] = self.optimizer_d.state_dict()
            if not self.use_constant_lr and self.lr_scheduler_d is not None:
                extra_state['scheduler_d_state_dict'] = self.lr_scheduler_d.state_dict()

        # Save using standardized format: checkpoint_{name}.pt
        save_full_checkpoint(
            model=self.model_raw,
            optimizer=self.optimizer,
            epoch=epoch,
            save_dir=self.save_dir,
            filename=f"checkpoint_{name}",
            model_config=model_config,
            scheduler=self.lr_scheduler if not self.use_constant_lr else None,
            ema=self.ema,
            extra_state=extra_state,
        )

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> int:
        """Load checkpoint to resume training.

        Uses CheckpointManager if available, otherwise falls back to legacy loading.

        Args:
            checkpoint_path: Path to checkpoint file.
            load_optimizer: Whether to load optimizer state.

        Returns:
            Epoch number from checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Use CheckpointManager if available
        if self.checkpoint_manager is not None:
            result = self.checkpoint_manager.load(
                checkpoint_path,
                strict=True,
                load_optimizer=load_optimizer,
            )
            epoch = result['epoch']
            if self.is_main_process:
                logger.info(f"Resuming from epoch {epoch + 1}")
            return epoch

        # Legacy loading (backward compatibility)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load model weights
        self.model_raw.load_state_dict(checkpoint['model_state_dict'])
        if self.is_main_process:
            logger.info(f"Loaded model weights from {checkpoint_path}")

        # Load discriminator weights
        if not self.disable_gan and self.discriminator_raw is not None:
            if 'discriminator_state_dict' in checkpoint:
                self.discriminator_raw.load_state_dict(checkpoint['discriminator_state_dict'])
                if self.is_main_process:
                    logger.info("Loaded discriminator weights")
            else:
                if self.is_main_process:
                    logger.warning("Checkpoint has no discriminator weights")

        # Load optimizer states
        if load_optimizer:
            if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Also try old key name for backwards compat
            elif 'optimizer_g_state_dict' in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_g_state_dict'])

            if not self.disable_gan and self.optimizer_d is not None:
                if 'optimizer_d_state_dict' in checkpoint:
                    self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

            # Load scheduler states
            if not self.use_constant_lr:
                if 'scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                elif 'scheduler_g_state_dict' in checkpoint and self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(checkpoint['scheduler_g_state_dict'])

                if not self.disable_gan and self.lr_scheduler_d is not None:
                    if 'scheduler_d_state_dict' in checkpoint:
                        self.lr_scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])

        # Load EMA state
        if self.use_ema and self.ema is not None:
            if 'ema_state_dict' in checkpoint:
                self.ema.load_state_dict(checkpoint['ema_state_dict'])
                if self.is_main_process:
                    logger.info("Loaded EMA state")

        epoch = checkpoint.get('epoch', 0)
        if self.is_main_process:
            logger.info(f"Resuming from epoch {epoch + 1}")

        return epoch

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
        timestamp = time.strftime("%H:%M:%S")
        epoch_pct = ((epoch + 1) / total_epochs) * 100

        # Format validation metrics
        val_gen = f"(v:{val_metrics.get('gen', 0):.4f})" if val_metrics else ""
        val_l1 = f"(v:{val_metrics.get('l1', 0):.4f})" if val_metrics else ""
        msssim_str = f"MS-SSIM: {val_metrics.get('msssim', 0):.3f}" if val_metrics.get('msssim') else ""

        logger.info(
            f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
            f"G: {avg_losses.get('gen', 0):.4f}{val_gen} | "
            f"L1: {avg_losses.get('recon', avg_losses.get('l1', 0)):.4f}{val_l1} | "
            f"D: {avg_losses.get('disc', 0):.4f} | "
            f"{msssim_str} | "
            f"Time: {elapsed_time:.1f}s"
        )

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
        """Load pretrained weights from checkpoint.

        This is the shared implementation for 2D trainers (VAE, VQ-VAE).
        3D trainers use _load_pretrained_weights_base() which handles
        prefix stripping for CheckpointedAutoencoder wrapper.

        Args:
            raw_model: The raw model to load weights into.
            raw_disc: The raw discriminator (can be None if GAN disabled).
            checkpoint_path: Path to the checkpoint file.
            model_name: Name for logging (e.g., "VAE", "VQ-VAE").
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                raw_model.load_state_dict(checkpoint['model_state_dict'])
                if self.is_main_process:
                    logger.info(f"Loaded {model_name} weights from {checkpoint_path}")
            if 'discriminator_state_dict' in checkpoint and raw_disc is not None:
                raw_disc.load_state_dict(checkpoint['discriminator_state_dict'])
                if self.is_main_process:
                    logger.info(f"Loaded discriminator weights from {checkpoint_path}")
        except FileNotFoundError:
            if self.is_main_process:
                logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")

    def _load_pretrained_weights_base(
        self,
        base_model: nn.Module,
        checkpoint_path: str,
        model_name: str = "model",
    ) -> None:
        """Load pretrained weights into base model (before checkpointing wrapper).

        Handles 'model.' prefix stripping for checkpointed model wrappers.
        Used by 3D trainers that wrap models with gradient checkpointing.

        Args:
            base_model: The base model (unwrapped) to load weights into.
            checkpoint_path: Path to the checkpoint file.
            model_name: Name for logging (e.g., "3D VAE", "3D VQ-VAE").
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Remove 'model.' prefix if present (from Checkpointed* wrappers)
                keys_with_prefix = [k for k in state_dict.keys() if k.startswith('model.')]
                if keys_with_prefix:
                    if len(keys_with_prefix) != len(state_dict) and self.is_main_process:
                        logger.warning(
                            f"Mixed prefix state: {len(keys_with_prefix)}/{len(state_dict)} keys "
                            "have 'model.' prefix. Stripping prefix from matching keys."
                        )
                    state_dict = {
                        k.replace('model.', '', 1) if k.startswith('model.') else k: v
                        for k, v in state_dict.items()
                    }
                base_model.load_state_dict(state_dict)
                if self.is_main_process:
                    logger.info(f"Loaded {model_name} weights from {checkpoint_path}")
        except FileNotFoundError:
            if self.is_main_process:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")

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
        """Create test evaluator for this trainer (2D or 3D).

        Factory method that creates a CompressionTestEvaluator (2D) or
        Compression3DTestEvaluator (3D) with trainer-specific callbacks.

        Returns:
            Configured test evaluator instance.
        """
        from medgen.evaluation import (
            Compression3DTestEvaluator,
            CompressionTestEvaluator,
            MetricsConfig,
        )

        # Check for seg_mode (set by subclasses)
        seg_mode = getattr(self, 'seg_mode', False)
        seg_loss_fn = getattr(self, 'seg_loss_fn', None)

        # Get modality name for single-modality suffix
        # Use empty string for seg_conditioned modes (no suffix needed)
        mode_name = self.cfg.mode.get('name', 'bravo')
        if mode_name.startswith('seg_conditioned'):
            mode_name = ''

        # Get image keys for per-channel metrics
        n_channels = self.cfg.mode.get('in_channels', 1)
        image_keys = None
        if n_channels > 1:
            image_keys = self.cfg.mode.get('image_keys', None)

        # Regional tracker factory (use seg-specific tracker for seg_mode)
        regional_factory = None
        if self.log_regional_losses:
            if seg_mode and hasattr(self, '_create_seg_regional_tracker'):
                regional_factory = self._create_seg_regional_tracker
            else:
                regional_factory = self._create_regional_tracker

        # 3D evaluator
        if self.spatial_dims == 3:
            metrics_config = MetricsConfig(
                compute_l1=not seg_mode,
                compute_psnr=not seg_mode,
                compute_lpips=not seg_mode,
                compute_msssim=self.log_msssim and not seg_mode,  # 2D slicewise
                compute_msssim_3d=self.log_msssim and not seg_mode,  # Volumetric
                compute_regional=self.log_regional_losses,
                seg_mode=seg_mode,
            )

            # Worst batch figure callback (3D version)
            worst_batch_fig_fn = self._create_worst_batch_figure

            return Compression3DTestEvaluator(
                model=self.model_raw,
                device=self.device,
                save_dir=self.save_dir,
                forward_fn=self._test_forward,
                weight_dtype=self.weight_dtype,
                writer=self.writer,
                metrics_config=metrics_config,
                is_cluster=self.is_cluster,
                regional_tracker_factory=regional_factory,
                worst_batch_figure_fn=worst_batch_fig_fn,
                image_keys=image_keys,
                seg_loss_fn=seg_loss_fn if seg_mode else None,
                modality_name=mode_name,
            )

        # 2D evaluator
        metrics_config = MetricsConfig(
            compute_l1=not seg_mode,
            compute_psnr=not seg_mode,
            compute_lpips=not seg_mode,
            compute_msssim=self.log_msssim and not seg_mode,
            compute_msssim_3d=False,  # Volume 3D MS-SSIM added via callback
            compute_regional=self.log_regional_losses,
            seg_mode=seg_mode,
        )

        # Volume 3D MS-SSIM callback (for 2D trainers reconstructing full volumes)
        def volume_3d_msssim() -> float | None:
            if seg_mode:
                return None
            return self._compute_volume_3d_msssim(epoch=0, data_split='test_new')

        # Worst batch figure callback
        worst_batch_fig_fn = self._create_worst_batch_figure

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
            seg_loss_fn=seg_loss_fn if seg_mode else None,
            modality_name=mode_name,
        )

    def evaluate_test_set(
        self,
        test_loader: DataLoader,
        checkpoint_name: str | None = None,
    ) -> dict[str, float]:
        """Evaluate compression model on test set.

        Uses CompressionTestEvaluator for unified test evaluation.

        Args:
            test_loader: Test data loader.
            checkpoint_name: Checkpoint to load ("best", "latest", or None).

        Returns:
            Dict with test metrics.
        """
        if not self.is_main_process:
            return {}

        evaluator = self._create_test_evaluator()
        return evaluator.evaluate(
            test_loader,
            checkpoint_name=checkpoint_name,
            get_eval_model=self._get_model_for_eval,
        )
