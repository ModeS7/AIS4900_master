"""Base class for 2D and 3D diffusion trainers.

This module provides DiffusionTrainerBase, which contains all shared functionality
between DiffusionTrainer (2D) and Diffusion3DTrainer (3D).

The goal is to eliminate code duplication and prevent divergence bugs where
2D and 3D trainers have different implementations of the same logic.

Shared functionality includes:
- Strategy creation (DDPM, RFlow)
- Timestep manipulation (jitter, curriculum)
- Noise augmentation
- Gradient noise injection
- EMA management
- Feature perturbation hooks
- Min-SNR weighting
- DC-AE 1.5 augmented diffusion helpers

Dimension-specific functionality (implemented in subclasses):
- setup_model() - model architecture differs
- train_step() - tensor shapes differ
- _create_mode() - supported modes differ
- _create_aug_diff_mask() - mask shapes differ
"""
import logging
import random
from abc import ABC, abstractmethod
from typing import Any

import torch
from ema_pytorch import EMA
from omegaconf import DictConfig
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from medgen.diffusion import (
    DDPMStrategy,
    DiffusionSpace,
    DiffusionStrategy,
    PixelSpace,
    RFlowStrategy,
)

from .base_config import StrategyConfig
from .base_trainer import BaseTrainer
from .diffusion_config import (
    AugmentedDiffusionConfig,
    TrainingTricksConfig,
)
from .results import BatchType, TrainingStepResult

logger = logging.getLogger(__name__)


class DiffusionTrainerBase(BaseTrainer, ABC):
    """Abstract base class for diffusion trainers.

    Provides shared functionality for both 2D and 3D diffusion training.
    Subclasses must implement dimension-specific methods.

    Args:
        cfg: Hydra configuration object.
        space: DiffusionSpace for pixel/latent operations. Defaults to PixelSpace.
    """

    def __init__(self, cfg: DictConfig, space: DiffusionSpace | None = None) -> None:
        super().__init__(cfg)

        self.space = space if space is not None else PixelSpace()

        # ─────────────────────────────────────────────────────────────────────
        # Extract typed configs
        # ─────────────────────────────────────────────────────────────────────
        sc = StrategyConfig.from_hydra(cfg)
        self._strategy_config = sc

        # ─────────────────────────────────────────────────────────────────────
        # Core diffusion config (shared between 2D and 3D)
        # ─────────────────────────────────────────────────────────────────────
        self.strategy_name: str = sc.name
        self.mode_name: str = cfg.mode.name
        self.num_timesteps: int = sc.num_train_timesteps

        # Initialize strategy (shared - both 2D and 3D use same strategies)
        self.strategy = self._create_strategy(self.strategy_name)

        # EMA configuration (shared)
        self.use_ema: bool = cfg.training.get('use_ema', False)
        self.ema_decay: float = cfg.training.get('ema', {}).get('decay', 0.9999)
        self.ema: EMA | None = None

        # Global step counter
        self._global_step: int = 0
        self._current_epoch: int = 0

        # ─────────────────────────────────────────────────────────────────────
        # Training tricks (from TrainingTricksConfig)
        # ─────────────────────────────────────────────────────────────────────
        tricks = TrainingTricksConfig.from_hydra(cfg)
        self._training_tricks = tricks

        self.use_min_snr: bool = tricks.min_snr.enabled
        self.min_snr_gamma: float = tricks.min_snr.gamma
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

        # RFlow-specific Min-SNR-γ (uses RFlow's own SNR formula)
        self.rflow_snr_gamma: float = sc.snr_gamma if self.strategy_name == 'rflow' else 0.0

        # ─────────────────────────────────────────────────────────────────────
        # DC-AE 1.5: Augmented Diffusion Training (from AugmentedDiffusionConfig)
        # ─────────────────────────────────────────────────────────────────────
        aug_diff = AugmentedDiffusionConfig.from_hydra(cfg)
        self._aug_diff_config = aug_diff
        self.augmented_diffusion_enabled: bool = aug_diff.enabled
        self.aug_diff_min_channels: int = aug_diff.min_channels
        self.aug_diff_channel_step: int = aug_diff.channel_step
        self._aug_diff_channel_steps: list[int] | None = None

        # ─────────────────────────────────────────────────────────────────────
        # Conditioning dropout (shared config)
        # ─────────────────────────────────────────────────────────────────────
        cond_dropout_cfg = cfg.training.get('conditioning_dropout', {})
        self.conditioning_dropout_prob: float = cond_dropout_cfg.get('prob', 0.15)

        # ─────────────────────────────────────────────────────────────────────
        # Logging config (from BaseTrainingConfig, set by BaseTrainer)
        # ─────────────────────────────────────────────────────────────────────
        log_cfg = cfg.training.get('logging', {})
        self.log_timestep_losses: bool = log_cfg.get('timestep_losses', True)

        # ─────────────────────────────────────────────────────────────────────
        # Model components (set during setup_model)
        # ─────────────────────────────────────────────────────────────────────
        self.model: nn.Module | None = None
        self.model_raw: nn.Module | None = None
        self.optimizer: AdamW | None = None
        self.lr_scheduler: Any | None = None

        # Validation loader (set in train())
        self.val_loader: DataLoader | None = None

        # Feature perturbation hooks (set in _setup_feature_perturbation)
        self._feature_hooks: list[Any] = []

        # Log config at initialization
        self._log_shared_config()

    @property
    @abstractmethod
    def spatial_dims(self) -> int:
        """Return spatial dimensions (2 or 3)."""
        pass

    @abstractmethod
    def setup_model(self, train_dataset: Dataset) -> None:
        """Initialize model, optimizer, and loss functions.

        Args:
            train_dataset: Training dataset for model config extraction.
        """
        pass

    @abstractmethod
    def train_step(self, batch: BatchType) -> TrainingStepResult:
        """Execute single training step.

        Args:
            batch: Input batch from dataloader.

        Returns:
            TrainingStepResult with loss values.
        """
        pass

    @abstractmethod
    def _create_mode(self, name: str) -> Any:
        """Create training mode based on name.

        Args:
            name: Mode name ('seg', 'bravo', 'dual', 'multi', 'seg_conditioned').

        Returns:
            TrainingMode instance.
        """
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Shared Methods: Strategy Creation
    # ─────────────────────────────────────────────────────────────────────────

    def _create_strategy(self, name: str) -> DiffusionStrategy:
        """Create diffusion strategy based on name.

        Args:
            name: Strategy name ('ddpm' or 'rflow').

        Returns:
            DiffusionStrategy instance.

        Raises:
            ValueError: If strategy name is unknown.
        """
        strategies: dict[str, type] = {
            'ddpm': DDPMStrategy,
            'rflow': RFlowStrategy,
        }
        if name not in strategies:
            raise ValueError(f"Unknown strategy: {name}. Choose from {list(strategies.keys())}")
        return strategies[name]()

    # ─────────────────────────────────────────────────────────────────────────
    # Shared Methods: Timestep Manipulation
    # ─────────────────────────────────────────────────────────────────────────

    def _get_curriculum_range(self, epoch: int) -> tuple[float, float] | None:
        """Get timestep range for curriculum learning. Override in subclass to enable."""
        return None

    def _apply_timestep_jitter(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Apply timestep jitter. Override in subclass to enable."""
        return timesteps

    # ─────────────────────────────────────────────────────────────────────────
    # Shared Methods: Noise Augmentation
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_noise_augmentation(
        self,
        noise: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Apply noise augmentation. Override in subclass to enable."""
        return noise

    # ─────────────────────────────────────────────────────────────────────────
    # Shared Methods: Gradient Noise
    # ─────────────────────────────────────────────────────────────────────────

    def _add_gradient_noise(self, step: int) -> None:
        """Add gradient noise. Override in subclass to enable."""
        return

    # ─────────────────────────────────────────────────────────────────────────
    # Shared Methods: EMA Management
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_ema(self, model: nn.Module) -> None:
        """Setup EMA wrapper. Override in subclass to enable."""
        return

    def _update_ema(self) -> None:
        """Update EMA model weights. Override in subclass to enable."""
        return

    # ─────────────────────────────────────────────────────────────────────────
    # Shared Methods: Feature Perturbation
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_feature_perturbation(self) -> None:
        """Setup feature perturbation hooks. Override in subclass to enable."""
        self._feature_hooks = []

    def _remove_feature_perturbation_hooks(self) -> None:
        """Remove feature perturbation hooks."""
        for handle in getattr(self, '_feature_hooks', []):
            handle.remove()
        self._feature_hooks = []

    # ─────────────────────────────────────────────────────────────────────────
    # Shared Methods: DC-AE 1.5 Augmented Diffusion
    # ─────────────────────────────────────────────────────────────────────────

    def _get_aug_diff_channel_steps(self, num_channels: int) -> list[int]:
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

        Note: Shape differs between 2D [1, C, 1, 1] and 3D [1, C, 1, 1, 1].
        This method creates the appropriate shape based on spatial_dims.

        Args:
            tensor: Input tensor to get shape from.

        Returns:
            Mask tensor for broadcasting.
        """
        C = tensor.shape[1]
        steps = self._get_aug_diff_channel_steps(C)
        c_prime = random.choice(steps)

        # Create mask with appropriate spatial dimensions
        if self.spatial_dims == 2:
            mask = torch.zeros(1, C, 1, 1, device=tensor.device, dtype=tensor.dtype)
            mask[:, :c_prime, :, :] = 1.0
        else:  # 3D
            mask = torch.zeros(1, C, 1, 1, 1, device=tensor.device, dtype=tensor.dtype)
            mask[:, :c_prime, :, :, :] = 1.0

        return mask

    # ─────────────────────────────────────────────────────────────────────────
    # Shared Methods: Min-SNR Weighting
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_snr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute Min-SNR loss weights for given timesteps.

        For RFlow (our primary strategy), uses t/(1-t) as SNR approximation.
        For DDPM, uses alpha_bar/(1-alpha_bar) from the scheduler.

        Args:
            timesteps: Tensor of timestep values (normalized to [0, 1] for RFlow).

        Returns:
            Tensor of SNR-based loss weights [B].
        """
        if self.strategy_name == 'ddpm' and hasattr(self, 'scheduler') and self.scheduler is not None:
            # DDPM: use alpha_bar from scheduler
            alphas_cumprod = self.scheduler.alphas_cumprod.to(timesteps.device)
            alpha_bar = alphas_cumprod[timesteps.long()]
            snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
        else:
            # RFlow: timesteps are in [0, 1], use (1-t)/t as SNR
            t_normalized = timesteps.float() / self.num_timesteps
            snr = (1.0 - t_normalized) / (t_normalized + 1e-8)

        # Clip SNR and compute weight: min(SNR, gamma) / SNR
        snr_clipped = torch.clamp(snr, max=self.min_snr_gamma)
        weights = snr_clipped / (snr + 1e-8)

        return weights

    # ─────────────────────────────────────────────────────────────────────────
    # Shared Methods: Self-Conditioning
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_self_conditioning_loss(
        self,
        model_input: torch.Tensor,
        timesteps: torch.Tensor,
        prediction: torch.Tensor,
        mode_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute self-conditioning consistency loss.

        With probability `prob`, runs model a second time and computes
        consistency loss between the two predictions.

        Args:
            model_input: Current model input tensor.
            timesteps: Current timesteps.
            prediction: Current prediction from main forward pass.
            mode_id: Optional mode ID for multi-modality (2D only).

        Returns:
            Consistency loss (0 if disabled or skipped this batch).
        """
        tt = self._training_tricks
        if not tt.self_cond.enabled:
            return torch.tensor(0.0, device=model_input.device)

        prob = tt.self_cond.prob

        # With probability (1-prob), skip self-conditioning
        if random.random() >= prob:
            return torch.tensor(0.0, device=model_input.device)

        # Get second prediction (detached first prediction as reference)
        with torch.no_grad():
            use_mode_embedding = getattr(self, 'use_mode_embedding', False)
            if use_mode_embedding and mode_id is not None:
                prediction_ref = self.model(model_input, timesteps, mode_id=mode_id)
            else:
                prediction_ref = self.model(x=model_input, timesteps=timesteps)
            prediction_ref = prediction_ref.detach()

        # Consistency loss: predictions should be similar
        import torch.nn.functional as F
        consistency_loss = F.mse_loss(prediction.float(), prediction_ref.float())

        return consistency_loss

    # ─────────────────────────────────────────────────────────────────────────
    # Logging Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _log_shared_config(self) -> None:
        """Log shared configuration at initialization."""
        if not self.is_main_process:
            return

        # Log Min-SNR config
        if self.use_min_snr:
            logger.info(f"Min-SNR weighting enabled (gamma={self.min_snr_gamma})")

        # Log RFlow Min-SNR-γ
        if self.rflow_snr_gamma > 0:
            logger.info(f"RFlow Min-SNR-γ weighting enabled (gamma={self.rflow_snr_gamma})")

        # Log EDM preconditioning
        if self._strategy_config.sigma_data > 0 and self.strategy_name == 'rflow':
            logger.info(f"EDM preconditioning enabled (sigma_data={self._strategy_config.sigma_data})")

        # Log augmented diffusion config
        if self.augmented_diffusion_enabled:
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

        # Log conditioning dropout
        if self.conditioning_dropout_prob > 0:
            logger.info(f"Conditioning dropout enabled: prob={self.conditioning_dropout_prob}")

        # Log training tricks from typed config
        tt = self._training_tricks

        if tt.gradient_noise.enabled:
            logger.info(
                f"Gradient noise enabled: sigma={tt.gradient_noise.sigma}, "
                f"decay={tt.gradient_noise.decay}"
            )

        if tt.curriculum.enabled:
            logger.info(
                f"Curriculum timestep scheduling enabled: "
                f"warmup_epochs={tt.curriculum.warmup_epochs}, "
                f"range [{tt.curriculum.min_t_start}-{tt.curriculum.max_t_start}] -> "
                f"[{tt.curriculum.min_t_end}-{tt.curriculum.max_t_end}]"
            )

        if tt.jitter.enabled:
            logger.info(f"Timestep jitter enabled: std={tt.jitter.std}")

        if tt.noise_augmentation.enabled:
            logger.info(f"Noise augmentation enabled: std={tt.noise_augmentation.std}")

        if tt.self_cond.enabled:
            logger.info(f"Self-conditioning enabled: prob={tt.self_cond.prob}")

    # ─────────────────────────────────────────────────────────────────────────
    # Dimension Helper Methods (for unified 2D/3D training)
    # ─────────────────────────────────────────────────────────────────────────

    def _expand_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        """Expand timesteps for broadcasting to spatial dims.

        Args:
            t: Timesteps [B] or [B, 1]

        Returns:
            Expanded timesteps [B, 1, 1, 1] for 2D or [B, 1, 1, 1, 1] for 3D
        """
        if t.ndim == 1:
            t = t.unsqueeze(1)
        for _ in range(self.spatial_dims):
            t = t.unsqueeze(-1)
        return t

    def _get_spatial_shape(self) -> tuple[int, ...]:
        """Get spatial dimensions as tuple.

        Returns:
            (H, W) for 2D or (D, H, W) for 3D
        """
        if self.spatial_dims == 2:
            return (self.image_size, self.image_size)
        else:
            return (self.volume_depth, self.volume_height, self.volume_width)

    def _get_noise_shape(self, batch_size: int, channels: int) -> tuple[int, ...]:
        """Get full tensor shape for noise generation.

        Args:
            batch_size: Batch size.
            channels: Number of channels.

        Returns:
            [B, C, H, W] for 2D or [B, C, D, H, W] for 3D
        """
        return (batch_size, channels) + self._get_spatial_shape()

    def _extract_center_slice(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract center slice from 3D volume. No-op for 2D.

        Args:
            tensor: [B, C, H, W] for 2D or [B, C, D, H, W] for 3D

        Returns:
            [B, C, H, W] (center slice for 3D, unchanged for 2D)
        """
        if self.spatial_dims == 2:
            return tensor
        center_idx = tensor.shape[2] // 2
        return tensor[:, :, center_idx, :, :]

    def _validate_tensor_shape(self, tensor: torch.Tensor, name: str) -> None:
        """Validate tensor has expected number of dimensions.

        Args:
            tensor: Tensor to validate.
            name: Name for error messages.

        Raises:
            ValueError: If tensor has wrong number of dimensions.
        """
        expected_ndim = 2 + self.spatial_dims  # [B, C] + spatial
        if tensor.ndim != expected_ndim:
            raise ValueError(
                f"{name} has {tensor.ndim} dims, expected {expected_ndim} "
                f"for spatial_dims={self.spatial_dims}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Metric/Component Factory Methods (for unified 2D/3D training)
    # ─────────────────────────────────────────────────────────────────────────

    def _create_msssim_fn(self):
        """Get dimension-appropriate MS-SSIM function.

        Returns:
            Function for computing MS-SSIM with spatial_dims preset.
        """
        from functools import partial

        from medgen.metrics import compute_msssim
        return partial(compute_msssim, spatial_dims=self.spatial_dims)

    def _create_lpips_fn(self):
        """Get dimension-appropriate LPIPS function.

        Returns:
            Function for computing LPIPS, or None if disabled.
        """
        log_lpips = getattr(self, 'log_lpips', False)
        if not log_lpips:
            return None
        if self.spatial_dims == 2:
            from medgen.metrics import compute_lpips
            return compute_lpips
        else:
            from medgen.metrics import compute_lpips_3d
            return compute_lpips_3d

    def _create_regional_tracker(self, loss_fn=None):
        """Create dimension-appropriate regional tracker.

        Args:
            loss_fn: Loss function for regional tracking.

        Returns:
            RegionalMetricsTracker instance.
        """
        if self.spatial_dims == 2:
            from medgen.metrics import RegionalMetricsTracker
            return RegionalMetricsTracker(
                image_size=self.image_size,
                loss_fn=loss_fn,
            )
        else:
            from medgen.metrics import RegionalMetricsTracker3D
            return RegionalMetricsTracker3D(
                volume_size=(self.volume_height, self.volume_width, self.volume_depth),
                loss_fn=loss_fn,
            )

