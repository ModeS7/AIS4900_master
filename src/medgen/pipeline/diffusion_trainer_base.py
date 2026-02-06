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
    DiffusionTrainerConfig,
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
        self.use_ema: bool = cfg.training.get('use_ema', True)
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

        self.use_min_snr: bool = tricks.min_snr_enabled
        self.min_snr_gamma: float = tricks.min_snr_gamma
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
        """Get timestep range for curriculum learning.

        Linearly interpolates from start range to end range over warmup_epochs.

        Args:
            epoch: Current training epoch.

        Returns:
            Tuple of (min_t, max_t) or None if curriculum disabled.
        """
        tt = self._training_tricks
        if not tt.curriculum_enabled:
            return None

        progress = min(1.0, epoch / tt.curriculum_warmup_epochs)

        # Linear interpolation from start to end range
        min_t = tt.curriculum_min_t_start + progress * (tt.curriculum_min_t_end - tt.curriculum_min_t_start)
        max_t = tt.curriculum_max_t_start + progress * (tt.curriculum_max_t_end - tt.curriculum_max_t_start)

        return (min_t, max_t)

    def _apply_timestep_jitter(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to timesteps for regularization.

        Increases noise-level diversity without changing output distribution.

        Args:
            timesteps: Original timesteps tensor.

        Returns:
            Jittered timesteps (clamped to valid range).
        """
        tt = self._training_tricks
        if not tt.jitter_enabled:
            return timesteps

        std = tt.jitter_std

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

    # ─────────────────────────────────────────────────────────────────────────
    # Shared Methods: Noise Augmentation
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_noise_augmentation(
        self,
        noise: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Add perturbation to noise vector for regularization.

        Increases noise diversity without affecting what model learns to output.

        Args:
            noise: Original noise tensor or dict of tensors (for dual mode).

        Returns:
            Perturbed noise (renormalized to maintain variance).
        """
        tt = self._training_tricks
        if not tt.noise_augmentation_enabled:
            return noise

        std = tt.noise_augmentation_std

        if isinstance(noise, dict):
            # Handle dual mode (2D only)
            perturbed = {}
            for k, v in noise.items():
                perturbation = torch.randn_like(v) * std
                perturbed_v = v + perturbation
                perturbed[k] = perturbed_v / perturbed_v.std() * v.std()
            return perturbed
        else:
            # Standard tensor (both 2D and 3D)
            perturbation = torch.randn_like(noise) * std
            perturbed = noise + perturbation
            return perturbed / perturbed.std() * noise.std()

    # ─────────────────────────────────────────────────────────────────────────
    # Shared Methods: Gradient Noise
    # ─────────────────────────────────────────────────────────────────────────

    def _add_gradient_noise(self, step: int) -> None:
        """Add Gaussian noise to gradients for regularization.

        Noise decays over training as: sigma / (1 + step)^decay
        Reference: "Adding Gradient Noise Improves Learning" (Neelakantan et al., 2015)

        Args:
            step: Current global training step.
        """
        tt = self._training_tricks
        if not tt.gradient_noise_enabled:
            return

        sigma = tt.gradient_noise_sigma
        decay = tt.gradient_noise_decay

        # Decay noise over training
        noise_scale = sigma / ((1 + step) ** decay)

        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad.add_(noise)

    # ─────────────────────────────────────────────────────────────────────────
    # Shared Methods: EMA Management
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_ema(self, model: nn.Module) -> None:
        """Setup EMA wrapper if enabled.

        Args:
            model: Model to wrap with EMA.
        """
        if self.use_ema:
            ema_cfg = self.cfg.training.get('ema', {})
            self.ema = EMA(
                model,
                beta=self.ema_decay,
                update_after_step=int(ema_cfg.get('update_after_step', 0)),
                update_every=int(ema_cfg.get('update_every', 1)),
            )
            if self.is_main_process:
                logger.info(f"EMA enabled with decay={self.ema_decay}")

    def _update_ema(self) -> None:
        """Update EMA model weights."""
        if self.ema is not None:
            self.ema.update()

    # ─────────────────────────────────────────────────────────────────────────
    # Shared Methods: Feature Perturbation
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_feature_perturbation(self) -> None:
        """Setup forward hooks for feature perturbation."""
        self._feature_hooks = []
        tt = self._training_tricks

        if not tt.feature_perturbation_enabled:
            return

        std = tt.feature_perturbation_std
        layers = tt.feature_perturbation_layers

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

        if self._feature_hooks and self.is_main_process:
            logger.info(
                f"Feature perturbation enabled: {len(self._feature_hooks)} hooks on {layers}"
            )

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
        if not tt.self_cond_enabled:
            return torch.tensor(0.0, device=model_input.device)

        prob = tt.self_cond_prob

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

        if tt.gradient_noise_enabled:
            logger.info(
                f"Gradient noise enabled: sigma={tt.gradient_noise_sigma}, "
                f"decay={tt.gradient_noise_decay}"
            )

        if tt.curriculum_enabled:
            logger.info(
                f"Curriculum timestep scheduling enabled: "
                f"warmup_epochs={tt.curriculum_warmup_epochs}, "
                f"range [{tt.curriculum_min_t_start}-{tt.curriculum_max_t_start}] -> "
                f"[{tt.curriculum_min_t_end}-{tt.curriculum_max_t_end}]"
            )

        if tt.jitter_enabled:
            logger.info(f"Timestep jitter enabled: std={tt.jitter_std}")

        if tt.noise_augmentation_enabled:
            logger.info(f"Noise augmentation enabled: std={tt.noise_augmentation_std}")

        if tt.self_cond_enabled:
            logger.info(f"Self-conditioning enabled: prob={tt.self_cond_prob}")

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

    def _create_score_aug(self, cfg: dict | None = None):
        """Create dimension-appropriate ScoreAug transform.

        Args:
            cfg: ScoreAug configuration dict (deprecated, use typed config).

        Returns:
            ScoreAugTransform instance, or None if disabled.
        """
        from .diffusion_config import ScoreAugConfig

        if cfg is not None:
            # Legacy dict path
            if not cfg.get('enabled', False):
                return None
            sa = cfg
        elif hasattr(self, '_diffusion_config'):
            sa = self._diffusion_config.score_aug
            if not sa.enabled:
                return None
        else:
            sa = ScoreAugConfig.from_hydra(self.cfg)
            if not sa.enabled:
                return None

        from medgen.augmentation import ScoreAugTransform

        # Support both dict and ScoreAugConfig
        def _get(key, default=None):
            if isinstance(sa, dict):
                return sa.get(key, default)
            return getattr(sa, key, default)

        return ScoreAugTransform(
            spatial_dims=self.spatial_dims,
            rotation=_get('rotation', True),
            flip=_get('flip', True),
            translation=_get('translation', False),
            cutout=_get('cutout', False),
            compose=_get('compose', False),
            compose_prob=_get('compose_prob', 0.5),
            v2_mode=_get('v2_mode', False),
            nondestructive_prob=_get('nondestructive_prob', 0.5),
            destructive_prob=_get('destructive_prob', 0.5),
            cutout_vs_pattern=_get('cutout_vs_pattern', 0.5),
            patterns_checkerboard=_get('patterns_checkerboard', True),
            patterns_grid_dropout=_get('patterns_grid_dropout', True),
            patterns_coarse_dropout=_get('patterns_coarse_dropout', True),
            patterns_patch_dropout=_get('patterns_patch_dropout', True),
        )

    def _create_sda(self, cfg: dict | None = None):
        """Create dimension-appropriate SDA transform.

        Args:
            cfg: SDA configuration dict (deprecated, use typed config).

        Returns:
            SDATransform instance, or None if disabled.
        """
        from .diffusion_config import SDAConfig

        if cfg is not None:
            # Legacy dict path
            if not cfg.get('enabled', False):
                return None
            prob = cfg.get('probability', 0.5)
        elif hasattr(self, '_diffusion_config'):
            sda = self._diffusion_config.sda
            if not sda.enabled:
                return None
            prob = sda.prob
        else:
            sda = SDAConfig.from_hydra(self.cfg)
            if not sda.enabled:
                return None
            prob = sda.prob

        if self.spatial_dims == 2:
            from medgen.augmentation import SDATransform
            return SDATransform(probability=prob)
        else:
            from medgen.augmentation import SDATransform3D
            return SDATransform3D(probability=prob)
