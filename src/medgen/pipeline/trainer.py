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
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from ema_pytorch import EMA
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from monai.networks.nets import DiffusionModelUNet

from medgen.core import ModeType, setup_distributed, create_warmup_cosine_scheduler, wrap_model_for_training
from .base_trainer import BaseTrainer
from .optimizers import SAM
from .results import TrainingStepResult
from medgen.models import create_diffusion_model, get_model_type, is_transformer_model
from .losses import PerceptualLoss
from .modes import ConditionalDualMode, ConditionalSingleMode, MultiModalityMode, SegmentationMode, TrainingMode
from .strategies import DDPMStrategy, RFlowStrategy, DiffusionStrategy
from .visualization import ValidationVisualizer
from .spaces import DiffusionSpace, PixelSpace
from .utils import (
    get_vram_usage,
    log_vram_to_tensorboard,
    log_epoch_summary,
    save_full_checkpoint,
    create_epoch_iterator,
)
from .metrics import (
    MetricsTracker,
    create_reconstruction_figure,
    RegionalMetricsTracker,
    compute_msssim,
    compute_psnr,
    compute_lpips,
)
from .tracking import FLOPsTracker

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

        # Perceptual weight (disabled for seg mode - binary masks)
        self.perceptual_weight: float = 0.0 if self.mode_name == 'seg' else cfg.training.perceptual_weight

        # Min-SNR weighting
        self.use_min_snr: bool = cfg.training.use_min_snr
        self.min_snr_gamma: float = cfg.training.min_snr_gamma

        # FP32 loss computation (set False to reproduce pre-Jan-7-2026 BF16 behavior)
        self.use_fp32_loss: bool = cfg.training.get('use_fp32_loss', True)
        logger.info(f"[DEBUG] use_fp32_loss = {self.use_fp32_loss} (from config override)")

        # SAM (Sharpness-Aware Minimization)
        sam_cfg = cfg.training.get('sam', {})
        self.use_sam: bool = sam_cfg.get('enabled', False)
        self.sam_rho: float = sam_cfg.get('rho', 0.05)
        self.sam_adaptive: bool = sam_cfg.get('adaptive', False)

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

        # Initialize model components (set during setup_model)
        self.perceptual_loss_fn: Optional[nn.Module] = None

        # Validation loader (set in train())
        self.val_loader: Optional[DataLoader] = None

        # Visualization helper (initialized in setup_model after strategy is ready)
        self.visualizer: Optional[ValidationVisualizer] = None

        # ScoreAug initialization (applies transforms to noisy data)
        self.score_aug = None
        self.use_omega_conditioning = False
        self.use_mode_intensity_scaling = False
        self._apply_mode_intensity_scale = None  # Function reference (lazy import)
        score_aug_cfg = cfg.training.get('score_aug', {})
        if score_aug_cfg.get('enabled', False):
            from medgen.data.score_aug import ScoreAugTransform
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
                from medgen.data.score_aug import apply_mode_intensity_scale
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

        # Mode embedding for multi-modality training
        self.use_mode_embedding = cfg.mode.get('use_mode_embedding', False)
        if self.use_mode_embedding and self.is_main_process:
            logger.info("Mode embedding enabled for multi-modality training")

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
                if self.is_main_process:
                    logger.warning(
                        "Omega/mode conditioning wrappers not yet supported for transformer models. "
                        "Disabling wrappers."
                    )
                self.use_omega_conditioning = False
                self.use_mode_embedding = False
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

        # Warn about DDP incompatibility with embedding wrappers
        if self.use_multi_gpu and (self.use_omega_conditioning or self.use_mode_embedding):
            if self.is_main_process:
                logger.warning(
                    "DDP is not compatible with embedding wrappers (ScoreAug, ModeEmbed). "
                    "Embeddings will NOT be synchronized across GPUs."
                )

        # Setup perceptual loss
        cache_dir = getattr(self.cfg.paths, 'cache_dir', None)
        self.perceptual_loss_fn = PerceptualLoss(
            spatial_dims=2,
            network_type="radimagenet_resnet50",
            cache_dir=cache_dir,
            pretrained=True,
            device=self.device,
            use_compile=use_compile,
        )

        # Compile fused forward pass setup
        compile_fused = self.cfg.training.get('compile_fused_forward', True)
        if self.use_multi_gpu:
            compile_fused = False
        elif self.space.scale_factor > 1:
            compile_fused = False
        elif self.use_min_snr:
            compile_fused = False
        elif self.score_aug is not None:
            compile_fused = False
        elif self.use_sam:
            compile_fused = False

        self._setup_compiled_forward(compile_fused)

        # Setup optimizer (with optional SAM wrapper)
        if self.use_sam:
            self.optimizer = SAM(
                self.model_raw.parameters(),
                base_optimizer=AdamW,
                rho=self.sam_rho,
                adaptive=self.sam_adaptive,
                lr=self.learning_rate,
            )
            if self.is_main_process:
                logger.info(f"Using SAM optimizer (rho={self.sam_rho}, adaptive={self.sam_adaptive})")
        else:
            self.optimizer = AdamW(self.model_raw.parameters(), lr=self.learning_rate)

        # Warmup + Cosine scheduler
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
        )

        # Save metadata
        if self.is_main_process:
            self._save_metadata()

        # Initialize metrics accumulators
        self.metrics.init_accumulators()

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

            # Encode to diffusion space (identity for PixelSpace)
            images = self.space.encode_batch(images)
            if labels is not None:
                labels = self.space.encode(labels)

            labels_dict = {'labels': labels}

            if isinstance(images, dict):
                noise = {key: torch.randn_like(img).to(self.device) for key, img in images.items()}
            else:
                noise = torch.randn_like(images).to(self.device)

            timesteps = self.strategy.sample_timesteps(images)
            noisy_images = self.strategy.add_noise(images, noise, timesteps)
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
                    if self.use_mode_embedding:
                        # Model is ModeEmbedModelWrapper, pass mode_id for conditioning
                        prediction = self.model(model_input, timesteps, mode_id=mode_id)
                    else:
                        prediction = self.strategy.predict_noise_or_velocity(self.model, model_input, timesteps)
                    mse_loss, predicted_clean = self.strategy.compute_loss(prediction, images, noise, noisy_images, timesteps)

                    if self.use_min_snr:
                        mse_loss = self._compute_min_snr_weighted_mse(
                            prediction, images, noise, timesteps
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
            self.optimizer.step()

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

        if self.writer is not None and self.is_main_process and not self.use_multi_gpu:
            self.writer.add_scalar('Loss/Total_train', avg_loss, epoch)
            self.writer.add_scalar('Loss/MSE_train', avg_mse, epoch)
            self.writer.add_scalar('Loss/Perceptual_train', avg_perceptual, epoch)
            self.writer.add_scalar('LR/Model', self.lr_scheduler.get_last_lr()[0], epoch)

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
        n_batches = 0

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
                n_batches += 1

                # Regional tracking (tumor vs background)
                if regional_tracker is not None and labels is not None:
                    regional_tracker.update(predicted_clean, images, labels)

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

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Loss/MSE_val', metrics['mse'], epoch)
            self.writer.add_scalar('Loss/Perceptual_val', metrics['perceptual'], epoch)
            self.writer.add_scalar('Loss/Total_val', metrics['total'], epoch)
            self.writer.add_scalar('Validation/MS-SSIM', metrics['msssim'], epoch)
            self.writer.add_scalar('Validation/PSNR', metrics['psnr'], epoch)
            if 'lpips' in metrics:
                self.writer.add_scalar('Validation/LPIPS', metrics['lpips'], epoch)

            # Log per-channel metrics for dual/multi modes
            for channel_key, channel_data in per_channel_metrics.items():
                count = channel_data['count']
                if count > 0:
                    self.writer.add_scalar(f'Validation/MS-SSIM_{channel_key}', channel_data['msssim'] / count, epoch)
                    self.writer.add_scalar(f'Validation/PSNR_{channel_key}', channel_data['psnr'] / count, epoch)
                    if self.metrics.log_lpips:
                        self.writer.add_scalar(f'Validation/LPIPS_{channel_key}', channel_data['lpips'] / count, epoch)

            # Log regional metrics (tumor vs background)
            if regional_tracker is not None:
                regional_tracker.log_to_tensorboard(self.writer, epoch, prefix='regional')

            # Compute and log volume 3D MS-SSIM
            if self.log_msssim:
                msssim_3d = self._compute_volume_3d_msssim(epoch, data_split='val')
                if msssim_3d is not None:
                    metrics['msssim_3d'] = msssim_3d
                    self.writer.add_scalar('Validation/MS-SSIM-3D', msssim_3d, epoch)

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

        # Measure FLOPs
        self._measure_model_flops(train_loader)

        if self.is_main_process and self.mode_name == 'seg':
            logger.info("Seg mode: perceptual loss disabled")

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

                    if self.writer is not None and self.use_multi_gpu:
                        self.writer.add_scalar('Loss/Total_train', avg_loss, epoch)
                        self.writer.add_scalar('Loss/MSE_train', avg_mse, epoch)
                        self.writer.add_scalar('Loss/Perceptual_train', avg_perceptual, epoch)
                        self.writer.add_scalar('LR/Model', self.lr_scheduler.get_last_lr()[0], epoch)

                    val_metrics, worst_val_data = self.compute_validation_losses(epoch)
                    log_figures = (epoch + 1) % self.figure_interval == 0

                    self.metrics.log_epoch(epoch, log_all=True)
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
        """Measure model FLOPs using the first batch."""
        if not self.metrics.log_flops:
            return

        try:
            batch = next(iter(train_loader))
            prepared = self.mode.prepare_batch(batch, self.device)
            images = prepared['images']
            labels_dict = {'labels': prepared.get('labels')}

            if isinstance(images, dict):
                noise = {key: torch.randn_like(img).to(self.device) for key, img in images.items()}
            else:
                noise = torch.randn_like(images).to(self.device)

            timesteps = self.strategy.sample_timesteps(images)
            noisy_images = self.strategy.add_noise(images, noise, timesteps)
            model_input = self.mode.format_model_input(noisy_images, labels_dict)

            self._flops_tracker.measure(
                self.model_raw,
                model_input,
                steps_per_epoch=len(train_loader),
                timesteps=timesteps,
                is_main_process=self.is_main_process,
            )
        except Exception as e:
            if self.is_main_process:
                logger.warning(f"Could not measure FLOPs: {e}")

    def _compute_per_modality_validation(self, epoch: int) -> None:
        """Compute per-modality validation metrics (placeholder)."""
        pass

    def _compute_volume_3d_msssim(self, epoch: int, data_split: str = 'val') -> Optional[float]:
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

        Returns:
            Average 3D MS-SSIM across all volumes, or None if unavailable.
        """
        if not self.log_msssim:
            return None

        # Import here to avoid circular imports
        from medgen.data.loaders.vae import create_vae_volume_validation_dataloader

        # Determine modality from mode
        # Use out_channels to determine volume channels (excludes conditioning)
        mode_name = self.cfg.mode.get('name', 'bravo')
        out_channels = self.cfg.mode.get('out_channels', 1)
        modality = 'dual' if out_channels > 1 else mode_name

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

        # Log to TensorBoard
        tb_prefix = f'test_{label}'
        if self.writer is not None:
            self.writer.add_scalar(f'{tb_prefix}/MSE', metrics['mse'], 0)
            self.writer.add_scalar(f'{tb_prefix}/MS-SSIM', metrics['msssim'], 0)
            self.writer.add_scalar(f'{tb_prefix}/PSNR', metrics['psnr'], 0)
            if 'lpips' in metrics:
                self.writer.add_scalar(f'{tb_prefix}/LPIPS', metrics['lpips'], 0)

            # Log timestep bin losses
            bin_size = self.num_timesteps // num_timestep_bins
            counts = timestep_loss_count.cpu()
            sums = timestep_loss_sum.cpu()
            for bin_idx in range(num_timestep_bins):
                bin_start = bin_idx * bin_size
                bin_end = (bin_idx + 1) * bin_size - 1
                count = counts[bin_idx].item()
                if count > 0:
                    avg_loss = (sums[bin_idx] / count).item()
                    self.writer.add_scalar(f'{tb_prefix}/Timestep/{bin_start}-{bin_end}', avg_loss, 0)

            # Log regional metrics
            if regional_tracker is not None:
                regional_tracker.log_to_tensorboard(self.writer, 0, prefix=f'{tb_prefix}_regional')

            # Compute and log volume 3D MS-SSIM
            if self.log_msssim:
                msssim_3d = self._compute_volume_3d_msssim(0, data_split='test_new')
                if msssim_3d is not None:
                    metrics['msssim_3d'] = msssim_3d
                    self.writer.add_scalar(f'{tb_prefix}/MS-SSIM-3D', msssim_3d, 0)

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
