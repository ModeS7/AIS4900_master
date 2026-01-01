"""
Diffusion model trainer module.

This module provides the DiffusionTrainer class for training diffusion models
with various strategies (DDPM, Rectified Flow) and modes (segmentation,
conditional single, conditional dual).
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
from .optimizers import SAM
from medgen.models import create_diffusion_model, get_model_type, is_transformer_model
from .losses import PerceptualLoss
from .modes import ConditionalDualMode, ConditionalSingleMode, MultiModalityMode, SegmentationMode, TrainingMode
from .strategies import DDPMStrategy, RFlowStrategy, DiffusionStrategy
from .visualization import ValidationVisualizer
from .spaces import DiffusionSpace, PixelSpace
from .utils import (
    get_vram_usage,
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


class DiffusionTrainer:
    """Unified diffusion model trainer composing strategy and mode.

    This trainer supports multiple diffusion strategies (DDPM, Rectified Flow)
    and training modes (segmentation, conditional single, conditional dual).
    It handles distributed training, mixed precision, and checkpoint management.

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
        self.cfg = cfg
        self.space = space if space is not None else PixelSpace()

        # Extract config values
        self.strategy_name: str = cfg.strategy.name
        self.mode_name: str = cfg.mode.name
        self.n_epochs: int = cfg.training.epochs
        self.batch_size: int = cfg.training.batch_size
        self.image_size: int = cfg.model.image_size
        self.learning_rate: float = cfg.training.learning_rate
        # Disable perceptual loss for seg mode (pretrained features don't apply to binary masks)
        self.perceptual_weight: float = 0.0 if self.mode_name == 'seg' else cfg.training.perceptual_weight
        self.num_timesteps: int = cfg.strategy.num_train_timesteps
        self.warmup_epochs: int = cfg.training.warmup_epochs
        self.eta_min: float = cfg.training.get('eta_min', 1e-6)
        # Compute val_interval: num_validations takes priority over val_interval
        num_validations = cfg.training.get('num_validations', None)
        if num_validations and num_validations > 0:
            self.val_interval: int = max(1, self.n_epochs // num_validations)
        else:
            self.val_interval: int = cfg.training.val_interval
        self.use_multi_gpu: bool = cfg.training.use_multi_gpu
        self.use_ema: bool = cfg.training.use_ema
        self.ema_decay: float = cfg.training.ema.decay
        self.use_min_snr: bool = cfg.training.use_min_snr
        self.min_snr_gamma: float = cfg.training.min_snr_gamma

        # SAM (Sharpness-Aware Minimization)
        sam_cfg = cfg.training.get('sam', {})
        self.use_sam: bool = sam_cfg.get('enabled', False)
        self.sam_rho: float = sam_cfg.get('rho', 0.05)
        self.sam_adaptive: bool = sam_cfg.get('adaptive', False)

        # Logging options
        logging_cfg = cfg.training.get('logging', {})
        self.log_grad_norm: bool = logging_cfg.get('grad_norm', True)
        self.log_psnr: bool = logging_cfg.get('psnr', True)
        self.log_lpips: bool = logging_cfg.get('lpips', True)
        self.log_msssim: bool = logging_cfg.get('msssim', True)
        self.log_regional_losses: bool = logging_cfg.get('regional_losses', True)
        self.log_flops: bool = logging_cfg.get('flops', True)

        # Determine if running on cluster
        self.is_cluster: bool = (cfg.paths.name == "cluster")

        # Setup device and distributed training
        if self.use_multi_gpu:
            self.rank, self.local_rank, self.world_size, self.device = self._setup_distributed()
            self.is_main_process: bool = (self.rank == 0)
        else:
            self.device: torch.device = torch.device("cuda")
            self.is_main_process = True
            self.rank: int = 0
            self.world_size: int = 1

        # Initialize strategy and mode
        self.strategy = self._create_strategy(self.strategy_name)
        self.mode = self._create_mode(self.mode_name)
        self.scheduler = self.strategy.setup_scheduler(self.num_timesteps, self.image_size)

        # Initialize logging and save directories
        if self.is_main_process:
            try:
                from hydra.core.hydra_config import HydraConfig
                self.save_dir = HydraConfig.get().runtime.output_dir
            except (ImportError, ValueError, AttributeError):
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                # Optional experiment name prefix from config (include underscore in value: "exp1_")
                exp_name = cfg.training.get('name', '')
                self.run_name = f"{exp_name}{self.strategy_name}_{self.image_size}_{timestamp}"
                # Structure: runs/diffusion_2d/{mode}/{run_name}
                self.save_dir = os.path.join(cfg.paths.model_dir, 'diffusion_2d', self.mode_name, self.run_name)

            tensorboard_dir = os.path.join(self.save_dir, "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer: Optional[SummaryWriter] = SummaryWriter(tensorboard_dir)
            self.best_loss: float = float('inf')
        else:
            self.writer = None
            self.run_name = ""
            self.save_dir = ""
            self.best_loss = float('inf')

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
        self.model: Optional[nn.Module] = None
        self.model_raw: Optional[nn.Module] = None
        self.ema: Optional[EMA] = None
        self.optimizer: Optional[AdamW] = None
        self.lr_scheduler: Optional[LRScheduler] = None
        self.perceptual_loss_fn: Optional[nn.Module] = None

        # Validation loader (set in train())
        self.val_loader: Optional[DataLoader] = None

        # FLOPs tracking using shared utility
        self._flops_tracker = FLOPsTracker()

        # Visualization helper (initialized in setup_model after strategy is ready)
        self.visualizer: Optional[ValidationVisualizer] = None

        # ScoreAug initialization (applies transforms to noisy data)
        # Reference: https://arxiv.org/abs/2508.07926
        self.score_aug = None
        self.use_omega_conditioning = False
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
                # Per paper: uniform sampling across enabled transforms + identity
                n_options = len(transforms) + 1  # +1 for identity
                logger.info(
                    f"ScoreAug enabled: transforms=[{', '.join(transforms)}], "
                    f"each with 1/{n_options} prob (uniform), "
                    f"omega_conditioning={self.use_omega_conditioning}"
                )

        # Mode embedding for multi-modality training
        self.use_mode_embedding = cfg.mode.get('use_mode_embedding', False)
        if self.use_mode_embedding and self.is_main_process:
            logger.info("Mode embedding enabled for multi-modality training")

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

        # Pass image_keys to ConditionalDualMode if configured
        if mode == ModeType.DUAL or mode == 'dual':
            image_keys = list(self.cfg.mode.image_keys) if 'image_keys' in self.cfg.mode else None
            return ConditionalDualMode(image_keys)

        # Pass image_keys to MultiModalityMode if configured
        if mode == 'multi':
            image_keys = list(self.cfg.mode.image_keys) if 'image_keys' in self.cfg.mode else None
            return MultiModalityMode(image_keys)

        return modes[mode]()

    def _setup_distributed(self) -> Tuple[int, int, int, torch.device]:
        """Setup distributed training with dynamic port allocation."""
        return setup_distributed()

    def setup_model(self, train_dataset: Dataset) -> None:
        """Initialize model, optimizer, and loss functions."""
        model_cfg = self.mode.get_model_config()

        # Adjust channels for latent space (PixelSpace returns unchanged)
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
            # SiT / transformer models use factory
            raw_model = create_diffusion_model(self.cfg, self.device, in_channels, out_channels)

            # Warn if embedding wrappers requested (not supported for transformers yet)
            if self.use_omega_conditioning or self.use_mode_embedding:
                if self.is_main_process:
                    logger.warning(
                        "Omega/mode conditioning wrappers not yet supported for transformer models. "
                        "Disabling wrappers."
                    )
                self.use_omega_conditioning = False
                self.use_mode_embedding = False
        else:
            # UNet via MONAI
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
            # MONAI DiffusionModelUNet time_embed output dim is 4 * channels[0]
            channels = tuple(self.cfg.model.channels)
            time_embed_dim = 4 * channels[0]
        else:
            time_embed_dim = None  # Not used for transformers

        # Handle embedding wrappers: omega, mode, or both
        # Wrap FIRST (modifies time_embed), then compile INNER model
        # This keeps encoding logic (with data-dependent branches) outside compiled graph
        if not self.is_transformer and self.use_omega_conditioning and self.use_mode_embedding:
            # Both omega + mode embedding
            from medgen.data.combined_embed import CombinedModelWrapper
            wrapper = CombinedModelWrapper(raw_model, embed_dim=time_embed_dim).to(self.device)
            if self.is_main_process:
                logger.info(f"Combined: wrapped model with omega + mode conditioning (embed_dim={time_embed_dim})")

            if use_compile:
                wrapper.model = torch.compile(wrapper.model, mode="default")
                if self.is_main_process:
                    logger.info("Single-GPU: Compiled inner UNet (combined wrapper uncompiled)")

            self.model = wrapper
            self.model_raw = wrapper

        elif not self.is_transformer and self.use_omega_conditioning:
            # Omega conditioning only (UNet)
            from medgen.data.score_aug import ScoreAugModelWrapper
            wrapper = ScoreAugModelWrapper(raw_model, embed_dim=time_embed_dim).to(self.device)
            if self.is_main_process:
                logger.info(f"ScoreAug: wrapped model with omega conditioning (embed_dim={time_embed_dim})")

            if use_compile:
                wrapper.model = torch.compile(wrapper.model, mode="default")
                if self.is_main_process:
                    logger.info("Single-GPU: Compiled inner UNet (omega wrapper uncompiled)")

            self.model = wrapper
            self.model_raw = wrapper

        elif not self.is_transformer and self.use_mode_embedding:
            # Mode embedding only (UNet)
            from medgen.data.mode_embed import ModeEmbedModelWrapper
            wrapper = ModeEmbedModelWrapper(raw_model, embed_dim=time_embed_dim).to(self.device)
            if self.is_main_process:
                logger.info(f"Mode embedding: wrapped model with mode conditioning (embed_dim={time_embed_dim})")

            if use_compile:
                wrapper.model = torch.compile(wrapper.model, mode="default")
                if self.is_main_process:
                    logger.info("Single-GPU: Compiled inner UNet (mode wrapper uncompiled)")

            self.model = wrapper
            self.model_raw = wrapper

        else:
            # No special conditioning - use standard wrap_model_for_training
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
                    "Embeddings will NOT be synchronized across GPUs. "
                    "For multi-GPU training with embeddings, consider using FSDP or single-GPU."
                )

        # Setup perceptual loss (shared wrapper handles multi-channel inputs)
        cache_dir = getattr(self.cfg.paths, 'cache_dir', None)
        self.perceptual_loss_fn = PerceptualLoss(
            spatial_dims=2,
            network_type="radimagenet_resnet50",
            cache_dir=cache_dir,
            pretrained=True,
            device=self.device,
            use_compile=use_compile,
        )

        # Compile fused forward pass (disabled for latent space, Min-SNR, and DDP)
        # - Latent space: perceptual loss needs pixel space decoding
        # - Min-SNR: requires per-sample loss weighting (compiled version incorrectly weights total loss)
        # - DDP: torch.compile can't trace DDP's internal logging (set_runtime_stats_and_log)
        compile_fused = self.cfg.training.get('compile_fused_forward', True)
        if self.use_multi_gpu:
            compile_fused = False
            if self.is_main_process:
                logger.info("Disabled compiled fused forward for DDP (multi-GPU)")
        elif self.space.scale_factor > 1:
            compile_fused = False
            if self.is_main_process:
                logger.info("Disabled compiled fused forward for latent space")
        elif self.use_min_snr:
            compile_fused = False
            if self.is_main_process:
                logger.info("Disabled compiled fused forward for Min-SNR weighting")
        elif self.score_aug is not None:
            compile_fused = False
            if self.is_main_process:
                logger.info("Disabled compiled fused forward for ScoreAug")
        elif self.use_sam:
            compile_fused = False
            if self.is_main_process:
                logger.info("Disabled compiled fused forward for SAM (incompatible with CUDA graphs)")
        elif compile_fused and self.is_main_process:
            logger.info("Compiling fused forward pass")
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

        # Warmup + Cosine scheduler (using shared utility)
        # Note: SAM wraps the base optimizer, scheduler works with base_optimizer's param_groups
        self.lr_scheduler = create_warmup_cosine_scheduler(
            self.optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=self.n_epochs,
            eta_min=self.eta_min,
        )

        # Create EMA wrapper if enabled
        if self.use_ema:
            self.ema = EMA(
                self.model_raw,
                beta=self.ema_decay,
                update_after_step=self.cfg.training.ema.update_after_step,
                update_every=self.cfg.training.ema.update_every,
            )
            if self.is_main_process:
                logger.info(f"EMA enabled with decay={self.ema_decay}")

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

        # Initialize metrics accumulators (explicit, not lazy)
        self.metrics.init_accumulators()

    def _setup_compiled_forward(self, enabled: bool) -> None:
        """Setup compiled forward functions for fused model + loss computation."""
        self._use_compiled_forward = enabled

        if not enabled:
            self._compiled_forward_single = None
            self._compiled_forward_dual = None
            return

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
            """Fused forward: model prediction + MSE loss + perceptual loss."""
            prediction = model(model_input, timesteps)

            if strategy_name == 'rflow':
                t_normalized = timesteps.float() / float(num_train_timesteps)
                t_expanded = t_normalized.view(-1, 1, 1, 1)
                predicted_clean = torch.clamp(noisy_images + t_expanded * prediction, 0, 1)
            else:
                predicted_clean = torch.clamp(noisy_images - prediction, 0, 1)

            if strategy_name == 'rflow':
                target = images - noise
            else:
                target = noise
            mse_loss = ((prediction - target) ** 2).mean()

            p_loss = perceptual_fn(predicted_clean.float(), images.float())
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
            """Fused forward for dual mode with separate tensor inputs."""
            prediction = model(model_input, timesteps)
            pred_0 = prediction[:, 0:1, :, :]
            pred_1 = prediction[:, 1:2, :, :]

            if strategy_name == 'rflow':
                t_normalized = timesteps.float() / float(num_train_timesteps)
                t_expanded = t_normalized.view(-1, 1, 1, 1)
                clean_0 = torch.clamp(noisy_0 + t_expanded * pred_0, 0, 1)
                clean_1 = torch.clamp(noisy_1 + t_expanded * pred_1, 0, 1)
            else:
                clean_0 = torch.clamp(noisy_0 - pred_0, 0, 1)
                clean_1 = torch.clamp(noisy_1 - pred_1, 0, 1)

            if strategy_name == 'rflow':
                target_0 = images_0 - noise_0
                target_1 = images_1 - noise_1
            else:
                target_0 = noise_0
                target_1 = noise_1

            mse_0 = ((pred_0 - target_0) ** 2).mean()
            mse_1 = ((pred_1 - target_1) ** 2).mean()
            mse_loss = (mse_0 + mse_1) / 2.0

            p_0 = perceptual_fn(clean_0.float(), images_0.float())
            p_1 = perceptual_fn(clean_1.float(), images_1.float())
            p_loss = (p_0 + p_1) / 2.0

            total_loss = mse_loss + perceptual_weight * p_loss

            return total_loss, mse_loss, p_loss, clean_0, clean_1

        self._compiled_forward_single = torch.compile(
            _forward_single, mode="reduce-overhead", fullgraph=True
        )
        self._compiled_forward_dual = torch.compile(
            _forward_dual, mode="reduce-overhead", fullgraph=True
        )

        if self.is_main_process:
            logger.info(f"Compiled fused forward functions for mode: {self.mode_name}")

    def _save_metadata(self) -> None:
        """Save training configuration to metadata.json."""
        os.makedirs(self.save_dir, exist_ok=True)

        config_path = os.path.join(self.save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        metadata = {
            'strategy': self.strategy_name,
            'mode': self.mode_name,
            'epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'image_size': self.image_size,
            'learning_rate': self.learning_rate,
            'perceptual_weight': self.perceptual_weight,
            'num_timesteps': self.num_timesteps,
            'warmup_epochs': self.warmup_epochs,
            'val_interval': self.val_interval,
            'multi_gpu': self.use_multi_gpu,
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay if self.use_ema else None,
            'use_min_snr': self.use_min_snr,
            'min_snr_gamma': self.min_snr_gamma if self.use_min_snr else None,
            'model': {
                'channels': list(self.cfg.model.channels),
                'attention_levels': list(self.cfg.model.attention_levels),
                'num_res_blocks': self.cfg.model.num_res_blocks,
                'num_head_channels': self.cfg.model.num_head_channels,
            },
            'created_at': datetime.now().isoformat(),
        }

        metadata_path = os.path.join(self.save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Config saved to: {config_path}")

    def _get_model_config(self) -> Dict[str, Any]:
        """Get model architecture config for checkpoint saving."""
        model_cfg = self.mode.get_model_config()
        return {
            'channels': list(self.cfg.model.channels),
            'attention_levels': list(self.cfg.model.attention_levels),
            'num_res_blocks': self.cfg.model.num_res_blocks,
            'num_head_channels': self.cfg.model.num_head_channels,
            'in_channels': model_cfg['in_channels'],
            'out_channels': model_cfg['out_channels'],
        }

    def _update_metadata_final(self, final_loss: float, final_mse: float, total_time: float) -> None:
        """Update metadata.json with final training results."""
        metadata_path = os.path.join(self.save_dir, 'metadata.json')

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        metadata['results'] = {
            'final_loss': float(final_loss),
            'final_mse': float(final_mse),
            'best_loss': float(self.best_loss),
            'total_time_seconds': float(total_time),
            'total_time_hours': float(total_time / 3600),
            'completed_at': datetime.now().isoformat(),
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

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

        if isinstance(images, dict):
            keys = list(images.keys())
            if self.strategy_name == 'rflow':
                target_0 = images[keys[0]] - noise[keys[0]]
                target_1 = images[keys[1]] - noise[keys[1]]
            else:
                target_0, target_1 = noise[keys[0]], noise[keys[1]]
            pred_0, pred_1 = prediction[:, 0:1, :, :], prediction[:, 1:2, :, :]
            mse_0 = ((pred_0 - target_0) ** 2).mean(dim=(1, 2, 3))
            mse_1 = ((pred_1 - target_1) ** 2).mean(dim=(1, 2, 3))
            mse_per_sample = (mse_0 + mse_1) / 2
        else:
            target = images - noise if self.strategy_name == 'rflow' else noise
            mse_per_sample = ((prediction - target) ** 2).mean(dim=(1, 2, 3))

        return (mse_per_sample * snr_weights).mean()

    def train_step(self, batch: torch.Tensor) -> Tuple[float, float, float]:
        """Execute a single training step."""
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

                    # Get prediction from augmented input
                    if self.use_omega_conditioning and self.use_mode_embedding:
                        # Model is CombinedModelWrapper, pass both omega and mode_id
                        prediction = self.model(aug_input, timesteps, omega=omega, mode_id=mode_id)
                    elif self.use_omega_conditioning:
                        # Model is ScoreAugModelWrapper, pass omega for conditioning
                        prediction = self.model(aug_input, timesteps, omega=omega)
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
                                # This is expected when omega contains rotation/flip codes
                                if self.perceptual_weight > 0:
                                    logger.debug("Perceptual loss skipped: non-invertible ScoreAug transform applied")
                                p_loss = torch.tensor(0.0, device=self.device)
                                predicted_clean = aug_clean  # Use augmented for metrics
                            else:
                                predicted_clean = inv_clean
                                p_loss = self.perceptual_loss_fn(predicted_clean, images)
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
                                p_loss = self.perceptual_loss_fn(predicted_clean, images)
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
                        p_loss = self.perceptual_loss_fn(pred_decoded, images_decoded)
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
                        prediction_2 = self.model(aug_input, timesteps, omega=omega)
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
                        p_loss_2 = self.perceptual_loss_fn(pred_decoded_2, images_decoded_2)
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

            return total_loss_val, mse_loss_val, p_loss_val
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

            return total_loss.item(), mse_loss.item(), p_loss.item()

    def _measure_model_flops(self, train_loader: DataLoader) -> None:
        """Measure model FLOPs using the first batch (one-time profiling)."""
        if not self.metrics.log_flops:
            return

        try:
            # Get first batch
            batch = next(iter(train_loader))
            prepared = self.mode.prepare_batch(batch, self.device)
            images = prepared['images']
            labels_dict = {'labels': prepared.get('labels')}

            # Create sample input like in train_step
            if isinstance(images, dict):
                noise = {key: torch.randn_like(img).to(self.device) for key, img in images.items()}
            else:
                noise = torch.randn_like(images).to(self.device)

            timesteps = self.strategy.sample_timesteps(images)
            noisy_images = self.strategy.add_noise(images, noise, timesteps)
            model_input = self.mode.format_model_input(noisy_images, labels_dict)

            # Measure FLOPs using shared utility
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

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Tuple[float, float, float]:
        """Train the model for one epoch."""
        self.model.train()
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_perceptual_loss = 0

        # Create progress bar iterator (tqdm for main process on non-cluster, plain iterator otherwise)
        epoch_iter = create_epoch_iterator(data_loader, epoch, self.is_cluster, self.is_main_process)

        for step, batch in enumerate(epoch_iter):
            loss, mse_loss, p_loss = self.train_step(batch)

            epoch_loss += loss
            epoch_mse_loss += mse_loss
            epoch_perceptual_loss += p_loss

            # Update progress bar if available (tqdm instance has set_postfix)
            if hasattr(epoch_iter, 'set_postfix'):
                epoch_iter.set_postfix(loss=f"{epoch_loss / (step + 1):.6f}")

            if epoch == 1 and step == 0 and self.is_main_process:
                logger.info(get_vram_usage(self.device))

        return epoch_loss / len(data_loader), epoch_mse_loss / len(data_loader), epoch_perceptual_loss / len(data_loader)

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
        # when perceptual loss (compiled with torch.compile) is called during validation
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
                    p_loss = self.perceptual_loss_fn(pred_decoded, images_decoded)
                else:
                    p_loss = torch.tensor(0.0, device=self.device)

                loss = mse_loss + self.perceptual_weight * p_loss
                loss_val = loss.item()

                total_mse += mse_loss.item()
                total_perc += p_loss.item()
                total_loss += loss_val

                # Track worst batch (only from full-sized batches to ensure consistent visualization)
                if loss_val > worst_loss and current_batch_size >= min_batch_size:
                    worst_loss = loss_val
                    # Convert to dict format for dual mode (consistent with VAE and create_reconstruction_figure)
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

                # Quality metrics (MS-SSIM, PSNR, optionally LPIPS) on predicted vs ground truth
                # Decode to pixel space for latent diffusion (metrics must be in pixel space)
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

        return metrics, worst_batch_data

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
                    mode_id = prepared.get('mode_id')  # For multi-modality mode

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
                    if self.use_mode_embedding and mode_id is not None:
                        prediction = model_to_use(model_input, timesteps, mode_id=mode_id)
                    else:
                        prediction = self.strategy.predict_noise_or_velocity(model_to_use, model_input, timesteps)
                    _, predicted_clean = self.strategy.compute_loss(prediction, images, noise, noisy_images, timesteps)

                    # Decode to pixel space for latent diffusion (metrics must be in pixel space)
                    if self.space.scale_factor > 1:
                        metrics_pred = self.space.decode(predicted_clean)
                        metrics_gt = self.space.decode(images)
                    else:
                        metrics_pred = predicted_clean
                        metrics_gt = images

                    # Compute metrics
                    total_psnr += compute_psnr(metrics_pred, metrics_gt)
                    if self.metrics.log_lpips:
                        total_lpips += compute_lpips(metrics_pred, metrics_gt, device=self.device)
                    total_msssim += compute_msssim(metrics_pred, metrics_gt)

                    # Regional tracking (tumor vs background)
                    if regional_tracker is not None and labels is not None:
                        regional_tracker.update(predicted_clean, images, labels)

                    n_batches += 1

            # Compute averages and log
            if n_batches > 0 and self.writer is not None:
                avg_psnr = total_psnr / n_batches
                avg_msssim = total_msssim / n_batches
                self.writer.add_scalar(f'Validation/PSNR_{modality}', avg_psnr, epoch)
                self.writer.add_scalar(f'Validation/MS-SSIM_{modality}', avg_msssim, epoch)
                if self.metrics.log_lpips:
                    avg_lpips = total_lpips / n_batches
                    self.writer.add_scalar(f'Validation/LPIPS_{modality}', avg_lpips, epoch)

                # Log regional metrics for this modality
                if regional_tracker is not None:
                    regional_tracker.log_to_tensorboard(
                        self.writer, epoch, prefix=f'regional_{modality}'
                    )

        model_to_use.train()

    def train(
        self,
        train_loader: DataLoader,
        train_dataset: Dataset,
        val_loader: Optional[DataLoader] = None
    ) -> None:
        """Execute the main training loop.

        Args:
            train_loader: Training dataloader.
            train_dataset: Training dataset (for visualization samples).
            val_loader: Optional validation dataloader for computing validation losses.
        """
        total_start = time.time()

        # Store validation loader
        self.val_loader = val_loader

        # Measure FLOPs using first batch (one-time profiling)
        self._measure_model_flops(train_loader)

        # Log training config
        if self.is_main_process and self.mode_name == 'seg':
            logger.info("Seg mode: perceptual loss disabled (pretrained features don't apply to binary masks)")

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

                    if self.writer is not None:
                        self.writer.add_scalar('Loss/Total_train', avg_loss, epoch)
                        self.writer.add_scalar('Loss/MSE_train', avg_mse, epoch)
                        self.writer.add_scalar('Loss/Perceptual_train', avg_perceptual, epoch)
                        self.writer.add_scalar('LR', self.lr_scheduler.get_last_lr()[0], epoch)

                    # Compute validation losses every epoch
                    val_metrics, worst_val_data = self.compute_validation_losses(epoch)

                    # Log metrics (grad norms every epoch, others at val_interval)
                    is_val_epoch = (epoch + 1) % self.val_interval == 0
                    self.metrics.log_epoch(epoch, log_all=is_val_epoch)

                    # Log FLOPs
                    self._flops_tracker.log_epoch(self.writer, epoch)

                    # Log worst validation batch at val_interval
                    if is_val_epoch and worst_val_data is not None:
                        self.visualizer.log_worst_batch(epoch, worst_val_data)

                    # Per-modality validation metrics (for multi-modality training)
                    if is_val_epoch:
                        self._compute_per_modality_validation(epoch)

                    # Generate samples at val_interval
                    if is_val_epoch or (epoch + 1) == self.n_epochs:
                        model_to_use = self.ema.ema_model if self.ema is not None else self.model_raw
                        self.visualizer.generate_samples(model_to_use, train_dataset, epoch)

                    model_config = self._get_model_config()

                    # Save checkpoints every epoch
                    save_full_checkpoint(
                        self.model_raw, self.optimizer, epoch, self.save_dir, "latest",
                        model_config=model_config, scheduler=self.lr_scheduler, ema=self.ema
                    )

                    # Use validation loss for best model selection (fallback to train if no val_loader)
                    loss_for_selection = val_metrics.get('total', avg_loss)
                    if loss_for_selection < self.best_loss:
                        self.best_loss = loss_for_selection
                        save_full_checkpoint(
                            self.model_raw, self.optimizer, epoch, self.save_dir, "best",
                            model_config=model_config, scheduler=self.lr_scheduler, ema=self.ema
                        )
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

    def close_writer(self) -> None:
        """Close TensorBoard writer. Call after all logging is complete."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

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
            checkpoint_path = os.path.join(self.save_dir, f"{checkpoint_name}.pt")
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

        # Use EMA model if available (whether from current state or loaded from checkpoint)
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

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test evaluation", ncols=100, disable=self.is_cluster):
                prepared = self.mode.prepare_batch(batch, self.device)
                images = prepared['images']
                labels = prepared.get('labels')
                mode_id = prepared.get('mode_id')  # For multi-modality mode
                batch_size = images[list(images.keys())[0]].shape[0] if isinstance(images, dict) else images.shape[0]

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

                with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    if self.use_mode_embedding and mode_id is not None:
                        prediction = model_to_use(model_input, timesteps, mode_id=mode_id)
                    else:
                        prediction = self.strategy.predict_noise_or_velocity(model_to_use, model_input, timesteps)
                    mse_loss, predicted_clean = self.strategy.compute_loss(prediction, images, noise, noisy_images, timesteps)

                # Compute metrics
                loss_val = mse_loss.item()
                total_mse += loss_val

                # Decode to pixel space for latent diffusion (metrics must be in pixel space)
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

                # Track worst batch (only from full-sized batches)
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
