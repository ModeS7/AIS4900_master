"""Shared typed configuration dataclasses for all trainers.

This module provides typed config dataclasses that replace scattered cfg.get()
calls throughout the codebase. Each config has a from_hydra() factory method
that centralizes default values in one place.

Dataclasses provided:
- PathsConfig: Data paths and environment detection
- ModelConfig: Diffusion model architecture (UNet or SiT)
- ModeConfig: Training mode (seg, bravo, dual, multi, etc.)
- StrategyConfig: Diffusion strategy (DDPM or RFlow)
- ProfilingConfig: PyTorch profiler settings
- BaseTrainingConfig: Core training parameters shared by all trainers
"""
import logging
from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# =============================================================================
# PathsConfig
# =============================================================================

@dataclass
class PathsConfig:
    """Data paths and environment configuration."""
    data_dir: str
    name: str = "local"
    model_dir: str = "runs"
    fov_mm: float = 240.0
    cache_dir: str = ".cache"

    @property
    def is_cluster(self) -> bool:
        return self.name == "cluster"

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'PathsConfig':
        """Extract paths config from Hydra DictConfig."""
        return cls(
            data_dir=cfg.paths.data_dir,
            name=cfg.paths.get('name', 'local'),
            model_dir=cfg.paths.get('model_dir', 'runs'),
            fov_mm=float(cfg.paths.get('fov_mm', 240.0)),
            cache_dir=cfg.paths.get('cache_dir', '.cache'),
        )


# =============================================================================
# ModelConfig
# =============================================================================

@dataclass
class ModelConfig:
    """Diffusion model architecture configuration."""
    type: str = "unet"
    spatial_dims: int = 2
    image_size: int = 128
    channels: list[int] = field(default_factory=lambda: [128, 256, 256])
    attention_levels: list[bool] = field(default_factory=lambda: [False, True, True])
    num_res_blocks: int = 1
    num_head_channels: int = 256
    # SiT-specific
    variant: str = "B"
    patch_size: int = 2
    conditioning: str = "concat"
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'ModelConfig':
        """Extract model config from Hydra DictConfig."""
        model_cfg = cfg.model
        return cls(
            type=model_cfg.get('type', 'unet'),
            spatial_dims=model_cfg.get('spatial_dims', 2),
            image_size=model_cfg.get('image_size', 128),
            channels=list(model_cfg.get('channels', [128, 256, 256])),
            attention_levels=list(model_cfg.get('attention_levels', [False, True, True])),
            num_res_blocks=model_cfg.get('num_res_blocks', 1),
            num_head_channels=model_cfg.get('num_head_channels', 256),
            variant=model_cfg.get('variant', 'B'),
            patch_size=model_cfg.get('patch_size', 2),
            conditioning=model_cfg.get('conditioning', 'concat'),
            mlp_ratio=model_cfg.get('mlp_ratio', 4.0),
            drop_rate=model_cfg.get('drop_rate', 0.0),
            drop_path_rate=model_cfg.get('drop_path_rate', 0.0),
        )


# =============================================================================
# ModeConfig
# =============================================================================

@dataclass
class ModeConfig:
    """Training mode configuration."""
    name: str
    is_conditional: bool = True
    in_channels: int = 2
    out_channels: int = 1
    image_keys: list[str] = field(default_factory=lambda: ["bravo"])
    cfg_dropout_prob: float = 0.0
    cond_channels: int | None = None
    # Mode embedding
    use_mode_embedding: bool = False
    mode_embedding_strategy: str = "full"
    mode_embedding_dropout: float = 0.2
    late_mode_start_level: int = 2
    # Size bin sub-config (seg_conditioned only)
    size_bins: dict | None = None
    # Latent seg conditioning
    latent_channels: int | None = None
    # Filesystem subdirectory (e.g., 'seg' for 'seg_conditioned')
    subdir: str | None = None

    @property
    def is_seg_mode(self) -> bool:
        return self.name in ('seg', 'seg_conditioned')

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'ModeConfig':
        """Extract mode config from Hydra DictConfig."""
        mode_cfg = cfg.mode
        size_bins_raw = mode_cfg.get('size_bins', None)
        size_bins = dict(size_bins_raw) if size_bins_raw else None

        return cls(
            name=mode_cfg.name,
            is_conditional=mode_cfg.get('is_conditional', True),
            in_channels=mode_cfg.get('in_channels', 2),
            out_channels=mode_cfg.get('out_channels', 1),
            image_keys=list(mode_cfg.get('image_keys', ['bravo'])),
            cfg_dropout_prob=mode_cfg.get('cfg_dropout_prob', 0.0),
            cond_channels=mode_cfg.get('cond_channels', None),
            use_mode_embedding=mode_cfg.get('use_mode_embedding', False),
            mode_embedding_strategy=mode_cfg.get('mode_embedding_strategy', 'full'),
            mode_embedding_dropout=mode_cfg.get('mode_embedding_dropout', 0.2),
            late_mode_start_level=mode_cfg.get('late_mode_start_level', 2),
            size_bins=size_bins,
            latent_channels=mode_cfg.get('latent_channels', None),
            subdir=mode_cfg.get('subdir', None),
        )


# =============================================================================
# StrategyConfig
# =============================================================================

@dataclass
class StrategyConfig:
    """Diffusion strategy configuration."""
    name: str
    num_train_timesteps: int = 1000
    prediction_type: str = "velocity"
    use_discrete_timesteps: bool = False
    sample_method: str = "logit-normal"
    use_timestep_transform: bool = True
    schedule: str = "cosine"

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'StrategyConfig':
        """Extract strategy config from Hydra DictConfig."""
        strat_cfg = cfg.strategy
        return cls(
            name=strat_cfg.name,
            num_train_timesteps=strat_cfg.get('num_train_timesteps', 1000),
            prediction_type=strat_cfg.get('prediction_type', 'velocity'),
            use_discrete_timesteps=strat_cfg.get('use_discrete_timesteps', False),
            sample_method=strat_cfg.get('sample_method', 'logit-normal'),
            use_timestep_transform=strat_cfg.get('use_timestep_transform', True),
            schedule=strat_cfg.get('schedule', 'cosine'),
        )


# =============================================================================
# ProfilingConfig
# =============================================================================

@dataclass
class ProfilingConfig:
    """PyTorch profiler configuration."""
    enabled: bool = False
    wait: int = 5
    warmup: int = 2
    active: int = 10
    repeat: int = 1
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    with_flops: bool = True

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'ProfilingConfig':
        """Extract profiling config from Hydra DictConfig."""
        profiling_cfg = cfg.training.get('profiling', {})
        return cls(
            enabled=profiling_cfg.get('enabled', False),
            wait=profiling_cfg.get('wait', 5),
            warmup=profiling_cfg.get('warmup', 2),
            active=profiling_cfg.get('active', 10),
            repeat=profiling_cfg.get('repeat', 1),
            record_shapes=profiling_cfg.get('record_shapes', True),
            profile_memory=profiling_cfg.get('profile_memory', True),
            with_stack=profiling_cfg.get('with_stack', False),
            with_flops=profiling_cfg.get('with_flops', True),
        )


# =============================================================================
# BaseTrainingConfig
# =============================================================================

@dataclass
class BaseTrainingConfig:
    """Core training parameters shared by ALL trainers (diffusion + compression).

    Covers fields used by BaseTrainer.__init__().
    """
    # Core
    n_epochs: int
    batch_size: int
    learning_rate: float = 1e-4
    warmup_epochs: int = 5
    gradient_clip_norm: float = 1.0
    limit_train_batches: int | None = None
    use_multi_gpu: bool = False
    # Figure interval
    num_figures: int = 20
    figure_interval: int | None = None
    # Verbosity (None = auto-detect from paths)
    verbose: bool | None = None
    # Logging sub-config
    log_grad_norm: bool = True
    log_psnr: bool = True
    log_lpips: bool = True
    log_msssim: bool = True
    log_regional_losses: bool = True
    log_flops: bool = True
    # Profiling sub-config
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    # Checkpoint management
    keep_last_n_checkpoints: int = 0
    # Experiment name (for fallback save dir)
    name: str = ""

    def get_figure_interval(self, n_epochs: int | None = None) -> int:
        """Compute figure interval from config."""
        epochs = n_epochs if n_epochs is not None else self.n_epochs
        if self.figure_interval is not None:
            return max(1, self.figure_interval)
        return max(1, epochs // max(1, self.num_figures))

    def get_verbose(self, is_cluster: bool) -> bool:
        """Resolve verbose setting (None = auto-detect)."""
        if self.verbose is not None:
            return self.verbose
        return not is_cluster

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'BaseTrainingConfig':
        """Extract base training config from Hydra DictConfig."""
        t = cfg.training
        logging_cfg = t.get('logging', {})

        # Support both 'epochs' (diffusion) and 'max_epochs' (compression) keys
        n_epochs = t.get('epochs', None) or t.get('max_epochs', None)
        if n_epochs is None:
            raise ValueError("Config must have training.epochs or training.max_epochs")

        return cls(
            # Core
            n_epochs=n_epochs,
            batch_size=t.batch_size,
            learning_rate=t.get('learning_rate', 1e-4),
            warmup_epochs=t.get('warmup_epochs', 5),
            gradient_clip_norm=t.get('gradient_clip_norm', 1.0),
            limit_train_batches=t.get('limit_train_batches', None),
            use_multi_gpu=t.get('use_multi_gpu', False),
            # Figure interval
            num_figures=t.get('num_figures', 20),
            figure_interval=t.get('figure_interval', None),
            # Verbosity
            verbose=t.get('verbose', None),
            # Logging
            log_grad_norm=logging_cfg.get('grad_norm', True),
            log_psnr=logging_cfg.get('psnr', True),
            log_lpips=logging_cfg.get('lpips', True),
            log_msssim=logging_cfg.get('msssim', True),
            log_regional_losses=logging_cfg.get('regional_losses', True),
            log_flops=logging_cfg.get('flops', True),
            # Profiling
            profiling=ProfilingConfig.from_hydra(cfg),
            # Checkpoint management
            keep_last_n_checkpoints=t.get('keep_last_n_checkpoints', 0),
            # Experiment name
            name=t.get('name', ''),
        )
