"""Configuration extraction for diffusion trainers.

This module provides:
- ScoreAugConfig: ScoreAug configuration dataclass
- SDAConfig: SDA configuration dataclass
- TrainingTricksConfig: Training tricks configuration dataclass
- ModeEmbeddingConfig: Mode embedding configuration dataclass
- SizeBinConfig: Size bin embedding configuration dataclass
- ControlNetConfig: ControlNet configuration dataclass
- GenerationMetricsConfig: Generation metrics configuration (re-export)
- DiffusionTrainerConfig: Complete diffusion trainer configuration dataclass
These classes consolidate the configuration extraction logic from DiffusionTrainer,
making it reusable and testable.
"""
import logging
from dataclasses import dataclass, field

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class ScoreAugConfig:
    """ScoreAug configuration."""
    enabled: bool = False
    rotation: bool = True
    flip: bool = True
    translation: bool = False
    cutout: bool = False
    compose: bool = False
    compose_prob: float = 0.5
    v2_mode: bool = False
    nondestructive_prob: float = 0.5
    destructive_prob: float = 0.5
    cutout_vs_pattern: float = 0.5
    patterns_checkerboard: bool = True
    patterns_grid_dropout: bool = True
    patterns_coarse_dropout: bool = True
    patterns_patch_dropout: bool = True
    use_omega_conditioning: bool = False
    use_mode_intensity_scaling: bool = False
    brightness: bool = False
    brightness_range: float = 1.2

    def __post_init__(self):
        for name in ('compose_prob', 'nondestructive_prob', 'destructive_prob', 'cutout_vs_pattern'):
            val = getattr(self, name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        if self.brightness_range <= 0:
            raise ValueError(f"brightness_range must be > 0, got {self.brightness_range}")
        if self.enabled:
            if (self.rotation or self.flip) and not self.use_omega_conditioning:
                raise ValueError(
                    "ScoreAug rotation/flip require omega conditioning. "
                    "Set training.score_aug.use_omega_conditioning=true"
                )
            if self.use_mode_intensity_scaling and not self.use_omega_conditioning:
                raise ValueError(
                    "Mode intensity scaling requires omega conditioning. "
                    "Set training.score_aug.use_omega_conditioning=true"
                )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'ScoreAugConfig':
        """Extract ScoreAug config from Hydra DictConfig."""
        score_aug_cfg = cfg.training.get('score_aug', {})
        patterns_cfg = score_aug_cfg.get('patterns', {})
        return cls(
            enabled=score_aug_cfg.get('enabled', False),
            rotation=score_aug_cfg.get('rotation', True),
            flip=score_aug_cfg.get('flip', True),
            translation=score_aug_cfg.get('translation', False),
            cutout=score_aug_cfg.get('cutout', False),
            compose=score_aug_cfg.get('compose', False),
            compose_prob=score_aug_cfg.get('compose_prob', 0.5),
            v2_mode=score_aug_cfg.get('v2_mode', False),
            nondestructive_prob=score_aug_cfg.get('nondestructive_prob', 0.5),
            destructive_prob=score_aug_cfg.get('destructive_prob', 0.5),
            cutout_vs_pattern=score_aug_cfg.get('cutout_vs_pattern', 0.5),
            patterns_checkerboard=patterns_cfg.get(
                'checkerboard', score_aug_cfg.get('patterns_checkerboard', True)),
            patterns_grid_dropout=patterns_cfg.get(
                'grid_dropout', score_aug_cfg.get('patterns_grid_dropout', True)),
            patterns_coarse_dropout=patterns_cfg.get(
                'coarse_dropout', score_aug_cfg.get('patterns_coarse_dropout', True)),
            patterns_patch_dropout=patterns_cfg.get(
                'patch_dropout', score_aug_cfg.get('patterns_patch_dropout', True)),
            use_omega_conditioning=score_aug_cfg.get('use_omega_conditioning', False),
            use_mode_intensity_scaling=score_aug_cfg.get('mode_intensity_scaling', False),
            brightness=score_aug_cfg.get('brightness', False),
            brightness_range=score_aug_cfg.get('brightness_range', 1.2),
        )


@dataclass
class SDAConfig:
    """SDA (Shifted Data Augmentation) configuration."""
    enabled: bool = False
    rotation: bool = True
    flip: bool = True
    noise_shift: float = 0.1
    prob: float = 0.5
    weight: float = 1.0

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'SDAConfig':
        """Extract SDA config from Hydra DictConfig."""
        sda_cfg = cfg.training.get('sda', {})
        return cls(
            enabled=sda_cfg.get('enabled', False),
            rotation=sda_cfg.get('rotation', True),
            flip=sda_cfg.get('flip', True),
            noise_shift=sda_cfg.get('noise_shift', 0.1),
            prob=sda_cfg.get('prob', 0.5),
            weight=sda_cfg.get('weight', 1.0),
        )


@dataclass
class GradientNoiseConfig:
    """Gradient noise injection configuration."""
    enabled: bool = False
    sigma: float = 0.01
    decay: float = 0.55

    def __post_init__(self):
        if not self.enabled:
            return
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if self.decay <= 0:
            raise ValueError(f"decay must be > 0, got {self.decay}")


@dataclass
class CurriculumConfig:
    """Curriculum timestep scheduling configuration."""
    enabled: bool = False
    warmup_epochs: int = 50
    min_t_start: float = 0.0
    max_t_start: float = 0.3
    min_t_end: float = 0.0
    max_t_end: float = 1.0

    def __post_init__(self):
        if not self.enabled:
            return
        if self.warmup_epochs <= 0:
            raise ValueError(f"warmup_epochs must be > 0, got {self.warmup_epochs}")
        if self.min_t_start > self.max_t_start:
            raise ValueError(f"min_t_start ({self.min_t_start}) > max_t_start ({self.max_t_start})")
        if self.min_t_end > self.max_t_end:
            raise ValueError(f"min_t_end ({self.min_t_end}) > max_t_end ({self.max_t_end})")
        for name, val in [('min_t_start', self.min_t_start), ('max_t_start', self.max_t_start),
                          ('min_t_end', self.min_t_end), ('max_t_end', self.max_t_end)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")


@dataclass
class JitterConfig:
    """Timestep jitter configuration."""
    enabled: bool = False
    std: float = 0.05

    def __post_init__(self):
        if not self.enabled:
            return
        if self.std <= 0:
            raise ValueError(f"std must be > 0, got {self.std}")


@dataclass
class MinSNRConfig:
    """Min-SNR loss weighting configuration."""
    enabled: bool = False
    gamma: float = 5.0

    def __post_init__(self):
        if not self.enabled:
            return
        if self.gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}")


@dataclass
class SelfCondConfig:
    """Self-conditioning via consistency configuration."""
    enabled: bool = False
    prob: float = 0.5
    consistency_weight: float = 0.1

    def __post_init__(self):
        if not self.enabled:
            return
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {self.prob}")
        if self.consistency_weight < 0:
            raise ValueError(f"consistency_weight must be >= 0, got {self.consistency_weight}")


@dataclass
class FeaturePerturbationConfig:
    """Feature perturbation configuration."""
    enabled: bool = False
    std: float = 0.1
    layers: list[str] = field(default_factory=lambda: ['mid'])

    def __post_init__(self):
        if not self.enabled:
            return
        if self.std <= 0:
            raise ValueError(f"std must be > 0, got {self.std}")


@dataclass
class NoiseAugConfig:
    """Input noise augmentation configuration."""
    enabled: bool = False
    std: float = 0.1

    def __post_init__(self):
        if not self.enabled:
            return
        if self.std <= 0:
            raise ValueError(f"std must be > 0, got {self.std}")


@dataclass
class TrainingTricksConfig:
    """Training tricks configuration with nested sub-configs."""
    gradient_noise: GradientNoiseConfig = field(default_factory=GradientNoiseConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    jitter: JitterConfig = field(default_factory=JitterConfig)
    min_snr: MinSNRConfig = field(default_factory=MinSNRConfig)
    self_cond: SelfCondConfig = field(default_factory=SelfCondConfig)
    feature_perturbation: FeaturePerturbationConfig = field(default_factory=FeaturePerturbationConfig)
    noise_augmentation: NoiseAugConfig = field(default_factory=NoiseAugConfig)

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'TrainingTricksConfig':
        """Extract training tricks config from Hydra DictConfig."""
        grad_noise_cfg = cfg.training.get('gradient_noise', {})
        curriculum_cfg = cfg.training.get('curriculum', {})
        jitter_cfg = cfg.training.get('timestep_jitter', {})
        self_cond_cfg = cfg.training.get('self_conditioning', {})
        feat_cfg = cfg.training.get('feature_perturbation', {})
        noise_aug_cfg = cfg.training.get('noise_augmentation', {})

        return cls(
            gradient_noise=GradientNoiseConfig(
                enabled=grad_noise_cfg.get('enabled', False),
                sigma=grad_noise_cfg.get('sigma', 0.01),
                decay=grad_noise_cfg.get('decay', 0.55),
            ),
            curriculum=CurriculumConfig(
                enabled=curriculum_cfg.get('enabled', False),
                warmup_epochs=curriculum_cfg.get('warmup_epochs', 50),
                min_t_start=curriculum_cfg.get('min_t_start', 0.0),
                max_t_start=curriculum_cfg.get('max_t_start', 0.3),
                min_t_end=curriculum_cfg.get('min_t_end', 0.0),
                max_t_end=curriculum_cfg.get('max_t_end', 1.0),
            ),
            jitter=JitterConfig(
                enabled=jitter_cfg.get('enabled', False),
                std=jitter_cfg.get('std', 0.05),
            ),
            min_snr=MinSNRConfig(
                enabled=cfg.training.get('use_min_snr', False),
                gamma=cfg.training.get('min_snr_gamma', 5.0),
            ),
            self_cond=SelfCondConfig(
                enabled=self_cond_cfg.get('enabled', False),
                prob=self_cond_cfg.get('prob', 0.5),
                consistency_weight=self_cond_cfg.get('consistency_weight', 0.1),
            ),
            feature_perturbation=FeaturePerturbationConfig(
                enabled=feat_cfg.get('enabled', False),
                std=feat_cfg.get('std', 0.1),
                layers=list(feat_cfg.get('layers', ['mid'])),
            ),
            noise_augmentation=NoiseAugConfig(
                enabled=noise_aug_cfg.get('enabled', False),
                std=noise_aug_cfg.get('std', 0.1),
            ),
        )


@dataclass
class ModeEmbeddingConfig:
    """Mode embedding configuration for multi-modality training."""
    enabled: bool = False
    strategy: str = 'full'
    dropout: float = 0.2
    late_start_level: int = 2

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'ModeEmbeddingConfig':
        """Extract mode embedding config from Hydra DictConfig."""
        return cls(
            enabled=cfg.mode.get('use_mode_embedding', False),
            strategy=cfg.mode.get('mode_embedding_strategy', 'full'),
            dropout=cfg.mode.get('mode_embedding_dropout', 0.2),
            late_start_level=cfg.mode.get('late_mode_start_level', 2),
        )


@dataclass
class SizeBinConfig:
    """Size bin embedding configuration for seg_conditioned mode."""
    enabled: bool = False
    edges: list[float] = field(default_factory=lambda: [0, 3, 6, 10, 15, 20, 30])
    num_bins: int = 7
    max_count: int = 10
    embed_dim: int = 32
    fov_mm: float = 240.0
    projection_hidden_dim: int = 0    # 0 = legacy 2-layer MLP
    projection_num_layers: int = 2    # only used when projection_hidden_dim > 0
    aux_loss_weight: float = 0.0     # 0 = disabled; auxiliary bin prediction loss weight

    def __post_init__(self):
        if not self.enabled:
            return
        if self.num_bins <= 0:
            raise ValueError(f"num_bins must be > 0, got {self.num_bins}")
        if self.max_count <= 0:
            raise ValueError(f"max_count must be > 0, got {self.max_count}")
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {self.embed_dim}")
        if self.fov_mm <= 0:
            raise ValueError(f"fov_mm must be > 0, got {self.fov_mm}")
        if list(self.edges) != sorted(self.edges):
            raise ValueError(f"edges must be sorted ascending, got {self.edges}")
        n_bounded = len(self.edges) - 1
        if self.num_bins not in (n_bounded, n_bounded + 1):
            raise ValueError(
                f"num_bins must be len(edges)-1 ({n_bounded}) or len(edges)-1+1 "
                f"({n_bounded + 1}, with overflow bin), got {self.num_bins}"
            )

    @classmethod
    def from_hydra(cls, cfg: DictConfig, mode_name: str) -> 'SizeBinConfig':
        """Extract size bin config from Hydra DictConfig."""
        enabled = mode_name.startswith('seg_conditioned')
        if not enabled:
            return cls(enabled=False)

        size_bin_cfg = cfg.mode.get('size_bins', {})
        edges = list(size_bin_cfg.get('edges', [0, 3, 6, 10, 15, 20, 30]))
        return cls(
            enabled=True,
            edges=edges,
            num_bins=size_bin_cfg.get('num_bins', len(edges) - 1),
            max_count=size_bin_cfg.get('max_count', 10),
            embed_dim=size_bin_cfg.get('embedding_dim', 32),
            fov_mm=float(size_bin_cfg.get('fov_mm', 240.0)),
            projection_hidden_dim=size_bin_cfg.get('projection_hidden_dim', 0),
            projection_num_layers=size_bin_cfg.get('projection_num_layers', 2),
            aux_loss_weight=float(size_bin_cfg.get('aux_loss_weight', 0.0)),
        )


@dataclass
class AugmentedDiffusionConfig:
    """DC-AE 1.5 Augmented Diffusion Training configuration."""
    enabled: bool = False
    min_channels: int = 16
    channel_step: int = 4

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'AugmentedDiffusionConfig':
        """Extract augmented diffusion config from Hydra DictConfig."""
        aug_diff_cfg = cfg.training.get('augmented_diffusion', {})
        return cls(
            enabled=aug_diff_cfg.get('enabled', False),
            min_channels=aug_diff_cfg.get('min_channels', 16),
            channel_step=aug_diff_cfg.get('channel_step', 4),
        )


@dataclass
class RegionalWeightingConfig:
    """Regional weighting configuration for loss computation."""
    enabled: bool = False
    tiny_weight: float = 2.5
    small_weight: float = 1.8
    medium_weight: float = 1.4
    large_weight: float = 1.2
    background_weight: float = 1.0

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'RegionalWeightingConfig':
        """Extract regional weighting config from Hydra DictConfig."""
        rw_cfg = cfg.training.get('regional_weighting', {})
        if not rw_cfg.get('enabled', False):
            return cls(enabled=False)

        weights = rw_cfg.get('weights', {})
        return cls(
            enabled=True,
            tiny_weight=weights.get('tiny', 2.5),
            small_weight=weights.get('small', 1.8),
            medium_weight=weights.get('medium', 1.4),
            large_weight=weights.get('large', 1.2),
            background_weight=rw_cfg.get('background_weight', 1.0),
        )


@dataclass
class ControlNetConfig:
    """ControlNet configuration."""
    enabled: bool = False
    stage1: bool = False
    freeze_unet: bool = True
    conditioning_scale: float = 1.0
    cfg_dropout_prob: float = 0.15
    gradient_checkpointing: bool = False

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'ControlNetConfig':
        """Extract ControlNet config from Hydra DictConfig."""
        controlnet_cfg = cfg.get('controlnet', {})
        return cls(
            enabled=controlnet_cfg.get('enabled', False),
            stage1=controlnet_cfg.get('stage1', False),
            freeze_unet=controlnet_cfg.get('freeze_unet', True),
            conditioning_scale=controlnet_cfg.get('conditioning_scale', 1.0),
            cfg_dropout_prob=controlnet_cfg.get('cfg_dropout_prob', 0.15),
            gradient_checkpointing=controlnet_cfg.get('gradient_checkpointing', False),
        )


@dataclass
class LoggingConfig:
    """Logging configuration."""
    msssim: bool = True
    psnr: bool = True
    lpips: bool = False
    timestep_region_losses: bool = True
    worst_batch: bool = True
    intermediate_steps: bool = True
    num_intermediate_steps: int = 5

    @classmethod
    def from_hydra(cls, cfg: DictConfig, is_seg_mode: bool = False) -> 'LoggingConfig':
        """Extract logging config from Hydra DictConfig."""
        logging_cfg = cfg.training.get('logging', {})
        # Disable LPIPS for seg modes (binary masks don't work with VGG features)
        lpips = False if is_seg_mode else logging_cfg.get('lpips', False)
        return cls(
            msssim=logging_cfg.get('msssim', True),
            psnr=logging_cfg.get('psnr', True),
            lpips=lpips,
            timestep_region_losses=logging_cfg.get('timestep_region_losses', True),
            worst_batch=logging_cfg.get('worst_batch', True),
            intermediate_steps=logging_cfg.get('intermediate_steps', True),
            num_intermediate_steps=logging_cfg.get('num_intermediate_steps', 5),
        )


@dataclass
class DiffusionTrainerConfig:
    """Complete diffusion trainer configuration.

    Combines all sub-configs into a single configuration object.
    """
    # Core
    spatial_dims: int
    mode_name: str
    strategy_name: str
    image_size: int
    num_timesteps: int

    # Training
    learning_rate: float
    batch_size: int
    n_epochs: int
    warmup_epochs: int
    weight_decay: float
    perceptual_weight: float
    use_fp32_loss: bool
    use_ema: bool
    ema_decay: float

    # Scheduler
    scheduler_type: str = 'cosine'
    eta_min: float = 1e-6

    # 3D-specific
    volume_depth: int = 160
    volume_height: int = 256
    volume_width: int = 256
    use_amp: bool = False
    use_gradient_checkpointing: bool = False

    def __post_init__(self):
        # Only validate fields unique to this config.
        # Fields like n_epochs, batch_size, learning_rate, warmup_epochs are
        # validated by BaseTrainingConfig; spatial_dims, image_size by ModelConfig;
        # num_timesteps by StrategyConfig.
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {self.ema_decay}")

    # Sub-configs
    score_aug: ScoreAugConfig = field(default_factory=ScoreAugConfig)
    sda: SDAConfig = field(default_factory=SDAConfig)
    training_tricks: TrainingTricksConfig = field(default_factory=TrainingTricksConfig)
    mode_embedding: ModeEmbeddingConfig = field(default_factory=ModeEmbeddingConfig)
    size_bin: SizeBinConfig = field(default_factory=SizeBinConfig)
    augmented_diffusion: AugmentedDiffusionConfig = field(default_factory=AugmentedDiffusionConfig)
    regional_weighting: RegionalWeightingConfig = field(default_factory=RegionalWeightingConfig)
    controlnet: ControlNetConfig = field(default_factory=ControlNetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_hydra(cls, cfg: DictConfig, spatial_dims: int = 2) -> 'DiffusionTrainerConfig':
        """Extract all config from Hydra DictConfig.

        Args:
            cfg: Hydra configuration object.
            spatial_dims: Spatial dimensions (2 or 3).

        Returns:
            Complete DiffusionTrainerConfig.
        """
        mode_name = cfg.mode.name
        is_seg_mode = mode_name in ('seg', 'seg_conditioned')

        # Extract dimension-specific size config
        if spatial_dims == 2:
            image_size = cfg.model.image_size
            volume_depth = 160
            volume_height = 256
            volume_width = 256
        else:
            image_size = cfg.volume.get('height', 256)
            volume_depth = cfg.volume.get('depth', 160)
            volume_height = cfg.volume.get('height', 256)
            volume_width = cfg.volume.get('width', 256)

        # Perceptual weight (disabled for seg modes)
        perceptual_weight = 0.0 if is_seg_mode else cfg.training.get('perceptual_weight', 0.0)

        # Optimizer settings
        optimizer_cfg = cfg.training.get('optimizer', {})
        weight_decay = optimizer_cfg.get('weight_decay', 0.0)

        # 3D-specific memory optimizations
        if spatial_dims == 3:
            use_amp = True
            use_gradient_checkpointing = cfg.training.get('gradient_checkpointing', True)
        else:
            use_amp = cfg.training.get('use_amp', False)
            use_gradient_checkpointing = cfg.training.get('gradient_checkpointing', False)

        n_epochs = cfg.training.get('epochs', None) or cfg.training.get('max_epochs', None)
        if n_epochs is None:
            raise ValueError("Config must have training.epochs or training.max_epochs")

        return cls(
            # Core
            spatial_dims=spatial_dims,
            mode_name=mode_name,
            strategy_name=cfg.strategy.name,
            image_size=image_size,
            num_timesteps=cfg.strategy.get('num_train_timesteps', 1000),
            # Training
            learning_rate=cfg.training.learning_rate,
            batch_size=cfg.training.batch_size,
            n_epochs=n_epochs,
            warmup_epochs=cfg.training.get('warmup_epochs', 5),
            weight_decay=weight_decay,
            perceptual_weight=perceptual_weight,
            use_fp32_loss=cfg.training.get('use_fp32_loss', True),
            use_ema=cfg.training.get('use_ema', False),
            ema_decay=cfg.training.get('ema', {}).get('decay', 0.9999),
            # Scheduler
            scheduler_type=cfg.training.get('scheduler', 'cosine'),
            eta_min=cfg.training.get('eta_min', 1e-6),
            # 3D-specific
            volume_depth=volume_depth,
            volume_height=volume_height,
            volume_width=volume_width,
            use_amp=use_amp,
            use_gradient_checkpointing=use_gradient_checkpointing,
            # Sub-configs
            score_aug=ScoreAugConfig.from_hydra(cfg),
            sda=SDAConfig.from_hydra(cfg),
            training_tricks=TrainingTricksConfig.from_hydra(cfg),
            mode_embedding=ModeEmbeddingConfig.from_hydra(cfg),
            size_bin=SizeBinConfig.from_hydra(cfg, mode_name),
            augmented_diffusion=AugmentedDiffusionConfig.from_hydra(cfg),
            regional_weighting=RegionalWeightingConfig.from_hydra(cfg),
            controlnet=ControlNetConfig.from_hydra(cfg),
            logging=LoggingConfig.from_hydra(cfg, is_seg_mode),
        )


def validate_score_aug_config(
    config: ScoreAugConfig,
    spatial_dims: int = 2,
    is_main_process: bool = True,
) -> None:
    """Validate ScoreAug configuration constraints that depend on spatial_dims.

    Note: Basic invariant checks (omega conditioning requirements, probability
    ranges) are now in ScoreAugConfig.__post_init__. This function only handles
    spatial_dims-dependent warnings.

    Args:
        config: ScoreAug configuration to validate.
        spatial_dims: Spatial dimensions (2 or 3).
        is_main_process: Whether this is the main process (for logging).
    """
    if not config.enabled:
        return

    # Mode intensity scaling not supported in 3D
    if config.use_mode_intensity_scaling and spatial_dims == 3:
        if is_main_process:
            logger.warning(
                "mode_intensity_scaling is not supported in 3D diffusion "
                "(requires mode_id from multi-modality mode). Ignoring."
            )


@dataclass
class LatentConfig:
    """Latent diffusion configuration."""
    enabled: bool = False
    scale_factor: int = 1
    compression_checkpoint: str | None = None
    slicewise_encoding: bool = False

    def __post_init__(self):
        if self.enabled and self.compression_checkpoint is None:
            raise ValueError(
                "latent.compression_checkpoint must be set when latent.enabled=true"
            )
        if self.scale_factor <= 0:
            raise ValueError(f"scale_factor must be > 0, got {self.scale_factor}")

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'LatentConfig':
        """Extract latent config from Hydra DictConfig."""
        latent_cfg = cfg.get('latent', {})
        return cls(
            enabled=latent_cfg.get('enabled', False),
            scale_factor=latent_cfg.get('scale_factor', 1),
            compression_checkpoint=latent_cfg.get('compression_checkpoint', None),
            slicewise_encoding=latent_cfg.get('slicewise_encoding', False),
        )
