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
- should_compile_fused: Helper to determine if compiled forward is safe

These classes consolidate the configuration extraction logic from DiffusionTrainer,
making it reusable and testable.
"""
from dataclasses import dataclass, field

from omegaconf import DictConfig


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

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'ScoreAugConfig':
        """Extract ScoreAug config from Hydra DictConfig."""
        score_aug_cfg = cfg.training.get('score_aug', {})
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
            patterns_checkerboard=score_aug_cfg.get('patterns_checkerboard', True),
            patterns_grid_dropout=score_aug_cfg.get('patterns_grid_dropout', True),
            patterns_coarse_dropout=score_aug_cfg.get('patterns_coarse_dropout', True),
            patterns_patch_dropout=score_aug_cfg.get('patterns_patch_dropout', True),
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
class TrainingTricksConfig:
    """Training tricks configuration."""
    # Gradient noise
    gradient_noise_enabled: bool = False
    gradient_noise_sigma: float = 0.01
    gradient_noise_decay: float = 0.55

    # Curriculum
    curriculum_enabled: bool = False
    curriculum_warmup_epochs: int = 50
    curriculum_min_t_start: float = 0.0
    curriculum_max_t_start: float = 0.3
    curriculum_min_t_end: float = 0.0
    curriculum_max_t_end: float = 1.0

    # Timestep jitter
    jitter_enabled: bool = False
    jitter_std: float = 0.05

    # Min-SNR
    min_snr_enabled: bool = False
    min_snr_gamma: float = 5.0

    # Self-conditioning
    self_cond_enabled: bool = False
    self_cond_prob: float = 0.5

    # Feature perturbation
    feature_perturbation_enabled: bool = False
    feature_perturbation_std: float = 0.1
    feature_perturbation_layers: list[str] = field(default_factory=lambda: ['mid'])

    # Noise augmentation
    noise_augmentation_enabled: bool = False
    noise_augmentation_std: float = 0.1

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'TrainingTricksConfig':
        """Extract training tricks config from Hydra DictConfig."""
        grad_noise_cfg = cfg.training.get('gradient_noise', {})
        curriculum_cfg = cfg.training.get('curriculum', {})
        jitter_cfg = cfg.training.get('timestep_jitter', {})
        min_snr_cfg = cfg.training.get('min_snr', {})
        self_cond_cfg = cfg.training.get('self_conditioning', {})
        feat_cfg = cfg.training.get('feature_perturbation', {})
        noise_aug_cfg = cfg.training.get('noise_augmentation', {})

        return cls(
            # Gradient noise
            gradient_noise_enabled=grad_noise_cfg.get('enabled', False),
            gradient_noise_sigma=grad_noise_cfg.get('sigma', 0.01),
            gradient_noise_decay=grad_noise_cfg.get('decay', 0.55),
            # Curriculum
            curriculum_enabled=curriculum_cfg.get('enabled', False),
            curriculum_warmup_epochs=curriculum_cfg.get('warmup_epochs', 50),
            curriculum_min_t_start=curriculum_cfg.get('min_t_start', 0.0),
            curriculum_max_t_start=curriculum_cfg.get('max_t_start', 0.3),
            curriculum_min_t_end=curriculum_cfg.get('min_t_end', 0.0),
            curriculum_max_t_end=curriculum_cfg.get('max_t_end', 1.0),
            # Timestep jitter
            jitter_enabled=jitter_cfg.get('enabled', False),
            jitter_std=jitter_cfg.get('std', 0.05),
            # Min-SNR
            min_snr_enabled=min_snr_cfg.get('enabled', False),
            min_snr_gamma=min_snr_cfg.get('gamma', 5.0),
            # Self-conditioning
            self_cond_enabled=self_cond_cfg.get('enabled', False),
            self_cond_prob=self_cond_cfg.get('prob', 0.5),
            # Feature perturbation
            feature_perturbation_enabled=feat_cfg.get('enabled', False),
            feature_perturbation_std=feat_cfg.get('std', 0.1),
            feature_perturbation_layers=list(feat_cfg.get('layers', ['mid'])),
            # Noise augmentation
            noise_augmentation_enabled=noise_aug_cfg.get('enabled', False),
            noise_augmentation_std=noise_aug_cfg.get('std', 0.1),
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

    @classmethod
    def from_hydra(cls, cfg: DictConfig, mode_name: str) -> 'SizeBinConfig':
        """Extract size bin config from Hydra DictConfig."""
        enabled = (mode_name == 'seg_conditioned')
        if not enabled:
            return cls(enabled=False)

        size_bin_cfg = cfg.mode.get('size_bins', {})
        edges = list(size_bin_cfg.get('edges', [0, 3, 6, 10, 15, 20, 30]))
        return cls(
            enabled=True,
            edges=edges,
            num_bins=size_bin_cfg.get('num_bins', len(edges)),
            max_count=size_bin_cfg.get('max_count_per_bin', 10),
            embed_dim=size_bin_cfg.get('embedding_dim', 32),
            fov_mm=float(size_bin_cfg.get('fov_mm', 240.0)),
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
            n_epochs=cfg.training.max_epochs,
            warmup_epochs=cfg.training.get('warmup_epochs', 5),
            weight_decay=weight_decay,
            perceptual_weight=perceptual_weight,
            use_fp32_loss=cfg.training.get('use_fp32_loss', True),
            use_ema=cfg.training.get('use_ema', True),
            ema_decay=cfg.training.ema.get('decay', 0.999),
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


def should_compile_fused(
    config: DiffusionTrainerConfig,
    use_multi_gpu: bool,
    latent_scale: int,
) -> bool:
    """Determine if compiled forward is safe to use.

    Compiled forward is disabled when using features that require
    extra model kwargs or variable control flow.

    Args:
        config: Diffusion trainer configuration.
        use_multi_gpu: Whether using DDP multi-GPU.
        latent_scale: Latent space scale factor.

    Returns:
        True if compiled forward can be used.
    """
    if use_multi_gpu:
        return False
    if latent_scale > 1:
        return False
    if config.training_tricks.min_snr_enabled:
        return False
    if config.score_aug.enabled:
        return False
    if config.sda.enabled:
        return False
    if config.mode_embedding.enabled:
        return False
    if config.score_aug.use_omega_conditioning:
        return False
    if config.augmented_diffusion.enabled:
        return False
    if config.controlnet.enabled:
        return False
    if config.size_bin.enabled:
        return False
    if config.regional_weighting.enabled:
        return False
    # Compiled forward only supports SEG, BRAVO, and DUAL modes
    return config.mode_name in ('seg', 'bravo', 'dual')


def validate_score_aug_config(
    config: ScoreAugConfig,
    spatial_dims: int = 2,
    is_main_process: bool = True,
) -> None:
    """Validate ScoreAug configuration constraints.

    Raises ValueError if configuration is invalid:
    - Rotation/flip transforms require omega conditioning
    - Mode intensity scaling requires omega conditioning

    Args:
        config: ScoreAug configuration to validate.
        spatial_dims: Spatial dimensions (2 or 3).
        is_main_process: Whether this is the main process (for logging).

    Raises:
        ValueError: If configuration constraints are violated.
    """
    if not config.enabled:
        return

    # Check spatial transforms require omega conditioning
    has_spatial_transforms = config.rotation or config.flip
    if has_spatial_transforms and not config.use_omega_conditioning:
        raise ValueError(
            "ScoreAug rotation/flip require omega conditioning (per ScoreAug paper). "
            "Gaussian noise is rotation-invariant, allowing the model to detect "
            "rotation from noise patterns and 'cheat' by inverting before denoising. "
            "Fix: Set training.score_aug.use_omega_conditioning=true"
        )

    # Mode intensity scaling requires omega conditioning
    if config.use_mode_intensity_scaling and not config.use_omega_conditioning:
        raise ValueError(
            "Mode intensity scaling requires omega conditioning. "
            "Fix: Set training.score_aug.use_omega_conditioning=true"
        )

    # Mode intensity scaling not supported in 3D
    if config.use_mode_intensity_scaling and spatial_dims == 3:
        import logging
        logger = logging.getLogger(__name__)
        if is_main_process:
            logger.warning(
                "mode_intensity_scaling is not supported in 3D diffusion "
                "(requires mode_id from multi-modality mode). Ignoring."
            )
