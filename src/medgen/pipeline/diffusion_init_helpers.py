"""Helper functions for DiffusionTrainer initialization.

These functions are extracted from DiffusionTrainer methods to reduce file size.
Each function takes a `trainer` (DiffusionTrainer instance) as its first argument
and accesses/sets trainer attributes directly.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import DiffusionTrainer

logger = logging.getLogger(__name__)


def setup_score_aug(trainer: DiffusionTrainer) -> None:
    """Initialize ScoreAug transform and related settings.

    Sets up:
    - trainer.score_aug: ScoreAugTransform instance or None
    - trainer.use_omega_conditioning: bool
    - trainer.use_mode_intensity_scaling: bool
    - trainer._apply_mode_intensity_scale: function reference or None

    Validates ScoreAug configuration constraints (rotation/flip require omega).
    """
    from .diffusion_config import ScoreAugConfig, validate_score_aug_config

    score_aug_cfg = ScoreAugConfig.from_hydra(trainer.cfg)

    trainer.score_aug = None
    trainer.use_omega_conditioning = False
    trainer.use_mode_intensity_scaling = False
    trainer._apply_mode_intensity_scale = None

    if not score_aug_cfg.enabled:
        return

    # Validate configuration constraints
    validate_score_aug_config(score_aug_cfg, trainer.spatial_dims, trainer.is_main_process)

    # Create ScoreAugTransform
    from medgen.augmentation import ScoreAugTransform
    trainer.score_aug = ScoreAugTransform(
        spatial_dims=trainer.spatial_dims,
        rotation=score_aug_cfg.rotation,
        flip=score_aug_cfg.flip,
        translation=score_aug_cfg.translation,
        cutout=score_aug_cfg.cutout,
        compose=score_aug_cfg.compose,
        compose_prob=score_aug_cfg.compose_prob,
        v2_mode=score_aug_cfg.v2_mode,
        nondestructive_prob=score_aug_cfg.nondestructive_prob,
        destructive_prob=score_aug_cfg.destructive_prob,
        cutout_vs_pattern=score_aug_cfg.cutout_vs_pattern,
        patterns_checkerboard=score_aug_cfg.patterns_checkerboard,
        patterns_grid_dropout=score_aug_cfg.patterns_grid_dropout,
        patterns_coarse_dropout=score_aug_cfg.patterns_coarse_dropout,
        patterns_patch_dropout=score_aug_cfg.patterns_patch_dropout,
    )

    trainer.use_omega_conditioning = score_aug_cfg.use_omega_conditioning

    # Mode intensity scaling (2D only)
    trainer.use_mode_intensity_scaling = score_aug_cfg.use_mode_intensity_scaling
    if trainer.use_mode_intensity_scaling:
        if trainer.spatial_dims == 3:
            if trainer.is_main_process:
                logger.warning(
                    "mode_intensity_scaling is not supported in 3D diffusion. Ignoring."
                )
            trainer.use_mode_intensity_scaling = False
        else:
            from medgen.augmentation import apply_mode_intensity_scale
            trainer._apply_mode_intensity_scale = apply_mode_intensity_scale

    # Log configuration
    if trainer.is_main_process:
        transforms = []
        if score_aug_cfg.rotation:
            transforms.append('rotation')
        if score_aug_cfg.flip:
            transforms.append('flip')
        if score_aug_cfg.translation:
            transforms.append('translation')
        if score_aug_cfg.cutout:
            transforms.append('cutout')
        if score_aug_cfg.brightness and trainer.spatial_dims == 2:
            transforms.append(f"brightness({score_aug_cfg.brightness_range})")
        n_options = len(transforms) + 1
        logger.info(
            f"ScoreAug {trainer.spatial_dims}D enabled: transforms=[{', '.join(transforms)}], "
            f"each with 1/{n_options} prob (uniform), "
            f"omega_conditioning={trainer.use_omega_conditioning}, "
            f"mode_intensity_scaling={trainer.use_mode_intensity_scaling}"
        )


def setup_sda(trainer: DiffusionTrainer) -> None:
    """Initialize SDA (Shifted Data Augmentation) transform.

    Sets up:
    - trainer.sda: SDATransform instance or None
    - trainer.sda_weight: float

    SDA is mutually exclusive with ScoreAug.
    """
    from .diffusion_config import SDAConfig

    sda_cfg = SDAConfig.from_hydra(trainer.cfg)

    trainer.sda = None
    trainer.sda_weight = 1.0

    if not sda_cfg.enabled:
        return

    # SDA and ScoreAug are mutually exclusive
    if trainer.score_aug is not None:
        if trainer.is_main_process:
            logger.warning("SDA and ScoreAug are mutually exclusive. Disabling SDA.")
        return

    from medgen.augmentation import SDATransform
    trainer.sda = SDATransform(
        rotation=sda_cfg.rotation,
        flip=sda_cfg.flip,
        noise_shift=sda_cfg.noise_shift,
        prob=sda_cfg.prob,
    )
    trainer.sda_weight = sda_cfg.weight

    if trainer.is_main_process:
        transforms = []
        if sda_cfg.rotation:
            transforms.append('rotation')
        if sda_cfg.flip:
            transforms.append('flip')
        logger.info(
            f"SDA {trainer.spatial_dims}D enabled: transforms=[{', '.join(transforms)}], "
            f"noise_shift={sda_cfg.noise_shift}, prob={sda_cfg.prob}, weight={trainer.sda_weight}"
        )


def setup_conditional_embeddings(trainer: DiffusionTrainer) -> None:
    """Initialize mode embedding and size bin embedding settings.

    Sets up:
    - trainer.use_mode_embedding: bool
    - trainer.mode_embedding_strategy: str
    - trainer.mode_embedding_dropout: float
    - trainer.late_mode_start_level: int
    - trainer.use_size_bin_embedding: bool
    - trainer.size_bin_num_bins: int
    - trainer.size_bin_max_count: int
    - trainer.size_bin_embed_dim: int
    """
    from .diffusion_config import ModeEmbeddingConfig, SizeBinConfig

    mode_cfg = ModeEmbeddingConfig.from_hydra(trainer.cfg)
    size_bin_cfg = SizeBinConfig.from_hydra(trainer.cfg, trainer.mode_name)

    # Mode embedding
    trainer.use_mode_embedding = mode_cfg.enabled
    trainer.mode_embedding_strategy = mode_cfg.strategy
    trainer.mode_embedding_dropout = mode_cfg.dropout
    trainer.late_mode_start_level = mode_cfg.late_start_level

    if trainer.use_mode_embedding and trainer.is_main_process:
        logger.info(
            f"Mode embedding enabled: strategy={trainer.mode_embedding_strategy}, "
            f"dropout={trainer.mode_embedding_dropout}, late_start_level={trainer.late_mode_start_level}"
        )

    # Size bin embedding
    trainer.use_size_bin_embedding = size_bin_cfg.enabled
    if trainer.use_size_bin_embedding:
        trainer.size_bin_num_bins = size_bin_cfg.num_bins
        trainer.size_bin_max_count = size_bin_cfg.max_count
        trainer.size_bin_embed_dim = size_bin_cfg.embed_dim
        trainer.size_bin_projection_hidden_dim = size_bin_cfg.projection_hidden_dim
        trainer.size_bin_projection_num_layers = size_bin_cfg.projection_num_layers
        if trainer.is_main_process:
            logger.info(
                f"Size bin embedding enabled: num_bins={trainer.size_bin_num_bins}, "
                f"max_count={trainer.size_bin_max_count}, embed_dim={trainer.size_bin_embed_dim}"
            )


def setup_augmented_diffusion(trainer: DiffusionTrainer) -> None:
    """Initialize DC-AE 1.5 augmented diffusion training settings.

    Sets up:
    - trainer.augmented_diffusion_enabled: bool
    - trainer.aug_diff_min_channels: int
    - trainer.aug_diff_channel_step: int
    - trainer._aug_diff_channel_steps: list[int] | None
    """
    from .diffusion_config import AugmentedDiffusionConfig

    aug_cfg = AugmentedDiffusionConfig.from_hydra(trainer.cfg)

    trainer.augmented_diffusion_enabled = aug_cfg.enabled
    trainer.aug_diff_min_channels = aug_cfg.min_channels
    trainer.aug_diff_channel_step = aug_cfg.channel_step
    trainer._aug_diff_channel_steps = None  # Computed lazily

    if trainer.augmented_diffusion_enabled and trainer.is_main_process:
        if trainer.space.scale_factor > 1:
            logger.info(
                f"DC-AE 1.5 Augmented Diffusion Training enabled: "
                f"min_channels={trainer.aug_diff_min_channels}, step={trainer.aug_diff_channel_step}"
            )
        else:
            logger.warning(
                "Augmented Diffusion Training enabled but using pixel space. "
                "This has no effect - only applies to latent diffusion."
            )


def setup_regional_weighting(trainer: DiffusionTrainer) -> None:
    """Initialize region-weighted loss computer.

    Sets up:
    - trainer.regional_weight_computer: RegionalWeightComputer | None
    """
    from medgen.losses import create_regional_weight_computer

    from .diffusion_config import RegionalWeightingConfig

    rw_cfg = RegionalWeightingConfig.from_hydra(trainer.cfg)

    trainer.regional_weight_computer = None

    if not rw_cfg.enabled:
        return

    if trainer.mode.is_conditional:
        trainer.regional_weight_computer = create_regional_weight_computer(trainer.cfg)
        if trainer.is_main_process:
            logger.info(
                f"Region-weighted loss enabled: "
                f"tiny={rw_cfg.tiny_weight}, small={rw_cfg.small_weight}, "
                f"medium={rw_cfg.medium_weight}, large={rw_cfg.large_weight}, "
                f"bg={rw_cfg.background_weight}"
            )
    else:
        if trainer.is_main_process:
            logger.warning(
                "Regional weighting enabled but mode is not conditional (seg mode). "
                "Skipping - regional weighting requires segmentation mask as conditioning."
            )


def setup_controlnet(trainer: DiffusionTrainer) -> None:
    """Initialize ControlNet configuration.

    Sets up:
    - trainer.use_controlnet: bool
    - trainer.controlnet_freeze_unet: bool
    - trainer.controlnet_scale: float
    - trainer.controlnet_cfg_dropout_prob: float
    - trainer.controlnet: nn.Module | None
    - trainer.controlnet_stage1: bool
    """
    from .diffusion_config import ControlNetConfig

    cn_cfg = ControlNetConfig.from_hydra(trainer.cfg)

    trainer.use_controlnet = cn_cfg.enabled
    trainer.controlnet_freeze_unet = cn_cfg.freeze_unet
    trainer.controlnet_scale = cn_cfg.conditioning_scale
    trainer.controlnet_cfg_dropout_prob = cn_cfg.cfg_dropout_prob
    trainer.controlnet = None
    trainer.controlnet_stage1 = cn_cfg.stage1

    if trainer.controlnet_stage1 and trainer.is_main_process:
        logger.info(
            "ControlNet Stage 1: Training unconditional UNet (in_channels=out_channels). "
            "Use this checkpoint for Stage 2 with controlnet.enabled=true"
        )

    if trainer.use_controlnet and trainer.is_main_process:
        logger.info(
            f"ControlNet Stage 2: freeze_unet={trainer.controlnet_freeze_unet}, "
            f"conditioning_scale={trainer.controlnet_scale}"
        )


def log_training_tricks_config(trainer: DiffusionTrainer) -> None:
    """Log configuration for various training tricks."""
    if not trainer.is_main_process:
        return

    # Use typed TrainingTricksConfig from base class
    tt = trainer._training_tricks

    if tt.gradient_noise.enabled:
        logger.info(
            f"Gradient noise injection enabled: "
            f"sigma={tt.gradient_noise.sigma}, decay={tt.gradient_noise.decay}"
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

    if tt.self_cond.enabled:
        logger.info(f"Self-conditioning enabled: prob={tt.self_cond.prob}")

    if tt.feature_perturbation.enabled:
        logger.info(
            f"Feature perturbation enabled: std={tt.feature_perturbation.std}, "
            f"layers={tt.feature_perturbation.layers}"
        )

    if tt.noise_augmentation.enabled:
        logger.info(f"Noise augmentation enabled: std={tt.noise_augmentation.std}")
