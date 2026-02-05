"""Training configuration dataclasses.

Provides type-safe access to training configuration including
EMA, optimizer, scheduler, and dataloader settings.
"""
from dataclasses import dataclass, field

from omegaconf import DictConfig


@dataclass
class EMAConfig:
    """Exponential Moving Average configuration.

    Attributes:
        enabled: Whether EMA is enabled.
        decay: EMA decay rate (single source of truth - 0.9999).
        update_after_step: Start EMA updates after this many steps.
        update_every: Update EMA every N steps.
    """
    enabled: bool = False
    decay: float = 0.9999  # Single source of truth for EMA decay
    update_after_step: int = 100
    update_every: int = 10

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'EMAConfig':
        """Extract EMA config from Hydra DictConfig.

        Args:
            cfg: Hydra configuration object.

        Returns:
            EMAConfig instance.
        """
        ema = cfg.training.get('ema', {})
        return cls(
            enabled=cfg.training.get('use_ema', False),
            decay=ema.get('decay', 0.9999),
            update_after_step=ema.get('update_after_step', 100),
            update_every=ema.get('update_every', 10),
        )


@dataclass
class OptimizerConfig:
    """Optimizer configuration.

    Attributes:
        betas: Adam beta parameters.
        weight_decay: Weight decay for regularization.
    """
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'OptimizerConfig':
        """Extract optimizer config from Hydra DictConfig.

        Args:
            cfg: Hydra configuration object.

        Returns:
            OptimizerConfig instance.
        """
        opt = cfg.training.get('optimizer', {})
        betas = opt.get('betas', [0.9, 0.999])
        return cls(
            betas=(betas[0], betas[1]),
            weight_decay=opt.get('weight_decay', 0.0),
        )


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration.

    Attributes:
        name: Scheduler type ('cosine', 'constant', or 'plateau').
        eta_min: Minimum learning rate for cosine scheduler.
        plateau_factor: Reduction factor for plateau scheduler.
        plateau_patience: Patience epochs for plateau scheduler.
        plateau_min_lr: Minimum LR for plateau scheduler.
    """
    name: str = 'cosine'
    eta_min: float = 1e-6
    plateau_factor: float = 0.5
    plateau_patience: int = 10
    plateau_min_lr: float = 1e-6

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'SchedulerConfig':
        """Extract scheduler config from Hydra DictConfig.

        Args:
            cfg: Hydra configuration object.

        Returns:
            SchedulerConfig instance.
        """
        training = cfg.training
        plateau = training.get('plateau', {})
        return cls(
            name=training.get('scheduler', 'cosine'),
            eta_min=training.get('eta_min', 1e-6),
            plateau_factor=plateau.get('factor', 0.5),
            plateau_patience=plateau.get('patience', 10),
            plateau_min_lr=plateau.get('min_lr', 1e-6),
        )


@dataclass
class DataLoaderConfig:
    """DataLoader configuration.

    Attributes:
        num_workers: Number of worker processes.
        prefetch_factor: Number of batches to prefetch per worker.
        pin_memory: Whether to pin memory for faster GPU transfer.
        persistent_workers: Whether to keep workers alive between epochs.
    """
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'DataLoaderConfig':
        """Extract dataloader config from Hydra DictConfig.

        Args:
            cfg: Hydra configuration object.

        Returns:
            DataLoaderConfig instance.
        """
        dl = cfg.training.get('dataloader', {})
        return cls(
            num_workers=dl.get('num_workers', 8),
            prefetch_factor=dl.get('prefetch_factor', 4),
            pin_memory=dl.get('pin_memory', True),
            persistent_workers=dl.get('persistent_workers', True),
        )


@dataclass
class LoggingConfig:
    """Logging configuration.

    Attributes:
        msssim: Whether to log MS-SSIM metrics.
        psnr: Whether to log PSNR metrics.
        lpips: Whether to log LPIPS metrics.
        grad_norm: Whether to log gradient norms.
        timestep_region_losses: Whether to log timestep-stratified losses.
        worst_batch: Whether to log worst batch visualizations.
        intermediate_steps: Whether to log intermediate denoising steps.
        num_intermediate_steps: Number of intermediate steps to log.
        regional_losses: Whether to log regional (tumor vs bg) losses.
        flops: Whether to log FLOPs metrics.
    """
    msssim: bool = True
    psnr: bool = True
    lpips: bool = False
    grad_norm: bool = True
    timestep_region_losses: bool = True
    worst_batch: bool = True
    intermediate_steps: bool = True
    num_intermediate_steps: int = 5
    regional_losses: bool = True
    flops: bool = True

    @classmethod
    def from_hydra(cls, cfg: DictConfig, is_seg_mode: bool = False) -> 'LoggingConfig':
        """Extract logging config from Hydra DictConfig.

        Args:
            cfg: Hydra configuration object.
            is_seg_mode: Whether training segmentation mode (disables LPIPS).

        Returns:
            LoggingConfig instance.
        """
        logging_cfg = cfg.training.get('logging', {})
        # Disable LPIPS for seg modes (binary masks don't work with VGG features)
        lpips = False if is_seg_mode else logging_cfg.get('lpips', False)
        return cls(
            msssim=logging_cfg.get('msssim', True),
            psnr=logging_cfg.get('psnr', True),
            lpips=lpips,
            grad_norm=logging_cfg.get('grad_norm', True),
            timestep_region_losses=logging_cfg.get('timestep_region_losses', True),
            worst_batch=logging_cfg.get('worst_batch', True),
            intermediate_steps=logging_cfg.get('intermediate_steps', True),
            num_intermediate_steps=logging_cfg.get('num_intermediate_steps', 5),
            regional_losses=logging_cfg.get('regional_losses', True),
            flops=logging_cfg.get('flops', True),
        )


@dataclass
class TrainingConfig:
    """Complete training configuration.

    Combines all sub-configs into a single configuration object
    providing type-safe access to all training parameters.

    Attributes:
        epochs: Total number of training epochs.
        batch_size: Training batch size.
        learning_rate: Base learning rate.
        gradient_clip_norm: Maximum gradient norm for clipping.
        warmup_epochs: Number of warmup epochs.
        augment: Whether to apply data augmentation.
        augment_type: Type of augmentation ('diffusion' or 'standard').
        verbose: Whether to show progress bars.
        use_compile: Whether to use torch.compile.
        compile_fused_forward: Whether to compile fused forward pass.
        use_multi_gpu: Whether to use distributed training.
        perceptual_weight: Weight for perceptual loss.
        use_fp32_loss: Whether to compute loss in FP32.
        gradient_checkpointing: Whether to use gradient checkpointing.
        limit_train_batches: Limit number of training batches (for debugging).
        ema: EMA configuration.
        optimizer: Optimizer configuration.
        scheduler: Scheduler configuration.
        dataloader: DataLoader configuration.
        logging: Logging configuration.
    """
    # Core parameters
    epochs: int
    batch_size: int
    learning_rate: float
    gradient_clip_norm: float = 1.0
    warmup_epochs: int = 5

    # Features
    augment: bool = False
    augment_type: str = 'diffusion'
    verbose: bool = True
    use_compile: bool = True
    compile_fused_forward: bool = True
    use_multi_gpu: bool = False
    perceptual_weight: float = 0.0
    use_fp32_loss: bool = True
    gradient_checkpointing: bool = False
    limit_train_batches: int | None = None

    # Sub-configs
    ema: EMAConfig = field(default_factory=EMAConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_hydra(cls, cfg: DictConfig, is_seg_mode: bool = False) -> 'TrainingConfig':
        """Extract all training config from Hydra DictConfig.

        Args:
            cfg: Hydra configuration object.
            is_seg_mode: Whether training segmentation mode (affects some defaults).

        Returns:
            Complete TrainingConfig instance.
        """
        training = cfg.training
        return cls(
            epochs=training.epochs,
            batch_size=training.batch_size,
            learning_rate=training.learning_rate,
            gradient_clip_norm=training.get('gradient_clip_norm', 1.0),
            warmup_epochs=training.get('warmup_epochs', 5),
            augment=training.get('augment', False),
            augment_type=training.get('augment_type', 'diffusion'),
            verbose=training.get('verbose', True),
            use_compile=training.get('use_compile', True),
            compile_fused_forward=training.get('compile_fused_forward', True),
            use_multi_gpu=training.get('use_multi_gpu', False),
            perceptual_weight=0.0 if is_seg_mode else training.get('perceptual_weight', 0.0),
            use_fp32_loss=training.get('use_fp32_loss', True),
            gradient_checkpointing=training.get('gradient_checkpointing', False),
            limit_train_batches=training.get('limit_train_batches'),
            ema=EMAConfig.from_hydra(cfg),
            optimizer=OptimizerConfig.from_hydra(cfg),
            scheduler=SchedulerConfig.from_hydra(cfg),
            dataloader=DataLoaderConfig.from_hydra(cfg),
            logging=LoggingConfig.from_hydra(cfg, is_seg_mode),
        )
