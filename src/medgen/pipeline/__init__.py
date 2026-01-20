"""Generative pipeline for training and inference (diffusion and VAE)."""

from .diffusion_trainer_base import DiffusionTrainerBase
from .trainer import DiffusionTrainer
from .vae_trainer import VAETrainer
from .vqvae_trainer import VQVAETrainer
from .dcae_trainer import DCAETrainer
# Diffusion components (from medgen.diffusion package)
from medgen.diffusion import (
    DDPMStrategy, RFlowStrategy, DiffusionStrategy,
    SegmentationMode, ConditionalSingleMode, ConditionalDualMode, TrainingMode,
    DiffusionSpace, PixelSpace, LatentSpace, load_vae_for_latent_space,
    # Model loading utilities
    load_diffusion_model, load_diffusion_model_with_metadata,
    detect_wrapper_type, LoadedModel,
)
# Evaluation utilities (from medgen.evaluation package)
from medgen.evaluation import (
    ValidationVisualizer,
    BaseTestEvaluator,
    CompressionTestEvaluator,
    Compression3DTestEvaluator,
    MetricsConfig,
    load_checkpoint_if_needed,
    save_test_results,
    ValidationRunner,
    ValidationConfig,
    ValidationResult,
)

# Metrics (from medgen.metrics package)
from medgen.metrics import (
    compute_msssim,
    compute_psnr,
    compute_lpips,
    RegionalMetricsTracker,
    create_reconstruction_figure,
)

# Tracking utilities (from medgen.metrics package)
from medgen.metrics import (
    GradientNormTracker,
    FLOPsTracker,
    measure_model_flops,
    WorstBatchTracker,
    create_worst_batch_figure,
)

# Optimizers (from optimizers/ subdirectory)
from .optimizers import SAM

# Training step result
from .results import TrainingStepResult

# Gradient checkpointing base class
from .checkpointing import BaseCheckpointedModel

# Checkpoint management
from .checkpoint_manager import CheckpointManager

__all__ = [
    # Trainers
    'DiffusionTrainerBase',
    'DiffusionTrainer',
    'VAETrainer',
    'VQVAETrainer',
    'DCAETrainer',
    # Diffusion strategies
    'DDPMStrategy',
    'RFlowStrategy',
    'DiffusionStrategy',
    # Training modes
    'SegmentationMode',
    'ConditionalSingleMode',
    'ConditionalDualMode',
    'TrainingMode',
    # Diffusion spaces
    'DiffusionSpace',
    'PixelSpace',
    'LatentSpace',
    'load_vae_for_latent_space',
    # Model loading utilities
    'load_diffusion_model',
    'load_diffusion_model_with_metadata',
    'detect_wrapper_type',
    'LoadedModel',
    # Metrics and visualization
    'ValidationVisualizer',
    'compute_msssim',
    'compute_psnr',
    'compute_lpips',
    'RegionalMetricsTracker',
    'create_reconstruction_figure',
    # Tracking utilities
    'GradientNormTracker',
    'FLOPsTracker',
    'measure_model_flops',
    'WorstBatchTracker',
    'create_worst_batch_figure',
    # Optimizers
    'SAM',
    # Training step result
    'TrainingStepResult',
    # Test evaluation
    'BaseTestEvaluator',
    'CompressionTestEvaluator',
    'Compression3DTestEvaluator',
    'MetricsConfig',
    'load_checkpoint_if_needed',
    'save_test_results',
    # Validation utilities
    'ValidationRunner',
    'ValidationConfig',
    'ValidationResult',
    # Gradient checkpointing
    'BaseCheckpointedModel',
    # Checkpoint management
    'CheckpointManager',
]
