"""Generative pipeline for training and inference (diffusion and VAE)."""

from .trainer import DiffusionTrainer
from .vae_trainer import VAETrainer
from .vae_3d_trainer import VAE3DTrainer
from .vqvae_trainer import VQVAETrainer
from .vqvae_3d_trainer import VQVAE3DTrainer
from .dcae_trainer import DCAETrainer
from .dcae_3d_trainer import DCAE3DTrainer
# Diffusion components (from medgen.diffusion package)
from medgen.diffusion import (
    DDPMStrategy, RFlowStrategy, DiffusionStrategy,
    SegmentationMode, ConditionalSingleMode, ConditionalDualMode, TrainingMode,
    DiffusionSpace, PixelSpace, LatentSpace, load_vae_for_latent_space,
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
    MetricsTracker,
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

__all__ = [
    # Trainers
    'DiffusionTrainer',
    'VAETrainer',
    'VAE3DTrainer',
    'VQVAETrainer',
    'VQVAE3DTrainer',
    'DCAETrainer',
    'DCAE3DTrainer',
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
    # Metrics and visualization
    'MetricsTracker',
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
]
