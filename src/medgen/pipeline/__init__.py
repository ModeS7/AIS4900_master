"""Generative pipeline for training and inference (diffusion and VAE)."""

from .trainer import DiffusionTrainer
from .vae_trainer import VAETrainer
from .vae_3d_trainer import VAE3DTrainer
from .vqvae_trainer import VQVAETrainer
from .vqvae_3d_trainer import VQVAE3DTrainer
from .progressive_vae_trainer import ProgressiveVAETrainer
from .dcae_trainer import DCAETrainer
from .plateau_detection import PlateauDetector
from .strategies import DDPMStrategy, RFlowStrategy, DiffusionStrategy
from .modes import SegmentationMode, ConditionalSingleMode, ConditionalDualMode, TrainingMode
from .spaces import DiffusionSpace, PixelSpace, LatentSpace, load_vae_for_latent_space
from .visualization import ValidationVisualizer

# Metrics (from metrics/ subdirectory)
from .metrics import (
    MetricsTracker,
    compute_msssim,
    compute_psnr,
    compute_lpips,
    RegionalMetricsTracker,
    create_reconstruction_figure,
)

# Tracking utilities (from tracking/ subdirectory)
from .tracking import (
    GradientNormTracker,
    FLOPsTracker,
    measure_model_flops,
    WorstBatchTracker,
    create_worst_batch_figure,
)

__all__ = [
    # Trainers
    'DiffusionTrainer',
    'VAETrainer',
    'VAE3DTrainer',
    'VQVAETrainer',
    'VQVAE3DTrainer',
    'ProgressiveVAETrainer',
    'DCAETrainer',
    'PlateauDetector',
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
]
