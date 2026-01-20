"""
Standalone metrics package for medical image generation.

This package provides:
- UnifiedMetrics: Single entry point for all metric tracking and TensorBoard logging
- SimpleLossAccumulator: Dynamic loss tracking without predefined config
- Quality metrics: PSNR, MS-SSIM (2D/3D), LPIPS (2D only), Dice, IoU
- Regional metrics: Per-tumor loss tracking by RANO-BM size categories
- Generation metrics: KID, CMMD, FID for evaluating generative models
- Visualization: Reconstruction figures and error heatmaps

Usage:
    from medgen.metrics import UnifiedMetrics, SimpleLossAccumulator

    metrics = UnifiedMetrics(
        writer=tensorboard_writer,
        mode='bravo',
        spatial_dims=3,
    )
"""

from .unified import SimpleLossAccumulator, UnifiedMetrics
from .quality import (
    compute_dice,
    compute_iou,
    compute_msssim,
    compute_msssim_2d_slicewise,
    compute_psnr,
    compute_lpips,
    compute_lpips_3d,
    reset_msssim_nan_warning,
    reset_lpips_nan_warning,
    clear_metric_caches,
    # Diversity metrics
    compute_lpips_diversity,
    compute_msssim_diversity,
    compute_lpips_diversity_3d,
    compute_msssim_diversity_3d,
)
from .regional import (
    BaseRegionalMetricsTracker,
    RegionalMetricsTracker,
    RegionalMetricsTracker3D,
    SegRegionalMetricsTracker,
)
from .figures import create_reconstruction_figure, figure_to_buffer
from .constants import TUMOR_SIZE_THRESHOLDS_MM, TUMOR_SIZE_CATEGORIES
from .feature_extractors import ResNet50Features, BiomedCLIPFeatures
from .generation import (
    GenerationMetricsConfig,
    GenerationMetrics,
    compute_kid,
    compute_cmmd,
    compute_fid,
    # 3D slice-wise (2.5D) generation metrics
    volumes_to_slices,
    extract_features_3d,
    compute_kid_3d,
    compute_cmmd_3d,
    compute_fid_3d,
)
from .tracking import (
    GradientNormTracker,
    FLOPsTracker,
    measure_model_flops,
    WorstBatchTracker,
    create_worst_batch_figure,
    create_worst_batch_figure_3d,
    CodebookTracker,
)

__all__ = [
    # Core
    'SimpleLossAccumulator',
    'UnifiedMetrics',
    # Quality metrics
    'compute_dice',
    'compute_iou',
    'compute_msssim',
    'compute_msssim_2d_slicewise',
    'compute_psnr',
    'compute_lpips',
    'compute_lpips_3d',
    'reset_msssim_nan_warning',
    'reset_lpips_nan_warning',
    'clear_metric_caches',
    # Diversity metrics
    'compute_lpips_diversity',
    'compute_msssim_diversity',
    'compute_lpips_diversity_3d',
    'compute_msssim_diversity_3d',
    # Regional metrics
    'BaseRegionalMetricsTracker',
    'RegionalMetricsTracker',
    'RegionalMetricsTracker3D',
    'SegRegionalMetricsTracker',
    # Constants
    'TUMOR_SIZE_THRESHOLDS_MM',
    'TUMOR_SIZE_CATEGORIES',
    # Visualization
    'create_reconstruction_figure',
    'figure_to_buffer',
    # Generation metrics
    'GenerationMetricsConfig',
    'GenerationMetrics',
    'compute_kid',
    'compute_cmmd',
    'compute_fid',
    'volumes_to_slices',
    'extract_features_3d',
    'compute_kid_3d',
    'compute_cmmd_3d',
    'compute_fid_3d',
    'ResNet50Features',
    'BiomedCLIPFeatures',
    # Tracking utilities
    'GradientNormTracker',
    'FLOPsTracker',
    'measure_model_flops',
    'WorstBatchTracker',
    'create_worst_batch_figure',
    'create_worst_batch_figure_3d',
    'CodebookTracker',
]
