"""Evaluation utilities for training and testing.

This package provides:
- Validation: Reusable validation loop components for compression trainers
- Evaluation: Test set evaluation with checkpoint loading and results saving
- Visualization: Training progress visualization and sample generation

Usage:
    from medgen.evaluation import (
        # Validation
        ValidationRunner,
        ValidationConfig,
        ValidationResult,
        # Test evaluation
        BaseTestEvaluator,
        CompressionTestEvaluator,
        Compression3DTestEvaluator,
        MetricsConfig,
        load_checkpoint_if_needed,
        save_test_results,
        # Visualization
        ValidationVisualizer,
    )
"""

from .validation import (
    ValidationRunner,
    ValidationConfig,
    ValidationResult,
)

from .evaluation import (
    BaseTestEvaluator,
    CompressionTestEvaluator,
    Compression3DTestEvaluator,
    MetricsConfig,
    load_checkpoint_if_needed,
    save_test_results,
)

from .visualization import ValidationVisualizer

__all__ = [
    # Validation
    'ValidationRunner',
    'ValidationConfig',
    'ValidationResult',
    # Test evaluation
    'BaseTestEvaluator',
    'CompressionTestEvaluator',
    'Compression3DTestEvaluator',
    'MetricsConfig',
    'load_checkpoint_if_needed',
    'save_test_results',
    # Visualization
    'ValidationVisualizer',
]
