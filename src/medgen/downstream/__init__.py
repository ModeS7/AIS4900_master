"""Downstream task evaluation for synthetic data utility.

This module provides training and evaluation for downstream segmentation models
to measure the utility of generated synthetic data.

Training Scenarios:
    - baseline: Train on real data only (control)
    - synthetic: Train on synthetic data only
    - mixed: Train on real + synthetic data

Evaluation:
    - Overall Dice score
    - Per-tumor-size Dice (0-10mm, 10-20mm, 20-30mm, 30+mm)
    - IoU metrics

Usage:
    from medgen.downstream import SegmentationTrainer, create_segmentation_dataloader

    trainer = SegmentationTrainer(cfg, spatial_dims=2)
    trainer.setup_model()
    trainer.train(train_loader, train_dataset, val_loader)
"""

from .segmentation_trainer import SegmentationTrainer
from .data import (
    create_segmentation_dataloader,
    create_segmentation_val_dataloader,
    create_segmentation_test_dataloader,
    SegmentationDataset,
    SyntheticDataset,
)

__all__ = [
    'SegmentationTrainer',
    'create_segmentation_dataloader',
    'create_segmentation_val_dataloader',
    'create_segmentation_test_dataloader',
    'SegmentationDataset',
    'SyntheticDataset',
]
