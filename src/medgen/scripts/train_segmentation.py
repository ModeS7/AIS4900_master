"""Training script for downstream segmentation evaluation.

Train segmentation models on real, synthetic, or mixed data to evaluate
the utility of generated synthetic data.

Usage:
    # Baseline (real data only)
    python -m medgen.scripts.train_segmentation scenario=baseline

    # Synthetic (generated data only)
    python -m medgen.scripts.train_segmentation scenario=synthetic \\
        data.synthetic_dir=runs/diffusion_3d/bravo/.../generated

    # Mixed (real + synthetic)
    python -m medgen.scripts.train_segmentation scenario=mixed \\
        data.synthetic_dir=runs/diffusion_3d/bravo/.../generated \\
        data.synthetic_ratio=0.5

    # 3D training
    python -m medgen.scripts.train_segmentation model.spatial_dims=3
"""
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from medgen.core import setup_cuda_optimizations
from medgen.downstream import (
    SegmentationTrainer,
    create_segmentation_dataloader,
    create_segmentation_test_dataloader,
    create_segmentation_val_dataloader,
)

# Enable CUDA optimizations at module import
setup_cuda_optimizations()

logger = logging.getLogger(__name__)


def validate_config(cfg: DictConfig) -> None:
    """Validate segmentation configuration."""
    errors = []

    # Check scenario
    scenario = cfg.get('scenario', 'baseline')
    if scenario not in ('baseline', 'synthetic', 'mixed'):
        errors.append(f"Invalid scenario: {scenario}. Must be 'baseline', 'synthetic', or 'mixed'")

    # Check synthetic_dir for synthetic/mixed scenarios
    if scenario in ('synthetic', 'mixed'):
        synthetic_dir = cfg.data.get('synthetic_dir')
        if not synthetic_dir:
            errors.append(
                f"data.synthetic_dir is required for scenario='{scenario}'. "
                "Specify path to generated NIfTI files."
            )

    # Check model config
    if 'model' not in cfg:
        errors.append("Missing 'model' config section")
    else:
        if cfg.model.get('image_size', 0) <= 0:
            errors.append("model.image_size must be positive")

    # Check training config
    if 'training' not in cfg:
        errors.append("Missing 'training' config section")
    else:
        if cfg.training.get('epochs', 0) <= 0:
            errors.append("training.epochs must be positive")
        if cfg.training.get('batch_size', 0) <= 0:
            errors.append("training.batch_size must be positive")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))


@hydra.main(version_base=None, config_path="../../../configs", config_name="segmentation")
def main(cfg: DictConfig) -> None:
    """Downstream segmentation training entry point."""
    # Validate config
    validate_config(cfg)

    scenario = cfg.get('scenario', 'baseline')
    spatial_dims = cfg.model.get('spatial_dims', 2)

    # Log configuration
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Log training header
    logger.info("")
    logger.info("=" * 60)
    logger.info("Downstream Segmentation Training")
    logger.info(f"Scenario: {scenario}")
    logger.info(f"Spatial dims: {spatial_dims}D")
    logger.info(f"Image size: {cfg.model.image_size}")
    logger.info(f"Batch size: {cfg.training.batch_size}")
    logger.info(f"Epochs: {cfg.training.epochs}")
    if scenario in ('synthetic', 'mixed'):
        logger.info(f"Synthetic dir: {cfg.data.synthetic_dir}")
    if scenario == 'mixed':
        logger.info(f"Synthetic ratio: {cfg.data.synthetic_ratio}")
    logger.info("=" * 60)
    logger.info("")

    # Create trainer
    if spatial_dims == 3:
        trainer = SegmentationTrainer.create_3d(cfg)
    else:
        trainer = SegmentationTrainer.create_2d(cfg)

    logger.info(f"Validation: every {cfg.training.get('val_every', 1)} epoch(s), "
             f"figures at interval {trainer.figure_interval}")

    # Create dataloaders
    train_loader, train_dataset = create_segmentation_dataloader(
        cfg=cfg,
        scenario=scenario,
        split='train',
        spatial_dims=spatial_dims,
    )
    logger.info(f"Training dataset: {len(train_dataset)} samples")

    # Validation dataloader (always uses real data)
    val_result = create_segmentation_val_dataloader(cfg, spatial_dims)
    if val_result is not None:
        val_loader, val_dataset = val_result
        logger.info(f"Validation dataset: {len(val_dataset)} samples")
    else:
        val_loader = None
        logger.info("No val/ directory found - using train samples for validation")

    # Setup model
    pretrained_checkpoint = cfg.get('pretrained_checkpoint', None)
    if pretrained_checkpoint:
        logger.info(f"Loading pretrained weights from: {pretrained_checkpoint}")
    trainer.setup_model(pretrained_checkpoint=pretrained_checkpoint)

    # Resume from checkpoint if specified
    start_epoch = 0
    resume_from = cfg.training.get('resume_from', None)
    if resume_from:
        result = trainer.checkpoint_manager.load(resume_from, strict=True, load_optimizer=True)
        start_epoch = result['epoch'] + 1
        # Restore best_dice from checkpoint
        ckpt = result['checkpoint']
        if 'best_dice' in ckpt:
            trainer.best_dice = ckpt['best_dice']
        logger.info(f"Resuming from epoch {start_epoch} (best_dice={trainer.best_dice:.4f})")

    # Train
    trainer.train(
        train_loader=train_loader,
        train_dataset=train_dataset,
        val_loader=val_loader,
        start_epoch=start_epoch,
    )

    # On wall-time signal: skip test eval, let SLURM script handle resubmission
    if getattr(trainer, '_sigterm_received', False):
        logger.info("Wall time signal received — skipping test evaluation")
        trainer.close_writer()
        return

    # Test evaluation
    test_result = create_segmentation_test_dataloader(cfg, spatial_dims)
    if test_result is not None:
        test_loader, test_dataset = test_result
        logger.info(f"Running test evaluation on {len(test_dataset)} samples...")

        # Set test loader as val_loader for evaluation
        trainer.val_loader = test_loader
        test_metrics = trainer.compute_validation_losses(
            epoch=trainer.n_epochs,
            log_figures=False,
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("Test Results")
        logger.info("=" * 60)
        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info("=" * 60)
    else:
        logger.info("No test_new/ directory found - skipping test evaluation")

    trainer.close_writer()

    # Mark training complete for SLURM job chaining
    from pathlib import Path
    (Path(trainer.save_dir) / '.training_complete').touch()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
