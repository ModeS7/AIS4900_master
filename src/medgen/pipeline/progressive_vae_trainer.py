"""
Progressive resolution VAE trainer.

Orchestrates training VAE at increasing resolutions (64 -> 128 -> 256) with
automatic plateau detection for phase transitions. Trains on multiple modalities
(bravo, flair, t1_pre, t1_gd) to create a pre-trained model for fine-tuning.
"""
import logging
import os
import shutil
import time
from datetime import datetime
from typing import Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from medgen.data import (
    create_multi_modality_dataloader,
    create_multi_modality_test_dataloader,
    create_multi_modality_validation_dataloader,
)
from medgen.pipeline.plateau_detection import PlateauDetector
from medgen.pipeline.vae_trainer import VAETrainer

log = logging.getLogger(__name__)


class ProgressiveVAETrainer:
    """Orchestrates progressive resolution VAE training.

    Trains VAE through multiple resolution phases, using plateau detection
    to determine when to transition to the next resolution.

    Args:
        cfg: Hydra configuration object.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.resolutions = list(cfg.progressive.resolutions)
        self.batch_sizes = {int(k): v for k, v in cfg.progressive.batch_sizes.items()}
        self.image_keys = list(cfg.modalities.image_keys)

        # Setup output directory
        try:
            from hydra.core.hydra_config import HydraConfig
            self.base_dir = HydraConfig.get().runtime.output_dir
        except (ImportError, ValueError, AttributeError):
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.base_dir = os.path.join(
                cfg.paths.model_dir, 'vae_2d', 'progressive', timestamp
            )

        os.makedirs(self.base_dir, exist_ok=True)

        # Training state
        self.current_phase = 0
        self.phase_epochs: Dict[int, int] = {}  # {resolution: epochs_completed}
        self.device = torch.device("cuda")

        # Plateau detector
        self.plateau_detector = PlateauDetector(
            window_size=cfg.progressive.plateau.window_size,
            min_improvement=cfg.progressive.plateau.min_improvement,
            min_epochs=cfg.progressive.plateau.min_epochs,
            patience=cfg.progressive.plateau.patience
        )

        # Save config
        config_path = os.path.join(self.base_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    def train(self) -> None:
        """Execute progressive training across all phases."""
        start_time = time.time()

        # Check for resume
        if self.cfg.progressive.resume_from:
            self._resume_from_checkpoint(self.cfg.progressive.resume_from)

        for phase_idx, resolution in enumerate(self.resolutions):
            if phase_idx < self.current_phase:
                continue  # Skip completed phases on resume

            self.current_phase = phase_idx
            is_final_phase = (resolution == self.resolutions[-1])

            log.info(f"\n{'=' * 60}")
            log.info(f"PHASE {phase_idx + 1}/{len(self.resolutions)}: {resolution}x{resolution}")
            log.info(f"Batch size: {self.batch_sizes[resolution]}")
            log.info(f"Final phase: {is_final_phase}")
            log.info(f"{'=' * 60}")

            # Run phase
            epochs_completed = self._train_phase(
                resolution=resolution,
                batch_size=self.batch_sizes[resolution],
                is_final=is_final_phase
            )

            self.phase_epochs[resolution] = epochs_completed
            log.info(f"Phase {resolution}x{resolution} completed after {epochs_completed} epochs")

        # Save final model
        self._save_final_model()

        # Run test evaluation on final model
        self._evaluate_final_model()

        total_time = time.time() - start_time
        log.info(f"\nProgressive training completed in {total_time / 3600:.2f} hours")
        log.info(f"Final model saved to: {os.path.join(self.base_dir, 'final_model.pt')}")

    def _train_phase(
        self,
        resolution: int,
        batch_size: int,
        is_final: bool
    ) -> int:
        """Train a single resolution phase.

        Args:
            resolution: Image resolution for this phase.
            batch_size: Batch size for this phase.
            is_final: Whether this is the final resolution phase.

        Returns:
            Number of epochs completed in this phase.
        """
        phase_dir = os.path.join(self.base_dir, f"phase_{resolution}")
        os.makedirs(phase_dir, exist_ok=True)

        # Create dataloader for this resolution
        log.info(f"Creating dataloader for {resolution}x{resolution}...")
        augment = self.cfg.training.get('augment', True)
        dataloader, dataset = create_multi_modality_dataloader(
            cfg=self.cfg,
            image_keys=self.image_keys,
            image_size=resolution,
            batch_size=batch_size,
            augment=augment
        )
        log.info(f"Training dataset: {len(dataset)} slices, {len(dataloader)} batches")

        # Create validation dataloader (if val/ directory exists)
        val_loader = None
        val_result = create_multi_modality_validation_dataloader(
            cfg=self.cfg,
            image_keys=self.image_keys,
            image_size=resolution,
            batch_size=batch_size
        )
        if val_result is not None:
            val_loader, val_dataset = val_result
            log.info(f"Validation dataset: {len(val_dataset)} slices")

        # Create modified config for this phase
        phase_cfg = self._create_phase_config(resolution, batch_size, phase_dir)

        # Create trainer
        trainer = VAETrainer(phase_cfg)

        # Load weights from previous phase (if not first)
        prev_checkpoint: Optional[str] = None
        if self.current_phase > 0:
            prev_resolution = self.resolutions[self.current_phase - 1]
            prev_checkpoint = os.path.join(
                self.base_dir, f"phase_{prev_resolution}", "best.pt"
            )
            if os.path.exists(prev_checkpoint):
                log.info(f"Loading weights from previous phase: {prev_checkpoint}")
            else:
                log.warning(f"Previous checkpoint not found: {prev_checkpoint}")
                prev_checkpoint = None

        # Setup model (with optional pretrained weights)
        trainer.setup_model(pretrained_checkpoint=prev_checkpoint)

        # Reset plateau detector for this phase
        self.plateau_detector.reset()

        # Configure training parameters based on phase type
        if is_final:
            # Final phase: fixed number of epochs, no early stopping
            max_epochs = self.cfg.progressive.final_phase.epochs
            early_stop_fn = None
        else:
            # Earlier phases: plateau detection for early stopping
            max_epochs = 500  # Safety limit

            def early_stop_fn(epoch, losses, val_metrics):
                self.plateau_detector.update(val_metrics.get('gen', losses['gen']), epoch)
                return self.plateau_detector.is_plateau()

        # Run training using VAETrainer.train() - handles all logging, checkpointing, etc.
        last_epoch = trainer.train(
            train_loader=dataloader,
            train_dataset=dataset,
            val_loader=val_loader,
            max_epochs=max_epochs,
            early_stop_fn=early_stop_fn,
        )

        # Save progressive state checkpoint
        self._save_progressive_checkpoint(last_epoch, resolution)

        # Cleanup trainer
        trainer.close_writer()

        return last_epoch + 1

    def _create_phase_config(
        self,
        resolution: int,
        batch_size: int,
        phase_dir: str
    ) -> DictConfig:
        """Create config for a specific phase.

        Args:
            resolution: Image resolution.
            batch_size: Batch size.
            phase_dir: Output directory for this phase.

        Returns:
            Modified config for this phase.
        """
        # Deep copy config
        phase_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))

        # Override resolution-specific settings
        phase_cfg.model.image_size = resolution
        phase_cfg.training.batch_size = batch_size

        # Set mode config for multi-modality (single channel, mixed modalities)
        phase_cfg.mode = OmegaConf.create({
            'name': 'multi_modality',
            'is_conditional': False,
            'in_channels': 1,  # Each image is single-channel
            'out_channels': 1,
            'image_keys': self.image_keys,
        })

        # Override paths to use phase directory
        phase_cfg.paths.model_dir = self.base_dir

        # Set explicit save directory for VAETrainer to use
        phase_cfg.save_dir_override = phase_dir

        return phase_cfg

    def _save_progressive_checkpoint(self, epoch: int, resolution: int) -> None:
        """Save checkpoint with progressive training state.

        Args:
            epoch: Current epoch within phase.
            resolution: Current resolution.
        """
        checkpoint = {
            'current_phase': self.current_phase,
            'resolution': resolution,
            'epoch': epoch,
            'phase_epochs': self.phase_epochs,
            'resolutions': self.resolutions,
        }

        path = os.path.join(self.base_dir, "progressive_state.pt")
        torch.save(checkpoint, path)
        log.info(f"Progressive state saved to: {path}")

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume from progressive checkpoint.

        Args:
            checkpoint_path: Path to progressive_state.pt.
        """
        if not os.path.exists(checkpoint_path):
            log.warning(f"Resume checkpoint not found: {checkpoint_path}")
            return

        state = torch.load(checkpoint_path, map_location='cpu')
        self.current_phase = state['current_phase']
        self.phase_epochs = state.get('phase_epochs', {})

        log.info(f"Resuming from phase {self.current_phase + 1}, resolution {state['resolution']}")

    def _save_final_model(self) -> None:
        """Copy final phase best model to top level."""
        final_resolution = self.resolutions[-1]
        src = os.path.join(self.base_dir, f"phase_{final_resolution}", "best.pt")
        dst = os.path.join(self.base_dir, "final_model.pt")

        if os.path.exists(src):
            shutil.copy(src, dst)
            log.info(f"Final model copied to: {dst}")
        else:
            log.warning(f"Final phase checkpoint not found: {src}")

    def _evaluate_final_model(self) -> None:
        """Run test evaluation on the final trained model.

        Creates a fresh trainer with the final resolution, loads the best
        checkpoint, and evaluates on the test set if it exists.
        """
        final_resolution = self.resolutions[-1]
        batch_size = self.batch_sizes[final_resolution]

        # Check if test set exists
        test_result = create_multi_modality_test_dataloader(
            cfg=self.cfg,
            image_keys=self.image_keys,
            image_size=final_resolution,
            batch_size=batch_size
        )

        if test_result is None:
            log.info("No test_new/ directory found - skipping test evaluation")
            return

        test_loader, test_dataset = test_result
        log.info(f"Test dataset: {len(test_dataset)} slices")

        # Create trainer for evaluation
        phase_dir = os.path.join(self.base_dir, f"phase_{final_resolution}")
        phase_cfg = self._create_phase_config(final_resolution, batch_size, phase_dir)

        trainer = VAETrainer(phase_cfg)
        trainer.setup_model()

        # Evaluate on best and latest checkpoints from final phase
        best_checkpoint = os.path.join(self.base_dir, "best.pt")
        latest_checkpoint = os.path.join(self.base_dir, "final_model.pt")

        # Evaluate best model
        if os.path.exists(best_checkpoint):
            trainer.load_checkpoint(best_checkpoint, load_optimizer=False)
            log.info(f"Loaded best model from: {best_checkpoint}")
            trainer.evaluate_test_set(test_loader, checkpoint_name="best")
        else:
            log.warning(f"Best checkpoint not found: {best_checkpoint}")

        # Evaluate latest/final model
        if os.path.exists(latest_checkpoint):
            trainer.load_checkpoint(latest_checkpoint, load_optimizer=False)
            log.info(f"Loaded final model from: {latest_checkpoint}")
            trainer.evaluate_test_set(test_loader, checkpoint_name="latest")
        else:
            log.warning(f"Final model not found: {latest_checkpoint}")

        # Close TensorBoard writer
        trainer.close_writer()
