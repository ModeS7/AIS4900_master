"""
Progressive resolution VAE training.

Trains VAE at increasing resolutions (64 -> 128 -> 256) with automatic
plateau detection for phase transitions. Trains on multiple modalities
(bravo, flair, t1_pre, t1_gd) to create a pre-trained model for fine-tuning.

Usage:
    # Full progressive training
    python -m medgen.scripts.train_vae_progressive

    # Resume from checkpoint
    python -m medgen.scripts.train_vae_progressive progressive.resume_from=/path/to/progressive_state.pt

    # Custom settings
    python -m medgen.scripts.train_vae_progressive \
        progressive.final_phase.epochs=100 \
        progressive.plateau.min_improvement=1.0
"""
import logging
import os
import shutil
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from medgen.core import setup_cuda_optimizations
from medgen.data import (
    create_multi_modality_dataloader,
    create_multi_modality_validation_dataloader,
    create_multi_modality_test_dataloader,
)
from medgen.diffusion.vae_trainer import VAETrainer, log_vae_epoch_summary

# Enable CUDA optimizations
setup_cuda_optimizations()

log = logging.getLogger(__name__)


class PlateauDetector:
    """Detect loss plateau for phase transition.

    Monitors training loss and detects when improvement rate falls below
    a threshold, indicating the model has converged at current resolution.

    Args:
        window_size: Number of epochs for rolling average.
        min_improvement: Minimum % improvement required (e.g., 0.5 = 0.5%).
        min_epochs: Minimum epochs before checking for plateau.
        patience: Epochs below threshold before declaring plateau.
    """

    def __init__(
        self,
        window_size: int = 10,
        min_improvement: float = 0.5,
        min_epochs: int = 20,
        patience: int = 5
    ):
        self.window_size = window_size
        self.min_improvement = min_improvement / 100.0  # Convert to fraction
        self.min_epochs = min_epochs
        self.patience = patience

        self.loss_history: deque = deque(maxlen=window_size * 2)
        self.epochs_without_improvement = 0
        self.best_window_avg = float('inf')

    def reset(self) -> None:
        """Reset detector for new phase."""
        self.loss_history.clear()
        self.epochs_without_improvement = 0
        self.best_window_avg = float('inf')

    def update(self, loss: float, epoch: int) -> bool:
        """Update with new loss and check for plateau.

        Args:
            loss: Current epoch loss.
            epoch: Current epoch number (0-indexed).

        Returns:
            True if plateau detected, False otherwise.
        """
        self.loss_history.append(loss)

        # Not enough history yet
        if epoch < self.min_epochs or len(self.loss_history) < self.window_size:
            return False

        # Calculate rolling average
        recent_losses = list(self.loss_history)[-self.window_size:]
        recent_avg = sum(recent_losses) / len(recent_losses)

        # Check improvement rate
        if self.best_window_avg > 0:
            improvement = (self.best_window_avg - recent_avg) / self.best_window_avg
        else:
            improvement = 0

        if improvement > self.min_improvement:
            self.best_window_avg = recent_avg
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        # Plateau detected if no improvement for patience epochs
        return self.epochs_without_improvement >= self.patience


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
        dataloader, dataset = create_multi_modality_dataloader(
            cfg=self.cfg,
            image_keys=self.image_keys,
            image_size=resolution,
            batch_size=batch_size
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
        prev_checkpoint = None
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

        # Set validation loader for metrics (if available)
        trainer.val_loader = val_loader

        # Measure FLOPs (once per phase)
        if trainer.log_flops and not trainer._flops_measured:
            try:
                first_batch = next(iter(dataloader))
                sample_images = trainer._prepare_batch(first_batch)
                trainer._measure_model_flops(sample_images)
            except Exception as e:
                log.warning(f"Could not measure FLOPs: {e}")

        # Reset plateau detector
        self.plateau_detector.reset()

        # Training loop with plateau detection
        epoch = 0
        max_epochs = 500  # Safety limit

        while epoch < max_epochs:
            epoch_start = time.time()

            # Train one epoch
            avg_losses = trainer.train_epoch(dataloader, epoch)

            # Step schedulers (only if not using constant LR)
            if hasattr(trainer, 'lr_scheduler_g') and trainer.lr_scheduler_g is not None:
                trainer.lr_scheduler_g.step()
            if hasattr(trainer, 'lr_scheduler_d') and trainer.lr_scheduler_d is not None:
                trainer.lr_scheduler_d.step()

            # Update EMA if enabled
            if trainer.ema is not None:
                trainer.ema.update()

            elapsed = time.time() - epoch_start

            # Compute validation metrics every epoch
            val_metrics = trainer.compute_validation_losses(epoch)

            # Log progress
            if is_final:
                total_epochs = self.cfg.progressive.final_phase.epochs
            else:
                total_epochs = "?"  # Unknown for plateau-based phases

            log_vae_epoch_summary(epoch, total_epochs if isinstance(total_epochs, int) else 999, avg_losses, val_metrics, elapsed)

            # Log to tensorboard
            if trainer.writer is not None:
                global_step = epoch
                trainer.writer.add_scalar('Loss/Generator_train', avg_losses['gen'], global_step)
                trainer.writer.add_scalar('Loss/L1_train', avg_losses['recon'], global_step)
                trainer.writer.add_scalar('Loss/Perceptual_train', avg_losses['perc'], global_step)
                trainer.writer.add_scalar('Loss/KL_train', avg_losses['kl'], global_step)
                lr_g = trainer.optimizer_g.param_groups[0]['lr']
                trainer.writer.add_scalar('LR/Generator', lr_g, global_step)

                # Log gradient norms
                trainer._log_grad_norms(global_step)

            # Visualization logging at intervals
            is_val_epoch = (epoch + 1) % self.cfg.training.val_interval == 0
            if is_val_epoch and trainer.writer is not None:
                # Log worst batch
                if trainer.log_worst_batch:
                    trainer._log_worst_batch(epoch)

                # Generate validation visualizations (metrics already logged above)
                trainer.generate_validation_visualizations(dataset, epoch)

            # Save best checkpoint (use validation loss for proper overfitting detection)
            val_gen_loss = val_metrics.get('gen', avg_losses['gen'])
            if val_gen_loss < trainer.best_loss:
                trainer.best_loss = val_gen_loss
                trainer._save_vae_checkpoint(epoch, "best")
                log.info(f"New best model saved (val G loss: {val_gen_loss:.6f})")

            # Check termination condition
            if is_final:
                # Final phase: fixed number of epochs
                if epoch >= self.cfg.progressive.final_phase.epochs - 1:
                    break
            else:
                # Earlier phases: plateau detection (use validation loss)
                if self.plateau_detector.update(val_gen_loss, epoch):
                    log.info(f"Plateau detected at epoch {epoch + 1}")
                    break

            # Save periodic checkpoints
            if (epoch + 1) % self.cfg.training.val_interval == 0:
                trainer._save_vae_checkpoint(epoch, f"epoch_{epoch + 1:04d}")

            epoch += 1

        # Save phase completion
        trainer._save_vae_checkpoint(epoch, "latest")
        self._save_progressive_checkpoint(epoch, resolution)

        # Cleanup trainer
        if trainer.writer is not None:
            trainer.writer.close()

        return epoch + 1

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


def validate_config(cfg: DictConfig) -> None:
    """Validate progressive training configuration.

    Args:
        cfg: Hydra configuration object.

    Raises:
        ValueError: If configuration is invalid.
    """
    errors = []

    # Check progressive config
    if not hasattr(cfg, 'progressive'):
        errors.append("Missing 'progressive' configuration section")
    else:
        if not cfg.progressive.resolutions:
            errors.append("progressive.resolutions must not be empty")

        for res in cfg.progressive.resolutions:
            if res not in cfg.progressive.batch_sizes:
                errors.append(f"Missing batch_size for resolution {res}")

    # Check modalities
    if not hasattr(cfg, 'modalities'):
        errors.append("Missing 'modalities' configuration section")
    elif not cfg.modalities.image_keys:
        errors.append("modalities.image_keys must not be empty")

    # Check VAE config
    if not hasattr(cfg, 'vae'):
        errors.append("Missing 'vae' configuration section")

    # Check paths
    if not os.path.exists(cfg.paths.data_dir):
        errors.append(f"Data directory does not exist: {cfg.paths.data_dir}")

    # Check CUDA
    if not torch.cuda.is_available():
        errors.append("CUDA is not available. Training requires GPU.")

    if errors:
        raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))


@hydra.main(version_base=None, config_path="../../../configs", config_name="train_vae_progressive")
def main(cfg: DictConfig) -> None:
    """Progressive VAE training entry point.

    Args:
        cfg: Hydra configuration object.
    """
    # Validate configuration
    validate_config(cfg)

    # Print configuration
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    print(f"\n{'=' * 60}")
    print("PROGRESSIVE VAE TRAINING")
    print(f"{'=' * 60}")
    print(f"Resolutions: {cfg.progressive.resolutions}")
    print(f"Modalities: {cfg.modalities.image_keys}")
    print(f"Plateau detection: window={cfg.progressive.plateau.window_size}, "
          f"min_improvement={cfg.progressive.plateau.min_improvement}%, "
          f"patience={cfg.progressive.plateau.patience}")
    print(f"Final phase epochs: {cfg.progressive.final_phase.epochs}")
    print(f"{'=' * 60}\n")

    # Create and run trainer
    trainer = ProgressiveVAETrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
