"""
Base trainer module providing shared infrastructure for all trainers.

This module provides the BaseTrainer abstract class that handles:
- Device and distributed training setup
- TensorBoard logging infrastructure
- Checkpoint directory management
- VRAM, FLOPs, and learning rate logging
- Common training loop skeleton

All concrete trainers (VAE, VQ-VAE, DC-AE, Diffusion) inherit from this base
or from BaseCompressionTrainer which extends it.
"""
import logging
import os
import signal
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from medgen.core import setup_distributed
from medgen.metrics import FLOPsTracker, GradientNormTracker
from medgen.pipeline.results import BatchType, TrainingStepResult

from .base_config import BaseTrainingConfig, PathsConfig
from .checkpoint_manager import CheckpointManager
from .utils import EpochTimeEstimator

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base trainer providing shared infrastructure for all trainers.

    This class handles:
    - Config extraction for common training parameters
    - Device setup (single GPU or distributed)
    - TensorBoard writer initialization
    - Save directory creation
    - VRAM, FLOPs, and learning rate logging
    - Training loop skeleton via template method pattern

    Subclasses must implement:
    - setup_model(): Initialize model, optimizer, scheduler
    - train_step(): Single training step
    - train_epoch(): Train for one epoch
    - compute_validation_losses(): Compute validation metrics
    - _save_checkpoint(): Save model checkpoint

    Subclasses can optionally override hook methods:
    - _on_training_start(): Called before training loop
    - _on_epoch_start(): Called at start of each epoch
    - _on_epoch_end(): Called at end of each epoch
    - _on_training_end(): Called after training loop
    - _prepare_batch(): Prepare batch for training

    Args:
        cfg: Hydra configuration object.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize base trainer with common configuration.

        Args:
            cfg: Hydra configuration object containing all settings.
        """
        self.cfg = cfg

        # ─────────────────────────────────────────────────────────────────────
        # Extract typed configs (single-source defaults)
        # ─────────────────────────────────────────────────────────────────────
        tc = BaseTrainingConfig.from_hydra(cfg)
        pc = PathsConfig.from_hydra(cfg)
        self._training_config = tc
        self._paths_config = pc

        # ─────────────────────────────────────────────────────────────────────
        # Common training config (from typed BaseTrainingConfig)
        # ─────────────────────────────────────────────────────────────────────
        self.n_epochs: int = tc.n_epochs
        self.batch_size: int = tc.batch_size
        self.learning_rate: float = tc.learning_rate
        self.warmup_epochs: int = tc.warmup_epochs
        self.figure_interval: int = tc.get_figure_interval()
        self.use_multi_gpu: bool = tc.use_multi_gpu
        self.gradient_clip_norm: float = tc.gradient_clip_norm
        self.limit_train_batches: int | None = tc.limit_train_batches

        # Gradient spike detection — skip optimizer step on anomalous gradients
        from .utils import GradientSkipDetector
        self._grad_skip_detector = GradientSkipDetector()

        # SIGTERM handling for graceful SLURM shutdown
        self._sigterm_received = False
        self._install_sigterm_handler()

        # Determine if running on cluster
        self.is_cluster: bool = pc.is_cluster

        # Verbose mode (auto-detect from paths.name if not set)
        self.verbose: bool = tc.get_verbose(self.is_cluster)

        # ─────────────────────────────────────────────────────────────────────
        # Logging config (from typed BaseTrainingConfig)
        # ─────────────────────────────────────────────────────────────────────
        self.log_grad_norm: bool = tc.log_grad_norm
        self.log_psnr: bool = tc.log_psnr
        self.log_lpips: bool = tc.log_lpips
        self.log_msssim: bool = tc.log_msssim
        self.log_regional_losses: bool = tc.log_regional_losses
        self.log_flops: bool = tc.log_flops

        # ─────────────────────────────────────────────────────────────────────
        # Profiling config (from typed ProfilingConfig)
        # ─────────────────────────────────────────────────────────────────────
        self._profiling_enabled: bool = tc.profiling.enabled
        self._profiling_config = tc.profiling
        self._profiler: torch.profiler.profile | None = None

        # ─────────────────────────────────────────────────────────────────────
        # Setup device and distributed training
        # ─────────────────────────────────────────────────────────────────────
        if self.use_multi_gpu:
            self.rank, self.local_rank, self.world_size, self.device = self._setup_distributed()
            self.is_main_process: bool = (self.rank == 0)
        else:
            self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.is_main_process = True
            self.rank: int = 0
            self.local_rank: int = 0
            self.world_size: int = 1

        # ─────────────────────────────────────────────────────────────────────
        # Initialize save directory and TensorBoard
        # ─────────────────────────────────────────────────────────────────────
        if self.is_main_process:
            self.save_dir = self._create_save_dir()
            self.writer: SummaryWriter | None = self._init_tensorboard()
            self.best_loss: float = float('inf')
        else:
            self.save_dir = ""
            self.writer = None
            self.best_loss = float('inf')

        # ─────────────────────────────────────────────────────────────────────
        # Initialize trackers
        # ─────────────────────────────────────────────────────────────────────
        self._flops_tracker = FLOPsTracker()
        self._grad_norm_tracker = GradientNormTracker()

        # ─────────────────────────────────────────────────────────────────────
        # Model placeholders (set in setup_model)
        # ─────────────────────────────────────────────────────────────────────
        self.model: nn.Module | None = None
        self.model_raw: nn.Module | None = None
        self.optimizer: Optimizer | None = None
        self.lr_scheduler: LRScheduler | None = None
        self.val_loader: DataLoader | None = None

        # ─────────────────────────────────────────────────────────────────────
        # Checkpoint manager (initialized in _setup_checkpoint_manager)
        # ─────────────────────────────────────────────────────────────────────
        self.checkpoint_manager: CheckpointManager | None = None

    def _setup_distributed(self) -> tuple[int, int, int, torch.device]:
        """Setup distributed training.

        Returns:
            Tuple of (rank, local_rank, world_size, device).
        """
        return setup_distributed()

    def _create_save_dir(self) -> str:
        """Create save directory for checkpoints and logs.

        Uses Hydra's output directory if available, otherwise creates
        a timestamped directory under the model directory.

        Returns:
            Path to the save directory.
        """
        # Check for explicit save_dir override (used by progressive training)
        if hasattr(self.cfg, 'save_dir_override') and self.cfg.save_dir_override:
            save_dir = self.cfg.save_dir_override
        else:
            try:
                from hydra.core.hydra_config import HydraConfig
                save_dir = HydraConfig.get().runtime.output_dir
            except (ImportError, ValueError, AttributeError):
                # Fallback to manual directory creation
                save_dir = self._create_fallback_save_dir()

        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _create_fallback_save_dir(self) -> str:
        """Create fallback save directory when Hydra is not available.

        Subclasses can override to customize the directory structure.

        Returns:
            Path to the save directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = self._training_config.name
        image_size = getattr(self.cfg.model, 'image_size', 256)
        run_name = f"{exp_name}{image_size}_{timestamp}"
        return os.path.join(self._paths_config.model_dir, 'runs', run_name)

    def _init_tensorboard(self) -> SummaryWriter | None:
        """Initialize TensorBoard writer.

        Returns:
            SummaryWriter instance or None if not main process.
        """
        if not self.is_main_process:
            return None

        tensorboard_dir = os.path.join(self.save_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        return SummaryWriter(tensorboard_dir)

    def _log_vram(self, epoch: int) -> None:
        """Log VRAM usage to TensorBoard.

        Args:
            epoch: Current epoch number.
        """
        if not self.is_main_process:
            return
        # Use unified metrics (all trainers must initialize _unified_metrics)
        if hasattr(self, '_unified_metrics') and self._unified_metrics is not None:
            self._unified_metrics.log_vram(epoch)

    def _log_flops(self, epoch: int) -> None:
        """Log FLOPs metrics to TensorBoard.

        Args:
            epoch: Current epoch number.
        """
        if self.is_main_process:
            # Use unified metrics (all trainers must initialize _unified_metrics)
            if hasattr(self, '_unified_metrics') and self._unified_metrics is not None:
                self._unified_metrics.log_flops_from_tracker(self._flops_tracker, epoch)

    def _log_learning_rate(
        self,
        epoch: int,
        scheduler: LRScheduler | None = None,
        prefix: str = "LR/Generator",
    ) -> None:
        """Log current learning rate to TensorBoard.

        Args:
            epoch: Current epoch number.
            scheduler: Learning rate scheduler (uses self.lr_scheduler if None).
            prefix: TensorBoard tag prefix.
        """
        if not self.is_main_process:
            return

        sched = scheduler if scheduler is not None else self.lr_scheduler
        if sched is not None:
            lr = sched.get_last_lr()[0]
            # Use unified metrics (all trainers must initialize _unified_metrics)
            if hasattr(self, '_unified_metrics') and self._unified_metrics is not None:
                self._unified_metrics.log_lr(lr, epoch, prefix)

    def close_writer(self) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def _setup_profiler(self) -> torch.profiler.profile | None:
        """Setup PyTorch profiler if enabled.

        Creates a profiler with Chrome trace export for performance analysis.
        Traces can be viewed in Perfetto UI (https://ui.perfetto.dev) or
        TensorBoard with torch-tb-profiler plugin.

        Returns:
            Profiler context manager or None if profiling disabled.
        """
        if not self._profiling_enabled or not self.is_main_process:
            return None

        pc = self._profiling_config
        trace_dir = os.path.join(self.save_dir, 'profiling')
        os.makedirs(trace_dir, exist_ok=True)

        logger.info(
            f"PyTorch profiler enabled: wait={pc.wait}, "
            f"warmup={pc.warmup}, active={pc.active}"
        )
        logger.info(f"Traces will be saved to: {trace_dir}")

        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=pc.wait,
                warmup=pc.warmup,
                active=pc.active,
                repeat=pc.repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
            record_shapes=pc.record_shapes,
            profile_memory=pc.profile_memory,
            with_stack=pc.with_stack,
            with_flops=pc.with_flops,
        )

    def _profiler_step(self) -> None:
        """Step the profiler to mark training step boundary.

        Call this after each training step in train_epoch().
        """
        if self._profiler is not None:
            self._profiler.step()

    def _cleanup_distributed(self) -> None:
        """Cleanup distributed training resources."""
        if self.use_multi_gpu and dist.is_initialized():
            dist.destroy_process_group()

    def _save_config(self) -> None:
        """Save training configuration to save directory."""
        if not self.is_main_process:
            return

        config_path = os.path.join(self.save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))

    # ─────────────────────────────────────────────────────────────────────────
    # Hook methods (can be overridden by subclasses)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_training_start(self) -> None:
        """Hook called before training loop starts.

        Default: Save config. Subclasses can extend.
        """
        if self.is_main_process:
            self._save_config()

    def _on_epoch_start(self, epoch: int) -> None:
        """Hook called at start of each epoch.

        Default: Set DDP sampler epoch, reset gradient tracker.
        Subclasses can extend.

        Args:
            epoch: Current epoch number.
        """
        # Set sampler epoch for proper shuffling in DDP
        if self.use_multi_gpu and hasattr(self, 'train_sampler') and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        # Reset gradient tracker for new epoch
        self._grad_norm_tracker.reset()

    def _on_epoch_end(
        self,
        epoch: int,
        avg_losses: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Hook called at end of each epoch.

        Default: Step scheduler, log learning rate, VRAM, FLOPs.
        Subclasses can extend.

        Args:
            epoch: Current epoch number.
            avg_losses: Dictionary of average training losses.
            val_metrics: Dictionary of validation metrics.
        """
        # Step scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Log learning rate
        self._log_learning_rate(epoch)

        # Log VRAM
        self._log_vram(epoch)

        # Log FLOPs
        self._log_flops(epoch)

    def _should_stop_early(self) -> bool:
        """Check if training should stop early.

        Default: Always returns False (no early stopping).
        Subclasses can override to implement early stopping logic.

        Returns:
            True if training should stop, False otherwise.
        """
        return False

    def _on_training_end(self, total_time: float) -> None:
        """Hook called after training loop ends.

        Default: Log completion, cleanup DDP.
        Note: Writer is NOT closed here - call close_writer() explicitly
        after test evaluation in your training script.

        Args:
            total_time: Total training time in seconds.
        """
        if self.is_main_process:
            hours = total_time / 3600
            logger.info(f"Training completed in {hours:.2f} hours")

        # Note: Don't close writer here - it's needed for test evaluation
        # Scripts should call trainer.close_writer() after test evaluation
        self._cleanup_distributed()

    def _prepare_batch(
        self,
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare batch for training.

        Handles different batch formats (dict, tuple, tensor).

        Args:
            batch: Input batch from dataloader.

        Returns:
            Tuple of (images, mask) where mask may be None.
        """
        # Handle dict batches (from NiFTIDataset)
        if isinstance(batch, dict):
            images = batch['image'].to(self.device, non_blocking=True)
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(self.device, non_blocking=True)
            return images, mask

        # Handle tuple batches (image, label) or (image, mask)
        if isinstance(batch, (tuple, list)):
            images = batch[0].to(self.device, non_blocking=True)
            mask = None
            if len(batch) > 1 and batch[1] is not None:
                second = batch[1]
                # Check if it's a mask (same spatial shape) or label (scalar)
                if isinstance(second, torch.Tensor) and second.shape[-2:] == images.shape[-2:]:
                    mask = second.to(self.device, non_blocking=True)
            return images, mask

        # Handle plain tensor batches
        return batch.to(self.device, non_blocking=True), None

    # ─────────────────────────────────────────────────────────────────────────
    # SIGTERM handling (graceful SLURM shutdown)
    # ─────────────────────────────────────────────────────────────────────────

    def _install_sigterm_handler(self) -> None:
        """Install SIGTERM handler for graceful SLURM shutdown.

        When SLURM's time limit is reached, it sends SIGTERM before SIGKILL.
        This handler sets a flag so the training loop can finish the current
        epoch, save a checkpoint, and exit cleanly.
        """
        def _sigterm_handler(signum: int, frame: Any) -> None:
            logger.warning("SIGTERM received — finishing current epoch then saving and exiting.")
            self._sigterm_received = True

        signal.signal(signal.SIGTERM, _sigterm_handler)

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint management
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_checkpoint_manager(self) -> None:
        """Setup checkpoint manager after model initialization.

        Call this method at the end of setup_model() to initialize the
        CheckpointManager with all required components.

        Subclasses can override to add discriminator/GAN components.
        """
        if not self.is_main_process:
            return

        self.checkpoint_manager = CheckpointManager(
            save_dir=self.save_dir,
            model=self.model_raw,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            ema=getattr(self, 'ema', None),
            config=self._get_model_config() if hasattr(self, '_get_model_config') else None,
            metric_name=self._get_best_metric_name(),
            keep_last_n=self._training_config.keep_last_n_checkpoints,
            device=self.device,
        )

    def _get_best_metric_name(self) -> str:
        """Return metric name for best checkpoint tracking.

        Override in subclasses if using different metric names.
        Default: 'gen' for compression trainers, 'total' for diffusion.

        Returns:
            Metric key string (e.g., 'gen', 'total').
        """
        return 'gen'

    def _get_checkpoint_extra_state(self) -> dict | None:
        """Return extra state to include in checkpoints.

        Override in subclasses to add trainer-specific state.
        Default: None.

        Returns:
            Dict of extra state or None.
        """
        return None

    def _handle_checkpoints(
        self,
        epoch: int,
        val_metrics: dict[str, float],
    ) -> None:
        """Handle checkpoint saving using CheckpointManager.

        Call this at the end of each epoch to save latest and optionally best
        checkpoints. Falls back to legacy _save_checkpoint() if manager not set.

        Args:
            epoch: Current epoch number.
            val_metrics: Validation metrics dict.
        """
        if not self.is_main_process:
            return

        extra_state = self._get_checkpoint_extra_state()

        try:
            if self.checkpoint_manager is not None:
                # Use CheckpointManager
                self.checkpoint_manager.save(epoch, val_metrics, name="latest", extra_state=extra_state)
                if self.checkpoint_manager.save_if_best(epoch, val_metrics, extra_state=extra_state):
                    logger.info(f"New best model saved (loss: {val_metrics.get(self._get_best_metric_name(), 0):.4f})")
            else:
                # Fallback to legacy method
                self._save_checkpoint(epoch, "latest")
                val_loss = val_metrics.get('gen', val_metrics.get('total', float('inf')))
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self._save_checkpoint(epoch, "best")
                    logger.info(f"New best model saved (loss: {val_loss:.4f})")
        except (RuntimeError, OSError) as e:
            logger.error(f"Checkpoint save failed at epoch {epoch}: {e}. Training continues.")

    # ─────────────────────────────────────────────────────────────────────────
    # Abstract methods (must be implemented by subclasses)
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def setup_model(self, pretrained_checkpoint: str | None = None) -> None:
        """Initialize model, optimizer, scheduler, and loss functions.

        Args:
            pretrained_checkpoint: Optional path to checkpoint for loading
                pretrained weights.
        """
        ...

    @abstractmethod
    def train_step(self, batch: BatchType) -> TrainingStepResult:
        """Execute single training step.

        Args:
            batch: Input batch from dataloader.

        Returns:
            TrainingStepResult with loss values for this step.
        """
        ...

    @abstractmethod
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary of average loss values for the epoch.
        """
        ...

    @abstractmethod
    def compute_validation_losses(
        self,
        epoch: int,
        log_figures: bool = True,
    ) -> dict[str, float]:
        """Compute validation metrics.

        Args:
            epoch: Current epoch number.
            log_figures: Whether to log figures (worst_batch, etc.).

        Returns:
            Dictionary of validation metrics.
        """
        ...

    @abstractmethod
    def _save_checkpoint(self, epoch: int, name: str) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            name: Checkpoint name (e.g., "latest", "best").
        """
        ...

    # ─────────────────────────────────────────────────────────────────────────
    # Metadata saving (template method pattern)
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def _get_trainer_type(self) -> str:
        """Return trainer type string for metadata (e.g., 'vae', 'vae_3d', 'diffusion').

        Returns:
            String identifier for this trainer type.
        """
        ...

    def _get_metadata_extra(self) -> dict[str, Any]:
        """Return trainer-specific metadata fields.

        Override in subclasses to add custom fields to metadata.json.
        Default returns empty dict.

        Returns:
            Dictionary of extra metadata fields.
        """
        return {}

    def _save_metadata(self) -> None:
        """Save training configuration and metadata to output directory.

        Template method: saves config.yaml and metadata.json.
        Subclasses customize via _get_trainer_type() and _get_metadata_extra().
        """
        import json
        os.makedirs(self.save_dir, exist_ok=True)

        # Save full Hydra config
        config_path = os.path.join(self.save_dir, 'config.yaml')
        OmegaConf.save(self.cfg, config_path)

        # Build metadata
        metadata = {
            'type': self._get_trainer_type(),
            'n_epochs': self.n_epochs,
        }
        metadata.update(self._get_metadata_extra())

        # Save metadata.json
        metadata_path = os.path.join(self.save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    # Template method for training loop
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        train_dataset: Dataset,
        val_loader: DataLoader | None = None,
        per_modality_val_loaders: dict[str, DataLoader] | None = None,
        start_epoch: int = 0,
        max_epochs: int | None = None,
    ) -> int:
        """Execute full training loop using template method pattern.

        Subclasses customize behavior via hooks and abstract methods.

        Args:
            train_loader: Training data loader.
            train_dataset: Training dataset.
            val_loader: Optional validation data loader.
            per_modality_val_loaders: Optional dict of per-modality validation loaders.
            start_epoch: Starting epoch (for resuming training).
            max_epochs: Optional max epochs override.

        Returns:
            Last completed epoch number.
        """
        n_epochs = max_epochs if max_epochs is not None else self.n_epochs
        self.val_loader = val_loader
        self.per_modality_val_loaders = per_modality_val_loaders

        # Store train sampler for DDP epoch setting
        if hasattr(train_loader, 'sampler'):
            self.train_sampler = getattr(train_loader.sampler, 'sampler', train_loader.sampler)
            if not hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler = None
        else:
            self.train_sampler = None

        self._on_training_start()

        # Measure FLOPs once at start
        if self.is_main_process and self.log_flops:
            try:
                sample_batch = next(iter(train_loader))
                sample_images, _ = self._prepare_batch(sample_batch)
                self._measure_model_flops(sample_images, len(train_loader))
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                logger.warning(f"FLOPs measurement failed (OOM or CUDA error): {e}")
            except StopIteration:
                logger.warning("FLOPs measurement failed: empty dataloader")
            except (ImportError, AssertionError) as e:
                logger.exception(f"FLOPs measurement failed unexpectedly: {e}")

        # Setup PyTorch profiler if enabled
        self._profiler = self._setup_profiler()
        if self._profiler is not None:
            self._profiler.__enter__()

        last_epoch = start_epoch
        total_start = time.time()

        # Time estimator for ETA calculation (excludes first epoch warmup)
        self._time_estimator = EpochTimeEstimator(n_epochs)

        try:
            for epoch in range(start_epoch, n_epochs):
                epoch_start = time.time()
                last_epoch = epoch

                self._on_epoch_start(epoch)

                # Training
                avg_losses = self.train_epoch(train_loader, epoch)

                # On SIGTERM: skip validation, just save checkpoint and exit
                if self._sigterm_received and self.is_main_process:
                    logger.info("SIGTERM: skipping validation, saving checkpoint and exiting.")
                    merged_metrics = {**avg_losses}
                    self._handle_checkpoints(epoch, merged_metrics)
                    break

                # Validation (main process only)
                val_metrics: dict[str, float] = {}
                if self.is_main_process:
                    log_figures = (epoch + 1) % self.figure_interval == 0
                    if self.val_loader is not None:
                        val_metrics = self.compute_validation_losses(epoch, log_figures)

                    self._on_epoch_end(epoch, avg_losses, val_metrics)

                    # Log epoch summary
                    elapsed = time.time() - epoch_start
                    self._log_epoch_summary(epoch, n_epochs, avg_losses, val_metrics, elapsed)

                    # Save checkpoints using unified handler
                    # Merge avg_losses into val_metrics for metric availability
                    merged_metrics = {**avg_losses, **val_metrics}
                    self._handle_checkpoints(epoch, merged_metrics)

                    # Check for early stopping
                    if self._should_stop_early():
                        logger.info("Stopping training early")
                        break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            # Cleanup profiler
            if self._profiler is not None:
                self._profiler.__exit__(None, None, None)
                trace_dir = os.path.join(self.save_dir, 'profiling')
                logger.info(f"Profiling traces saved to: {trace_dir}")
                self._profiler = None

            total_time = time.time() - total_start
            self._on_training_end(total_time)

        return last_epoch

    def _measure_model_flops(
        self,
        sample_images: torch.Tensor,
        steps_per_epoch: int,
    ) -> None:
        """Measure model FLOPs using sample input.

        Subclasses can override to handle special model architectures.

        Args:
            sample_images: Sample input batch for FLOPs measurement.
            steps_per_epoch: Number of training steps per epoch.
        """
        if not self.log_flops or self.model_raw is None:
            return

        self._flops_tracker.measure(
            model=self.model_raw,
            sample_input=sample_images[:1],
            steps_per_epoch=steps_per_epoch,
            timesteps=None,
            is_main_process=self.is_main_process,
        )

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: dict[str, float],
        val_metrics: dict[str, float],
        elapsed_time: float,
    ) -> None:
        """Log epoch completion summary.

        Subclasses can override to customize the log format.

        Args:
            epoch: Current epoch number.
            total_epochs: Total number of epochs.
            avg_losses: Dictionary of average training losses.
            val_metrics: Dictionary of validation metrics.
            elapsed_time: Time taken for the epoch in seconds.
        """
        timestamp = time.strftime("%H:%M:%S")
        epoch_pct = ((epoch + 1) / total_epochs) * 100

        # Build loss string from available losses
        loss_parts = []
        for key in ['gen', 'total', 'loss']:
            if key in avg_losses:
                val_str = f"(v:{val_metrics.get(key, 0):.4f})" if key in val_metrics else ""
                loss_parts.append(f"{key.upper()}: {avg_losses[key]:.4f}{val_str}")
                break

        # Add validation metrics if available
        if 'msssim' in val_metrics:
            loss_parts.append(f"MS-SSIM: {val_metrics['msssim']:.3f}")
        if 'psnr' in val_metrics:
            loss_parts.append(f"PSNR: {val_metrics['psnr']:.2f}")

        loss_parts.append(f"Time: {elapsed_time:.1f}s")

        # Add ETA from time estimator
        if hasattr(self, '_time_estimator') and self._time_estimator is not None:
            self._time_estimator.update(elapsed_time)
            eta_str = self._time_estimator.get_eta_string()
            if eta_str:
                loss_parts.append(eta_str)

        loss_str = " | ".join(loss_parts) if loss_parts else "No losses"

        logger.info(
            f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
            f"{loss_str}"
        )
