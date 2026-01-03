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
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from medgen.core import setup_distributed
from .tracking import FLOPsTracker, GradientNormTracker
from .utils import log_vram_to_tensorboard

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
        # Extract common training config
        # ─────────────────────────────────────────────────────────────────────
        self.n_epochs: int = cfg.training.epochs
        self.batch_size: int = cfg.training.batch_size
        self.learning_rate: float = cfg.training.get('learning_rate', 1e-4)
        self.warmup_epochs: int = cfg.training.warmup_epochs
        figure_count: int = cfg.training.get('figure_count', 20)
        self.figure_interval: int = max(1, self.n_epochs // figure_count)
        self.use_multi_gpu: bool = cfg.training.get('use_multi_gpu', False)
        self.gradient_clip_norm: float = cfg.training.get('gradient_clip_norm', 1.0)
        self.limit_train_batches: Optional[int] = cfg.training.get('limit_train_batches', None)

        # Determine if running on cluster
        self.is_cluster: bool = (cfg.paths.name == "cluster")

        # ─────────────────────────────────────────────────────────────────────
        # Extract logging config
        # ─────────────────────────────────────────────────────────────────────
        logging_cfg = cfg.training.get('logging', {})
        self.log_grad_norm: bool = logging_cfg.get('grad_norm', True)
        self.log_psnr: bool = logging_cfg.get('psnr', True)
        self.log_lpips: bool = logging_cfg.get('lpips', True)
        self.log_msssim: bool = logging_cfg.get('msssim', True)
        self.log_regional_losses: bool = logging_cfg.get('regional_losses', True)
        self.log_flops: bool = logging_cfg.get('flops', True)

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
            self.writer: Optional[SummaryWriter] = self._init_tensorboard()
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
        self.model: Optional[nn.Module] = None
        self.model_raw: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.lr_scheduler: Optional[LRScheduler] = None
        self.val_loader: Optional[DataLoader] = None

    def _setup_distributed(self) -> Tuple[int, int, int, torch.device]:
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
        exp_name = self.cfg.training.get('name', '')
        image_size = self.cfg.model.get('image_size', 256)
        run_name = f"{exp_name}{image_size}_{timestamp}"
        return os.path.join(self.cfg.paths.model_dir, 'runs', run_name)

    def _init_tensorboard(self) -> Optional[SummaryWriter]:
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
        if self.is_main_process and self.writer is not None:
            log_vram_to_tensorboard(self.writer, self.device, epoch)

    def _log_flops(self, epoch: int) -> None:
        """Log FLOPs metrics to TensorBoard.

        Args:
            epoch: Current epoch number.
        """
        if self.is_main_process:
            self._flops_tracker.log_epoch(self.writer, epoch)

    def _log_learning_rate(
        self,
        epoch: int,
        scheduler: Optional[LRScheduler] = None,
        prefix: str = "LR/Generator",
    ) -> None:
        """Log current learning rate to TensorBoard.

        Args:
            epoch: Current epoch number.
            scheduler: Learning rate scheduler (uses self.lr_scheduler if None).
            prefix: TensorBoard tag prefix.
        """
        if not self.is_main_process or self.writer is None:
            return

        sched = scheduler if scheduler is not None else self.lr_scheduler
        if sched is not None:
            lr = sched.get_last_lr()[0]
            self.writer.add_scalar(prefix, lr, epoch)

    def close_writer(self) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

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
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
    # Abstract methods (must be implemented by subclasses)
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize model, optimizer, scheduler, and loss functions.

        Args:
            pretrained_checkpoint: Optional path to checkpoint for loading
                pretrained weights.
        """
        ...

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Execute single training step.

        Args:
            batch: Input batch from dataloader.

        Returns:
            Dictionary of loss values for this step.
        """
        ...

    @abstractmethod
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
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
    ) -> Dict[str, float]:
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
    # Template method for training loop
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        train_dataset: Dataset,
        val_loader: Optional[DataLoader] = None,
        per_modality_val_loaders: Optional[Dict[str, DataLoader]] = None,
        start_epoch: int = 0,
        max_epochs: Optional[int] = None,
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
            except Exception as e:
                logger.warning(f"FLOPs measurement failed: {e}")

        last_epoch = start_epoch
        total_start = time.time()

        try:
            for epoch in range(start_epoch, n_epochs):
                epoch_start = time.time()
                last_epoch = epoch

                self._on_epoch_start(epoch)

                # Training
                avg_losses = self.train_epoch(train_loader, epoch)

                # Validation (main process only)
                val_metrics: Dict[str, float] = {}
                if self.is_main_process:
                    log_figures = (epoch + 1) % self.figure_interval == 0
                    if self.val_loader is not None:
                        val_metrics = self.compute_validation_losses(epoch, log_figures)

                    self._on_epoch_end(epoch, avg_losses, val_metrics)

                    # Log epoch summary
                    elapsed = time.time() - epoch_start
                    self._log_epoch_summary(epoch, n_epochs, avg_losses, val_metrics, elapsed)

                    # Save checkpoints
                    self._save_checkpoint(epoch, "latest")

                    # Save best checkpoint
                    val_loss = val_metrics.get('gen', avg_losses.get('gen', float('inf')))
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self._save_checkpoint(epoch, "best")
                        logger.info(f"New best model saved (loss: {val_loss:.4f})")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
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
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
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

        loss_str = " | ".join(loss_parts) if loss_parts else "No losses"

        logger.info(
            f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
            f"{loss_str} | Time: {elapsed_time:.1f}s"
        )
