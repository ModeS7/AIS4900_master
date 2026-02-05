"""Centralized checkpoint management for training.

This module provides the CheckpointManager class that handles all checkpoint
save/load/resume logic in a single place. It replaces scattered checkpoint
logic across trainers.

Features:
- Model, optimizer, scheduler, EMA state management
- Discriminator state (for GAN training)
- Automatic best/latest tracking
- Optional checkpoint cleanup
- Resume from interruption
"""
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)

__all__ = ['CheckpointManager']


class CheckpointManager:
    """Centralized checkpoint management for all trainers.

    Handles saving and loading of model checkpoints with support for:
    - Model, optimizer, scheduler, EMA state
    - Discriminator state (for GAN training)
    - Automatic best/latest checkpoint tracking
    - Optional periodic checkpoint cleanup
    - Resume from interruption

    Example:
        >>> manager = CheckpointManager(
        ...     save_dir="./checkpoints",
        ...     model=model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     metric_name='gen',
        ...     metric_mode='min',
        ... )
        >>> # Save after each epoch
        >>> manager.save(epoch, metrics, name="latest")
        >>> manager.save_if_best(epoch, metrics)
        >>> # Resume training
        >>> start_epoch = manager.resume()
    """

    VERSION = "1.0"

    def __init__(
        self,
        save_dir: str,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        ema: Any | None = None,
        config: dict | None = None,
        discriminator: nn.Module | None = None,
        optimizer_d: Optimizer | None = None,
        scheduler_d: LRScheduler | None = None,
        metric_name: str = 'total',
        metric_mode: str = 'min',
        keep_last_n: int = 0,
        device: torch.device | None = None,
    ):
        """Initialize CheckpointManager.

        Args:
            save_dir: Directory to save checkpoints.
            model: Main model (generator for GAN training).
            optimizer: Optimizer for main model.
            scheduler: Optional learning rate scheduler.
            ema: Optional EMA wrapper (from ema-pytorch).
            config: Optional model configuration dict.
            discriminator: Optional discriminator model for GAN training.
            optimizer_d: Optional discriminator optimizer.
            scheduler_d: Optional discriminator scheduler.
            metric_name: Metric key for best checkpoint tracking (e.g., 'gen', 'total').
            metric_mode: 'min' for loss (lower is better), 'max' for accuracy.
            keep_last_n: Keep N latest periodic checkpoints (0 = keep all).
            device: Device for loading checkpoints.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Required components
        self.model = model
        self.optimizer = optimizer

        # Optional components
        self.scheduler = scheduler
        self.ema = ema
        self.config = config

        # GAN components
        self.discriminator = discriminator
        self.optimizer_d = optimizer_d
        self.scheduler_d = scheduler_d

        # Tracking
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self._best_metric = float('inf') if metric_mode == 'min' else float('-inf')

        # Cleanup
        self.keep_last_n = keep_last_n
        self._epoch_checkpoints: list[str] = []

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def best_metric(self) -> float:
        """Current best metric value."""
        return self._best_metric

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.metric_mode == 'min':
            return current < best
        return current > best

    def _get_model_state(self, model: nn.Module) -> dict:
        """Get state dict, handling DDP/compiled models."""
        if hasattr(model, 'module'):
            return model.module.state_dict()
        return model.state_dict()

    def save(
        self,
        epoch: int,
        metrics: dict[str, float] | None = None,
        name: str = "latest",
        extra_state: dict[str, Any] | None = None,
    ) -> str:
        """Save checkpoint with given name.

        Args:
            epoch: Current epoch number.
            metrics: Optional metrics dict (for logging).
            name: Checkpoint name (e.g., "latest", "best", "epoch_10").
            extra_state: Optional extra state to include in checkpoint.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint: dict[str, Any] = {
            'epoch': epoch,
            'model_state_dict': self._get_model_state(self.model),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self._best_metric,
            'metric_name': self.metric_name,
            'checkpoint_manager_version': self.VERSION,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        if self.config is not None:
            checkpoint['config'] = self.config

        # GAN components
        if self.discriminator is not None:
            checkpoint['discriminator_state_dict'] = self._get_model_state(self.discriminator)
        if self.optimizer_d is not None:
            checkpoint['optimizer_d_state_dict'] = self.optimizer_d.state_dict()
        if self.scheduler_d is not None:
            checkpoint['scheduler_d_state_dict'] = self.scheduler_d.state_dict()

        # Extra state (e.g., trainer-specific flags)
        if extra_state is not None:
            checkpoint.update(extra_state)

        path = self.save_dir / f"checkpoint_{name}.pt"
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint: {path}")

        return str(path)

    def save_if_best(
        self,
        epoch: int,
        metrics: dict[str, float],
        extra_state: dict[str, Any] | None = None,
    ) -> bool:
        """Save 'best' checkpoint if metric improved.

        Args:
            epoch: Current epoch number.
            metrics: Metrics dict containing the tracked metric.
            extra_state: Optional extra state to include in checkpoint.

        Returns:
            True if new best checkpoint was saved.
        """
        current = metrics.get(self.metric_name)
        if current is None:
            logger.warning(f"Metric '{self.metric_name}' not in metrics: {list(metrics.keys())}")
            return False

        # Skip invalid metrics (e.g., 0.0 from empty validation)
        if current <= 0:
            return False

        if self._is_better(current, self._best_metric):
            self._best_metric = current
            self.save(epoch, metrics, name="best", extra_state=extra_state)
            logger.info(f"New best {self.metric_name}: {current:.4f}")
            return True
        return False

    def save_periodic(
        self,
        epoch: int,
        metrics: dict[str, float] | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> str | None:
        """Save periodic checkpoint and cleanup old ones.

        Args:
            epoch: Current epoch number.
            metrics: Optional metrics dict.
            extra_state: Optional extra state to include.

        Returns:
            Path to saved checkpoint, or None if keep_last_n <= 0.
        """
        if self.keep_last_n <= 0:
            return None

        path = self.save(epoch, metrics, name=f"epoch_{epoch}", extra_state=extra_state)
        self._epoch_checkpoints.append(path)
        self._cleanup_old_checkpoints()
        return path

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old periodic checkpoints beyond keep_last_n."""
        while len(self._epoch_checkpoints) > self.keep_last_n:
            old_path = self._epoch_checkpoints.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)
                logger.debug(f"Removed old checkpoint: {old_path}")

    def load(
        self,
        path: str,
        strict: bool = True,
        load_optimizer: bool = True,
    ) -> dict[str, Any]:
        """Load checkpoint from path.

        Args:
            path: Path to checkpoint file.
            strict: Whether to strictly enforce state dict key matching.
            load_optimizer: Whether to load optimizer state.

        Returns:
            Metadata dict with 'epoch', 'config', 'best_metric'.

        Raises:
            ValueError: If checkpoint is missing required model_state_dict.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Validate checkpoint version
        self._validate_checkpoint_version(checkpoint)

        # Validate checkpoint
        if 'model_state_dict' not in checkpoint:
            raise ValueError(f"Invalid checkpoint: missing model_state_dict in {path}")

        # Load model
        self._load_model_state(self.model, checkpoint['model_state_dict'], strict)

        # Load optimizer
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load EMA
        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])

        # Load GAN components
        if self.discriminator is not None and 'discriminator_state_dict' in checkpoint:
            self._load_model_state(self.discriminator, checkpoint['discriminator_state_dict'], strict)
        if self.optimizer_d is not None and load_optimizer and 'optimizer_d_state_dict' in checkpoint:
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        if self.scheduler_d is not None and 'scheduler_d_state_dict' in checkpoint:
            self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])

        # Restore tracking state
        if 'best_metric' in checkpoint:
            self._best_metric = checkpoint['best_metric']

        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', '?')})")

        return {
            'epoch': checkpoint.get('epoch', 0),
            'config': checkpoint.get('config'),
            'best_metric': checkpoint.get('best_metric'),
            'checkpoint': checkpoint,  # Full checkpoint for extra state access
        }

    def _validate_checkpoint_version(self, checkpoint: dict) -> None:
        """Validate checkpoint version compatibility.

        Args:
            checkpoint: Loaded checkpoint dict.

        Raises:
            ValueError: If major version mismatch (breaking changes).
        """
        saved_version = checkpoint.get('checkpoint_manager_version')
        if saved_version is None:
            logger.debug("Legacy checkpoint without version info")
            return

        try:
            major_saved, minor_saved = map(int, saved_version.split('.'))
            major_curr, minor_curr = map(int, self.VERSION.split('.'))
        except ValueError:
            logger.warning(f"Invalid version format: {saved_version}")
            return

        if major_saved != major_curr:
            raise ValueError(
                f"Incompatible checkpoint version: saved={saved_version}, current={self.VERSION}. "
                f"Major version mismatch indicates breaking changes."
            )
        if minor_saved > minor_curr:
            logger.warning(
                f"Checkpoint from newer version: saved={saved_version}, current={self.VERSION}. "
                f"Some saved features may be ignored."
            )

    def _load_model_state(
        self,
        model: nn.Module,
        state_dict: dict,
        strict: bool,
    ) -> None:
        """Load state dict, handling DDP/compiled models."""
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict, strict=strict)
        else:
            model.load_state_dict(state_dict, strict=strict)

    def resume(self, path: str | None = None) -> int:
        """Resume training from checkpoint.

        If path is None, auto-detects latest checkpoint in save_dir.

        Args:
            path: Optional explicit checkpoint path.

        Returns:
            Start epoch (checkpoint epoch + 1).
        """
        if path is None:
            path = self._find_latest_checkpoint()

        if path is None or not os.path.exists(path):
            logger.info("No checkpoint found, starting from epoch 0")
            return 0

        result = self.load(path, strict=True, load_optimizer=True)
        start_epoch = result['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")
        return start_epoch

    def _find_latest_checkpoint(self) -> str | None:
        """Find latest checkpoint in save_dir."""
        latest = self.save_dir / "checkpoint_latest.pt"
        if latest.exists():
            return str(latest)

        # Fallback: find highest epoch checkpoint
        epoch_files = list(self.save_dir.glob("checkpoint_epoch_*.pt"))
        if not epoch_files:
            return None

        def get_epoch(p: Path) -> int:
            try:
                return int(p.stem.split('_')[-1])
            except ValueError:
                return -1

        return str(max(epoch_files, key=get_epoch))

    def update_components(
        self,
        model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        ema: Any | None = None,
        discriminator: nn.Module | None = None,
        optimizer_d: Optimizer | None = None,
        scheduler_d: LRScheduler | None = None,
    ) -> None:
        """Update managed components.

        Use this after model setup if components weren't available during init.

        Args:
            model: New model to manage.
            optimizer: New optimizer to manage.
            scheduler: New scheduler to manage.
            ema: New EMA wrapper to manage.
            discriminator: New discriminator to manage.
            optimizer_d: New discriminator optimizer to manage.
            scheduler_d: New discriminator scheduler to manage.
        """
        if model is not None:
            self.model = model
        if optimizer is not None:
            self.optimizer = optimizer
        if scheduler is not None:
            self.scheduler = scheduler
        if ema is not None:
            self.ema = ema
        if discriminator is not None:
            self.discriminator = discriminator
        if optimizer_d is not None:
            self.optimizer_d = optimizer_d
        if scheduler_d is not None:
            self.scheduler_d = scheduler_d
