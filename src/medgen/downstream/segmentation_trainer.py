"""
Downstream segmentation trainer for evaluating synthetic data utility.

Uses MONAI's SegResNet model with combined BCE + Dice loss.
Tracks per-tumor-size Dice scores using SegRegionalMetricsTracker.
"""
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from monai.networks.nets import SegResNet

from medgen.core import create_warmup_cosine_scheduler
from medgen.losses import SegmentationLoss
from medgen.metrics import SimpleLossAccumulator
from medgen.metrics.regional import SegRegionalMetricsTracker
from medgen.pipeline.base_trainer import BaseTrainer
from medgen.pipeline.results import TrainingStepResult
from medgen.pipeline.utils import create_epoch_iterator, get_vram_usage

logger = logging.getLogger(__name__)


class SegmentationTrainer(BaseTrainer):
    """Trainer for downstream segmentation task.

    Uses MONAI SegResNet for tumor segmentation and tracks per-size Dice scores
    using SegRegionalMetricsTracker with RANO-BM clinical thresholds.

    Extends BaseTrainer with:
    - SegResNet model initialization
    - BCE + Dice + Boundary loss
    - Per-tumor-size metrics (tiny, small, medium, large)

    Args:
        cfg: Hydra configuration object.
        spatial_dims: 2 for 2D, 3 for 3D.
    """

    def __init__(self, cfg: DictConfig, spatial_dims: int = 2) -> None:
        self.spatial_dims = spatial_dims
        super().__init__(cfg)

        # Extract segmentation-specific config
        self.in_channels: int = cfg.model.get('in_channels', 1)
        self.out_channels: int = cfg.model.get('out_channels', 1)
        self.init_filters: int = cfg.model.get('init_filters', 32)
        self.blocks_down: Tuple[int, ...] = tuple(cfg.model.get('blocks_down', [1, 2, 2, 4]))
        self.blocks_up: Tuple[int, ...] = tuple(cfg.model.get('blocks_up', [1, 1, 1]))
        self.dropout_prob: float = cfg.model.get('dropout_prob', 0.2)

        # Image size for regional metrics
        self.image_size: int = cfg.model.image_size

        # Validation frequency
        self.val_every: int = cfg.training.get('val_every', 1)

        # Early stopping
        self.patience: int = cfg.training.get('patience', 20)
        self._epochs_without_improvement: int = 0

        # Precision config - bf16 AMP for both 2D and 3D
        # Requires PyTorch 2.10+ (2.9 had 3D conv regression)
        self.use_amp = True
        self.weight_dtype = torch.bfloat16

        # Loss accumulator
        self._loss_accumulator = SimpleLossAccumulator()

    @classmethod
    def create_2d(cls, cfg: DictConfig) -> 'SegmentationTrainer':
        """Create 2D segmentation trainer."""
        return cls(cfg, spatial_dims=2)

    @classmethod
    def create_3d(cls, cfg: DictConfig) -> 'SegmentationTrainer':
        """Create 3D segmentation trainer."""
        return cls(cfg, spatial_dims=3)

    def _get_trainer_type(self) -> str:
        """Return trainer type for metadata."""
        return 'segmentation_3d' if self.spatial_dims == 3 else 'segmentation'

    def _get_best_metric_name(self) -> str:
        """Return metric name for best checkpoint tracking."""
        return 'dice'  # Track validation Dice

    def _create_fallback_save_dir(self) -> str:
        """Create fallback save directory."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = self.cfg.training.get('name', '')
        scenario = self.cfg.get('scenario', 'baseline')
        run_name = f"{exp_name}{scenario}_{timestamp}"
        return os.path.join(
            self.cfg.paths.model_dir,
            'segmentation',
            f'{self.spatial_dims}d',
            scenario,
            run_name
        )

    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize SegResNet model, optimizer, and loss functions.

        Args:
            pretrained_checkpoint: Optional path to checkpoint for resuming.
        """
        # Create SegResNet model
        self.model_raw = SegResNet(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            init_filters=self.init_filters,
            blocks_down=self.blocks_down,
            blocks_up=self.blocks_up,
            dropout_prob=self.dropout_prob,
        ).to(self.device)

        self.model = self.model_raw

        # Optional: torch.compile for speed
        if self.cfg.training.get('use_compile', False):
            self.model = torch.compile(self.model, mode='default')
            if self.is_main_process:
                logger.info("Model compiled with torch.compile")

        # Create segmentation loss
        self.loss_fn = SegmentationLoss(
            bce_weight=self.cfg.training.get('ce_weight', 1.0),
            dice_weight=self.cfg.training.get('dice_weight', 1.0),
            boundary_weight=0.5,
            spatial_dims=self.spatial_dims,
        )

        # Create regional metrics tracker
        fov_mm = self.cfg.evaluation.get('fov_mm', 240.0)
        self.regional_tracker = SegRegionalMetricsTracker(
            image_size=self.image_size,
            fov_mm=fov_mm,
            device=self.device,
        )

        # Setup optimizer
        self.optimizer = AdamW(
            self.model_raw.parameters(),
            lr=self.learning_rate,
            weight_decay=self.cfg.training.get('weight_decay', 1e-5),
        )

        # Setup scheduler
        self.lr_scheduler = create_warmup_cosine_scheduler(
            self.optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=self.n_epochs,
            eta_min=1e-6,
        )

        # Load checkpoint if provided
        if pretrained_checkpoint:
            self._load_checkpoint(pretrained_checkpoint)

        # Setup checkpoint manager
        self._setup_checkpoint_manager()

        # Log model info
        if self.is_main_process:
            n_params = sum(p.numel() for p in self.model_raw.parameters())
            logger.info(f"SegResNet initialized: {n_params / 1e6:.1f}M parameters")
            logger.info(f"  spatial_dims: {self.spatial_dims}")
            logger.info(f"  init_filters: {self.init_filters}")
            logger.info(f"  blocks_down: {self.blocks_down}")
            logger.info(f"  blocks_up: {self.blocks_up}")
            logger.info("  AMP: bf16")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model_raw.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model_raw.load_state_dict(checkpoint['state_dict'])
        else:
            self.model_raw.load_state_dict(checkpoint)

        logger.info(f"Loaded checkpoint: {checkpoint_path}")

    def train_step(self, batch: Any) -> TrainingStepResult:
        """Execute single training step.

        Args:
            batch: Input batch with 'image' and 'seg' keys.

        Returns:
            TrainingStepResult with loss values.
        """
        images = batch['image'].to(self.device, non_blocking=True)
        targets = batch['seg'].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=self.use_amp, dtype=self.weight_dtype):
            logits = self.model(images)
            total_loss, breakdown = self.loss_fn(logits.float(), targets.float())

        total_loss.backward()

        # Gradient clipping
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model_raw.parameters(),
                max_norm=self.gradient_clip_norm,
            )

        self.optimizer.step()

        return TrainingStepResult(
            total_loss=total_loss.item(),
            reconstruction_loss=breakdown['bce'],
            perceptual_loss=breakdown['dice'],
            regularization_loss=breakdown['boundary'],
        )

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
            Dict with average losses.
        """
        self.model.train()
        self._loss_accumulator.reset()

        epoch_iter = create_epoch_iterator(
            train_loader, epoch, self.is_cluster, self.is_main_process,
            limit_batches=self.limit_train_batches,
        )

        for step, batch in enumerate(epoch_iter):
            result = self.train_step(batch)
            losses = {
                'loss': result.total_loss,
                'bce': result.reconstruction_loss,
                'dice_loss': result.perceptual_loss,
                'boundary': result.regularization_loss,
            }

            self._loss_accumulator.update(losses)

            if hasattr(epoch_iter, 'set_postfix'):
                avg_so_far = self._loss_accumulator.compute()
                epoch_iter.set_postfix(
                    loss=f"{avg_so_far.get('loss', 0):.4f}",
                    bce=f"{avg_so_far.get('bce', 0):.4f}",
                    dice=f"{avg_so_far.get('dice_loss', 0):.4f}",
                )

            if epoch == 0 and step == 0 and self.is_main_process:
                logger.info(get_vram_usage(self.device))

        avg_losses = self._loss_accumulator.compute()

        # Log training losses to TensorBoard
        if self.writer is not None:
            for key, value in avg_losses.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)

        return avg_losses

    def compute_validation_losses(
        self,
        epoch: int,
        log_figures: bool = True,
    ) -> Dict[str, float]:
        """Compute validation metrics.

        Args:
            epoch: Current epoch number.
            log_figures: Whether to log prediction figures.

        Returns:
            Dict with validation metrics including per-size Dice.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        self.regional_tracker.reset()

        val_losses = {'loss': 0.0, 'bce': 0.0, 'dice_loss': 0.0}
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                targets = batch['seg'].to(self.device, non_blocking=True)

                with autocast('cuda', enabled=self.use_amp, dtype=self.weight_dtype):
                    logits = self.model(images)
                    total_loss, breakdown = self.loss_fn(logits.float(), targets.float())

                val_losses['loss'] += total_loss.item()
                val_losses['bce'] += breakdown['bce']
                val_losses['dice_loss'] += breakdown['dice']
                n_batches += 1

                # Update regional tracker
                self.regional_tracker.update(
                    prediction=logits,
                    target=targets,
                    apply_sigmoid=True,
                )

        # Average losses
        for key in val_losses:
            val_losses[key] /= max(n_batches, 1)

        # Compute per-size metrics
        regional_metrics = self.regional_tracker.compute()

        # Merge metrics
        metrics = {**val_losses, **regional_metrics}

        # Log to TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)

            # Log predictions figure
            if log_figures and n_batches > 0:
                self._log_predictions_figure(epoch)

        # Log summary
        if self.is_main_process:
            dice_str = f"Dice: {metrics.get('dice', 0):.4f}"
            size_str = (
                f"tiny: {metrics.get('dice_tiny', 0):.3f}, "
                f"small: {metrics.get('dice_small', 0):.3f}, "
                f"medium: {metrics.get('dice_medium', 0):.3f}, "
                f"large: {metrics.get('dice_large', 0):.3f}"
            )
            logger.info(f"Validation - {dice_str} | {size_str}")

        return metrics

    def _log_predictions_figure(self, epoch: int) -> None:
        """Log sample predictions to TensorBoard."""
        if self.val_loader is None or self.writer is None:
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        self.model.eval()

        # Get one batch
        batch = next(iter(self.val_loader))
        images = batch['image'].to(self.device)
        targets = batch['seg'].to(self.device)

        with torch.no_grad():
            logits = self.model(images)
            preds = torch.sigmoid(logits) > 0.5

        # Take first 4 samples
        n_samples = min(4, images.shape[0])

        if self.spatial_dims == 2:
            fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
            if n_samples == 1:
                axes = axes.reshape(1, -1)

            for i in range(n_samples):
                img = images[i, 0].cpu().numpy()
                tgt = targets[i, 0].cpu().numpy()
                pred = preds[i, 0].cpu().numpy()

                axes[i, 0].imshow(img, cmap='gray')
                axes[i, 0].set_title('Input')
                axes[i, 0].axis('off')

                axes[i, 1].imshow(tgt, cmap='gray')
                axes[i, 1].set_title('Ground Truth')
                axes[i, 1].axis('off')

                axes[i, 2].imshow(pred, cmap='gray')
                axes[i, 2].set_title('Prediction')
                axes[i, 2].axis('off')
        else:
            # For 3D, show middle slices
            fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
            if n_samples == 1:
                axes = axes.reshape(1, -1)

            for i in range(n_samples):
                mid_slice = images.shape[2] // 2
                img = images[i, 0, mid_slice].cpu().numpy()
                tgt = targets[i, 0, mid_slice].cpu().numpy()
                pred = preds[i, 0, mid_slice].cpu().numpy()

                axes[i, 0].imshow(img, cmap='gray')
                axes[i, 0].set_title('Input (mid slice)')
                axes[i, 0].axis('off')

                axes[i, 1].imshow(tgt, cmap='gray')
                axes[i, 1].set_title('Ground Truth')
                axes[i, 1].axis('off')

                axes[i, 2].imshow(pred, cmap='gray')
                axes[i, 2].set_title('Prediction')
                axes[i, 2].axis('off')

        plt.tight_layout()
        self.writer.add_figure('val/predictions', fig, epoch)
        plt.close(fig)

    def _save_checkpoint(self, epoch: int, name: str) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            name: Checkpoint name (e.g., 'latest', 'best').
        """
        if not self.is_main_process:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model_raw.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'best_loss': self.best_loss,
            'config': OmegaConf.to_container(self.cfg),
        }

        path = os.path.join(self.save_dir, f'checkpoint_{name}.pt')
        torch.save(checkpoint, path)

    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for checkpoint."""
        return {
            'spatial_dims': self.spatial_dims,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'init_filters': self.init_filters,
            'blocks_down': list(self.blocks_down),
            'blocks_up': list(self.blocks_up),
            'dropout_prob': self.dropout_prob,
        }

    def _on_epoch_end(
        self,
        epoch: int,
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """Hook called at end of each epoch."""
        super()._on_epoch_end(epoch, avg_losses, val_metrics)

        # Early stopping check
        val_dice = val_metrics.get('dice', 0)
        if val_dice > 0 and val_dice > self.best_loss:
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        if self._epochs_without_improvement >= self.patience:
            logger.info(
                f"Early stopping triggered: no improvement for {self.patience} epochs"
            )
            # Note: actual stopping would need to be handled in train()

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
        elapsed_time: float,
    ) -> None:
        """Log epoch completion summary."""
        timestamp = time.strftime("%H:%M:%S")
        epoch_pct = ((epoch + 1) / total_epochs) * 100

        train_loss = avg_losses.get('loss', 0)
        val_dice = val_metrics.get('dice', 0)

        parts = [
            f"Loss: {train_loss:.4f}",
            f"Val Dice: {val_dice:.4f}",
            f"Time: {elapsed_time:.1f}s",
        ]

        # Add ETA from time estimator
        if hasattr(self, '_time_estimator') and self._time_estimator is not None:
            self._time_estimator.update(elapsed_time)
            eta_str = self._time_estimator.get_eta_string()
            if eta_str:
                parts.append(eta_str)

        logger.info(
            f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
            + " | ".join(parts)
        )
