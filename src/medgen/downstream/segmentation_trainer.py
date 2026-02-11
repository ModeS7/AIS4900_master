"""
Downstream segmentation trainer for evaluating synthetic data utility.

Uses MONAI's SegResNet model with combined BCE + Dice loss.
Tracks per-tumor-size Dice scores using SegRegionalMetricsTracker.
"""
import logging
import os
import time
from typing import Any

import torch
from monai.networks.nets import SegResNet
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from medgen.core import create_warmup_cosine_scheduler
from medgen.losses import SegmentationLoss
from medgen.metrics import SimpleLossAccumulator, UnifiedMetrics
from medgen.metrics.mc_dropout import MCDropoutEvaluator
from medgen.metrics.regional import SegRegionalMetricsTracker
from medgen.metrics.seg_metrics import GlobalSegMetrics
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

        # Best Dice tracking (Dice: higher is better, range [0, 1])
        self.best_dice: float = 0.0

        # Extract segmentation-specific config
        self.in_channels: int = cfg.model.get('in_channels', 1)
        self.out_channels: int = cfg.model.get('out_channels', 1)
        self.init_filters: int = cfg.model.get('init_filters', 32)
        self.blocks_down: tuple[int, ...] = tuple(cfg.model.get('blocks_down', [1, 2, 2, 4]))
        self.blocks_up: tuple[int, ...] = tuple(cfg.model.get('blocks_up', [1, 1, 1]))
        self.dropout_prob: float = cfg.model.get('dropout_prob', 0.2)

        # Image size for regional metrics
        self.image_size: int = cfg.model.image_size

        # Validation frequency
        self.val_every: int = cfg.training.get('val_every', 1)

        # Early stopping
        self.patience: int = cfg.training.get('patience', 20)
        self._epochs_without_improvement: int = 0
        self._stop_early_flag: bool = False

        # Precision config - bf16 AMP for both 2D and 3D
        # Requires PyTorch 2.10+ (2.9 had 3D conv regression)
        self.use_amp = True
        self.weight_dtype = torch.bfloat16

        # Loss accumulator
        self._loss_accumulator = SimpleLossAccumulator()

        # Seg mode flag for consistency with compression trainers
        self.seg_mode = True

        # Evaluation config
        self.compute_hd95: bool = cfg.evaluation.get('compute_hd95', True)
        self.mc_dropout_samples: int = cfg.evaluation.get('mc_dropout_samples', 0)

    @classmethod
    def create_2d(cls, cfg: DictConfig) -> 'SegmentationTrainer':
        """Create 2D segmentation trainer."""
        return cls(cfg, spatial_dims=2)

    @classmethod
    def create_3d(cls, cfg: DictConfig) -> 'SegmentationTrainer':
        """Create 3D segmentation trainer."""
        return cls(cfg, spatial_dims=3)

    def _init_unified_metrics(self) -> None:
        """Initialize UnifiedMetrics for segmentation logging.

        Creates UnifiedMetrics instance for consistent TensorBoard paths
        across all trainers (matching compression trainer seg_mode).
        """
        self._unified_metrics = UnifiedMetrics(
            writer=self.writer,
            mode='seg',
            spatial_dims=self.spatial_dims,
            modality=None,
            device=self.device,
            enable_regional=False,  # Use SegRegionalMetricsTracker directly
            enable_codebook=False,
            image_size=self.image_size,
            fov_mm=self.cfg.evaluation.get('fov_mm', 240.0),
        )

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
        run_name = f"{exp_name}{timestamp}"
        return os.path.join(
            self.cfg.paths.model_dir,
            'segmentation',
            f'{self.spatial_dims}d',
            run_name
        )

    def setup_model(self, pretrained_checkpoint: str | None = None) -> None:
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

        # Create global metrics tracker (precision, recall, HD95)
        self.global_metrics = GlobalSegMetrics(
            compute_hd95=self.compute_hd95,
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

        # Initialize unified metrics (after writer is set up in parent)
        self._init_unified_metrics()

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

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        if 'model_state_dict' in checkpoint:
            self.model_raw.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model_raw.load_state_dict(checkpoint['state_dict'])
        else:
            self.model_raw.load_state_dict(checkpoint)

        # Restore best_dice if present in checkpoint
        if 'best_dice' in checkpoint:
            self.best_dice = checkpoint['best_dice']

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

        # Gradient clipping + norm tracking
        if self.gradient_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model_raw.parameters(),
                max_norm=self.gradient_clip_norm,
            )
            if self.log_grad_norm:
                self._grad_norm_tracker.update(grad_norm.item())

        self.optimizer.step()

        return TrainingStepResult(
            total_loss=total_loss.item(),
            reconstruction_loss=breakdown['bce'].item(),
            perceptual_loss=breakdown['dice'].item(),
            regularization_loss=breakdown['boundary'].item(),
        )

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

        # Log training losses using unified metrics
        if hasattr(self, '_unified_metrics') and self._unified_metrics is not None:
            # Map keys for consistency: dice_loss -> dice
            seg_losses = {
                'bce': avg_losses.get('bce', 0),
                'dice': avg_losses.get('dice_loss', 0),
                'boundary': avg_losses.get('boundary', 0),
                'gen': avg_losses.get('loss', 0),
            }
            self._unified_metrics.log_seg_training(seg_losses, epoch)

        return avg_losses

    def compute_validation_losses(
        self,
        epoch: int,
        log_figures: bool = True,
    ) -> dict[str, float]:
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
        self.global_metrics.reset()

        val_losses = {'loss': 0.0, 'bce': 0.0, 'dice_loss': 0.0}
        n_batches = 0
        worst_loss = 0.0
        worst_batch_data: dict[str, Any] | None = None

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                targets = batch['seg'].to(self.device, non_blocking=True)

                with autocast('cuda', enabled=self.use_amp, dtype=self.weight_dtype):
                    logits = self.model(images)
                    total_loss, breakdown = self.loss_fn(logits.float(), targets.float())

                batch_loss = total_loss.item()
                val_losses['loss'] += batch_loss
                val_losses['bce'] += breakdown['bce'].item()
                val_losses['dice_loss'] += breakdown['dice'].item()
                n_batches += 1

                # Track worst batch (highest loss)
                if batch_loss > worst_loss:
                    worst_loss = batch_loss
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    worst_batch_data = {
                        'target': targets.cpu(),
                        'prediction': preds.cpu(),
                        'loss': batch_loss,
                        'breakdown': {
                            'bce': breakdown['bce'].item(),
                            'dice': breakdown['dice'].item(),
                        },
                    }

                # Update regional tracker
                self.regional_tracker.update(
                    prediction=logits,
                    target=targets,
                    apply_sigmoid=True,
                )

                # Update global metrics (precision, recall, HD95)
                self.global_metrics.update(logits, targets, apply_sigmoid=True)

        # Average losses
        for key in val_losses:
            val_losses[key] /= max(n_batches, 1)

        # Compute per-size metrics
        regional_metrics = self.regional_tracker.compute()

        # Compute global metrics
        global_results = self.global_metrics.compute()

        # Merge metrics
        metrics = {**val_losses, **regional_metrics, **global_results}

        # MC Dropout confidence estimation (on first batch only)
        mc_metrics: dict[str, float] = {}
        if self.mc_dropout_samples > 0 and log_figures:
            mc_metrics, mc_mean_pred, mc_uncertainty = self._run_mc_dropout()
            metrics.update(mc_metrics)

        # Log to TensorBoard using unified metrics
        if hasattr(self, '_unified_metrics') and self._unified_metrics is not None:
            # Log seg validation metrics for consistent paths
            seg_metrics = {
                'bce': val_losses.get('bce', 0),
                'dice_score': regional_metrics.get('dice', 0),  # dice METRIC, not loss
                'boundary': 0,  # Not tracked in val
                'gen': val_losses.get('loss', 0),
                'iou': regional_metrics.get('iou', 0),
            }
            self._unified_metrics.log_seg_validation(seg_metrics, epoch)

        # Log global + MC metrics to TensorBoard
        if self.writer is not None:
            for key in ('precision', 'recall', 'hd95'):
                if key in global_results:
                    self.writer.add_scalar(f'Validation/{key}', global_results[key], epoch)
            for key in ('confidence_tp', 'confidence_fp', 'confidence_fn', 'ece'):
                if key in mc_metrics:
                    self.writer.add_scalar(f'Validation/{key}', mc_metrics[key], epoch)

        # Log worst batch figure (Ground Truth vs Prediction vs |Diff|)
        if log_figures and worst_batch_data is not None and self._unified_metrics is not None:
            self._unified_metrics.log_worst_batch(
                original=worst_batch_data['target'],
                reconstructed=worst_batch_data['prediction'],
                loss=worst_batch_data['loss'],
                epoch=epoch,
                phase='val',
                display_metrics=worst_batch_data['breakdown'],
            )

        # Log regional metrics (per-size dice) separately
        if self.writer is not None:
            self.regional_tracker.log_to_tensorboard(self.writer, epoch, prefix='regional_seg')

            # Log predictions figure
            if log_figures and n_batches > 0:
                if self.mc_dropout_samples > 0 and mc_metrics:
                    self._log_predictions_figure(epoch, mc_mean_pred, mc_uncertainty)
                else:
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
            global_str = (
                f"Prec: {metrics.get('precision', 0):.3f}, "
                f"Rec: {metrics.get('recall', 0):.3f}"
            )
            if 'hd95' in metrics:
                global_str += f", HD95: {metrics['hd95']:.1f}"
            logger.info(f"Validation - {dice_str} | {size_str} | {global_str}")

        return metrics

    def _run_mc_dropout(self) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
        """Run MC Dropout on first validation batch.

        Returns:
            Tuple of (confidence_metrics, mean_prediction, uncertainty_map).
        """
        mc_eval = MCDropoutEvaluator(self.model, self.mc_dropout_samples, self.device)

        batch = next(iter(self.val_loader))
        images = batch['image'].to(self.device)
        targets = batch['seg'].to(self.device)

        mean_pred, uncertainty = mc_eval.predict_with_uncertainty(images)
        conf_metrics = mc_eval.compute_confidence_metrics(mean_pred, uncertainty, targets)

        return conf_metrics, mean_pred, uncertainty

    def _log_predictions_figure(
        self,
        epoch: int,
        mc_mean_pred: torch.Tensor | None = None,
        mc_uncertainty: torch.Tensor | None = None,
    ) -> None:
        """Log sample predictions to TensorBoard.

        Columns: Input | Ground Truth | Prediction | Confidence | Uncertainty
        If MC dropout disabled, shows 4 columns (sigmoid as confidence).
        If MC dropout enabled, shows 5 columns with MC-based confidence + uncertainty.

        Args:
            epoch: Current epoch number.
            mc_mean_pred: MC mean prediction [B, 1, ...] (optional).
            mc_uncertainty: MC variance map [B, 1, ...] (optional).
        """
        if self.val_loader is None or self.writer is None:
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        self.model.eval()

        has_mc = mc_mean_pred is not None and mc_uncertainty is not None

        # Get one batch for predictions
        batch = next(iter(self.val_loader))
        images = batch['image'].to(self.device)
        targets = batch['seg'].to(self.device)

        with torch.no_grad():
            logits = self.model(images)
            sigmoid_pred = torch.sigmoid(logits)

        if has_mc:
            pred_binary = (mc_mean_pred > 0.5)
            confidence = mc_mean_pred
            uncertainty = mc_uncertainty
            n_cols = 5
        else:
            pred_binary = (sigmoid_pred > 0.5)
            confidence = sigmoid_pred
            uncertainty = None
            n_cols = 4

        n_samples = min(4, images.shape[0])
        fig, axes = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            if self.spatial_dims == 3:
                s = images.shape[2] // 2  # middle slice
                img = images[i, 0, s].cpu().numpy()
                tgt = targets[i, 0, s].cpu().numpy()
                pred = pred_binary[i, 0, s].float().cpu().numpy()
                conf = confidence[i, 0, s].cpu().numpy()
                unc = uncertainty[i, 0, s].cpu().numpy() if uncertainty is not None else None
                slice_label = ' (mid slice)'
            else:
                img = images[i, 0].cpu().numpy()
                tgt = targets[i, 0].cpu().numpy()
                pred = pred_binary[i, 0].float().cpu().numpy()
                conf = confidence[i, 0].cpu().numpy()
                unc = uncertainty[i, 0].cpu().numpy() if uncertainty is not None else None
                slice_label = ''

            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f'Input{slice_label}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(tgt, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

            im = axes[i, 3].imshow(conf, cmap='viridis', vmin=0, vmax=1)
            axes[i, 3].set_title('Confidence' if has_mc else 'Sigmoid')
            axes[i, 3].axis('off')
            fig.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

            if has_mc and unc is not None:
                im = axes[i, 4].imshow(unc, cmap='hot')
                axes[i, 4].set_title('Uncertainty')
                axes[i, 4].axis('off')
                fig.colorbar(im, ax=axes[i, 4], fraction=0.046, pad=0.04)

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
            'best_dice': self.best_dice,
            'config': OmegaConf.to_container(self.cfg),
        }

        path = os.path.join(self.save_dir, f'checkpoint_{name}.pt')
        from medgen.pipeline.utils import _safe_torch_save
        _safe_torch_save(checkpoint, path)

    def _get_model_config(self) -> dict[str, Any]:
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
        avg_losses: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Hook called at end of each epoch."""
        super()._on_epoch_end(epoch, avg_losses, val_metrics)

        # Log gradient norms
        if self.log_grad_norm and self._unified_metrics is not None:
            self._unified_metrics.log_grad_norm_from_tracker(
                self._grad_norm_tracker, epoch, prefix='training/grad_norm'
            )

        # Early stopping check (compare against best_dice, not best_loss)
        val_dice = val_metrics.get('dice', 0)
        if val_dice > 0 and val_dice > self.best_dice:
            self.best_dice = val_dice
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        if self._epochs_without_improvement >= self.patience:
            logger.info(
                f"Early stopping triggered: no improvement for {self.patience} epochs"
            )
            self._stop_early_flag = True

    def _should_stop_early(self) -> bool:
        """Check if training should stop due to lack of improvement.

        Returns:
            True if no improvement for `patience` epochs.
        """
        return self._stop_early_flag

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: dict[str, float],
        val_metrics: dict[str, float],
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
