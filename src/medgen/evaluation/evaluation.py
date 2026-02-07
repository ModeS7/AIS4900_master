"""Test evaluation utilities for compression and diffusion trainers.

Provides shared functionality for test set evaluation:
- Checkpoint loading
- Results saving (JSON + TensorBoard)
- Metric computation orchestration

This module extracts duplicated evaluation code from trainers.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if TYPE_CHECKING:
    from medgen.metrics.unified import UnifiedMetrics

from medgen.core.dict_utils import get_with_fallbacks

# Import logging helpers from submodule
from .evaluation_logging import (
    log_metrics_to_tensorboard,
    log_test_header,
    log_test_per_modality,
    log_test_results,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Utilities
# =============================================================================

def load_checkpoint_if_needed(
    checkpoint_name: str | None,
    save_dir: str,
    model: nn.Module,
    device: torch.device,
) -> str:
    """Load checkpoint weights if specified.

    Args:
        checkpoint_name: "best", "latest", or None for current weights.
        save_dir: Directory containing checkpoints.
        model: Model to load weights into.
        device: Device to load weights to.

    Returns:
        Label string for logging ("best", "latest", or "current").
    """
    if checkpoint_name is not None:
        checkpoint_path = os.path.join(save_dir, f"checkpoint_{checkpoint_name}.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded {checkpoint_name} checkpoint for test evaluation")
        else:
            logger.warning(f"Checkpoint {checkpoint_path} not found, using current weights")
            return "current"

    return checkpoint_name or "current"


def save_test_results(
    metrics: dict[str, float],
    label: str,
    save_dir: str,
) -> str:
    """Save test results to JSON file.

    Args:
        metrics: Dictionary of metric name -> value.
        label: Checkpoint label ("best", "latest", "current").
        save_dir: Directory to save results to.

    Returns:
        Path to saved JSON file.
    """
    results_path = os.path.join(save_dir, f'test_results_{label}.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    return results_path


# =============================================================================
# Metrics Configuration
# =============================================================================

@dataclass
class MetricsConfig:
    """Configuration for which metrics to compute during test evaluation."""

    compute_l1: bool = True
    compute_psnr: bool = True
    compute_lpips: bool = True
    compute_msssim: bool = True
    compute_msssim_3d: bool = False  # For 3D trainers
    compute_regional: bool = False
    seg_mode: bool = False  # Compute Dice/IoU instead of image metrics


# =============================================================================
# Test Evaluator Base Class
# =============================================================================

class BaseTestEvaluator(ABC):
    """Base class for test set evaluation.

    Provides template method pattern for test evaluation:
    1. Load checkpoint
    2. Setup accumulators
    3. Loop through test data (subclass implements metrics computation)
    4. Save and log results

    Subclasses implement:
    - _compute_batch_metrics(): Trainer-specific metrics computation
    - _create_worst_batch_figure(): Trainer-specific visualization
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        save_dir: str,
        writer: SummaryWriter | None = None,
        metrics_config: MetricsConfig | None = None,
        is_cluster: bool = False,
        worst_batch_figure_fn: Callable[[dict[str, Any]], Any] | None = None,
        modality_name: str | None = None,
        unified_metrics: "UnifiedMetrics | None" = None,
    ):
        """Initialize test evaluator.

        Args:
            model: Model to evaluate (raw, not wrapped).
            device: Device for evaluation.
            save_dir: Directory for saving results.
            writer: Optional TensorBoard writer.
            metrics_config: Which metrics to compute.
            is_cluster: If True, disable tqdm progress bar.
            worst_batch_figure_fn: Optional callable to create worst batch figure.
            modality_name: Optional modality name for single-modality suffix
                (e.g., 'bravo', 'seg'). If set and not 'multi_modality' or 'dual',
                metrics are logged with this suffix.
            unified_metrics: Optional UnifiedMetrics instance for centralized logging.
        """
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.writer = writer
        self.metrics_config = metrics_config or MetricsConfig()
        self.is_cluster = is_cluster
        self.worst_batch_figure_fn = worst_batch_figure_fn
        self.modality_name = modality_name
        self._unified_metrics = unified_metrics

    def evaluate(
        self,
        test_loader: DataLoader,
        checkpoint_name: str | None = None,
        get_eval_model: Callable[[], nn.Module] | None = None,
    ) -> dict[str, float]:
        """Run test evaluation.

        Args:
            test_loader: DataLoader for test set.
            checkpoint_name: "best", "latest", or None.
            get_eval_model: Optional callable to get evaluation model (e.g., EMA).

        Returns:
            Dictionary of metric name -> value.
        """
        # Load checkpoint
        label = load_checkpoint_if_needed(
            checkpoint_name, self.save_dir, self.model, self.device
        )
        log_test_header(label)

        # Select model (EMA or raw)
        if checkpoint_name is None and get_eval_model is not None:
            model_to_use = get_eval_model()
        else:
            model_to_use = self.model
        model_to_use.eval()

        # Run evaluation loop
        metrics, worst_batch_data = self._run_evaluation_loop(
            test_loader, model_to_use
        )

        # Log results
        log_test_results(metrics, label, metrics.get('n_samples', 0))
        save_test_results(metrics, label, self.save_dir)

        # Log to TensorBoard with modality suffix for single-modality modes
        # Empty string means no suffix (e.g., seg_conditioned modes)
        is_single_modality = (
            self.modality_name is not None
            and self.modality_name != ''
            and self.modality_name not in ('multi_modality', 'dual')
        )
        if is_single_modality:
            # Use per-modality logging for single-modality modes (e.g., test_best/PSNR_bravo)
            log_test_per_modality(
                self.writer, metrics, f'test_{label}', self.modality_name,
                unified_metrics=self._unified_metrics,
            )
        else:
            # Aggregate logging for multi-modality/dual (e.g., test_best/PSNR)
            log_metrics_to_tensorboard(
                self.writer, metrics, f'test_{label}',
                unified_metrics=self._unified_metrics,
            )

        # Log worst batch figure
        if worst_batch_data is not None and (self.writer is not None or self._unified_metrics is not None):
            self._log_worst_batch(worst_batch_data, label)

        # Restore training mode
        model_to_use.train()

        return metrics

    def _run_evaluation_loop(
        self,
        test_loader: DataLoader,
        model: nn.Module,
    ) -> tuple[dict[str, float], Any | None]:
        """Run the evaluation loop over all batches.

        Args:
            test_loader: Test data loader.
            model: Model to use for evaluation.

        Returns:
            Tuple of (metrics dict, worst_batch_data or None).
        """
        # Initialize accumulators
        accumulators = self._init_accumulators()
        n_batches = 0
        n_samples = 0
        worst_loss = 0.0
        worst_batch_data = None

        with torch.inference_mode():
            for batch in tqdm(
                test_loader,
                desc="Test evaluation",
                ncols=100,
                disable=self.is_cluster
            ):
                # Prepare batch (subclass may override)
                batch_data = self._prepare_batch(batch)
                batch_size = self._get_batch_size(batch_data)

                # Compute metrics for this batch
                batch_metrics = self._compute_batch_metrics(model, batch_data)

                # Accumulate metrics
                for key, value in batch_metrics.items():
                    if key in accumulators:
                        accumulators[key] += value

                # Track worst batch by loss (L1/MSE for images, 1-Dice for seg)
                if self.metrics_config.seg_mode:
                    # Seg mode: worst batch = lowest Dice (use 1-Dice as loss)
                    batch_loss = 1.0 - batch_metrics.get('dice', 1.0)
                else:
                    loss_key = 'l1' if 'l1' in batch_metrics else 'mse'
                    batch_loss = batch_metrics.get(loss_key, 0.0)
                if batch_loss > worst_loss:
                    worst_loss = batch_loss
                    worst_batch_data = self._capture_worst_batch(
                        batch_data, batch_metrics
                    )

                n_batches += 1
                n_samples += batch_size

                # Free GPU tensors stored for worst batch capture
                self._current_batch = None

        # Compute averages
        metrics = {key: val / n_batches for key, val in accumulators.items()}
        metrics['n_samples'] = n_samples

        # Add any additional metrics (e.g., volume-level 3D MS-SSIM)
        self._add_additional_metrics(metrics)

        return metrics, worst_batch_data

    def _init_accumulators(self) -> dict[str, float]:
        """Initialize metric accumulators based on config."""
        accumulators = {}
        if self.metrics_config.seg_mode:
            # Segmentation mode: Dice, IoU, and loss components
            accumulators['dice'] = 0.0
            accumulators['iou'] = 0.0
            accumulators['bce'] = 0.0
            accumulators['boundary'] = 0.0
            accumulators['gen'] = 0.0
        else:
            # Standard image metrics
            if self.metrics_config.compute_l1:
                accumulators['l1'] = 0.0
            if self.metrics_config.compute_psnr:
                accumulators['psnr'] = 0.0
            if self.metrics_config.compute_lpips:
                accumulators['lpips'] = 0.0
            if self.metrics_config.compute_msssim:
                accumulators['msssim'] = 0.0
        return accumulators

    def _prepare_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare batch for evaluation, handling dict, tuple, or tensor inputs.

        Handles three input formats:
        - Dict: Extracts 'image'/'images' and 'mask'/'seg' keys
        - Tuple/List: First element is images, second (optional) is mask
        - Tensor: Uses as images directly with no mask

        Args:
            batch: Raw batch from DataLoader.

        Returns:
            Tuple of (images_tensor, mask_tensor_or_None).
        """
        if isinstance(batch, dict):
            images = get_with_fallbacks(batch, 'image', 'images')
            mask = get_with_fallbacks(batch, 'seg', 'mask')
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
            mask = batch[1] if len(batch) > 1 else None
        else:
            images = batch
            mask = None

        images = images.to(self.device, non_blocking=True)
        if mask is not None:
            mask = mask.to(self.device, non_blocking=True)

        return images, mask

    @abstractmethod
    def _get_batch_size(self, batch_data: Any) -> int:
        """Get batch size from prepared batch data."""
        pass

    @abstractmethod
    def _compute_batch_metrics(
        self,
        model: nn.Module,
        batch_data: Any,
    ) -> dict[str, float]:
        """Compute metrics for a single batch. Subclass implements."""
        pass

    def _capture_worst_batch(
        self,
        batch_data: Any,
        batch_metrics: dict[str, float],
    ) -> Any | None:
        """Capture data for worst batch visualization. Can be overridden."""
        return None

    def _add_additional_metrics(self, metrics: dict[str, float]) -> None:
        """Add additional metrics after main loop. Can be overridden."""
        pass

    def _log_worst_batch(self, worst_batch_data: Any, label: str) -> None:
        """Log worst batch figure to TensorBoard and save as PNG.

        Args:
            worst_batch_data: Dict with 'original', 'generated', 'loss' keys.
            label: 'best' or 'latest' for filename.
        """
        import matplotlib.pyplot as plt

        if self.worst_batch_figure_fn is None:
            return
        if self.writer is None and self._unified_metrics is None:
            return

        # Use UnifiedMetrics if available (preferred)
        if self._unified_metrics is not None:
            self._unified_metrics.log_worst_batch(
                original=worst_batch_data['original'],
                reconstructed=worst_batch_data['generated'],
                loss=worst_batch_data['loss'],
                epoch=0,
                tag_prefix=f'test_{label}',
                save_path=os.path.join(self.save_dir, f'test_worst_batch_{label}.png'),
            )
            return

        # Fallback to direct writer calls for backward compatibility
        # Log to TensorBoard
        fig = self.worst_batch_figure_fn(worst_batch_data)
        self.writer.add_figure(f'test_{label}/worst_batch', fig, 0)
        plt.close(fig)

        # Save as PNG
        fig = self.worst_batch_figure_fn(worst_batch_data)
        fig_path = os.path.join(self.save_dir, f'test_worst_batch_{label}.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Test worst batch saved to: {fig_path}")


# =============================================================================
# 2D Compression Test Evaluator
# =============================================================================

class CompressionTestEvaluator(BaseTestEvaluator):
    """Test evaluator for 2D compression models (VAE, VQVAE, DCAE).

    Computes: L1, MS-SSIM, PSNR, LPIPS, optional volume 3D MS-SSIM.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        save_dir: str,
        forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
        weight_dtype: torch.dtype = torch.bfloat16,
        writer: SummaryWriter | None = None,
        metrics_config: MetricsConfig | None = None,
        is_cluster: bool = False,
        regional_tracker_factory: Callable[[], Any] | None = None,
        volume_3d_msssim_fn: Callable[[], float | None] | None = None,
        worst_batch_figure_fn: Callable[[dict[str, Any]], Any] | None = None,
        image_keys: list[str] | None = None,
        seg_loss_fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict[str, float]]] | None = None,
        modality_name: str | None = None,
        unified_metrics: "UnifiedMetrics | None" = None,
    ):
        """Initialize 2D compression evaluator.

        Args:
            model: Model to evaluate.
            device: Device for evaluation.
            save_dir: Directory for saving results.
            forward_fn: Callable that takes (model, images) and returns reconstruction.
                This is trainer-specific (VAE returns recon, VQVAE returns recon from (recon, vq_loss)).
            weight_dtype: Data type for autocast.
            writer: Optional TensorBoard writer.
            metrics_config: Which metrics to compute.
            is_cluster: If True, disable tqdm progress bar.
            regional_tracker_factory: Optional factory to create regional tracker.
            volume_3d_msssim_fn: Optional callable to compute volume-level 3D MS-SSIM.
            worst_batch_figure_fn: Optional callable to create worst batch figure.
            image_keys: Optional list of channel names for per-channel metrics (e.g., ['t1_pre', 't1_gd']).
            seg_loss_fn: Optional segmentation loss function for seg_mode (returns total, breakdown).
            modality_name: Optional modality name for single-modality suffix (e.g., 'bravo', 'seg').
            unified_metrics: Optional UnifiedMetrics instance for centralized logging.
        """
        super().__init__(
            model, device, save_dir, writer, metrics_config, is_cluster,
            worst_batch_figure_fn=worst_batch_figure_fn,
            modality_name=modality_name,
            unified_metrics=unified_metrics,
        )
        self.forward_fn = forward_fn
        self.weight_dtype = weight_dtype
        self.regional_tracker_factory = regional_tracker_factory
        self.volume_3d_msssim_fn = volume_3d_msssim_fn
        self.image_keys = image_keys
        self.seg_loss_fn = seg_loss_fn
        self._regional_tracker: Any | None = None
        # Per-channel metric accumulators
        self._per_channel_metrics: dict[str, dict[str, float]] = {}
        self._per_channel_count: int = 0

    def evaluate(
        self,
        test_loader: DataLoader,
        checkpoint_name: str | None = None,
        get_eval_model: Callable[[], nn.Module] | None = None,
    ) -> dict[str, float]:
        """Run test evaluation with optional regional tracking."""
        # Create regional tracker if configured
        if self.regional_tracker_factory is not None:
            self._regional_tracker = self.regional_tracker_factory()

        # Reset per-channel accumulators
        self._per_channel_metrics = {}
        self._per_channel_count = 0

        result = super().evaluate(test_loader, checkpoint_name, get_eval_model)

        label = checkpoint_name or "current"

        # Log regional metrics to TensorBoard with modality suffix for single-modality
        if self._regional_tracker is not None and (self.writer is not None or self._unified_metrics is not None):
            is_single_modality = (
                self.modality_name is not None
                and self.modality_name != ''
                and self.modality_name not in ('multi_modality', 'dual')
            )
            if is_single_modality:
                regional_prefix = f'test_{label}_regional_{self.modality_name}'
            else:
                regional_prefix = f'test_{label}_regional'
            # Use UnifiedMetrics if available (preferred)
            if self._unified_metrics is not None:
                self._unified_metrics.log_test_regional(
                    self._regional_tracker, prefix=regional_prefix
                )
            else:
                self._regional_tracker.log_to_tensorboard(
                    self.writer, 0, prefix=regional_prefix
                )

        # Log per-channel metrics to TensorBoard using unified helper
        if self._per_channel_count > 0 and self.writer is not None:
            for key, channel_metrics in self._per_channel_metrics.items():
                modality_metrics = {}
                if self.metrics_config.compute_msssim:
                    modality_metrics['msssim'] = channel_metrics['msssim'] / self._per_channel_count
                if self.metrics_config.compute_psnr:
                    modality_metrics['psnr'] = channel_metrics['psnr'] / self._per_channel_count
                if self.metrics_config.compute_lpips:
                    modality_metrics['lpips'] = channel_metrics['lpips'] / self._per_channel_count
                log_test_per_modality(self.writer, modality_metrics, f'test_{label}', key)

        return result

    def _prepare_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare batch for evaluation, handling extra seg channel in tensor.

        For dual/multi-modality modes, test dataloader may return concatenated
        tensor [B, C+1, H, W] where last channel is seg mask for regional metrics.
        This method splits it based on expected image_keys count.
        """
        # First use base class logic to extract images and mask
        images, mask = super()._prepare_batch(batch)

        # If image_keys is set and tensor has extra channel, split out seg
        if self.image_keys is not None and mask is None:
            expected_channels = len(self.image_keys)
            actual_channels = images.shape[1]
            if actual_channels == expected_channels + 1:
                # Last channel is seg mask
                mask = images[:, -1:, :, :]
                images = images[:, :expected_channels, :, :]

        return images, mask

    def _get_batch_size(self, batch_data: tuple[torch.Tensor, torch.Tensor | None]) -> int:
        """Get batch size from prepared batch."""
        images, _ = batch_data
        return images.shape[0]

    def _compute_batch_metrics(
        self,
        model: nn.Module,
        batch_data: tuple[torch.Tensor, torch.Tensor | None],
    ) -> dict[str, float]:
        """Compute metrics for a single batch."""
        from torch.amp import autocast

        from medgen.metrics import (
            compute_dice,
            compute_iou,
            compute_lpips,
            compute_msssim,
            compute_psnr,
        )

        images, mask = batch_data

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            reconstructed = self.forward_fn(model, images)

        metrics = {}

        # Segmentation mode: compute Dice/IoU and seg losses
        if self.metrics_config.seg_mode:
            metrics['dice'] = compute_dice(reconstructed, images, apply_sigmoid=True)
            metrics['iou'] = compute_iou(reconstructed, images, apply_sigmoid=True)
            # Compute BCE, Boundary and total seg loss if seg_loss_fn available
            if self.seg_loss_fn is not None:
                seg_loss, seg_breakdown = self.seg_loss_fn(reconstructed, images)
                metrics['bce'] = seg_breakdown.get('bce', 0.0)
                metrics['boundary'] = seg_breakdown.get('boundary', 0.0)
                metrics['gen'] = seg_loss.item()
        else:
            # Standard image metrics
            # L1 loss
            if self.metrics_config.compute_l1:
                metrics['l1'] = torch.abs(reconstructed - images).mean().item()

            # MS-SSIM
            if self.metrics_config.compute_msssim:
                metrics['msssim'] = compute_msssim(reconstructed.float(), images.float())

            # PSNR
            if self.metrics_config.compute_psnr:
                metrics['psnr'] = compute_psnr(reconstructed, images)

            # LPIPS
            if self.metrics_config.compute_lpips:
                metrics['lpips'] = compute_lpips(
                    reconstructed.float(), images.float(), device=self.device
                )

        # Regional tracking
        if self._regional_tracker is not None and mask is not None:
            self._regional_tracker.update(reconstructed, images, mask)

        # Per-channel metrics for dual/multi-modality mode
        if self.image_keys is not None and len(self.image_keys) > 1:
            n_channels = images.shape[1]
            if n_channels == len(self.image_keys):
                for i, key in enumerate(self.image_keys):
                    img_ch = images[:, i:i+1]
                    rec_ch = reconstructed[:, i:i+1]

                    if key not in self._per_channel_metrics:
                        self._per_channel_metrics[key] = {'msssim': 0.0, 'psnr': 0.0, 'lpips': 0.0}

                    if self.metrics_config.compute_msssim:
                        self._per_channel_metrics[key]['msssim'] += compute_msssim(rec_ch.float(), img_ch.float())
                    if self.metrics_config.compute_psnr:
                        self._per_channel_metrics[key]['psnr'] += compute_psnr(rec_ch, img_ch)
                    if self.metrics_config.compute_lpips:
                        self._per_channel_metrics[key]['lpips'] += compute_lpips(rec_ch.float(), img_ch.float(), device=self.device)

                self._per_channel_count += 1

        # Store for worst batch capture
        self._current_batch = {
            'images': images,
            'reconstructed': reconstructed,
        }

        return metrics

    def _capture_worst_batch(
        self,
        batch_data: tuple[torch.Tensor, torch.Tensor | None],
        batch_metrics: dict[str, float],
    ) -> dict[str, Any] | None:
        """Capture worst batch data for visualization."""
        if not hasattr(self, '_current_batch'):
            return None

        # Build loss breakdown based on mode
        if self.metrics_config.seg_mode:
            loss = 1.0 - batch_metrics.get('dice', 1.0)  # 1-Dice as loss
            loss_breakdown = {
                'Dice': batch_metrics.get('dice', 0.0),
                'IoU': batch_metrics.get('iou', 0.0),
            }
        else:
            loss = get_with_fallbacks(batch_metrics, 'l1', 'mse', default=0.0)
            loss_breakdown = {
                'L1': batch_metrics.get('l1', 0.0),
            }

        return {
            'original': self._current_batch['images'].cpu(),
            'generated': self._current_batch['reconstructed'].float().cpu(),
            'loss': loss,
            'loss_breakdown': loss_breakdown,
        }

    def _add_additional_metrics(self, metrics: dict[str, float]) -> None:
        """Add volume-level 3D MS-SSIM if configured."""
        if self.volume_3d_msssim_fn is not None:
            msssim_3d = self.volume_3d_msssim_fn()
            if msssim_3d is not None:
                metrics['msssim_3d'] = msssim_3d


# =============================================================================
# Re-exports from submodules for backward compatibility
# =============================================================================

from .evaluation_3d import (  # noqa: F401
    Compression3DTestEvaluator,
    create_compression_test_evaluator,
)
