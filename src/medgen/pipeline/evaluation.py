"""Test evaluation utilities for compression and diffusion trainers.

Provides shared functionality for test set evaluation:
- Checkpoint loading
- Results saving (JSON + TensorBoard)
- Metric computation orchestration

This module extracts ~500 lines of duplicated code from:
- BaseCompressionTrainer.evaluate_test_set()
- BaseCompression3DTrainer.evaluate_test_set()
- DiffusionTrainer.evaluate_test_set()
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Utilities
# =============================================================================

def load_checkpoint_if_needed(
    checkpoint_name: Optional[str],
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
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded {checkpoint_name} checkpoint for test evaluation")
        else:
            logger.warning(f"Checkpoint {checkpoint_path} not found, using current weights")
            return "current"

    return checkpoint_name or "current"


def save_test_results(
    metrics: Dict[str, float],
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


def log_test_header(label: str) -> None:
    """Log test evaluation header."""
    logger.info("=" * 60)
    logger.info(f"EVALUATING ON TEST SET ({label.upper()} MODEL)")
    logger.info("=" * 60)


def log_test_results(
    metrics: Dict[str, float],
    label: str,
    n_samples: int,
) -> None:
    """Log test results to console.

    Args:
        metrics: Dictionary of metric name -> value.
        label: Checkpoint label.
        n_samples: Number of test samples evaluated.
    """
    logger.info(f"Test Results - {label} ({n_samples} samples):")
    if 'l1' in metrics:
        logger.info(f"  L1 Loss: {metrics['l1']:.6f}")
    if 'mse' in metrics:
        logger.info(f"  MSE Loss: {metrics['mse']:.6f}")
    if 'msssim' in metrics:
        logger.info(f"  MS-SSIM: {metrics['msssim']:.4f}")
    if 'msssim_3d' in metrics:
        logger.info(f"  MS-SSIM-3D: {metrics['msssim_3d']:.4f}")
    if 'psnr' in metrics:
        logger.info(f"  PSNR:    {metrics['psnr']:.2f} dB")
    if 'lpips' in metrics:
        logger.info(f"  LPIPS:   {metrics['lpips']:.4f}")


def log_metrics_to_tensorboard(
    writer: Optional[SummaryWriter],
    metrics: Dict[str, float],
    prefix: str,
    step: int = 0,
) -> None:
    """Log metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter (can be None).
        metrics: Dictionary of metric name -> value.
        prefix: Prefix for metric names (e.g., "test_best").
        step: Global step for TensorBoard.
    """
    if writer is None:
        return

    # Map metric names to TensorBoard display names
    name_map = {
        'l1': 'L1',
        'mse': 'MSE',
        'msssim': 'MS-SSIM',
        'msssim_3d': 'MS-SSIM-3D',
        'psnr': 'PSNR',
        'lpips': 'LPIPS',
    }

    for key, value in metrics.items():
        if key in name_map:
            writer.add_scalar(f'{prefix}/{name_map[key]}', value, step)
        elif key != 'n_samples':  # Skip sample count
            writer.add_scalar(f'{prefix}/{key}', value, step)


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
        writer: Optional[SummaryWriter] = None,
        metrics_config: Optional[MetricsConfig] = None,
        is_cluster: bool = False,
        worst_batch_figure_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
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
        """
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.writer = writer
        self.metrics_config = metrics_config or MetricsConfig()
        self.is_cluster = is_cluster
        self.worst_batch_figure_fn = worst_batch_figure_fn

    def evaluate(
        self,
        test_loader: DataLoader,
        checkpoint_name: Optional[str] = None,
        get_eval_model: Optional[Callable[[], nn.Module]] = None,
    ) -> Dict[str, float]:
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
        log_metrics_to_tensorboard(self.writer, metrics, f'test_{label}')

        # Log worst batch figure
        if worst_batch_data is not None and self.writer is not None:
            self._log_worst_batch(worst_batch_data, label)

        # Restore training mode
        model_to_use.train()

        return metrics

    def _run_evaluation_loop(
        self,
        test_loader: DataLoader,
        model: nn.Module,
    ) -> Tuple[Dict[str, float], Optional[Any]]:
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

                # Track worst batch by L1/MSE loss
                loss_key = 'l1' if 'l1' in batch_metrics else 'mse'
                batch_loss = batch_metrics.get(loss_key, 0.0)
                if batch_loss > worst_loss:
                    worst_loss = batch_loss
                    worst_batch_data = self._capture_worst_batch(
                        batch_data, batch_metrics
                    )

                n_batches += 1
                n_samples += batch_size

        # Compute averages
        metrics = {key: val / n_batches for key, val in accumulators.items()}
        metrics['n_samples'] = n_samples

        # Add any additional metrics (e.g., volume-level 3D MS-SSIM)
        self._add_additional_metrics(metrics)

        return metrics, worst_batch_data

    def _init_accumulators(self) -> Dict[str, float]:
        """Initialize metric accumulators based on config."""
        accumulators = {}
        if self.metrics_config.compute_l1:
            accumulators['l1'] = 0.0
        if self.metrics_config.compute_psnr:
            accumulators['psnr'] = 0.0
        if self.metrics_config.compute_lpips:
            accumulators['lpips'] = 0.0
        if self.metrics_config.compute_msssim:
            accumulators['msssim'] = 0.0
        return accumulators

    def _prepare_batch(self, batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
            images = batch.get('image', batch.get('images'))
            mask = batch.get('mask', batch.get('seg'))
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
    ) -> Dict[str, float]:
        """Compute metrics for a single batch. Subclass implements."""
        pass

    def _capture_worst_batch(
        self,
        batch_data: Any,
        batch_metrics: Dict[str, float],
    ) -> Optional[Any]:
        """Capture data for worst batch visualization. Can be overridden."""
        return None

    def _add_additional_metrics(self, metrics: Dict[str, float]) -> None:
        """Add additional metrics after main loop. Can be overridden."""
        pass

    def _log_worst_batch(self, worst_batch_data: Any, label: str) -> None:
        """Log worst batch figure to TensorBoard and save as PNG.

        Args:
            worst_batch_data: Dict with 'original', 'generated', 'loss' keys.
            label: 'best' or 'latest' for filename.
        """
        import matplotlib.pyplot as plt

        if self.worst_batch_figure_fn is None or self.writer is None:
            return

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
        writer: Optional[SummaryWriter] = None,
        metrics_config: Optional[MetricsConfig] = None,
        is_cluster: bool = False,
        regional_tracker_factory: Optional[Callable[[], Any]] = None,
        volume_3d_msssim_fn: Optional[Callable[[], Optional[float]]] = None,
        worst_batch_figure_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
        image_keys: Optional[List[str]] = None,
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
        """
        super().__init__(
            model, device, save_dir, writer, metrics_config, is_cluster,
            worst_batch_figure_fn=worst_batch_figure_fn
        )
        self.forward_fn = forward_fn
        self.weight_dtype = weight_dtype
        self.regional_tracker_factory = regional_tracker_factory
        self.volume_3d_msssim_fn = volume_3d_msssim_fn
        self.image_keys = image_keys
        self._regional_tracker: Optional[Any] = None
        # Per-channel metric accumulators
        self._per_channel_metrics: Dict[str, Dict[str, float]] = {}
        self._per_channel_count: int = 0

    def evaluate(
        self,
        test_loader: DataLoader,
        checkpoint_name: Optional[str] = None,
        get_eval_model: Optional[Callable[[], nn.Module]] = None,
    ) -> Dict[str, float]:
        """Run test evaluation with optional regional tracking."""
        # Create regional tracker if configured
        if self.regional_tracker_factory is not None:
            self._regional_tracker = self.regional_tracker_factory()

        # Reset per-channel accumulators
        self._per_channel_metrics = {}
        self._per_channel_count = 0

        result = super().evaluate(test_loader, checkpoint_name, get_eval_model)

        label = checkpoint_name or "current"

        # Log regional metrics to TensorBoard
        if self._regional_tracker is not None and self.writer is not None:
            self._regional_tracker.log_to_tensorboard(
                self.writer, 0, prefix=f'test_{label}_regional'
            )

        # Log per-channel metrics to TensorBoard
        if self._per_channel_count > 0 and self.writer is not None:
            for key, metrics in self._per_channel_metrics.items():
                if self.metrics_config.compute_msssim:
                    self.writer.add_scalar(f'test_{label}/MS-SSIM_{key}', metrics['msssim'] / self._per_channel_count, 0)
                if self.metrics_config.compute_psnr:
                    self.writer.add_scalar(f'test_{label}/PSNR_{key}', metrics['psnr'] / self._per_channel_count, 0)
                if self.metrics_config.compute_lpips:
                    self.writer.add_scalar(f'test_{label}/LPIPS_{key}', metrics['lpips'] / self._per_channel_count, 0)

        return result

    def _get_batch_size(self, batch_data: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> int:
        """Get batch size from prepared batch."""
        images, _ = batch_data
        return images.shape[0]

    def _compute_batch_metrics(
        self,
        model: nn.Module,
        batch_data: Tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> Dict[str, float]:
        """Compute metrics for a single batch."""
        from torch.amp import autocast
        from .metrics import compute_lpips, compute_msssim, compute_psnr

        images, mask = batch_data

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            reconstructed = self.forward_fn(model, images)

        metrics = {}

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
        batch_data: Tuple[torch.Tensor, Optional[torch.Tensor]],
        batch_metrics: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Capture worst batch data for visualization."""
        if not hasattr(self, '_current_batch'):
            return None

        return {
            'original': self._current_batch['images'].cpu(),
            'generated': self._current_batch['reconstructed'].float().cpu(),
            'loss': batch_metrics.get('l1', batch_metrics.get('mse', 0.0)),
            'loss_breakdown': {
                'L1': batch_metrics.get('l1', 0.0),
            },
        }

    def _add_additional_metrics(self, metrics: Dict[str, float]) -> None:
        """Add volume-level 3D MS-SSIM if configured."""
        if self.volume_3d_msssim_fn is not None:
            msssim_3d = self.volume_3d_msssim_fn()
            if msssim_3d is not None:
                metrics['msssim_3d'] = msssim_3d


# =============================================================================
# 3D Compression Test Evaluator
# =============================================================================

class Compression3DTestEvaluator(BaseTestEvaluator):
    """Test evaluator for 3D volumetric compression models (VAE-3D, VQVAE-3D).

    Computes: L1, MS-SSIM-3D (volumetric), MS-SSIM (2D slicewise), PSNR, LPIPS-3D.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        save_dir: str,
        forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
        weight_dtype: torch.dtype = torch.bfloat16,
        writer: Optional[SummaryWriter] = None,
        metrics_config: Optional[MetricsConfig] = None,
        is_cluster: bool = False,
        regional_tracker_factory: Optional[Callable[[], Any]] = None,
        worst_batch_figure_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
        image_keys: Optional[List[str]] = None,
    ):
        """Initialize 3D compression evaluator.

        Args:
            model: Model to evaluate.
            device: Device for evaluation.
            save_dir: Directory for saving results.
            forward_fn: Callable that takes (model, volumes) and returns reconstruction.
            weight_dtype: Data type for autocast.
            writer: Optional TensorBoard writer.
            metrics_config: Which metrics to compute.
            is_cluster: If True, disable tqdm progress bar.
            regional_tracker_factory: Optional factory to create 3D regional tracker.
            worst_batch_figure_fn: Optional callable to create 3D worst batch figure.
            image_keys: Optional list of channel names for per-channel metrics.
        """
        super().__init__(
            model, device, save_dir, writer, metrics_config, is_cluster,
            worst_batch_figure_fn=worst_batch_figure_fn
        )
        self.forward_fn = forward_fn
        self.weight_dtype = weight_dtype
        self.regional_tracker_factory = regional_tracker_factory
        self.image_keys = image_keys
        self._regional_tracker: Optional[Any] = None
        self._per_channel_metrics: Dict[str, Dict[str, float]] = {}
        self._per_channel_count: int = 0

    def _init_accumulators(self) -> Dict[str, float]:
        """Initialize accumulators including 3D MS-SSIM."""
        accumulators = super()._init_accumulators()
        if self.metrics_config.compute_msssim_3d:
            accumulators['msssim_3d'] = 0.0
        return accumulators

    def evaluate(
        self,
        test_loader: DataLoader,
        checkpoint_name: Optional[str] = None,
        get_eval_model: Optional[Callable[[], nn.Module]] = None,
    ) -> Dict[str, float]:
        """Run test evaluation with optional regional tracking."""
        if self.regional_tracker_factory is not None:
            self._regional_tracker = self.regional_tracker_factory()

        # Reset per-channel accumulators
        self._per_channel_metrics = {}
        self._per_channel_count = 0

        result = super().evaluate(test_loader, checkpoint_name, get_eval_model)

        label = checkpoint_name or "current"

        if self._regional_tracker is not None and self.writer is not None:
            self._regional_tracker.log_to_tensorboard(
                self.writer, 0, prefix=f'test_{label}_regional'
            )

        # Log per-channel metrics to TensorBoard
        if self._per_channel_count > 0 and self.writer is not None:
            for key, metrics in self._per_channel_metrics.items():
                if self.metrics_config.compute_msssim:
                    self.writer.add_scalar(f'test_{label}/MS-SSIM_{key}', metrics['msssim'] / self._per_channel_count, 0)
                if self.metrics_config.compute_msssim_3d:
                    self.writer.add_scalar(f'test_{label}/MS-SSIM-3D_{key}', metrics['msssim_3d'] / self._per_channel_count, 0)
                if self.metrics_config.compute_psnr:
                    self.writer.add_scalar(f'test_{label}/PSNR_{key}', metrics['psnr'] / self._per_channel_count, 0)
                if self.metrics_config.compute_lpips:
                    self.writer.add_scalar(f'test_{label}/LPIPS_{key}', metrics['lpips'] / self._per_channel_count, 0)

        return result

    def _get_batch_size(self, batch_data: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> int:
        """Get batch size from prepared batch."""
        images, _ = batch_data
        return images.shape[0]

    def _compute_batch_metrics(
        self,
        model: nn.Module,
        batch_data: Tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> Dict[str, float]:
        """Compute metrics for a single 3D batch."""
        from torch.amp import autocast
        from .metrics import (
            compute_lpips_3d,
            compute_msssim,
            compute_msssim_2d_slicewise,
            compute_psnr,
        )

        images, mask = batch_data

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            reconstructed = self.forward_fn(model, images)

        metrics = {}

        # L1 loss
        if self.metrics_config.compute_l1:
            metrics['l1'] = torch.abs(reconstructed - images).mean().item()

        # MS-SSIM (2D slicewise for comparison with 2D trainers)
        if self.metrics_config.compute_msssim:
            metrics['msssim'] = compute_msssim_2d_slicewise(
                reconstructed.float(), images.float()
            )

        # MS-SSIM-3D (volumetric)
        if self.metrics_config.compute_msssim_3d:
            metrics['msssim_3d'] = compute_msssim(
                reconstructed.float(), images.float(), spatial_dims=3
            )

        # PSNR
        if self.metrics_config.compute_psnr:
            metrics['psnr'] = compute_psnr(reconstructed, images)

        # LPIPS (3D batched)
        if self.metrics_config.compute_lpips:
            metrics['lpips'] = compute_lpips_3d(
                reconstructed.float(), images.float(), device=self.device
            )

        # Regional tracking
        if self._regional_tracker is not None and mask is not None:
            self._regional_tracker.update(reconstructed.float(), images.float(), mask)

        # Per-channel metrics for multi-modality mode
        if self.image_keys is not None and len(self.image_keys) > 1:
            n_channels = images.shape[1]
            if n_channels == len(self.image_keys):
                for i, key in enumerate(self.image_keys):
                    img_ch = images[:, i:i+1]
                    rec_ch = reconstructed[:, i:i+1]

                    if key not in self._per_channel_metrics:
                        self._per_channel_metrics[key] = {'msssim': 0.0, 'msssim_3d': 0.0, 'psnr': 0.0, 'lpips': 0.0}

                    if self.metrics_config.compute_msssim:
                        self._per_channel_metrics[key]['msssim'] += compute_msssim_2d_slicewise(rec_ch.float(), img_ch.float())
                    if self.metrics_config.compute_msssim_3d:
                        self._per_channel_metrics[key]['msssim_3d'] += compute_msssim(rec_ch.float(), img_ch.float(), spatial_dims=3)
                    if self.metrics_config.compute_psnr:
                        self._per_channel_metrics[key]['psnr'] += compute_psnr(rec_ch, img_ch)
                    if self.metrics_config.compute_lpips:
                        self._per_channel_metrics[key]['lpips'] += compute_lpips_3d(rec_ch.float(), img_ch.float(), device=self.device)

                self._per_channel_count += 1

        # Store for worst batch capture
        self._current_batch = {
            'images': images,
            'reconstructed': reconstructed,
        }

        return metrics

    def _capture_worst_batch(
        self,
        batch_data: Tuple[torch.Tensor, Optional[torch.Tensor]],
        batch_metrics: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Capture worst batch data for 3D visualization."""
        if not hasattr(self, '_current_batch'):
            return None

        return {
            'original': self._current_batch['images'].cpu(),
            'generated': self._current_batch['reconstructed'].float().cpu(),
            'loss': batch_metrics.get('l1', 0.0),
        }
