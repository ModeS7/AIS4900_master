"""
Unified metrics system for ALL trainers (compression + diffusion).

Provides:
- UnifiedMetrics: Single entry point for all metric tracking and TensorBoard logging
- SimpleLossAccumulator: Dynamic loss tracking without predefined config

Usage:
    # In trainer __init__
    self._unified_metrics = UnifiedMetrics(
        writer=tensorboard_writer,
        mode='bravo',           # or 'seg', 'dual', 'multi', etc.
        spatial_dims=3,         # 2 or 3
        modality='bravo',       # For TensorBoard suffix
        device='cuda',
    )
    self._loss_accumulator = SimpleLossAccumulator()

    # Training loop
    for batch in train_loader:
        loss = train_step(batch)
        losses = {'MSE': loss.item(), 'Perceptual': perc.item()}
        self._loss_accumulator.update(losses)

    avg_losses = self._loss_accumulator.compute()
    for key, value in avg_losses.items():
        self._unified_metrics.update_loss(key, value, phase='train')
    self._unified_metrics.log_training(epoch)
    self._unified_metrics.reset_training()

    # Validation loop
    for batch in val_loader:
        pred, gt = validate_step(batch)
        self._unified_metrics.update_psnr(pred, gt)
        self._unified_metrics.update_lpips(pred, gt)
        self._unified_metrics.update_msssim(pred, gt)

    self._unified_metrics.log_validation(epoch)
    self._unified_metrics.reset_validation()
"""
import logging
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from medgen.pipeline.utils import EpochTimeEstimator

logger = logging.getLogger(__name__)


# =============================================================================
# Simple Loss Accumulator
# =============================================================================

class SimpleLossAccumulator:
    """Simple loss accumulator that doesn't require a predefined config.

    Dynamically tracks any loss keys it sees during training.
    Use this for trainers that don't use TrainerMetricsConfig.

    Usage:
        accumulator = SimpleLossAccumulator()
        accumulator.reset()

        for batch in loader:
            losses = {'l1': 0.5, 'perc': 0.1, 'kl': 0.01}
            accumulator.update(losses)

        avg_losses = accumulator.compute()
    """

    def __init__(self) -> None:
        """Initialize empty accumulator."""
        self._accumulators: dict[str, float] = {}
        self._count: int = 0

    def reset(self) -> None:
        """Reset accumulators for new epoch."""
        self._accumulators.clear()
        self._count = 0

    def update(self, losses: dict[str, float | torch.Tensor]) -> None:
        """Accumulate losses from a single step.

        Args:
            losses: Dictionary of loss values. All keys are tracked dynamically.
                   Values can be float or 0-dim tensors.
        """
        for key, val in losses.items():
            if isinstance(val, torch.Tensor):
                val = val.item()
            if key not in self._accumulators:
                self._accumulators[key] = 0.0
            self._accumulators[key] += val
        self._count += 1

    def compute(self) -> dict[str, float]:
        """Compute average losses over accumulated steps.

        Returns:
            Dictionary of averaged losses.
        """
        if self._count == 0:
            return {}
        return {key: val / self._count for key, val in self._accumulators.items()}


# =============================================================================
# Unified Metrics - Single Entry Point for All Trainers
# =============================================================================

class UnifiedMetrics:
    """
    Single entry point for all metrics across ALL trainers.

    Provides:
    - Per-metric method calls (update_psnr, update_lpips, etc.)
    - Mode-aware behavior (auto-skip PSNR for seg mode)
    - Dimension-aware (3D metrics computed slice-wise if metric is 2D-only)
    - Modality suffix on all validation metrics (LPIPS_bravo, PSNR_flair)
    - No manual add_scalar in trainers - all logging through this class

    Usage:
        metrics = UnifiedMetrics(
            writer=tensorboard_writer,
            mode='bravo',           # or 'seg', 'dual', 'multi', etc.
            spatial_dims=3,         # 2 or 3
            modality='bravo',       # For TensorBoard suffix
            device='cuda',
        )

        # Training loop
        for batch in train_loader:
            loss = train_step(batch)
            metrics.update_loss('MSE', loss.item())
            metrics.update_grad_norm(grad_norm)

        metrics.update_vram()
        metrics.log_training(epoch)
        metrics.reset_training()

        # Validation loop
        for batch in val_loader:
            pred, gt = validate_step(batch)
            metrics.update_psnr(pred, gt)
            metrics.update_lpips(pred, gt)
            metrics.update_msssim(pred, gt)
            metrics.update_regional(pred, gt, mask)

        metrics.log_validation(epoch)
        metrics.reset_validation()

        # Test evaluation
        metrics.evaluate_test(test_loader, prefix='test_best')
    """

    def __init__(
        self,
        writer: SummaryWriter | None,
        mode: str,
        spatial_dims: int = 2,
        modality: str | None = None,
        device: torch.device | None = None,
        # Optional feature flags
        enable_regional: bool = False,
        enable_codebook: bool = False,
        codebook_size: int = 512,
        num_timestep_bins: int = 10,
        # Regional tracker params (2D)
        image_size: int = 256,
        fov_mm: float = 240.0,
        # Regional tracker params (3D)
        volume_size: tuple[int, int, int] | None = None,
        # Logging config flags (from MetricsTracker)
        log_grad_norm: bool = True,
        log_timestep_losses: bool = True,
        log_regional_losses: bool = True,
        log_msssim: bool = True,
        log_psnr: bool = True,
        log_lpips: bool = False,
        log_flops: bool = True,
        # SNR weight config
        strategy_name: str = 'ddpm',
        num_train_timesteps: int = 1000,
        use_min_snr: bool = False,
        min_snr_gamma: float = 5.0,
    ):
        """Initialize unified metrics.

        Args:
            writer: TensorBoard SummaryWriter (may be None for DDP non-main).
            mode: Training mode ('seg', 'bravo', 'dual', 'multi', etc.).
            spatial_dims: 2 or 3 for 2D or 3D models.
            modality: Modality name for TensorBoard suffix (e.g., 'bravo', 'flair').
            device: Device for computation (defaults to cuda if available).
            enable_regional: Enable regional metrics tracking.
            enable_codebook: Enable codebook tracking (VQ-VAE).
            codebook_size: Size of VQ codebook (if enable_codebook=True).
            num_timestep_bins: Number of bins for timestep loss tracking.
            image_size: Image size for 2D regional tracker (default 256).
            fov_mm: Field of view in mm for regional tracker (default 240.0).
            volume_size: (H, W, D) for 3D regional tracker.
            log_grad_norm: Enable gradient norm logging.
            log_timestep_losses: Enable timestep loss logging.
            log_regional_losses: Enable regional loss logging.
            log_msssim: Enable MS-SSIM logging.
            log_psnr: Enable PSNR logging.
            log_lpips: Enable LPIPS logging.
            log_flops: Enable FLOPs logging.
            strategy_name: Diffusion strategy ('ddpm' or 'rflow').
            num_train_timesteps: Number of training timesteps for SNR computation.
            use_min_snr: Enable min-SNR loss weighting.
            min_snr_gamma: Gamma value for min-SNR weighting.
        """
        self.writer = writer
        self.mode = mode
        self.spatial_dims = spatial_dims
        self.modality = modality
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_timestep_bins = num_timestep_bins
        self.image_size = image_size
        self.fov_mm = fov_mm
        self.volume_size = volume_size or (256, 256, 160)

        # Mode-aware flags
        self.is_seg_mode = mode in ('seg', 'seg_compression')
        self.uses_image_quality = not self.is_seg_mode

        # Logging config flags (consolidated from MetricsTracker)
        self.log_grad_norm = log_grad_norm
        self.log_timestep_losses = log_timestep_losses
        self.log_regional_losses = log_regional_losses
        self.log_msssim = log_msssim
        self.log_psnr = log_psnr
        self.log_lpips = log_lpips
        self.log_flops = log_flops

        # SNR weight config
        self.strategy_name = strategy_name
        self.num_train_timesteps = num_train_timesteps
        self.use_min_snr = use_min_snr
        self.min_snr_gamma = min_snr_gamma
        self.scheduler: Any | None = None  # Set via set_scheduler()

        # Initialize accumulators
        self._init_accumulators()

        # Initialize optional components
        self._regional_tracker = None
        if enable_regional:
            self._init_regional()

        self._codebook_tracker = None
        if enable_codebook:
            self._init_codebook(codebook_size)

    def _init_accumulators(self):
        """Initialize all metric accumulators."""
        # Training accumulators
        self._train_losses: dict[str, dict[str, float]] = {}
        self._grad_norm_sum = 0.0
        self._grad_norm_max = 0.0
        self._grad_norm_count = 0
        self._current_lr: float | None = None

        # Validation quality metrics
        self._val_psnr_sum = 0.0
        self._val_psnr_count = 0
        self._val_lpips_sum = 0.0
        self._val_lpips_count = 0
        self._val_msssim_sum = 0.0
        self._val_msssim_count = 0
        self._val_msssim_3d_sum = 0.0
        self._val_msssim_3d_count = 0

        # Validation seg metrics
        self._val_dice_sum = 0.0
        self._val_dice_count = 0
        self._val_iou_sum = 0.0
        self._val_iou_count = 0

        # Validation losses
        self._val_losses: dict[str, dict[str, float]] = {}

        # Timestep losses (validation only)
        self._val_timesteps = self._create_timestep_storage()

        # Timestep-region tracking (for heatmap visualization)
        self._tr_tumor_sum = [0.0] * self.num_timestep_bins
        self._tr_tumor_count = [0] * self.num_timestep_bins
        self._tr_bg_sum = [0.0] * self.num_timestep_bins
        self._tr_bg_count = [0] * self.num_timestep_bins

        # History tracking for JSON export
        self._regional_history: dict[str, Any] = {}
        self._timestep_history: dict[str, Any] = {}
        self._timestep_region_history: dict[str, Any] = {}

        # Resource metrics
        self._vram_allocated = 0.0
        self._vram_reserved = 0.0
        self._vram_max = 0.0
        self._flops_epoch = 0.0
        self._flops_total = 0.0

    def _create_timestep_storage(self) -> dict[str, Any]:
        """Create storage for timestep losses."""
        return {
            'sums': [0.0] * self.num_timestep_bins,
            'counts': [0] * self.num_timestep_bins,
        }

    def _init_regional(self):
        """Initialize regional metrics tracker."""
        if self.spatial_dims == 3:
            from .regional import RegionalMetricsTracker3D
            self._regional_tracker = RegionalMetricsTracker3D(
                volume_size=self.volume_size,
                fov_mm=self.fov_mm,
                device=self.device,
            )
        else:
            from .regional import RegionalMetricsTracker
            self._regional_tracker = RegionalMetricsTracker(
                image_size=self.image_size,
                fov_mm=self.fov_mm,
                device=self.device,
            )

    def _init_codebook(self, codebook_size: int):
        """Initialize codebook tracker for VQ-VAE."""
        try:
            from .tracking import CodebookTracker
            self._codebook_tracker = CodebookTracker(
                num_embeddings=codebook_size,
                device=self.device,
            )
        except ImportError:
            logger.warning("CodebookTracker not available, codebook metrics disabled")

    # =========================================================================
    # Quality Metric Methods
    # =========================================================================

    def update_psnr(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute and accumulate PSNR.

        Works for both 2D and 3D data.
        Auto-skips for seg mode.

        Args:
            pred: Predicted tensor [B, C, (D), H, W].
            gt: Ground truth tensor [B, C, (D), H, W].

        Returns:
            Computed PSNR value (0.0 if skipped).
        """
        if not self.uses_image_quality:
            return 0.0
        from .quality import compute_psnr
        val = compute_psnr(pred, gt)
        self._val_psnr_sum += val
        self._val_psnr_count += 1
        return val

    def update_lpips(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute and accumulate LPIPS.

        3D data uses slice-wise computation (2.5D approach).
        Auto-skips for seg mode.

        Args:
            pred: Predicted tensor [B, C, (D), H, W].
            gt: Ground truth tensor [B, C, (D), H, W].

        Returns:
            Computed LPIPS value (0.0 if skipped).
        """
        if not self.uses_image_quality:
            return 0.0
        from .dispatch import compute_lpips_dispatch
        val = compute_lpips_dispatch(pred, gt, self.spatial_dims, device=self.device)
        self._val_lpips_sum += val
        self._val_lpips_count += 1
        return val

    def update_msssim(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute and accumulate MS-SSIM.

        3D data uses 2D slice-wise computation for efficiency.
        Auto-skips for seg mode.

        Args:
            pred: Predicted tensor [B, C, (D), H, W].
            gt: Ground truth tensor [B, C, (D), H, W].

        Returns:
            Computed MS-SSIM value (0.0 if skipped).
        """
        if not self.uses_image_quality:
            return 0.0
        from .dispatch import compute_msssim_dispatch
        val = compute_msssim_dispatch(pred, gt, self.spatial_dims)
        self._val_msssim_sum += val
        self._val_msssim_count += 1
        return val

    def update_msssim_3d(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute true 3D MS-SSIM (native, not slice-wise).

        Only works for 3D data. Auto-skips for seg mode or 2D.

        Args:
            pred: Predicted tensor [B, C, D, H, W].
            gt: Ground truth tensor [B, C, D, H, W].

        Returns:
            Computed 3D MS-SSIM value (0.0 if skipped).
        """
        if not self.uses_image_quality or self.spatial_dims != 3:
            return 0.0
        from .quality import compute_msssim
        val = compute_msssim(pred, gt, spatial_dims=3)
        self._val_msssim_3d_sum += val
        self._val_msssim_3d_count += 1
        return val

    def update_dice(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute and accumulate Dice coefficient.

        Works for both image and seg modes.

        Args:
            pred: Predicted tensor [B, C, (D), H, W].
            gt: Ground truth tensor [B, C, (D), H, W].

        Returns:
            Computed Dice value.
        """
        from .quality import compute_dice
        val = compute_dice(pred, gt)
        self._val_dice_sum += val
        self._val_dice_count += 1
        return val

    def update_iou(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute and accumulate IoU.

        Works for both image and seg modes.

        Args:
            pred: Predicted tensor [B, C, (D), H, W].
            gt: Ground truth tensor [B, C, (D), H, W].

        Returns:
            Computed IoU value.
        """
        from .quality import compute_iou
        val = compute_iou(pred, gt)
        self._val_iou_sum += val
        self._val_iou_count += 1
        return val

    # =========================================================================
    # Segmentation Metrics Methods
    # =========================================================================

    def log_seg_training(self, metrics: dict[str, float], epoch: int) -> None:
        """Log segmentation training losses.

        Provides consistent TensorBoard paths for segmentation training metrics
        across all trainers (compression seg_mode and SegmentationTrainer).

        Args:
            metrics: Dictionary of training metrics. Keys: 'bce', 'dice', 'boundary', 'gen'.
            epoch: Current epoch number.

        TensorBoard paths:
            - Loss/BCE_train
            - Loss/Dice_train
            - Loss/Boundary_train
            - Loss/Generator_train
        """
        if self.writer is None:
            return
        if 'bce' in metrics:
            self.writer.add_scalar('Loss/BCE_train', metrics['bce'], epoch)
        if 'dice' in metrics:
            self.writer.add_scalar('Loss/Dice_train', metrics['dice'], epoch)
        if 'boundary' in metrics:
            self.writer.add_scalar('Loss/Boundary_train', metrics['boundary'], epoch)
        if 'gen' in metrics:
            self.writer.add_scalar('Loss/Generator_train', metrics['gen'], epoch)

    def log_seg_validation(self, metrics: dict[str, float], epoch: int) -> None:
        """Log segmentation validation metrics.

        Provides consistent TensorBoard paths for segmentation validation metrics
        across all trainers (compression seg_mode and SegmentationTrainer).

        Args:
            metrics: Dictionary of validation metrics. Keys: 'bce', 'dice_score', 'boundary', 'gen', 'iou'.
            epoch: Current epoch number.

        TensorBoard paths:
            - Loss/BCE_val
            - Loss/Dice_val
            - Loss/Boundary_val
            - Loss/Generator_val
            - Validation/IoU
        """
        if self.writer is None:
            return
        if 'bce' in metrics:
            self.writer.add_scalar('Loss/BCE_val', metrics['bce'], epoch)
        if 'dice_score' in metrics:
            self.writer.add_scalar('Loss/Dice_val', metrics['dice_score'], epoch)
        if 'boundary' in metrics:
            self.writer.add_scalar('Loss/Boundary_val', metrics['boundary'], epoch)
        if 'gen' in metrics:
            self.writer.add_scalar('Loss/Generator_val', metrics['gen'], epoch)
        if 'iou' in metrics:
            self.writer.add_scalar('Validation/IoU', metrics['iou'], epoch)

    # =========================================================================
    # Loss Methods
    # =========================================================================

    def update_loss(self, key: str, value: float, phase: str = 'train'):
        """Accumulate a loss value.

        Args:
            key: Loss key (e.g., 'MSE', 'Total', 'KL', 'Perceptual').
            value: Loss value to accumulate.
            phase: 'train' or 'val'.
        """
        storage = self._train_losses if phase == 'train' else self._val_losses
        if key not in storage:
            storage[key] = {'sum': 0.0, 'count': 0}
        storage[key]['sum'] += value
        storage[key]['count'] += 1

    # =========================================================================
    # Training Diagnostic Methods
    # =========================================================================

    def update_grad_norm(self, value: float):
        """Accumulate gradient norm.

        Args:
            value: Gradient norm value.
        """
        self._grad_norm_sum += value
        self._grad_norm_max = max(self._grad_norm_max, value)
        self._grad_norm_count += 1

    def update_lr(self, value: float):
        """Store current learning rate.

        Args:
            value: Learning rate value.
        """
        self._current_lr = value

    def log_lr(self, lr: float, epoch: int, prefix: str = 'LR/Generator'):
        """Log learning rate to TensorBoard immediately.

        Use this for immediate LR logging (e.g., from base trainer).
        For deferred logging as part of training metrics, use update_lr() + log_training().

        Args:
            lr: Learning rate value.
            epoch: Current epoch number.
            prefix: TensorBoard tag (default: 'LR/Generator').
        """
        if self.writer is None:
            return
        self.writer.add_scalar(prefix, lr, epoch)

    def set_scheduler(self, scheduler: Any) -> None:
        """Set scheduler reference for SNR weight computation.

        Args:
            scheduler: Diffusion scheduler with alphas_cumprod attribute.
        """
        self.scheduler = scheduler

    def compute_snr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute Min-SNR loss weights for given timesteps.

        Args:
            timesteps: Tensor of timestep indices.

        Returns:
            Tensor of SNR-based loss weights.
        """
        if self.strategy_name == 'ddpm' and self.scheduler is not None:
            alphas_cumprod = self.scheduler.alphas_cumprod.to(timesteps.device)
            alpha_bar = alphas_cumprod[timesteps]
            snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
        else:
            # For RFlow: use continuous timesteps directly
            t_normalized = timesteps.float() / self.num_train_timesteps
            snr = (1.0 - t_normalized) / (t_normalized + 1e-8)

        snr_clipped = torch.clamp(snr, max=self.min_snr_gamma)
        weights = snr_clipped / (snr + 1e-8)

        return weights

    def update_vram(self):
        """Capture current VRAM usage."""
        if torch.cuda.is_available():
            self._vram_allocated = torch.cuda.memory_allocated(self.device) / 1e9
            self._vram_reserved = torch.cuda.memory_reserved(self.device) / 1e9
            self._vram_max = torch.cuda.max_memory_allocated(self.device) / 1e9

    def log_vram(self, epoch: int, prefix: str = 'VRAM'):
        """Log VRAM usage to TensorBoard immediately.

        Captures current VRAM stats and logs them. Use this for immediate
        logging (e.g., from base trainer). For deferred logging as part of
        training metrics, use update_vram() + log_training().

        Args:
            epoch: Current epoch number.
            prefix: TensorBoard tag prefix (default: 'VRAM').
        """
        if self.writer is None or not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9

        self.writer.add_scalar(f'{prefix}/allocated_GB', allocated, epoch)
        self.writer.add_scalar(f'{prefix}/reserved_GB', reserved, epoch)
        self.writer.add_scalar(f'{prefix}/max_allocated_GB', max_allocated, epoch)

    def update_flops(self, tflops_epoch: float, tflops_total: float):
        """Store FLOPs values.

        Args:
            tflops_epoch: TFLOPs for this epoch.
            tflops_total: Cumulative TFLOPs.
        """
        self._flops_epoch = tflops_epoch
        self._flops_total = tflops_total

    def update_timestep_loss(self, t: float, loss: float):
        """Accumulate validation loss for timestep bin.

        Args:
            t: Timestep value (0.0 to 1.0).
            loss: Loss value at this timestep.
        """
        bin_idx = min(int(t * self.num_timestep_bins), self.num_timestep_bins - 1)
        self._val_timesteps['sums'][bin_idx] += loss
        self._val_timesteps['counts'][bin_idx] += 1

    def update_timestep_region_loss(
        self,
        t: float,
        tumor_loss: float,
        bg_loss: float,
        tumor_count: int = 1,
        bg_count: int = 1,
    ):
        """Accumulate timestep-region losses for heatmap visualization.

        Args:
            t: Timestep value (0.0 to 1.0).
            tumor_loss: Total loss on tumor pixels.
            bg_loss: Total loss on background pixels.
            tumor_count: Number of tumor pixels (for averaging).
            bg_count: Number of background pixels (for averaging).
        """
        bin_idx = min(int(t * self.num_timestep_bins), self.num_timestep_bins - 1)
        self._tr_tumor_sum[bin_idx] += tumor_loss
        self._tr_tumor_count[bin_idx] += tumor_count
        self._tr_bg_sum[bin_idx] += bg_loss
        self._tr_bg_count[bin_idx] += bg_count

    # =========================================================================
    # Regional and Codebook Methods
    # =========================================================================

    def update_regional(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Update regional metrics tracker.

        Args:
            pred: Predicted tensor.
            gt: Ground truth tensor.
            mask: Segmentation mask for regional analysis.
        """
        if self._regional_tracker is not None:
            self._regional_tracker.update(pred, gt, mask)

    def update_codebook(self, indices: torch.Tensor):
        """Update codebook usage tracker.

        Args:
            indices: Codebook indices from VQ-VAE quantization.
        """
        if self._codebook_tracker is not None:
            self._codebook_tracker.update_fast(indices)

    # =========================================================================
    # Logging Methods
    # =========================================================================

    def log_training(self, epoch: int):
        """Log all training metrics to TensorBoard."""
        from .unified_logging import log_training
        log_training(self, epoch)

    def log_validation(self, epoch: int):
        """Log all validation metrics to TensorBoard."""
        from .unified_logging import log_validation
        log_validation(self, epoch)

    def log_generation(self, epoch: int, results: dict[str, float]):
        """Log generation metrics to TensorBoard."""
        from .unified_logging import log_generation
        log_generation(self, epoch, results)

    def log_test(self, metrics: dict[str, float], prefix: str = 'test_best'):
        """Log test evaluation metrics."""
        from .unified_logging import log_test
        log_test(self, metrics, prefix)

    def log_test_generation(self, results: dict[str, float], prefix: str = 'test_best') -> dict[str, float]:
        """Log test generation metrics (FID, KID, CMMD, diversity)."""
        from .unified_logging import log_test_generation
        return log_test_generation(self, results, prefix)

    def log_test_regional(self, regional_tracker: Any, prefix: str = 'test_best'):
        """Log regional metrics with modality suffix."""
        from .unified_logging import log_test_regional
        log_test_regional(self, regional_tracker, prefix)

    def log_validation_regional(self, regional_tracker: Any, epoch: int, modality_override: str | None = None):
        """Log regional metrics for validation (supports per-modality tracking)."""
        from .unified_logging import log_validation_regional
        log_validation_regional(self, regional_tracker, epoch, modality_override)

    def log_test_timesteps(self, timestep_bins: dict[str, float], prefix: str = 'test_best'):
        """Log timestep bin losses."""
        from .unified_logging import log_test_timesteps
        log_test_timesteps(self, timestep_bins, prefix)

    def log_per_channel_validation(self, channel_metrics: dict[str, dict[str, float]], epoch: int):
        """Log per-channel validation (dual/multi modes)."""
        from .unified_logging import log_per_channel_validation
        log_per_channel_validation(self, channel_metrics, epoch)

    def log_per_modality_validation(self, metrics: dict[str, float], modality: str, epoch: int):
        """Log per-modality validation."""
        from .unified_logging import log_per_modality_validation
        log_per_modality_validation(self, metrics, modality, epoch)

    def log_regularization_loss(self, loss_type: str, weighted_loss: float, epoch: int, unweighted_loss: float | None = None):
        """Log regularization losses (KL for VAE, VQ for VQVAE)."""
        from .unified_logging import log_regularization_loss
        log_regularization_loss(self, loss_type, weighted_loss, epoch, unweighted_loss)

    def log_codebook_metrics(self, codebook_tracker: Any, epoch: int, prefix: str = 'Codebook') -> dict[str, float]:
        """Log codebook metrics from external tracker (VQVAE)."""
        from .unified_logging import log_codebook_metrics
        return log_codebook_metrics(self, codebook_tracker, epoch, prefix)

    def update_validation_batch(
        self,
        psnr: float,
        msssim: float,
        lpips: float | None = None,
        msssim_3d: float | None = None,
        dice: float | None = None,
        iou: float | None = None,
    ):
        """Update validation metrics from pre-computed results.

        Replaces direct state access (self._unified_metrics._val_psnr_sum = ...).

        Args:
            psnr: Pre-computed PSNR value.
            msssim: Pre-computed MS-SSIM value.
            lpips: Optional pre-computed LPIPS value.
            msssim_3d: Optional pre-computed 3D MS-SSIM value.
            dice: Optional pre-computed Dice value.
            iou: Optional pre-computed IoU value.
        """
        self._val_psnr_sum += psnr
        self._val_psnr_count += 1
        self._val_msssim_sum += msssim
        self._val_msssim_count += 1

        if lpips is not None:
            self._val_lpips_sum += lpips
            self._val_lpips_count += 1
        if msssim_3d is not None:
            self._val_msssim_3d_sum += msssim_3d
            self._val_msssim_3d_count += 1
        if dice is not None:
            self._val_dice_sum += dice
            self._val_dice_count += 1
        if iou is not None:
            self._val_iou_sum += iou
            self._val_iou_count += 1

    def set_validation_metrics(self, metrics: dict[str, float]) -> None:
        """Set validation metrics from a dictionary of pre-computed values.

        Use this when you have pre-aggregated metrics (e.g., from external
        validation loops) and want to set them directly for logging.

        Unlike update_validation_batch() which accumulates, this sets
        values directly with count=1.

        Args:
            metrics: Dictionary with optional keys:
                'psnr', 'msssim', 'lpips', 'msssim_3d', 'dice', 'dice_score', 'iou'
        """
        if 'psnr' in metrics:
            self._val_psnr_sum = metrics['psnr']
            self._val_psnr_count = 1
        if 'msssim' in metrics:
            self._val_msssim_sum = metrics['msssim']
            self._val_msssim_count = 1
        if 'lpips' in metrics:
            self._val_lpips_sum = metrics['lpips']
            self._val_lpips_count = 1
        if 'msssim_3d' in metrics:
            self._val_msssim_3d_sum = metrics['msssim_3d']
            self._val_msssim_3d_count = 1
        # Support both 'dice' and 'dice_score' keys
        dice_val = metrics.get('dice') if 'dice' in metrics else metrics.get('dice_score')
        if dice_val is not None:
            self._val_dice_sum = dice_val
            self._val_dice_count = 1
        if 'iou' in metrics:
            self._val_iou_sum = metrics['iou']
            self._val_iou_count = 1

    def log_timestep_region_heatmap(self, epoch: int):
        """Log 2D heatmap of loss by timestep bin and region."""
        from .unified_logging import log_timestep_region_heatmap
        log_timestep_region_heatmap(self, epoch)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def get_validation_metrics(self) -> dict[str, float]:
        """Collect current validation metrics as a dict.

        Returns:
            Dict of metric name -> averaged value.
        """
        metrics = {}

        if self.uses_image_quality:
            if self._val_psnr_count > 0:
                metrics['PSNR'] = self._val_psnr_sum / self._val_psnr_count
            if self._val_msssim_count > 0:
                metrics['MS-SSIM'] = self._val_msssim_sum / self._val_msssim_count
            if self._val_lpips_count > 0:
                metrics['LPIPS'] = self._val_lpips_sum / self._val_lpips_count
            if self._val_msssim_3d_count > 0:
                metrics['MS-SSIM-3D'] = self._val_msssim_3d_sum / self._val_msssim_3d_count

        if self._val_dice_count > 0:
            metrics['Dice'] = self._val_dice_sum / self._val_dice_count
        if self._val_iou_count > 0:
            metrics['IoU'] = self._val_iou_sum / self._val_iou_count

        # Add losses
        for key, data in self._val_losses.items():
            if data['count'] > 0:
                metrics[key] = data['sum'] / data['count']

        return metrics

    def get_training_losses(self) -> dict[str, float]:
        """Collect current training losses as a dict.

        Returns:
            Dict of loss name -> averaged value.
        """
        return {
            key: data['sum'] / data['count']
            for key, data in self._train_losses.items()
            if data['count'] > 0
        }

    # =========================================================================
    # Reset Methods
    # =========================================================================

    def reset_training(self):
        """Reset training accumulators for next epoch."""
        self._train_losses.clear()
        self._grad_norm_sum = 0.0
        self._grad_norm_max = 0.0
        self._grad_norm_count = 0
        self._current_lr = None
        self._vram_allocated = 0.0
        self._vram_reserved = 0.0
        self._vram_max = 0.0
        if self._codebook_tracker is not None:
            self._codebook_tracker.reset()

    def reset_validation(self):
        """Reset validation accumulators for next epoch."""
        self._val_losses.clear()
        self._val_psnr_sum = self._val_psnr_count = 0
        self._val_lpips_sum = self._val_lpips_count = 0
        self._val_msssim_sum = self._val_msssim_count = 0
        self._val_msssim_3d_sum = self._val_msssim_3d_count = 0
        self._val_dice_sum = self._val_dice_count = 0
        self._val_iou_sum = self._val_iou_count = 0
        self._val_timesteps = self._create_timestep_storage()
        # Reset timestep-region accumulators
        self._tr_tumor_sum = [0.0] * self.num_timestep_bins
        self._tr_tumor_count = [0] * self.num_timestep_bins
        self._tr_bg_sum = [0.0] * self.num_timestep_bins
        self._tr_bg_count = [0] * self.num_timestep_bins
        if self._regional_tracker is not None:
            self._regional_tracker.reset()

    def reset_all(self):
        """Reset everything."""
        self.reset_training()
        self.reset_validation()

    def record_epoch_history(self, epoch: int):
        """Record current epoch data to history for JSON export."""
        from .unified_history import record_epoch_history
        record_epoch_history(self, epoch)

    def save_json_histories(self, save_dir: str):
        """Save all history data to JSON files."""
        from .unified_history import save_json_histories
        save_json_histories(self, save_dir)

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def log_reconstruction_figure(self, original: torch.Tensor, reconstructed: torch.Tensor, epoch: int, mask: torch.Tensor | None = None, timesteps: torch.Tensor | None = None, tag: str = 'Figures/reconstruction', max_samples: int = 8, metrics: dict[str, float] | None = None, save_path: str | None = None):
        """Log reconstruction comparison figure to TensorBoard."""
        from .unified_visualization import log_reconstruction_figure
        log_reconstruction_figure(self, original, reconstructed, epoch, mask, timesteps, tag, max_samples, metrics, save_path)

    def log_worst_batch(self, original: torch.Tensor, reconstructed: torch.Tensor, loss: float, epoch: int, phase: str = 'train', mask: torch.Tensor | None = None, timesteps: torch.Tensor | None = None, tag_prefix: str | None = None, save_path: str | None = None, display_metrics: dict[str, float] | None = None):
        """Log worst batch visualization to TensorBoard."""
        from .unified_visualization import log_worst_batch
        log_worst_batch(self, original, reconstructed, loss, epoch, phase, mask, timesteps, tag_prefix, save_path, display_metrics)

    def log_denoising_trajectory(self, trajectory: list, epoch: int, tag: str = 'denoising_trajectory'):
        """Log denoising step visualization to TensorBoard."""
        from .unified_visualization import log_denoising_trajectory
        log_denoising_trajectory(self, trajectory, epoch, tag)

    def log_generated_samples(self, samples: torch.Tensor, epoch: int, tag: str = 'Generated_Samples', nrow: int = 4, num_slices: int = 8):
        """Log generated samples grid to TensorBoard."""
        from .unified_visualization import log_generated_samples
        log_generated_samples(self, samples, epoch, tag, nrow, num_slices)

    def log_latent_samples(self, samples: torch.Tensor, epoch: int, tag: str = 'Latent_Samples', num_slices: int = 8):
        """Log latent space samples to TensorBoard (before decoding)."""
        from .unified_visualization import log_latent_samples
        log_latent_samples(self, samples, epoch, tag, num_slices)

    def log_latent_trajectory(self, trajectory: list, epoch: int, tag: str = 'denoising_trajectory'):
        """Log latent space denoising trajectory to TensorBoard."""
        from .unified_visualization import log_latent_trajectory
        log_latent_trajectory(self, trajectory, epoch, tag)

    def log_test_figure(self, original: torch.Tensor, reconstructed: torch.Tensor, prefix: str = 'test_best', mask: torch.Tensor | None = None, metrics: dict[str, float] | None = None):
        """Log test evaluation figure to TensorBoard."""
        from .unified_visualization import log_test_figure
        log_test_figure(self, original, reconstructed, prefix, mask, metrics)

    def _extract_center_slice(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract center slice from 3D volume."""
        from .unified_visualization import _extract_center_slice
        return _extract_center_slice(self, tensor)

    def _extract_multiple_slices(self, tensor: torch.Tensor, num_slices: int = 8) -> torch.Tensor:
        """Extract multiple evenly-spaced slices from 3D volume."""
        from .unified_visualization import _extract_multiple_slices
        return _extract_multiple_slices(self, tensor, num_slices)

    # =========================================================================
    # Tracker Integration Methods
    # =========================================================================

    def log_flops_from_tracker(self, flops_tracker: Any, epoch: int) -> None:
        """Log FLOPs metrics from FLOPsTracker."""
        from .unified_logging import log_flops_from_tracker
        log_flops_from_tracker(self, flops_tracker, epoch)

    def log_grad_norm_from_tracker(self, grad_tracker: Any, epoch: int, prefix: str = 'training/grad_norm') -> None:
        """Log gradient norm stats from GradientNormTracker."""
        from .unified_logging import log_grad_norm_from_tracker
        log_grad_norm_from_tracker(self, grad_tracker, epoch, prefix)

    def log_sample_images(self, images: torch.Tensor, tag: str, epoch: int) -> None:
        """Log image grid to TensorBoard using add_images."""
        from .unified_logging import log_sample_images
        log_sample_images(self, images, tag, epoch)

    # =========================================================================
    # Console Logging
    # =========================================================================

    def log_console_summary(self, epoch: int, total_epochs: int, elapsed_time: float, time_estimator: "EpochTimeEstimator | None" = None):
        """Log epoch completion summary to console."""
        from .unified_history import log_console_summary
        log_console_summary(self, epoch, total_epochs, elapsed_time, time_estimator)
