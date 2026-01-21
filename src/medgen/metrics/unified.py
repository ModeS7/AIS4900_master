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
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

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
        self._accumulators: Dict[str, float] = {}
        self._count: int = 0

    def reset(self) -> None:
        """Reset accumulators for new epoch."""
        self._accumulators.clear()
        self._count = 0

    def update(self, losses: Dict[str, Union[float, torch.Tensor]]) -> None:
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

    def compute(self) -> Dict[str, float]:
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
        writer: Optional[SummaryWriter],
        mode: str,
        spatial_dims: int = 2,
        modality: Optional[str] = None,
        device: Optional[torch.device] = None,
        # Optional feature flags
        enable_regional: bool = False,
        enable_codebook: bool = False,
        codebook_size: int = 512,
        num_timestep_bins: int = 10,
        # Regional tracker params (2D)
        image_size: int = 256,
        fov_mm: float = 240.0,
        # Regional tracker params (3D)
        volume_size: Optional[Tuple[int, int, int]] = None,
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
        self.scheduler: Optional[Any] = None  # Set via set_scheduler()

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
        self._train_losses: Dict[str, Dict[str, float]] = {}
        self._grad_norm_sum = 0.0
        self._grad_norm_max = 0.0
        self._grad_norm_count = 0
        self._current_lr: Optional[float] = None

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
        self._val_losses: Dict[str, Dict[str, float]] = {}

        # Timestep losses (validation only)
        self._val_timesteps = self._create_timestep_storage()

        # Timestep-region tracking (for heatmap visualization)
        self._tr_tumor_sum = [0.0] * self.num_timestep_bins
        self._tr_tumor_count = [0] * self.num_timestep_bins
        self._tr_bg_sum = [0.0] * self.num_timestep_bins
        self._tr_bg_count = [0] * self.num_timestep_bins

        # History tracking for JSON export
        self._regional_history: Dict[str, Any] = {}
        self._timestep_history: Dict[str, Any] = {}
        self._timestep_region_history: Dict[str, Any] = {}

        # Resource metrics
        self._vram_allocated = 0.0
        self._vram_reserved = 0.0
        self._vram_max = 0.0
        self._flops_epoch = 0.0
        self._flops_total = 0.0

    def _create_timestep_storage(self) -> Dict[str, Any]:
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
        from .quality import compute_lpips, compute_lpips_3d
        if self.spatial_dims == 3:
            val = compute_lpips_3d(pred, gt, device=self.device)
        else:
            val = compute_lpips(pred, gt)
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
        from .quality import compute_msssim, compute_msssim_2d_slicewise
        if self.spatial_dims == 3:
            val = compute_msssim_2d_slicewise(pred, gt)
        else:
            val = compute_msssim(pred, gt, spatial_dims=2)
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
        """Log all training metrics to TensorBoard.

        Args:
            epoch: Current epoch number.
        """
        if self.writer is None:
            return

        # Losses
        for key, data in self._train_losses.items():
            if data['count'] > 0:
                self.writer.add_scalar(
                    f'Loss/{key}_train',
                    data['sum'] / data['count'],
                    epoch,
                )

        # LR
        if self._current_lr is not None:
            self.writer.add_scalar('LR/Generator', self._current_lr, epoch)

        # Grad norm
        if self._grad_norm_count > 0:
            self.writer.add_scalar(
                'training/grad_norm_avg',
                self._grad_norm_sum / self._grad_norm_count,
                epoch,
            )
            self.writer.add_scalar('training/grad_norm_max', self._grad_norm_max, epoch)

        # VRAM
        if self._vram_allocated > 0:
            self.writer.add_scalar('VRAM/allocated_GB', self._vram_allocated, epoch)
            self.writer.add_scalar('VRAM/reserved_GB', self._vram_reserved, epoch)
            self.writer.add_scalar('VRAM/max_allocated_GB', self._vram_max, epoch)

        # FLOPs
        if self._flops_epoch > 0:
            self.writer.add_scalar('FLOPs/TFLOPs_epoch', self._flops_epoch, epoch)
            self.writer.add_scalar('FLOPs/TFLOPs_total', self._flops_total, epoch)

        # Codebook
        if self._codebook_tracker is not None:
            self._codebook_tracker.log_to_tensorboard(self.writer, epoch)

    def log_validation(self, epoch: int):
        """Log all validation metrics to TensorBoard.

        Args:
            epoch: Current epoch number.
        """
        if self.writer is None:
            return

        suffix = f'_{self.modality}' if self.modality else ''

        # Losses
        for key, data in self._val_losses.items():
            if data['count'] > 0:
                self.writer.add_scalar(
                    f'Loss/{key}_val',
                    data['sum'] / data['count'],
                    epoch,
                )

        # Quality metrics (image modes)
        if self.uses_image_quality:
            if self._val_psnr_count > 0:
                self.writer.add_scalar(
                    f'Validation/PSNR{suffix}',
                    self._val_psnr_sum / self._val_psnr_count,
                    epoch,
                )
            if self._val_msssim_count > 0:
                self.writer.add_scalar(
                    f'Validation/MS-SSIM{suffix}',
                    self._val_msssim_sum / self._val_msssim_count,
                    epoch,
                )
            if self._val_lpips_count > 0:
                self.writer.add_scalar(
                    f'Validation/LPIPS{suffix}',
                    self._val_lpips_sum / self._val_lpips_count,
                    epoch,
                )
            if self._val_msssim_3d_count > 0:
                self.writer.add_scalar(
                    f'Validation/MS-SSIM-3D{suffix}',
                    self._val_msssim_3d_sum / self._val_msssim_3d_count,
                    epoch,
                )

        # Seg metrics (always log if computed)
        if self._val_dice_count > 0:
            self.writer.add_scalar(
                f'Validation/Dice{suffix}',
                self._val_dice_sum / self._val_dice_count,
                epoch,
            )
        if self._val_iou_count > 0:
            self.writer.add_scalar(
                f'Validation/IoU{suffix}',
                self._val_iou_sum / self._val_iou_count,
                epoch,
            )

        # Validation timesteps (format: Timestep/0.0-0.1, Timestep/0.1-0.2, etc.)
        for i in range(self.num_timestep_bins):
            if self._val_timesteps['counts'][i] > 0:
                # Convert bin index to normalized timestep range [0.0, 1.0]
                bin_start = i / self.num_timestep_bins
                bin_end = (i + 1) / self.num_timestep_bins
                avg = self._val_timesteps['sums'][i] / self._val_timesteps['counts'][i]
                self.writer.add_scalar(f'Timestep/{bin_start:.1f}-{bin_end:.1f}', avg, epoch)

        # Regional
        if self._regional_tracker is not None:
            prefix = f'regional{suffix}' if suffix else 'regional'
            self._regional_tracker.log_to_tensorboard(self.writer, epoch, prefix=prefix)

    def log_generation(self, epoch: int, results: Dict[str, float]):
        """Log generation metrics to TensorBoard.

        Args:
            epoch: Current epoch number.
            results: Dict of generation metric results (KID, CMMD, etc.).
                Keys starting with 'Diversity/' go to 'Generation_Diversity/' section.
        """
        if self.writer is None:
            return
        for key, value in results.items():
            if key.startswith('Diversity/'):
                # Diversity metrics get their own section
                metric_name = key[len('Diversity/'):]  # Remove 'Diversity/' prefix
                self.writer.add_scalar(f'Generation_Diversity/{metric_name}', value, epoch)
            else:
                self.writer.add_scalar(f'Generation/{key}', value, epoch)

    def log_test(self, metrics: Dict[str, float], prefix: str = 'test_best'):
        """Log test evaluation metrics.

        Args:
            metrics: Dict of metric name -> value.
            prefix: 'test_best' or 'test_latest'.
        """
        if self.writer is None:
            return
        suffix = f'_{self.modality}' if self.modality else ''
        for key, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{key}{suffix}', value, 0)

    def log_test_generation(
        self,
        results: Dict[str, float],
        prefix: str = 'test_best',
    ) -> Dict[str, float]:
        """Log test generation metrics (FID, KID, CMMD, diversity).

        Paths:
        - Generation: {prefix}_generation/{key}{suffix} (e.g., test_best_generation/FID_bravo)
        - Diversity: {prefix}_diversity/{key}{suffix} (e.g., test_best_diversity/LPIPS_bravo)

        Args:
            results: Dict of metric name -> value from generation metrics computation.
                Keys starting with 'Diversity/' go to diversity section.
            prefix: 'test_best' or 'test_latest'.

        Returns:
            Dict for JSON export (gen_fid, gen_diversity_lpips, etc.)
        """
        exported = {}
        if self.writer is None:
            return exported

        suffix = f'_{self.modality}' if self.modality else ''

        for key, value in results.items():
            if key.startswith('Diversity/'):
                # Diversity metrics get their own section
                metric_name = key[len('Diversity/'):]
                self.writer.add_scalar(f'{prefix}_diversity/{metric_name}{suffix}', value, 0)
                exported[f'gen_diversity_{metric_name.lower()}'] = value
            else:
                self.writer.add_scalar(f'{prefix}_generation/{key}{suffix}', value, 0)
                exported[f'gen_{key.lower()}'] = value

        return exported

    def log_test_regional(
        self,
        regional_tracker: Any,
        prefix: str = 'test_best',
    ):
        """Log regional metrics with modality suffix.

        Path: {prefix}_regional_{modality} for single-modality modes,
              {prefix}_regional for multi-modality modes.

        Args:
            regional_tracker: RegionalMetricsTracker instance with accumulated metrics.
            prefix: 'test_best' or 'test_latest'.
        """
        if self.writer is None or regional_tracker is None:
            return

        # Determine if single modality based on mode
        is_single_modality = self.mode not in ('multi_modality', 'dual', 'multi')

        if is_single_modality and self.modality:
            regional_prefix = f'{prefix}_regional_{self.modality}'
        else:
            regional_prefix = f'{prefix}_regional'

        regional_tracker.log_to_tensorboard(self.writer, 0, prefix=regional_prefix)

    def log_validation_regional(
        self,
        regional_tracker: Any,
        epoch: int,
        modality_override: Optional[str] = None,
    ):
        """Log regional metrics for validation (supports per-modality tracking).

        Used for per-modality validation where each modality has its own tracker.
        For regular validation, use update_regional() + log_validation() instead.

        Path: regional_{modality} where modality comes from modality_override or self.modality.

        Args:
            regional_tracker: RegionalMetricsTracker instance with accumulated metrics.
            epoch: Current epoch number.
            modality_override: Optional modality name to use in prefix (for per-modality validation).
                If None, uses self.modality.
        """
        if self.writer is None or regional_tracker is None:
            return

        modality = modality_override or self.modality
        if modality:
            regional_prefix = f'regional_{modality}'
        else:
            regional_prefix = 'regional'

        regional_tracker.log_to_tensorboard(self.writer, epoch, prefix=regional_prefix)

    def log_test_timesteps(
        self,
        timestep_bins: Dict[str, float],
        prefix: str = 'test_best',
    ):
        """Log timestep bin losses.

        Path: {prefix}_timestep/{bin_name}{suffix}

        Args:
            timestep_bins: Dict mapping bin names (e.g., '0.0-0.1') to avg loss values.
            prefix: 'test_best' or 'test_latest'.
        """
        if self.writer is None or not timestep_bins:
            return

        suffix = f'_{self.modality}' if self.modality else ''

        for bin_name, loss in timestep_bins.items():
            self.writer.add_scalar(f'{prefix}_timestep/{bin_name}{suffix}', loss, 0)

    def log_per_channel_validation(
        self,
        channel_metrics: Dict[str, Dict[str, float]],
        epoch: int,
    ):
        """Log per-channel validation (dual/multi modes).

        Args:
            channel_metrics: Dict mapping channel names to metric dicts.
                e.g., {'t1_pre': {'psnr': 30.0, 'msssim': 0.95, 'lpips': 0.1, 'count': 10}, ...}
        Paths: Validation/PSNR_t1_pre, Validation/MS-SSIM_t1_pre, etc.
        """
        if self.writer is None:
            return

        for channel_key, channel_data in channel_metrics.items():
            count = channel_data.get('count', 0)
            if count > 0:
                suffix = f'_{channel_key}'
                if 'psnr' in channel_data:
                    avg_psnr = channel_data['psnr'] / count
                    self.writer.add_scalar(f'Validation/PSNR{suffix}', avg_psnr, epoch)
                if 'msssim' in channel_data:
                    avg_msssim = channel_data['msssim'] / count
                    self.writer.add_scalar(f'Validation/MS-SSIM{suffix}', avg_msssim, epoch)
                if 'lpips' in channel_data and channel_data.get('lpips', 0) > 0:
                    avg_lpips = channel_data['lpips'] / count
                    self.writer.add_scalar(f'Validation/LPIPS{suffix}', avg_lpips, epoch)

    def log_per_modality_validation(
        self,
        metrics: Dict[str, float],
        modality: str,
        epoch: int,
    ):
        """Log per-modality validation.

        Args:
            metrics: Dict of metric name -> value.
                Image quality: {'psnr': 30.0, 'msssim': 0.95, 'lpips': 0.1, 'msssim_3d': 0.92}
                Segmentation: {'dice': 0.85, 'iou': 0.75}
            modality: Modality name (e.g., 'bravo', 't1_pre').
            epoch: Current epoch number.
        Paths: Validation/PSNR_bravo, Validation/MS-SSIM-3D_bravo, Validation/Dice_bravo, etc.
        """
        if self.writer is None:
            return

        suffix = f'_{modality}'

        # Image quality metrics
        if 'psnr' in metrics and metrics['psnr'] is not None:
            self.writer.add_scalar(f'Validation/PSNR{suffix}', metrics['psnr'], epoch)
        if 'msssim' in metrics and metrics['msssim'] is not None:
            self.writer.add_scalar(f'Validation/MS-SSIM{suffix}', metrics['msssim'], epoch)
        if 'lpips' in metrics and metrics['lpips'] is not None:
            self.writer.add_scalar(f'Validation/LPIPS{suffix}', metrics['lpips'], epoch)
        if 'msssim_3d' in metrics and metrics['msssim_3d'] is not None:
            self.writer.add_scalar(f'Validation/MS-SSIM-3D{suffix}', metrics['msssim_3d'], epoch)

        # Segmentation metrics
        if 'dice' in metrics and metrics['dice'] is not None:
            self.writer.add_scalar(f'Validation/Dice{suffix}', metrics['dice'], epoch)
        if 'iou' in metrics and metrics['iou'] is not None:
            self.writer.add_scalar(f'Validation/IoU{suffix}', metrics['iou'], epoch)

    def log_regularization_loss(
        self,
        loss_type: str,
        weighted_loss: float,
        epoch: int,
        unweighted_loss: Optional[float] = None,
    ):
        """Log regularization losses (KL for VAE, VQ for VQVAE).

        Provides consistent TensorBoard paths for regularization losses
        across all compression trainers.

        Args:
            loss_type: Type of regularization loss ('KL' or 'VQ').
            weighted_loss: The weighted loss value (as used in total loss).
            epoch: Current epoch number.
            unweighted_loss: Optional unweighted loss value for monitoring.

        Paths:
            - Loss/{loss_type}_val (e.g., Loss/KL_val, Loss/VQ_val)
            - Loss/{loss_type}_unweighted_val (if unweighted provided)
        """
        if self.writer is None:
            return

        self.writer.add_scalar(f'Loss/{loss_type}_val', weighted_loss, epoch)

        if unweighted_loss is not None:
            self.writer.add_scalar(f'Loss/{loss_type}_unweighted_val', unweighted_loss, epoch)

    def log_codebook_metrics(
        self,
        codebook_tracker: Any,
        epoch: int,
        prefix: str = 'Codebook',
    ) -> Dict[str, float]:
        """Log codebook metrics from external tracker (VQVAE).

        Delegates to the codebook_tracker's log_to_tensorboard() method
        while ensuring the logging goes through a unified writer reference.

        Args:
            codebook_tracker: CodebookTracker instance with accumulated metrics.
            epoch: Current epoch number.
            prefix: TensorBoard prefix for codebook metrics (default: 'Codebook').

        Returns:
            Dict with codebook metrics (perplexity, utilization, etc.)
            for downstream use (e.g., adding to returned metrics dict).

        Paths:
            - {prefix}/perplexity
            - {prefix}/utilization
            - {prefix}/active_codes
        """
        if self.writer is None or codebook_tracker is None:
            return {}

        return codebook_tracker.log_to_tensorboard(self.writer, epoch, prefix=prefix)

    def update_validation_batch(
        self,
        psnr: float,
        msssim: float,
        lpips: Optional[float] = None,
        msssim_3d: Optional[float] = None,
        dice: Optional[float] = None,
        iou: Optional[float] = None,
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
        self._val_psnr_sum = psnr
        self._val_psnr_count = 1
        self._val_msssim_sum = msssim
        self._val_msssim_count = 1

        if lpips is not None:
            self._val_lpips_sum = lpips
            self._val_lpips_count = 1
        if msssim_3d is not None:
            self._val_msssim_3d_sum = msssim_3d
            self._val_msssim_3d_count = 1
        if dice is not None:
            self._val_dice_sum = dice
            self._val_dice_count = 1
        if iou is not None:
            self._val_iou_sum = iou
            self._val_iou_count = 1

    def log_timestep_region_heatmap(self, epoch: int):
        """Log 2D heatmap of loss by timestep bin and region.

        Creates a visualization showing how loss varies across timesteps
        for tumor vs background regions.

        Args:
            epoch: Current epoch number.
        """
        if self.writer is None:
            return

        # Check if we have any data
        if not any(self._tr_tumor_count) and not any(self._tr_bg_count):
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        heatmap_data = np.zeros((self.num_timestep_bins, 2))
        labels_timestep = []

        for i in range(self.num_timestep_bins):
            bin_start = i / self.num_timestep_bins
            bin_end = (i + 1) / self.num_timestep_bins
            labels_timestep.append(f'{bin_start:.1f}-{bin_end:.1f}')

            # Tumor column
            if self._tr_tumor_count[i] > 0:
                heatmap_data[i, 0] = self._tr_tumor_sum[i] / self._tr_tumor_count[i]

            # Background column
            if self._tr_bg_count[i] > 0:
                heatmap_data[i, 1] = self._tr_bg_sum[i] / self._tr_bg_count[i]

        fig, ax = plt.subplots(figsize=(6, 10))
        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Tumor', 'Background'])
        ax.set_yticks(range(self.num_timestep_bins))
        ax.set_yticklabels(labels_timestep)
        ax.set_xlabel('Region')
        ax.set_ylabel('Timestep Range')
        ax.set_title(f'Loss by Timestep & Region (Epoch {epoch})')
        plt.colorbar(im, ax=ax, label='MSE Loss')

        # Add text annotations
        for i in range(self.num_timestep_bins):
            for j in range(2):
                ax.text(j, i, f'{heatmap_data[i, j]:.4f}',
                        ha='center', va='center', color='black', fontsize=8)

        plt.tight_layout()
        self.writer.add_figure('loss/timestep_region_heatmap', fig, epoch)
        plt.close(fig)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def get_validation_metrics(self) -> Dict[str, float]:
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

    def get_training_losses(self) -> Dict[str, float]:
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
        """Record current epoch data to history for JSON export.

        Call this BEFORE reset_validation() to capture the epoch's data.

        Args:
            epoch: Current epoch number.
        """
        # Regional history
        if self._regional_tracker is not None:
            metrics = self._regional_tracker.compute()
            if metrics:
                self._regional_history[str(epoch)] = {
                    'tumor': metrics.get('tumor', 0),
                    'background': metrics.get('background', 0),
                    'tumor_bg_ratio': metrics.get('ratio', 0),
                    'by_size': {
                        'tiny': metrics.get('tumor_size_tiny', 0),
                        'small': metrics.get('tumor_size_small', 0),
                        'medium': metrics.get('tumor_size_medium', 0),
                        'large': metrics.get('tumor_size_large', 0),
                    }
                }

        # Timestep history
        epoch_timesteps = {}
        for i in range(self.num_timestep_bins):
            if self._val_timesteps['counts'][i] > 0:
                bin_start = i / self.num_timestep_bins
                bin_end = (i + 1) / self.num_timestep_bins
                bin_name = f'{bin_start:.1f}-{bin_end:.1f}'
                epoch_timesteps[bin_name] = self._val_timesteps['sums'][i] / self._val_timesteps['counts'][i]
        if epoch_timesteps:
            self._timestep_history[str(epoch)] = epoch_timesteps

        # Timestep-region history
        epoch_tr = {}
        for i in range(self.num_timestep_bins):
            bin_start = i / self.num_timestep_bins
            bin_label = f'{bin_start:.1f}'
            tumor_avg = self._tr_tumor_sum[i] / max(self._tr_tumor_count[i], 1) if self._tr_tumor_count[i] > 0 else 0.0
            bg_avg = self._tr_bg_sum[i] / max(self._tr_bg_count[i], 1) if self._tr_bg_count[i] > 0 else 0.0
            epoch_tr[bin_label] = {
                'tumor': tumor_avg,
                'background': bg_avg,
            }
        if any(self._tr_tumor_count) or any(self._tr_bg_count):
            self._timestep_region_history[str(epoch)] = epoch_tr

    def save_json_histories(self, save_dir: str):
        """Save all history data to JSON files.

        Args:
            save_dir: Directory to save JSON files to.
        """
        import json
        import os

        if self._regional_history:
            filepath = os.path.join(save_dir, 'regional_losses.json')
            with open(filepath, 'w') as f:
                json.dump(self._regional_history, f, indent=2)

        if self._timestep_history:
            filepath = os.path.join(save_dir, 'timestep_losses.json')
            with open(filepath, 'w') as f:
                json.dump(self._timestep_history, f, indent=2)

        if self._timestep_region_history:
            filepath = os.path.join(save_dir, 'timestep_region_losses.json')
            with open(filepath, 'w') as f:
                json.dump(self._timestep_region_history, f, indent=2)

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def log_reconstruction_figure(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        epoch: int,
        mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        tag: str = 'Figures/reconstruction',
        max_samples: int = 8,
        metrics: Optional[Dict[str, float]] = None,
        save_path: Optional[str] = None,
    ):
        """Log reconstruction comparison figure to TensorBoard.

        Args:
            original: Ground truth tensor [B, C, (D), H, W].
            reconstructed: Reconstructed tensor [B, C, (D), H, W].
            epoch: Current epoch number.
            mask: Optional segmentation mask for contour overlay.
            timesteps: Optional timesteps for column titles (diffusion).
            tag: TensorBoard tag for the figure.
            max_samples: Maximum number of samples to display.
            metrics: Optional dict of metrics to show in subtitle.
            save_path: Optional path to save figure as PNG file.
        """
        if self.writer is None and save_path is None:
            return

        from .figures import create_reconstruction_figure
        import matplotlib.pyplot as plt

        # Handle 3D volumes - extract multiple slices for visualization
        if self.spatial_dims == 3:
            original = self._extract_multiple_slices(original, num_slices=max_samples)
            reconstructed = self._extract_multiple_slices(reconstructed, num_slices=max_samples)
            if mask is not None:
                mask = self._extract_multiple_slices(mask, num_slices=max_samples)
            # For 3D, slices are from same volume - show timestep in metrics instead of per-column
            if timesteps is not None and len(timesteps) > 0:
                t_val = timesteps[0].item() if hasattr(timesteps[0], 'item') else timesteps[0]
                metrics = metrics.copy() if metrics else {}
                metrics['t'] = t_val
            timesteps = None  # Don't show per-column (all slices have same timestep)

        fig = create_reconstruction_figure(
            original=original,
            generated=reconstructed,
            timesteps=timesteps,
            mask=mask,
            max_samples=max_samples,
            metrics=metrics,
        )

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_figure(tag, fig, epoch)

        # Save to file if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.close(fig)

    def log_worst_batch(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        loss: float,
        epoch: int,
        phase: str = 'train',
        mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        tag_prefix: Optional[str] = None,
        save_path: Optional[str] = None,
        display_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log worst batch visualization to TensorBoard.

        Args:
            original: Ground truth tensor.
            reconstructed: Reconstructed tensor.
            loss: Loss value for this batch.
            epoch: Current epoch number.
            phase: 'train' or 'val' (used if tag_prefix not provided).
            mask: Optional segmentation mask.
            timesteps: Optional timesteps (diffusion).
            tag_prefix: Optional custom prefix (e.g., 'Test_best'). If provided,
                tag becomes '{tag_prefix}/worst_batch' instead of 'worst_batch/{phase}'.
            save_path: Optional path to save figure as PNG file.
            display_metrics: Optional metrics dict to display (e.g., {'MS-SSIM': 0.95}).
                If not provided, uses {'loss': loss}.
        """
        if self.writer is None and save_path is None:
            return

        # Determine tag - use TensorBoard standard format: Validation/worst_batch
        if tag_prefix is not None:
            tag = f'{tag_prefix}/worst_batch'
        else:
            phase_cap = phase.capitalize()  # val -> Validation, train -> Train
            tag = f'{phase_cap}/worst_batch'

        # Determine metrics to display
        metrics = display_metrics if display_metrics is not None else {'loss': loss}

        self.log_reconstruction_figure(
            original=original,
            reconstructed=reconstructed,
            epoch=epoch,
            mask=mask,
            timesteps=timesteps,
            tag=tag,
            metrics=metrics,
            save_path=save_path,
        )

    def log_denoising_trajectory(
        self,
        trajectory: list,
        epoch: int,
        tag: str = 'denoising_trajectory',
    ):
        """Log denoising step visualization to TensorBoard.

        Args:
            trajectory: List of tensors showing denoising progress.
            epoch: Current epoch number.
            tag: TensorBoard tag prefix.
        """
        if self.writer is None or not trajectory:
            return

        import matplotlib.pyplot as plt
        import numpy as np

        # Stack trajectory into single tensor
        steps = len(trajectory)
        sample = trajectory[0]

        # Handle 3D - take center slice
        if self.spatial_dims == 3 and sample.dim() == 5:
            trajectory = [self._extract_center_slice(t) for t in trajectory]
            sample = trajectory[0]

        # Create figure showing progression
        fig, axes = plt.subplots(1, steps, figsize=(2.5 * steps, 3))
        if steps == 1:
            axes = [axes]

        for i, step_tensor in enumerate(trajectory):
            # Take first sample, first channel
            if isinstance(step_tensor, torch.Tensor):
                img = step_tensor[0, 0].cpu().float().numpy()
            else:
                img = step_tensor[0, 0]
            axes[i].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f'Step {i}', fontsize=8)
            axes[i].axis('off')

        fig.tight_layout()
        self.writer.add_figure(f'{tag}/progression', fig, epoch)
        plt.close(fig)

    def log_generated_samples(
        self,
        samples: torch.Tensor,
        epoch: int,
        tag: str = 'Generated_Samples',
        nrow: int = 4,
        num_slices: int = 8,
    ):
        """Log generated samples grid to TensorBoard.

        For 3D volumes, shows multiple evenly-spaced slices per sample.

        Args:
            samples: Generated samples [B, C, (D), H, W].
            epoch: Current epoch number.
            tag: TensorBoard tag.
            nrow: Number of images per row in grid (2D only).
            num_slices: Number of slices to show for 3D volumes.
        """
        if self.writer is None:
            return

        from torchvision.utils import make_grid

        # Handle 3D - show multiple slices per sample
        if self.spatial_dims == 3 and samples.dim() == 5:
            self._log_generated_samples_3d(samples, epoch, tag, num_slices)
            return

        # 2D: simple grid
        samples = torch.clamp(samples.float(), 0, 1)
        grid = make_grid(samples, nrow=nrow, normalize=False, padding=2)
        self.writer.add_image(tag, grid, epoch)

    def _log_generated_samples_3d(
        self,
        samples: torch.Tensor,
        epoch: int,
        tag: str,
        num_slices: int = 8,
    ):
        """Log 3D generated samples with multiple slices per sample.

        Creates a figure with one row per sample, showing evenly-spaced
        depth slices across columns.

        Args:
            samples: 5D tensor [B, C, D, H, W].
            epoch: Current epoch number.
            tag: TensorBoard tag.
            num_slices: Number of slices per sample.
        """
        import matplotlib.pyplot as plt

        B, C, D, H, W = samples.shape
        samples = torch.clamp(samples.float(), 0, 1).cpu()

        # Calculate slice indices (evenly spaced, avoiding edges)
        margin = max(1, D // (num_slices + 2))
        indices = torch.linspace(margin, D - margin - 1, num_slices).long().tolist()

        # Create figure: rows = samples, cols = slices
        fig, axes = plt.subplots(B, num_slices, figsize=(num_slices * 2, B * 2))

        # Handle single sample case
        if B == 1:
            axes = axes.reshape(1, -1)

        for b in range(B):
            for s, slice_idx in enumerate(indices):
                ax = axes[b, s]
                # Get slice and handle channels
                slice_img = samples[b, :, slice_idx, :, :]
                if C == 1:
                    ax.imshow(slice_img[0].numpy(), cmap='gray', vmin=0, vmax=1)
                else:
                    # Multi-channel: show first channel or RGB if 3
                    ax.imshow(slice_img[0].numpy(), cmap='gray', vmin=0, vmax=1)

                ax.set_xticks([])
                ax.set_yticks([])

                # Label top row with slice indices
                if b == 0:
                    ax.set_title(f'z={slice_idx}', fontsize=8)

                # Label left column with sample number
                if s == 0:
                    ax.set_ylabel(f'Sample {b+1}', fontsize=8)

        plt.tight_layout()
        self.writer.add_figure(tag, fig, epoch)
        plt.close(fig)

    def log_test_figure(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        prefix: str = 'test_best',
        mask: Optional[torch.Tensor] = None,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Log test evaluation figure to TensorBoard.

        Args:
            original: Ground truth tensor.
            reconstructed: Reconstructed tensor.
            prefix: 'test_best' or 'test_latest'.
            mask: Optional segmentation mask.
            metrics: Optional dict of metrics for subtitle.
        """
        if self.writer is None:
            return

        self.log_reconstruction_figure(
            original=original,
            reconstructed=reconstructed,
            epoch=0,  # Test uses step 0
            mask=mask,
            tag=f'{prefix}/reconstruction',
            metrics=metrics,
        )

    def _extract_center_slice(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract center slice from 3D volume.

        Args:
            tensor: 5D tensor [B, C, D, H, W].

        Returns:
            4D tensor [B, C, H, W] with center depth slice.
        """
        if tensor.dim() == 5:
            depth = tensor.shape[2]
            center_idx = depth // 2
            return tensor[:, :, center_idx, :, :]
        return tensor

    def _extract_multiple_slices(
        self,
        tensor: torch.Tensor,
        num_slices: int = 8,
    ) -> torch.Tensor:
        """Extract multiple evenly-spaced slices from 3D volume.

        For 3D worst_batch visualization, extracts N slices from the depth
        dimension and returns them as a batch of 2D images.

        Args:
            tensor: 5D tensor [B, C, D, H, W] (typically B=1 for 3D).
            num_slices: Number of slices to extract.

        Returns:
            4D tensor [N, C, H, W] with N evenly-spaced slices.
        """
        if tensor.dim() != 5:
            return tensor

        B, C, D, H, W = tensor.shape
        # Calculate evenly spaced indices (avoid edges)
        margin = D // (num_slices + 1)
        indices = [margin + i * (D - 2 * margin) // (num_slices - 1) for i in range(num_slices)]
        indices = [min(max(0, idx), D - 1) for idx in indices]  # Clamp to valid range

        # Extract slices from first volume (3D typically has batch_size=1)
        slices = [tensor[0, :, idx, :, :] for idx in indices]
        return torch.stack(slices, dim=0)  # [num_slices, C, H, W]

    # =========================================================================
    # Tracker Integration Methods
    # =========================================================================

    def log_flops_from_tracker(self, flops_tracker: Any, epoch: int) -> None:
        """Log FLOPs metrics from FLOPsTracker.

        Centralizes FLOPs logging through UnifiedMetrics instead of
        direct writer.add_scalar calls in trainers.

        Args:
            flops_tracker: FLOPsTracker instance.
            epoch: Current epoch number.
        """
        if self.writer is None or flops_tracker is None:
            return
        if not getattr(flops_tracker, '_measured', False):
            return
        if flops_tracker.forward_flops == 0:
            return

        completed_epochs = epoch + 1
        self.writer.add_scalar('FLOPs/TFLOPs_epoch', flops_tracker.get_tflops_epoch(), epoch)
        self.writer.add_scalar('FLOPs/TFLOPs_total', flops_tracker.get_tflops_total(completed_epochs), epoch)
        self.writer.add_scalar('FLOPs/TFLOPs_bs1', flops_tracker.get_tflops_bs1(), epoch)

    def log_grad_norm_from_tracker(
        self,
        grad_tracker: Any,
        epoch: int,
        prefix: str = 'training/grad_norm',
    ) -> None:
        """Log gradient norm stats from GradientNormTracker.

        Centralizes gradient norm logging through UnifiedMetrics instead of
        direct writer.add_scalar calls in trainers.

        Args:
            grad_tracker: GradientNormTracker instance.
            epoch: Current epoch number.
            prefix: TensorBoard prefix (default: 'training/grad_norm').
        """
        if self.writer is None or grad_tracker is None:
            return
        if grad_tracker.count == 0:
            return

        self.writer.add_scalar(f'{prefix}_avg', grad_tracker.get_avg(), epoch)
        self.writer.add_scalar(f'{prefix}_max', grad_tracker.get_max(), epoch)

    def log_sample_images(
        self,
        images: torch.Tensor,
        tag: str,
        epoch: int,
    ) -> None:
        """Log image grid to TensorBoard using add_images.

        Centralizes image logging through UnifiedMetrics.

        Args:
            images: Tensor of images [N, C, H, W] normalized to [0, 1].
            tag: TensorBoard tag (e.g., 'Generated_Images').
            epoch: Current epoch number.
        """
        if self.writer is None:
            return
        self.writer.add_images(tag, images, epoch)

    # =========================================================================
    # Console Logging
    # =========================================================================

    def log_console_summary(
        self,
        epoch: int,
        total_epochs: int,
        elapsed_time: float,
        time_estimator: Optional["EpochTimeEstimator"] = None,
    ):
        """Log epoch completion summary to console.

        Args:
            epoch: Current epoch number (0-indexed).
            total_epochs: Total number of epochs.
            elapsed_time: Time taken for epoch in seconds.
            time_estimator: Optional estimator for ETA calculation.
        """
        import time as time_module

        timestamp = time_module.strftime("%H:%M:%S")
        epoch_pct = ((epoch + 1) / total_epochs) * 100

        train_losses = self.get_training_losses()
        val_metrics = self.get_validation_metrics()

        # Build loss string
        loss_parts = []
        total_loss = train_losses.get('Total') or train_losses.get('MSE') or 0
        val_total = val_metrics.get('MSE') or val_metrics.get('Total') or 0
        loss_parts.append(f"Loss: {total_loss:.4f}")
        if val_total > 0:
            loss_parts[-1] += f"(v:{val_total:.4f})"

        for key in ['MSE', 'Perceptual', 'KL', 'VQ', 'BCE', 'Dice']:
            if key in train_losses and key != 'Total':
                loss_parts.append(f"{key}: {train_losses[key]:.4f}")

        # Build validation metrics string
        metric_parts = []
        if self.uses_image_quality:
            if 'MS-SSIM' in val_metrics:
                metric_parts.append(f"MS-SSIM: {val_metrics['MS-SSIM']:.3f}")
            if 'MS-SSIM-3D' in val_metrics:
                metric_parts.append(f"MS-SSIM-3D: {val_metrics['MS-SSIM-3D']:.3f}")
            if 'PSNR' in val_metrics:
                metric_parts.append(f"PSNR: {val_metrics['PSNR']:.2f}")
            if 'LPIPS' in val_metrics:
                metric_parts.append(f"LPIPS: {val_metrics['LPIPS']:.3f}")
        else:
            if 'Dice' in val_metrics:
                metric_parts.append(f"Dice: {val_metrics['Dice']:.3f}")
            if 'IoU' in val_metrics:
                metric_parts.append(f"IoU: {val_metrics['IoU']:.3f}")

        # Combine all parts
        all_parts = loss_parts + metric_parts
        all_parts.append(f"Time: {elapsed_time:.1f}s")

        # Add ETA if estimator provided
        if time_estimator is not None:
            time_estimator.update(elapsed_time)
            eta_str = time_estimator.get_eta_string()
            if eta_str:
                all_parts.append(eta_str)

        logger.info(
            f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
            + " | ".join(all_parts)
        )
