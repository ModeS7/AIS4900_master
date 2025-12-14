"""
Metrics tracking and computation for diffusion model training.

This module provides GPU-efficient metrics tracking with lazy initialization
to avoid CPU-GPU synchronization bottlenecks during training.
"""
import json
import logging
import math
import os
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim_skimage
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class MetricsTracker:
    """GPU-efficient metrics tracking with lazy initialization.

    Accumulates metrics on GPU during training steps, transferring to CPU
    only at epoch boundaries for logging. This eliminates CPU-GPU sync
    bottlenecks that would otherwise slow down training.

    Args:
        cfg: Hydra configuration object.
        device: PyTorch device for GPU accumulators.
        writer: Optional TensorBoard SummaryWriter.
        save_dir: Directory for saving JSON logs.
        is_main_process: Whether this is the main process (for logging).
        is_conditional: Whether the training mode uses conditioning (mask).
    """

    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device,
        writer: Optional[SummaryWriter],
        save_dir: str,
        is_main_process: bool = True,
        is_conditional: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.writer = writer
        self.save_dir = save_dir
        self.is_main_process = is_main_process
        self.is_conditional = is_conditional

        # Extract config
        self.image_size: int = cfg.model.image_size
        self.num_timesteps: int = cfg.strategy.num_train_timesteps
        self.strategy_name: str = cfg.strategy.name
        self.use_min_snr: bool = cfg.training.use_min_snr
        self.min_snr_gamma: float = cfg.training.min_snr_gamma

        # Logging config (with defaults for backward compatibility)
        logging_cfg = cfg.training.get('logging', {})
        self.log_grad_norm: bool = logging_cfg.get('grad_norm', True)
        self.log_timestep_losses: bool = logging_cfg.get('timestep_losses', True)
        self.log_regional_losses: bool = logging_cfg.get('regional_losses', True)
        self.log_timestep_region: bool = logging_cfg.get('timestep_region_losses', True)
        self.log_ssim: bool = logging_cfg.get('ssim', True)
        self.log_psnr: bool = logging_cfg.get('psnr', True)
        self.log_lpips: bool = logging_cfg.get('lpips', True)
        self.log_boundary_sharpness: bool = logging_cfg.get('boundary_sharpness', True)
        self.log_worst_batch: bool = logging_cfg.get('worst_batch', True)
        self.log_flops: bool = logging_cfg.get('flops', True)

        # LPIPS model (initialized lazily)
        self._lpips_model: Optional[Any] = None
        self._lpips_initialized: bool = False

        # FLOPs tracking
        self._flops_measured: bool = False
        self.forward_flops: int = 0  # FLOPs for one forward pass
        self.epoch_flops: int = 0  # Accumulated FLOPs this epoch
        self.epoch_steps: int = 0  # Steps this epoch
        self.total_flops: int = 0  # Total FLOPs across all epochs

        # Timestep loss tracking
        self.num_timestep_bins: int = 10
        self._timestep_accum_initialized: bool = False
        self.timestep_loss_sum: Optional[torch.Tensor] = None
        self.timestep_loss_count: Optional[torch.Tensor] = None

        # Regional loss tracking (tumor vs background)
        self._regional_accum_initialized: bool = False
        self.tumor_loss_sum: Optional[torch.Tensor] = None
        self.tumor_loss_count: Optional[torch.Tensor] = None
        self.bg_loss_sum: Optional[torch.Tensor] = None
        self.bg_loss_count: Optional[torch.Tensor] = None

        # Tumor size thresholds and tracking
        self.tumor_size_thresholds = self._compute_tumor_size_thresholds()
        self.tumor_size_loss_sum: Dict[str, Optional[torch.Tensor]] = {
            size: None for size in self.tumor_size_thresholds.keys()
        }
        self.tumor_size_loss_count: Dict[str, Optional[torch.Tensor]] = {
            size: None for size in self.tumor_size_thresholds.keys()
        }

        # Gradient norm tracking
        self.grad_norm_sum: Optional[torch.Tensor] = None
        self.grad_norm_max: Optional[torch.Tensor] = None
        self.grad_norm_count: int = 0

        # Worst batch tracking
        self.worst_batch_loss: float = 0.0
        self.worst_batch_data: Optional[Dict[str, Any]] = None

        # 2D timestep-region loss tracking
        self._timestep_region_accum_initialized: bool = False
        self.timestep_region_loss_sum: Optional[torch.Tensor] = None
        self.timestep_region_loss_count: Optional[torch.Tensor] = None

        # Scheduler reference (set externally for SNR computation)
        self.scheduler: Optional[Any] = None

    def set_scheduler(self, scheduler: Any) -> None:
        """Set scheduler reference for SNR weight computation."""
        self.scheduler = scheduler

    def measure_forward_flops(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> int:
        """Measure FLOPs for a single forward pass using torch.profiler.

        This should be called once during setup with a representative batch.
        The measured FLOPs are stored and used to estimate total FLOPs per step
        (forward + backward ≈ 3x forward FLOPs).

        Args:
            model: The diffusion model (UNet).
            sample_input: Sample model input tensor [B, C, H, W] (includes conditioning).
            timesteps: Sample timesteps tensor [B].

        Returns:
            FLOPs for one forward pass.
        """
        if not self.log_flops:
            return 0

        if self._flops_measured:
            return self.forward_flops

        model.eval()
        with torch.no_grad():
            from torch.profiler import profile, ProfilerActivity

            # Run profiler to count FLOPs
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_flops=True,
            ) as prof:
                _ = model(x=sample_input, timesteps=timesteps)

            # Sum all FLOPs from profiler events
            total_flops = sum(e.flops for e in prof.key_averages() if e.flops > 0)

        model.train()

        self.forward_flops = total_flops
        self._flops_measured = True

        if self.is_main_process and total_flops > 0:
            batch_size = sample_input.shape[0]
            flops_per_sample = total_flops / batch_size
            gflops_per_sample = flops_per_sample / 1e9
            logger.info(f"Model FLOPs measured: {gflops_per_sample:.2f} GFLOPs/sample "
                       f"(forward), ~{gflops_per_sample * 3:.2f} GFLOPs/sample (train step)")

        return total_flops

    def _compute_tumor_size_thresholds(self) -> Dict[str, Tuple[float, float]]:
        """Compute tumor size thresholds based on image resolution.

        Clinical definitions (diameter):
            - tiny:   <10mm  (often non-measurable per RANO-BM)
            - small:  10-20mm (small metastases, SRS alone)
            - medium: 20-30mm (SRS candidates)
            - large:  >30mm  (often surgical)

        Returns:
            Dictionary mapping size names to (low, high) area percentage ranges.
        """
        fov_mm = self.cfg.paths.get('fov_mm', 240.0)
        mm_per_pixel = fov_mm / self.image_size
        total_pixels = self.image_size ** 2

        diameter_thresholds = {
            'tiny': (0, 10),
            'small': (10, 20),
            'medium': (20, 30),
            'large': (30, 150),
        }

        thresholds = {}
        for size_name, (d_low, d_high) in diameter_thresholds.items():
            d_low_px = d_low / mm_per_pixel
            d_high_px = d_high / mm_per_pixel
            area_low = math.pi * (d_low_px / 2) ** 2
            area_high = math.pi * (d_high_px / 2) ** 2
            pct_low = area_low / total_pixels
            pct_high = area_high / total_pixels
            thresholds[size_name] = (pct_low, pct_high)

        if self.is_main_process:
            logger.info(f"Tumor size thresholds for {self.image_size}px ({mm_per_pixel:.2f} mm/px):")
            for name, (low, high) in thresholds.items():
                logger.info(f"  {name}: {low*100:.3f}% - {high*100:.3f}%")

        return thresholds

    def _init_regional_accumulators(self) -> None:
        """Initialize GPU accumulators for regional loss tracking."""
        self.tumor_loss_sum = torch.tensor(0.0, device=self.device)
        self.tumor_loss_count = torch.tensor(0, device=self.device, dtype=torch.long)
        self.bg_loss_sum = torch.tensor(0.0, device=self.device)
        self.bg_loss_count = torch.tensor(0, device=self.device, dtype=torch.long)

        for size in self.tumor_size_thresholds.keys():
            self.tumor_size_loss_sum[size] = torch.tensor(0.0, device=self.device)
            self.tumor_size_loss_count[size] = torch.tensor(0, device=self.device, dtype=torch.long)

        self._regional_accum_initialized = True

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
            t_normalized = timesteps.float() / self.num_timesteps
            snr = (1.0 - t_normalized) / (t_normalized + 1e-8)

        snr_clipped = torch.clamp(snr, max=self.min_snr_gamma)
        weights = snr_clipped / (snr + 1e-8)

        return weights

    def compute_regional_losses(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[float, float]:
        """Compute MSE loss separately for tumor and background regions.

        Args:
            predicted: Predicted clean image [B, C, H, W].
            target: Ground truth image [B, C, H, W].
            mask: Binary segmentation mask [B, 1, H, W].

        Returns:
            Tuple of (tumor_loss, background_loss).
        """
        if predicted.shape[1] > 1:
            mask = mask.expand(-1, predicted.shape[1], -1, -1)

        sq_error = (predicted - target) ** 2

        tumor_mask = (mask > 0.5).float()
        tumor_pixels = tumor_mask.sum() + 1e-8
        tumor_loss = (sq_error * tumor_mask).sum() / tumor_pixels

        bg_mask = 1.0 - tumor_mask
        bg_pixels = bg_mask.sum() + 1e-8
        bg_loss = (sq_error * bg_mask).sum() / bg_pixels

        return tumor_loss.item(), bg_loss.item()

    def track_step(
        self,
        timesteps: torch.Tensor,
        predicted_clean: Union[torch.Tensor, Dict[str, torch.Tensor]],
        images: Union[torch.Tensor, Dict[str, torch.Tensor]],
        mask: Optional[torch.Tensor],
        grad_norm: Optional[torch.Tensor],
        loss: float,
    ) -> None:
        """Track metrics for a single training step.

        All accumulation happens on GPU to avoid sync bottlenecks.

        Args:
            timesteps: Diffusion timesteps for batch [B].
            predicted_clean: Model's predicted clean images.
            images: Ground truth images.
            mask: Segmentation mask (None for seg mode).
            grad_norm: Gradient norm from clipping.
            loss: Total loss value.
        """
        # Track gradient norm
        if self.log_grad_norm and grad_norm is not None:
            if self.grad_norm_sum is None:
                self.grad_norm_sum = torch.tensor(0.0, device=self.device)
                self.grad_norm_max = torch.tensor(0.0, device=self.device)
            self.grad_norm_sum = self.grad_norm_sum + grad_norm
            self.grad_norm_max = torch.maximum(self.grad_norm_max, grad_norm)
            self.grad_norm_count += 1

        # Track timestep losses
        if self.log_timestep_losses:
            self._track_timestep_loss_batch(timesteps, predicted_clean, images)

        # Track regional losses for conditional modes
        if self.is_conditional and mask is not None:
            if self.log_regional_losses:
                self._track_regional_losses(predicted_clean, images, mask)
            if self.log_timestep_region:
                self._track_timestep_region_loss(timesteps, predicted_clean, images, mask)

        # Track worst batch (keep on GPU, defer .cpu() to get_worst_batch_data)
        if self.log_worst_batch and loss > self.worst_batch_loss:
            self.worst_batch_loss = loss
            self.worst_batch_data = {
                'images': images.detach().clone() if not isinstance(images, dict) else {k: v.detach().clone() for k, v in images.items()},
                'mask': mask.detach().clone() if mask is not None else None,
                'predicted': predicted_clean.detach().clone() if not isinstance(predicted_clean, dict) else {k: v.detach().clone() for k, v in predicted_clean.items()},
                'timesteps': timesteps.detach().clone(),
                'loss': loss,
            }

        # Track FLOPs (no GPU sync - just integer arithmetic)
        if self.log_flops and self._flops_measured:
            # Training step ≈ 3x forward FLOPs (forward + backward)
            # forward_flops was measured for one batch, so just multiply by 3
            self.epoch_flops += self.forward_flops * 3
            self.epoch_steps += 1

    def _track_timestep_loss_batch(
        self,
        timesteps: torch.Tensor,
        predicted_clean: Union[torch.Tensor, Dict[str, torch.Tensor]],
        images: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> None:
        """Vectorized timestep loss tracking - no CPU sync."""
        if not self._timestep_accum_initialized:
            self.timestep_loss_sum = torch.zeros(self.num_timestep_bins, device=self.device)
            self.timestep_loss_count = torch.zeros(self.num_timestep_bins, device=self.device, dtype=torch.long)
            self._timestep_accum_initialized = True

        if isinstance(predicted_clean, dict):
            pred = torch.cat(list(predicted_clean.values()), dim=1)
            img = torch.cat(list(images.values()), dim=1)
        else:
            pred, img = predicted_clean, images

        mse_per_sample = ((pred - img) ** 2).mean(dim=(1, 2, 3))

        bin_size = self.num_timesteps // self.num_timestep_bins
        bin_indices = (timesteps // bin_size).clamp(max=self.num_timestep_bins - 1).long()

        self.timestep_loss_sum.scatter_add_(0, bin_indices, mse_per_sample)
        ones = torch.ones_like(bin_indices)
        self.timestep_loss_count.scatter_add_(0, bin_indices, ones)

    def _track_regional_losses(
        self,
        predicted_clean: Union[torch.Tensor, Dict[str, torch.Tensor]],
        images: Union[torch.Tensor, Dict[str, torch.Tensor]],
        mask: torch.Tensor
    ) -> None:
        """Vectorized tracking of losses by region and tumor size."""
        if not self._regional_accum_initialized:
            self._init_regional_accumulators()

        if isinstance(predicted_clean, dict):
            pred = torch.cat(list(predicted_clean.values()), dim=1)
            img = torch.cat(list(images.values()), dim=1)
        else:
            pred, img = predicted_clean, images

        sq_error = (pred - img) ** 2

        tumor_mask = (mask > 0.5).float()
        if sq_error.shape[1] > 1:
            tumor_mask_expanded = tumor_mask.expand_as(sq_error)
        else:
            tumor_mask_expanded = tumor_mask
        bg_mask_expanded = 1.0 - tumor_mask_expanded

        tumor_pixels = tumor_mask.sum(dim=(1, 2, 3))
        total_pixels = mask.shape[2] * mask.shape[3]

        tumor_pixels_safe = tumor_pixels.clamp(min=1)
        bg_pixels = total_pixels - tumor_pixels
        bg_pixels_safe = bg_pixels.clamp(min=1)

        tumor_loss_per_sample = (sq_error * tumor_mask_expanded).sum(dim=(1, 2, 3)) / tumor_pixels_safe
        bg_loss_per_sample = (sq_error * bg_mask_expanded).sum(dim=(1, 2, 3)) / bg_pixels_safe

        has_tumor_float = (tumor_pixels > 10).float()
        num_valid = has_tumor_float.sum().long()

        self.tumor_loss_sum += (tumor_loss_per_sample * has_tumor_float).sum()
        self.tumor_loss_count += num_valid
        self.bg_loss_sum += (bg_loss_per_sample * has_tumor_float).sum()
        self.bg_loss_count += num_valid

        tumor_ratios = tumor_pixels / total_pixels
        for size_name, (low, high) in self.tumor_size_thresholds.items():
            size_mask = has_tumor_float * ((tumor_ratios >= low) & (tumor_ratios < high)).float()
            self.tumor_size_loss_sum[size_name] += (tumor_loss_per_sample * size_mask).sum()
            self.tumor_size_loss_count[size_name] += size_mask.sum().long()

    def _track_timestep_region_loss(
        self,
        timesteps: torch.Tensor,
        predicted_clean: Union[torch.Tensor, Dict[str, torch.Tensor]],
        images: Union[torch.Tensor, Dict[str, torch.Tensor]],
        mask: torch.Tensor
    ) -> None:
        """Vectorized 2D loss tracking by timestep bin AND region.

        Uses separate 1D accumulators to avoid scatter_add_ on sliced 2D tensors
        which causes sync overhead.
        """
        if not self._timestep_region_accum_initialized:
            # Use separate 1D accumulators for tumor and background
            self._tr_tumor_sum = torch.zeros(self.num_timestep_bins, device=self.device)
            self._tr_tumor_count = torch.zeros(self.num_timestep_bins, device=self.device, dtype=torch.long)
            self._tr_bg_sum = torch.zeros(self.num_timestep_bins, device=self.device)
            self._tr_bg_count = torch.zeros(self.num_timestep_bins, device=self.device, dtype=torch.long)
            # Keep 2D tensors for logging (assembled from 1D at log time)
            self.timestep_region_loss_sum = None
            self.timestep_region_loss_count = None
            self._timestep_region_accum_initialized = True

        if isinstance(predicted_clean, dict):
            pred = torch.cat(list(predicted_clean.values()), dim=1)
            img = torch.cat(list(images.values()), dim=1)
        else:
            pred, img = predicted_clean, images

        sq_error = (pred - img) ** 2
        tumor_mask = (mask > 0.5).float()
        if sq_error.shape[1] > 1:
            tumor_mask_expanded = tumor_mask.expand_as(sq_error)
        else:
            tumor_mask_expanded = tumor_mask
        bg_mask_expanded = 1.0 - tumor_mask_expanded

        tumor_pixels = tumor_mask.sum(dim=(1, 2, 3))
        total_pixels = mask.shape[2] * mask.shape[3]

        tumor_pixels_safe = tumor_pixels.clamp(min=1)
        bg_pixels_safe = (total_pixels - tumor_pixels).clamp(min=1)

        tumor_loss = (sq_error * tumor_mask_expanded).sum(dim=(1, 2, 3)) / tumor_pixels_safe
        bg_loss = (sq_error * bg_mask_expanded).sum(dim=(1, 2, 3)) / bg_pixels_safe

        has_tumor = (tumor_pixels > 10).float()

        bin_size = self.num_timesteps // self.num_timestep_bins
        bin_indices = (timesteps // bin_size).clamp(max=self.num_timestep_bins - 1).long()

        masked_tumor_loss = tumor_loss * has_tumor
        masked_bg_loss = bg_loss * has_tumor
        has_tumor_long = has_tumor.long()

        # Use 1D scatter_add_ (much faster than sliced 2D)
        self._tr_tumor_sum.scatter_add_(0, bin_indices, masked_tumor_loss)
        self._tr_tumor_count.scatter_add_(0, bin_indices, has_tumor_long)
        self._tr_bg_sum.scatter_add_(0, bin_indices, masked_bg_loss)
        self._tr_bg_count.scatter_add_(0, bin_indices, has_tumor_long)

    def log_epoch(self, epoch: int, log_all: bool = False) -> None:
        """Log all accumulated metrics for the epoch.

        Args:
            epoch: Current epoch number.
            log_all: If True, log all metrics. If False, only log grad norms.
        """
        if not self.is_main_process:
            return

        # Always log grad norms and FLOPs (lightweight)
        if self.log_grad_norm:
            self._log_grad_norms(epoch)
        if self.log_flops:
            self._log_flops(epoch)

        # Other metrics only at val_interval
        if log_all:
            if self.log_timestep_losses:
                self._log_timestep_losses(epoch)
            if self.log_regional_losses:
                self._log_regional_losses(epoch)
            if self.log_timestep_region:
                self._log_timestep_region_losses(epoch)

    def _log_grad_norms(self, epoch: int) -> None:
        """Log gradient norm statistics to TensorBoard."""
        if self.grad_norm_count == 0 or self.writer is None or self.grad_norm_sum is None:
            return

        avg_grad_norm = (self.grad_norm_sum / self.grad_norm_count).item()
        max_grad_norm = self.grad_norm_max.item()

        self.writer.add_scalar('training/grad_norm_avg', avg_grad_norm, epoch)
        self.writer.add_scalar('training/grad_norm_max', max_grad_norm, epoch)

        self.grad_norm_sum = torch.tensor(0.0, device=self.device)
        self.grad_norm_max = torch.tensor(0.0, device=self.device)
        self.grad_norm_count = 0

    def _log_flops(self, epoch: int) -> None:
        """Log FLOPs statistics to TensorBoard and update totals."""
        if self.epoch_steps == 0 or not self._flops_measured:
            return

        # Calculate metrics
        tflops_epoch = self.epoch_flops / 1e12
        gflops_per_step = (self.epoch_flops / self.epoch_steps) / 1e9
        self.total_flops += self.epoch_flops

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('compute/TFLOPs_epoch', tflops_epoch, epoch)
            self.writer.add_scalar('compute/GFLOPs_per_step', gflops_per_step, epoch)
            self.writer.add_scalar('compute/TFLOPs_total', self.total_flops / 1e12, epoch)

        # Log to console on first epoch
        if epoch == 0:
            logger.info(f"Epoch compute: {tflops_epoch:.2f} TFLOPs ({gflops_per_step:.1f} GFLOPs/step)")

        # Reset epoch counters
        self.epoch_flops = 0
        self.epoch_steps = 0

    def _log_timestep_losses(self, epoch: int) -> None:
        """Save timestep loss distribution to JSON file."""
        if not self._timestep_accum_initialized or self.timestep_loss_sum is None:
            return

        counts = self.timestep_loss_count.cpu()
        total_count = counts.sum().item()
        if total_count == 0:
            return

        sums = self.timestep_loss_sum.cpu()
        bin_size = self.num_timesteps // self.num_timestep_bins

        epoch_data = {}
        for bin_idx in range(self.num_timestep_bins):
            bin_start = bin_idx * bin_size
            bin_end = (bin_idx + 1) * bin_size - 1
            bin_label = f"{bin_start:04d}-{bin_end:04d}"
            count = counts[bin_idx].item()
            if count > 0:
                epoch_data[bin_label] = (sums[bin_idx] / count).item()
            else:
                epoch_data[bin_label] = 0.0

        filepath = os.path.join(self.save_dir, 'timestep_losses.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                all_data = json.load(f)
        else:
            all_data = {}

        all_data[str(epoch)] = epoch_data

        with open(filepath, 'w') as f:
            json.dump(all_data, f, indent=2)

        # Reset accumulators
        self.timestep_loss_sum = torch.zeros(self.num_timestep_bins, device=self.device)
        self.timestep_loss_count = torch.zeros(self.num_timestep_bins, device=self.device, dtype=torch.long)

    def _log_regional_losses(self, epoch: int) -> None:
        """Save regional loss data to JSON and TensorBoard."""
        if not self._regional_accum_initialized or self.tumor_loss_count is None:
            return

        tumor_count = self.tumor_loss_count.item()
        if tumor_count == 0:
            return

        avg_tumor_loss = (self.tumor_loss_sum / tumor_count).item()
        avg_bg_loss = (self.bg_loss_sum / self.bg_loss_count).item()
        tumor_bg_ratio = avg_tumor_loss / (avg_bg_loss + 1e-8)

        size_losses = {}
        for size_name in self.tumor_size_thresholds.keys():
            count = self.tumor_size_loss_count[size_name].item()
            if count > 0:
                size_losses[size_name] = (self.tumor_size_loss_sum[size_name] / count).item()
            else:
                size_losses[size_name] = 0.0

        if self.writer is not None:
            self.writer.add_scalar('loss/tumor_region', avg_tumor_loss, epoch)
            self.writer.add_scalar('loss/background_region', avg_bg_loss, epoch)
            self.writer.add_scalar('loss/tumor_bg_ratio', tumor_bg_ratio, epoch)

            for size_name, loss_val in size_losses.items():
                self.writer.add_scalar(f'loss/tumor_size_{size_name}', loss_val, epoch)

        epoch_data = {
            'tumor': avg_tumor_loss,
            'background': avg_bg_loss,
            'tumor_bg_ratio': tumor_bg_ratio,
            'by_size': size_losses,
        }

        filepath = os.path.join(self.save_dir, 'regional_losses.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                all_data = json.load(f)
        else:
            all_data = {}

        all_data[str(epoch)] = epoch_data

        with open(filepath, 'w') as f:
            json.dump(all_data, f, indent=2)

        # Reset accumulators
        self.tumor_loss_sum = torch.tensor(0.0, device=self.device)
        self.tumor_loss_count = torch.tensor(0, device=self.device, dtype=torch.long)
        self.bg_loss_sum = torch.tensor(0.0, device=self.device)
        self.bg_loss_count = torch.tensor(0, device=self.device, dtype=torch.long)
        for size in self.tumor_size_thresholds.keys():
            self.tumor_size_loss_sum[size] = torch.tensor(0.0, device=self.device)
            self.tumor_size_loss_count[size] = torch.tensor(0, device=self.device, dtype=torch.long)

    def _log_timestep_region_losses(self, epoch: int) -> None:
        """Log 2D heatmap of loss by timestep bin and region."""
        if self.writer is None:
            return

        if not self._timestep_region_accum_initialized:
            return

        # Get counts from 1D accumulators
        tumor_counts = self._tr_tumor_count.cpu()
        bg_counts = self._tr_bg_count.cpu()
        total_count = tumor_counts.sum().item()
        if total_count == 0:
            return

        tumor_sums = self._tr_tumor_sum.cpu()
        bg_sums = self._tr_bg_sum.cpu()
        bin_size = self.num_timesteps // self.num_timestep_bins

        heatmap_data = np.zeros((self.num_timestep_bins, 2))
        labels_timestep = []

        for bin_idx in range(self.num_timestep_bins):
            bin_start = bin_idx * bin_size
            bin_end = (bin_idx + 1) * bin_size - 1
            labels_timestep.append(f'{bin_start}-{bin_end}')

            # Tumor column
            t_count = tumor_counts[bin_idx].item()
            if t_count > 0:
                heatmap_data[bin_idx, 0] = (tumor_sums[bin_idx] / t_count).item()

            # Background column
            b_count = bg_counts[bin_idx].item()
            if b_count > 0:
                heatmap_data[bin_idx, 1] = (bg_sums[bin_idx] / b_count).item()

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

        for i in range(self.num_timestep_bins):
            for j in range(2):
                ax.text(j, i, f'{heatmap_data[i, j]:.4f}',
                        ha='center', va='center', color='black', fontsize=8)

        plt.tight_layout()
        self.writer.add_figure('loss/timestep_region_heatmap', fig, epoch)
        plt.close(fig)

        epoch_data = {}
        for bin_idx in range(self.num_timestep_bins):
            bin_start = bin_idx * bin_size
            bin_label = f'{bin_start:04d}'
            epoch_data[bin_label] = {
                'tumor': float(heatmap_data[bin_idx, 0]),
                'background': float(heatmap_data[bin_idx, 1]),
            }

        filepath = os.path.join(self.save_dir, 'timestep_region_losses.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                all_data = json.load(f)
        else:
            all_data = {}

        all_data[str(epoch)] = epoch_data

        with open(filepath, 'w') as f:
            json.dump(all_data, f, indent=2)

        # Reset 1D accumulators
        self._tr_tumor_sum = torch.zeros(self.num_timestep_bins, device=self.device)
        self._tr_tumor_count = torch.zeros(self.num_timestep_bins, device=self.device, dtype=torch.long)
        self._tr_bg_sum = torch.zeros(self.num_timestep_bins, device=self.device)
        self._tr_bg_count = torch.zeros(self.num_timestep_bins, device=self.device, dtype=torch.long)

    def get_worst_batch_data(self) -> Optional[Dict[str, Any]]:
        """Get the worst batch data and reset for next epoch.

        Moves tensors to CPU here (deferred from track_step to avoid sync).
        """
        if self.worst_batch_data is None:
            self.worst_batch_loss = 0.0
            return None

        # Move to CPU now (only happens once per val_interval, not every step)
        data = self.worst_batch_data
        cpu_data = {
            'images': data['images'].cpu() if not isinstance(data['images'], dict) else {k: v.cpu() for k, v in data['images'].items()},
            'mask': data['mask'].cpu() if data['mask'] is not None else None,
            'predicted': data['predicted'].cpu() if not isinstance(data['predicted'], dict) else {k: v.cpu() for k, v in data['predicted'].items()},
            'timesteps': data['timesteps'].cpu(),
            'loss': data['loss'],
        }

        self.worst_batch_loss = 0.0
        self.worst_batch_data = None
        return cpu_data

    def compute_ssim(self, generated: torch.Tensor, reference: torch.Tensor) -> float:
        """Compute SSIM between generated and reference images.

        Args:
            generated: Generated images [B, 1, H, W].
            reference: Reference images [B, 1, H, W].

        Returns:
            Average SSIM across batch.
        """
        ssim_values = []
        gen_np = generated.cpu().numpy()
        ref_np = reference.cpu().numpy()

        for i in range(gen_np.shape[0]):
            gen_img = gen_np[i, 0]
            ref_img = ref_np[i, 0]
            gen_img = np.clip(gen_img, 0, 1)
            ref_img = np.clip(ref_img, 0, 1)
            ssim_val = ssim_skimage(gen_img, ref_img, data_range=1.0)
            ssim_values.append(ssim_val)

        return float(np.mean(ssim_values))

    def compute_psnr(self, generated: torch.Tensor, reference: torch.Tensor) -> float:
        """Compute PSNR between generated and reference images.

        Args:
            generated: Generated images [B, 1, H, W].
            reference: Reference images [B, 1, H, W].

        Returns:
            Average PSNR across batch.
        """
        gen_np = np.clip(generated.cpu().numpy(), 0, 1)
        ref_np = np.clip(reference.cpu().numpy(), 0, 1)

        mse = np.mean((gen_np - ref_np) ** 2)
        if mse < 1e-10:
            return 100.0

        psnr = 10 * np.log10(1.0 / mse)
        return float(psnr)

    def _init_lpips(self) -> None:
        """Initialize LPIPS model lazily on first use."""
        if self._lpips_initialized:
            return

        if not self.log_lpips:
            self._lpips_initialized = True
            return

        try:
            import lpips
            self._lpips_model = lpips.LPIPS(net='alex', verbose=False).to(self.device)
            self._lpips_model.eval()
            for param in self._lpips_model.parameters():
                param.requires_grad = False
            if self.is_main_process:
                logger.info("LPIPS metric initialized (AlexNet)")
        except ImportError:
            if self.is_main_process:
                logger.warning("lpips package not installed - LPIPS metric disabled")
            self.log_lpips = False

        self._lpips_initialized = True

    def compute_lpips(self, generated: torch.Tensor, reference: torch.Tensor) -> float:
        """Compute LPIPS (perceptual similarity) between generated and reference images.

        Args:
            generated: Generated images [B, C, H, W].
            reference: Reference images [B, C, H, W].

        Returns:
            Average LPIPS across batch (lower is better, 0 = identical).
        """
        # Initialize LPIPS model lazily
        if not self._lpips_initialized:
            self._init_lpips()

        if self._lpips_model is None:
            return 0.0

        # LPIPS expects images in [-1, 1] range and RGB (3 channels)
        gen = generated.float()
        ref = reference.float()

        # Normalize to [-1, 1]
        gen = gen * 2.0 - 1.0
        ref = ref * 2.0 - 1.0

        # Handle single channel by replicating to 3 channels
        if gen.shape[1] == 1:
            gen = gen.repeat(1, 3, 1, 1)
            ref = ref.repeat(1, 3, 1, 1)
        elif gen.shape[1] > 3:
            # For multi-channel, just use first channel replicated
            gen = gen[:, :1].repeat(1, 3, 1, 1)
            ref = ref[:, :1].repeat(1, 3, 1, 1)

        with torch.no_grad():
            lpips_values = self._lpips_model(gen, ref)

        return float(lpips_values.mean().item())

    def compute_boundary_sharpness(
        self,
        generated: torch.Tensor,
        mask: torch.Tensor,
        dilation_pixels: int = 3
    ) -> float:
        """Compute boundary sharpness in tumor regions.

        Args:
            generated: Generated images [B, 1, H, W].
            mask: Segmentation masks [B, 1, H, W].
            dilation_pixels: Pixels to dilate mask for boundary region.

        Returns:
            Average boundary sharpness.
        """
        sharpness_values = []
        gen_np = generated.cpu().numpy()
        mask_np = mask.cpu().numpy()

        for i in range(gen_np.shape[0]):
            img = gen_np[i, 0]
            m = (mask_np[i, 0] > 0.5).astype(np.float32)

            if m.sum() < 10:
                continue

            dilated = ndimage.binary_dilation(m, iterations=dilation_pixels)
            eroded = ndimage.binary_erosion(m, iterations=dilation_pixels)
            boundary = (dilated.astype(np.float32) - eroded.astype(np.float32))

            if boundary.sum() < 1:
                continue

            grad_x = ndimage.sobel(img, axis=0)
            grad_y = ndimage.sobel(img, axis=1)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

            boundary_grad = (grad_mag * boundary).sum() / (boundary.sum() + 1e-8)
            sharpness_values.append(boundary_grad)

        return float(np.mean(sharpness_values)) if sharpness_values else 0.0
