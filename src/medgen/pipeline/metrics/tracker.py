"""
Metrics tracking for diffusion model training.

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
from scipy.ndimage import label as scipy_label
from skimage.measure import regionprops
from torch.utils.tensorboard import SummaryWriter

from .quality import (
    compute_msssim as _compute_msssim,
    compute_psnr as _compute_psnr,
    compute_lpips as _compute_lpips,
)
from .constants import TUMOR_SIZE_THRESHOLDS_MM


def _compute_max_feret_diameter(mask_np: np.ndarray, mm_per_pixel: float) -> float:
    """Compute maximum Feret diameter across all tumors in a mask.

    Uses connected component analysis and skimage regionprops to find the
    longest edge-to-edge distance (Feret diameter) across all tumor regions.

    Args:
        mask_np: Binary 2D mask array (H, W).
        mm_per_pixel: Millimeters per pixel for size conversion.

    Returns:
        Maximum Feret diameter in mm, or 0.0 if no valid tumors found.
    """
    labeled, num_tumors = scipy_label(mask_np)
    if num_tumors == 0:
        return 0.0

    regions = regionprops(labeled)
    max_feret_mm = 0.0

    for region in regions:
        # Skip tiny fragments (<5 pixels)
        if region.area < 5:
            continue
        feret_px = region.feret_diameter_max
        feret_mm = feret_px * mm_per_pixel
        max_feret_mm = max(max_feret_mm, feret_mm)

    return max_feret_mm


def _classify_tumor_size_feret(diameter_mm: float) -> str:
    """Classify tumor by Feret diameter using RANO-BM thresholds.

    Args:
        diameter_mm: Feret diameter (longest axis) in millimeters.

    Returns:
        Size category: 'tiny', 'small', 'medium', or 'large'.
    """
    for size_name, (low, high) in TUMOR_SIZE_THRESHOLDS_MM.items():
        if low <= diameter_mm < high:
            return size_name
    return 'large'  # Fallback for very large tumors

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
        self.log_msssim: bool = logging_cfg.get('msssim', True)
        self.log_psnr: bool = logging_cfg.get('psnr', True)
        self.log_lpips: bool = logging_cfg.get('lpips', False)  # Off by default (2D only, slower)
        self.log_boundary_sharpness: bool = logging_cfg.get('boundary_sharpness', True)
        self.log_flops: bool = logging_cfg.get('flops', True)

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

        # 2D timestep-region loss tracking
        self._timestep_region_accum_initialized: bool = False
        self.timestep_region_loss_sum: Optional[torch.Tensor] = None
        self.timestep_region_loss_count: Optional[torch.Tensor] = None

        # Scheduler reference (set externally for SNR computation)
        self.scheduler: Optional[Any] = None

    def set_scheduler(self, scheduler: Any) -> None:
        """Set scheduler reference for SNR weight computation."""
        self.scheduler = scheduler

    def init_accumulators(self) -> None:
        """Initialize all GPU accumulators.

        Call this after construction, before training begins. Makes initialization
        explicit rather than lazy (hidden in first track_step call).
        """
        # Timestep loss accumulators
        if not self._timestep_accum_initialized:
            self.timestep_loss_sum = torch.zeros(self.num_timestep_bins, device=self.device)
            self.timestep_loss_count = torch.zeros(self.num_timestep_bins, device=self.device, dtype=torch.long)
            self._timestep_accum_initialized = True

        # Regional loss accumulators
        if not self._regional_accum_initialized:
            self._init_regional_accumulators()

        # Timestep-region accumulators
        if not self._timestep_region_accum_initialized:
            self._tr_tumor_sum = torch.zeros(self.num_timestep_bins, device=self.device)
            self._tr_tumor_count = torch.zeros(self.num_timestep_bins, device=self.device, dtype=torch.long)
            self._tr_bg_sum = torch.zeros(self.num_timestep_bins, device=self.device)
            self._tr_bg_count = torch.zeros(self.num_timestep_bins, device=self.device, dtype=torch.long)
            self.timestep_region_loss_sum = None
            self.timestep_region_loss_count = None
            self._timestep_region_accum_initialized = True

        # Gradient norm accumulators
        if self.grad_norm_sum is None:
            self.grad_norm_sum = torch.tensor(0.0, device=self.device)
            self.grad_norm_max = torch.tensor(0.0, device=self.device)

    def _compute_tumor_size_thresholds(self) -> Dict[str, Tuple[float, float]]:
        """Get RANO-BM tumor size thresholds and compute mm_per_pixel.

        Clinical definitions (Feret diameter):
            - tiny:   <10mm  (often non-measurable per RANO-BM)
            - small:  10-20mm (small metastases, SRS alone)
            - medium: 20-30mm (SRS candidates)
            - large:  >30mm  (often surgical)

        Returns:
            Dictionary mapping size names to (low, high) mm thresholds.
        """
        fov_mm = self.cfg.paths.get('fov_mm', 240.0)
        self.mm_per_pixel = fov_mm / self.image_size

        if self.is_main_process:
            logger.info(f"Tumor size tracking: {self.image_size}px ({self.mm_per_pixel:.2f} mm/px)")
            logger.info("  Using Feret diameter (RANO-BM thresholds):")
            for name, (low, high) in TUMOR_SIZE_THRESHOLDS_MM.items():
                logger.info(f"    {name}: {low}-{high}mm")

        return TUMOR_SIZE_THRESHOLDS_MM

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
    ) -> None:
        """Track metrics for a single training step.

        All accumulation happens on GPU to avoid sync bottlenecks.

        Args:
            timesteps: Diffusion timesteps for batch [B].
            predicted_clean: Model's predicted clean images.
            images: Ground truth images.
            mask: Segmentation mask (None for seg mode).
            grad_norm: Gradient norm from clipping.
        """
        # Track gradient norm
        if self.log_grad_norm and grad_norm is not None:
            if self.grad_norm_sum is None:
                self.grad_norm_sum = torch.tensor(0.0, device=self.device)
                self.grad_norm_max = torch.tensor(0.0, device=self.device)
            # Ensure grad_norm is on correct device (clip_grad_norm_ returns CPU tensor)
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.to(self.device)
            else:
                grad_norm = torch.tensor(grad_norm, device=self.device)
            self.grad_norm_sum = self.grad_norm_sum + grad_norm
            self.grad_norm_max = torch.maximum(self.grad_norm_max, grad_norm)
            self.grad_norm_count += 1

        # Track timestep losses
        if self.log_timestep_losses:
            self._track_timestep_loss_batch(timesteps, predicted_clean, images)

        # Track regional losses
        # For conditional modes (bravo, dual): use conditioning mask
        # For seg mode: use ground truth seg as mask (images IS the seg)
        regional_mask = mask
        if not self.is_conditional and mask is None:
            # Seg mode: use ground truth seg as the mask
            if isinstance(images, dict):
                regional_mask = list(images.values())[0]
            else:
                regional_mask = images

        if regional_mask is not None:
            if self.log_regional_losses:
                self._track_regional_losses(predicted_clean, images, regional_mask)
            if self.log_timestep_region:
                self._track_timestep_region_loss(timesteps, predicted_clean, images, regional_mask)

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
        """Track losses by region and tumor size using Feret diameter classification.

        Uses connected component analysis to compute Feret diameter (longest axis)
        for tumor size classification, matching the validation metric approach.
        """
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

        tumor_pixels_safe = tumor_pixels.clamp(min=1)
        bg_pixels = mask.shape[2] * mask.shape[3] - tumor_pixels
        bg_pixels_safe = bg_pixels.clamp(min=1)

        tumor_loss_per_sample = (sq_error * tumor_mask_expanded).sum(dim=(1, 2, 3)) / tumor_pixels_safe
        bg_loss_per_sample = (sq_error * bg_mask_expanded).sum(dim=(1, 2, 3)) / bg_pixels_safe

        has_tumor = (tumor_pixels > 10)
        has_tumor_float = has_tumor.float()
        num_valid = has_tumor_float.sum().long()

        self.tumor_loss_sum += (tumor_loss_per_sample * has_tumor_float).sum()
        self.tumor_loss_count += num_valid
        self.bg_loss_sum += (bg_loss_per_sample * has_tumor_float).sum()
        self.bg_loss_count += num_valid

        # Classify by Feret diameter (per-sample, requires connected components)
        batch_size = mask.shape[0]
        for i in range(batch_size):
            if not has_tumor[i]:
                continue

            # Compute max Feret diameter for this sample
            mask_np = mask[i, 0].cpu().numpy() > 0.5
            feret_mm = _compute_max_feret_diameter(mask_np, self.mm_per_pixel)

            if feret_mm > 0:
                size_cat = _classify_tumor_size_feret(feret_mm)
                sample_loss = tumor_loss_per_sample[i].item()
                self.tumor_size_loss_sum[size_cat] += sample_loss
                self.tumor_size_loss_count[size_cat] += 1

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

    def log_epoch(self, epoch: int, log_all: bool = False, is_figure_epoch: bool = False) -> None:
        """Log all accumulated metrics for the epoch.

        Args:
            epoch: Current epoch number.
            log_all: If True, log all metrics. If False, only log grad norms.
            is_figure_epoch: If True, log expensive visualizations (timestep-region heatmap).
        """
        if not self.is_main_process:
            return

        # Always log grad norms (lightweight)
        if self.log_grad_norm:
            self._log_grad_norms(epoch)

        # Other metrics when log_all=True (every epoch by default)
        if log_all:
            if self.log_timestep_losses:
                self._log_timestep_losses(epoch)
            if self.log_regional_losses:
                self._log_regional_losses(epoch)

        # Timestep-region heatmap only at figure_interval (expensive)
        if is_figure_epoch and self.log_timestep_region:
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

    def _log_timestep_losses(self, epoch: int) -> None:
        """Save timestep loss distribution to JSON file and TensorBoard."""
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
                avg_loss = (sums[bin_idx] / count).item()
                epoch_data[bin_label] = avg_loss
                # Log to TensorBoard under Timestep/ branch
                if self.writer is not None:
                    self.writer.add_scalar(f'Timestep/{bin_start}-{bin_end}', avg_loss, epoch)
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

        bg_count = self.bg_loss_count.item()
        if bg_count == 0:
            return

        avg_tumor_loss = (self.tumor_loss_sum / tumor_count).item()
        avg_bg_loss = (self.bg_loss_sum / bg_count).item()
        tumor_bg_ratio = avg_tumor_loss / (avg_bg_loss + 1e-8)

        size_losses = {}
        for size_name in self.tumor_size_thresholds.keys():
            count = self.tumor_size_loss_count[size_name].item()
            if count > 0:
                size_losses[size_name] = (self.tumor_size_loss_sum[size_name] / count).item()
            else:
                size_losses[size_name] = 0.0

        # Note: Training regional losses not logged to TensorBoard - they're meaningless
        # for diffusion (each batch has different noise levels). Only validation/test
        # regional metrics matter. JSON file still saved for offline analysis.

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

        # Reset accumulators (zero existing tensors to avoid memory allocation)
        self.tumor_loss_sum.zero_()
        self.tumor_loss_count.zero_()
        self.bg_loss_sum.zero_()
        self.bg_loss_count.zero_()
        for size in self.tumor_size_thresholds.keys():
            self.tumor_size_loss_sum[size].zero_()
            self.tumor_size_loss_count[size].zero_()

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

    def compute_msssim(self, generated: torch.Tensor, reference: torch.Tensor) -> float:
        """Compute MS-SSIM between generated and reference images.

        Multi-Scale Structural Similarity replaces both SSIM and LPIPS.
        Uses shared implementation from quality module.

        Args:
            generated: Generated images [B, C, H, W].
            reference: Reference images [B, C, H, W].

        Returns:
            Average MS-SSIM across batch (higher is better, 1.0 = identical).
        """
        if not self.log_msssim:
            return 0.0
        return _compute_msssim(generated, reference)

    def compute_psnr(self, generated: torch.Tensor, reference: torch.Tensor) -> float:
        """Compute PSNR between generated and reference images.

        Uses shared implementation from quality module.

        Args:
            generated: Generated images [B, C, H, W].
            reference: Reference images [B, C, H, W].

        Returns:
            Average PSNR across batch.
        """
        return _compute_psnr(generated, reference)

    def compute_lpips(self, generated: torch.Tensor, reference: torch.Tensor) -> float:
        """Compute LPIPS (perceptual distance) between generated and reference images.

        Uses MONAI's PerceptualLoss with RadImageNet pretrained features.
        Only works with 2D images. For 3D, use MS-SSIM instead.

        Args:
            generated: Generated images [B, C, H, W].
            reference: Reference images [B, C, H, W].

        Returns:
            Average LPIPS across batch (lower is better, 0 = identical).
        """
        if not self.log_lpips:
            return 0.0
        cache_dir = getattr(self.cfg.paths, 'cache_dir', None)
        return _compute_lpips(generated, reference, cache_dir=cache_dir)

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
        gen_np = generated.cpu().float().numpy()
        mask_np = mask.cpu().float().numpy()

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
