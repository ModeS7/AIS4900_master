"""
Metrics tracking for diffusion model training.

This module provides GPU-efficient metrics tracking for gradient norms
during training. Timestep and regional metrics are tracked during validation
in trainer.py (not here) to avoid misleading training metrics.
"""
import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from .quality import (
    compute_msssim as _compute_msssim,
    compute_psnr as _compute_psnr,
    compute_lpips as _compute_lpips,
)

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Metrics tracking for diffusion training.

    Tracks gradient norms during training. Timestep and regional metrics
    are now tracked during validation in trainer.py (not here).

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
        # Note: timestep_losses and regional_losses flags are used by trainer.py
        # for validation metrics, not for training tracking here
        logging_cfg = cfg.training.get('logging', {})
        self.log_grad_norm: bool = logging_cfg.get('grad_norm', True)
        self.log_timestep_losses: bool = logging_cfg.get('timestep_losses', True)
        self.log_regional_losses: bool = logging_cfg.get('regional_losses', True)
        self.log_msssim: bool = logging_cfg.get('msssim', True)
        self.log_psnr: bool = logging_cfg.get('psnr', True)
        self.log_lpips: bool = logging_cfg.get('lpips', False)
        self.log_boundary_sharpness: bool = logging_cfg.get('boundary_sharpness', True)
        self.log_flops: bool = logging_cfg.get('flops', True)

        # Gradient norm tracking
        self.grad_norm_sum: Optional[torch.Tensor] = None
        self.grad_norm_max: Optional[torch.Tensor] = None
        self.grad_norm_count: int = 0

        # Scheduler reference (set externally for SNR computation)
        self.scheduler: Optional[Any] = None

    def set_scheduler(self, scheduler: Any) -> None:
        """Set scheduler reference for SNR weight computation."""
        self.scheduler = scheduler

    def init_accumulators(self) -> None:
        """Initialize gradient norm accumulators."""
        # Gradient norm accumulators
        if self.grad_norm_sum is None:
            self.grad_norm_sum = torch.tensor(0.0, device=self.device)
            self.grad_norm_max = torch.tensor(0.0, device=self.device)

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

        # Note: Timestep and regional losses are now tracked during validation only
        # (training metrics are misleading due to overfitting)

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

        # Note: Timestep and regional losses are logged from validation loop in trainer.py
        # (training metrics removed - they're misleading due to overfitting)

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
