"""Validation runner for compression trainers.

Extracts the validation loop logic from BaseCompressionTrainer and
BaseCompression3DTrainer into a reusable component.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from .metrics import (
    compute_lpips,
    compute_lpips_3d,
    compute_msssim,
    compute_msssim_2d_slicewise,
    compute_psnr,
)
from .tracking import WorstBatchTracker

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation metrics computation.

    Args:
        log_msssim: Whether to compute MS-SSIM.
        log_psnr: Whether to compute PSNR.
        log_lpips: Whether to compute LPIPS.
        log_regional_losses: Whether to track regional losses.
        weight_dtype: Data type for autocast.
        use_compile: Whether to use torch.compile.
        spatial_dims: 2 for 2D images, 3 for 3D volumes.
    """

    log_msssim: bool = True
    log_psnr: bool = True
    log_lpips: bool = True
    log_regional_losses: bool = False
    weight_dtype: torch.dtype = torch.bfloat16
    use_compile: bool = False
    spatial_dims: int = 2  # 2 for 2D images, 3 for 3D volumes


@dataclass
class ValidationResult:
    """Result from a validation run."""

    metrics: Dict[str, float]
    worst_batch_data: Optional[Dict[str, Any]] = None
    regional_tracker: Optional[Any] = None


class ValidationRunner:
    """Runs validation loop for compression models.

    Encapsulates the common validation logic shared between 2D and 3D
    compression trainers, including:
    - Metric accumulation
    - Worst batch tracking
    - Regional metrics tracking
    - Quality metrics computation (MS-SSIM, PSNR, LPIPS)

    Usage:
        runner = ValidationRunner(
            config=ValidationConfig(log_msssim=True),
            device=device,
            forward_fn=lambda model, x: model(x)[0],  # VAE returns (recon, mean, logvar)
            loss_fn=lambda recon, target: (l1, perc, reg, total),
        )

        result = runner.run(
            val_loader=val_loader,
            model=model,
            perceptual_weight=0.1,
            epoch=epoch,
        )
    """

    def __init__(
        self,
        config: ValidationConfig,
        device: torch.device,
        forward_fn: Callable[[nn.Module, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        perceptual_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        regional_tracker_factory: Optional[Callable[[], Any]] = None,
        prepare_batch_fn: Optional[Callable[[Any], Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
    ):
        """Initialize validation runner.

        Args:
            config: Validation configuration.
            device: Device for computation.
            forward_fn: Function that takes (model, images) and returns
                (reconstruction, regularization_loss). This is trainer-specific.
            perceptual_loss_fn: Optional perceptual loss function.
            regional_tracker_factory: Optional factory to create regional tracker.
            prepare_batch_fn: Optional function to prepare batch data.
        """
        self.config = config
        self.device = device
        self.forward_fn = forward_fn
        self.perceptual_loss_fn = perceptual_loss_fn
        self.regional_tracker_factory = regional_tracker_factory
        self.prepare_batch_fn = prepare_batch_fn or self._default_prepare_batch

    def _default_prepare_batch(
        self,
        batch: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Default batch preparation."""
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

    def run(
        self,
        val_loader: DataLoader,
        model: nn.Module,
        perceptual_weight: float = 0.0,
        log_figures: bool = True,
    ) -> ValidationResult:
        """Run validation loop.

        Args:
            val_loader: Validation data loader.
            model: Model to evaluate (should already be in eval mode).
            perceptual_weight: Weight for perceptual loss.
            log_figures: Whether to track worst batch for visualization.

        Returns:
            ValidationResult with metrics and optional tracking data.
        """
        is_3d = self.config.spatial_dims == 3

        # Initialize accumulators
        total_l1 = 0.0
        total_perc = 0.0
        total_reg = 0.0
        total_gen = 0.0
        total_msssim = 0.0
        total_msssim_3d = 0.0  # For 3D: volumetric MS-SSIM
        total_psnr = 0.0
        total_lpips = 0.0
        n_batches = 0

        # Worst batch tracking
        worst_batch_tracker = WorstBatchTracker(enabled=log_figures)

        # Regional tracker
        regional_tracker = None
        if self.config.log_regional_losses and self.regional_tracker_factory is not None:
            regional_tracker = self.regional_tracker_factory()

        # Mark CUDA graph step boundary
        if self.config.use_compile:
            torch.compiler.cudagraph_mark_step_begin()

        with torch.inference_mode():
            for batch in val_loader:
                images, mask = self.prepare_batch_fn(batch)

                with autocast('cuda', enabled=True, dtype=self.config.weight_dtype):
                    # Forward pass (subclass-specific)
                    reconstruction, reg_loss = self.forward_fn(model, images)

                    # L1 reconstruction loss
                    l1_loss = torch.abs(reconstruction - images).mean()

                    # Perceptual loss
                    if self.perceptual_loss_fn is not None and perceptual_weight > 0:
                        p_loss = self.perceptual_loss_fn(reconstruction, images)
                    else:
                        p_loss = torch.tensor(0.0, device=self.device)

                    # Total generator loss
                    g_loss = l1_loss + perceptual_weight * p_loss + reg_loss

                # Accumulate losses
                loss_val = g_loss.item()
                total_l1 += l1_loss.item()
                total_perc += p_loss.item()
                total_reg += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
                total_gen += loss_val

                # Track worst batch
                worst_batch_tracker.update(
                    loss=loss_val,
                    original=images,
                    generated=reconstruction,
                    loss_breakdown={
                        'L1': l1_loss.item(),
                        'Perc': p_loss.item(),
                        'Reg': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
                    },
                )

                # Quality metrics - use appropriate functions based on spatial_dims
                if self.config.log_msssim:
                    if is_3d:
                        # 3D: compute both volumetric and slice-by-slice MS-SSIM
                        total_msssim_3d += compute_msssim(
                            reconstruction.float(), images.float(), spatial_dims=3
                        )
                        total_msssim += compute_msssim_2d_slicewise(
                            reconstruction.float(), images.float()
                        )
                    else:
                        # 2D: standard MS-SSIM
                        total_msssim += compute_msssim(reconstruction, images)
                if self.config.log_psnr:
                    total_psnr += compute_psnr(reconstruction, images)
                if self.config.log_lpips:
                    if is_3d:
                        total_lpips += compute_lpips_3d(
                            reconstruction.float(), images.float(), device=self.device
                        )
                    else:
                        total_lpips += compute_lpips(
                            reconstruction, images, device=self.device
                        )

                # Regional tracking
                if regional_tracker is not None and mask is not None:
                    regional_tracker.update(reconstruction, images, mask)

                n_batches += 1

        # Compute averages
        metrics = {
            'l1': total_l1 / n_batches if n_batches > 0 else 0.0,
            'perc': total_perc / n_batches if n_batches > 0 else 0.0,
            'reg': total_reg / n_batches if n_batches > 0 else 0.0,
            'gen': total_gen / n_batches if n_batches > 0 else 0.0,
        }
        if self.config.log_psnr:
            metrics['psnr'] = total_psnr / n_batches if n_batches > 0 else 0.0
        if self.config.log_lpips:
            metrics['lpips'] = total_lpips / n_batches if n_batches > 0 else 0.0
        if self.config.log_msssim:
            # msssim is always slice-by-slice 2D (for comparison with 2D trainers)
            metrics['msssim'] = total_msssim / n_batches if n_batches > 0 else 0.0
            if is_3d:
                # msssim_3d is volumetric 3D MS-SSIM (captures cross-slice structure)
                metrics['msssim_3d'] = total_msssim_3d / n_batches if n_batches > 0 else 0.0

        # Get worst batch data
        worst_batch_data = worst_batch_tracker.get_and_reset()

        return ValidationResult(
            metrics=metrics,
            worst_batch_data=worst_batch_data,
            regional_tracker=regional_tracker,
        )
