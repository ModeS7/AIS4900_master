"""
Visualization utilities for diffusion model training.

This module provides validation sample generation and visualization helpers
for monitoring training progress.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.amp import autocast
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from .modes import TrainingMode
from .strategies import DiffusionStrategy
from .metrics import MetricsTracker

logger = logging.getLogger(__name__)


class ValidationVisualizer:
    """Validation sample generation and visualization.

    Handles generation of validation samples, intermediate denoising steps,
    and worst batch visualization for training monitoring.

    Args:
        cfg: Hydra configuration object.
        strategy: Diffusion strategy instance.
        mode: Training mode instance.
        metrics: MetricsTracker instance for quality metrics.
        writer: Optional TensorBoard SummaryWriter.
        save_dir: Directory for saving visualizations.
        device: PyTorch device.
        is_main_process: Whether this is the main process.
    """

    def __init__(
        self,
        cfg: DictConfig,
        strategy: DiffusionStrategy,
        mode: TrainingMode,
        metrics: MetricsTracker,
        writer: Optional[SummaryWriter],
        save_dir: str,
        device: torch.device,
        is_main_process: bool = True,
    ) -> None:
        self.cfg = cfg
        self.strategy = strategy
        self.mode = mode
        self.metrics = metrics
        self.writer = writer
        self.save_dir = save_dir
        self.device = device
        self.is_main_process = is_main_process

        self.mode_name: str = cfg.mode.name
        self.strategy_name: str = cfg.strategy.name
        self.image_size: int = cfg.model.image_size
        self.num_timesteps: int = cfg.strategy.num_train_timesteps

        # Logging config
        logging_cfg = cfg.training.get('logging', {})
        self.log_ssim: bool = logging_cfg.get('ssim', True)
        self.log_psnr: bool = logging_cfg.get('psnr', True)
        self.log_boundary_sharpness: bool = logging_cfg.get('boundary_sharpness', True)
        self.log_intermediate_steps: bool = logging_cfg.get('intermediate_steps', True)
        self.num_intermediate_steps: int = logging_cfg.get('num_intermediate_steps', 5)

    def log_worst_batch(self, epoch: int, data: Dict[str, Any]) -> None:
        """Save visualization of the worst (highest loss) batch.

        Layout:
        - Dual mode: 4x8 grid (8 samples) - rows: GT_pre, Pred_pre, GT_gd, Pred_gd
        - Single mode: 2x16 grid (16 samples) - rows: GT, Pred

        Args:
            epoch: Current epoch number.
            data: Dictionary with 'images', 'predicted', 'mask', 'loss' keys.
        """
        if self.writer is None:
            return

        mask = data['mask']
        is_dual = isinstance(data['images'], dict)

        if is_dual:
            keys = list(data['images'].keys())
            images_pre = data['images'][keys[0]]
            images_gd = data['images'][keys[1]]
            pred_pre = data['predicted'][keys[0]]
            pred_gd = data['predicted'][keys[1]]

            num_show = min(8, images_pre.shape[0])
            fig, axes = plt.subplots(4, num_show, figsize=(2 * num_show, 8))
            fig.suptitle(f'Worst Batch - Epoch {epoch} (Loss: {data["loss"]:.6f})', fontsize=12)

            for i in range(num_show):
                axes[0, i].imshow(images_pre[i, 0].numpy(), cmap='gray')
                axes[0, i].axis('off')
                if i == 0:
                    axes[0, i].set_ylabel('GT Pre', fontsize=10)

                axes[1, i].imshow(pred_pre[i, 0].numpy(), cmap='gray')
                if mask is not None:
                    axes[1, i].contour(mask[i, 0].numpy(), colors='red', linewidths=0.5, alpha=0.7)
                axes[1, i].axis('off')
                if i == 0:
                    axes[1, i].set_ylabel('Pred Pre', fontsize=10)

                axes[2, i].imshow(images_gd[i, 0].numpy(), cmap='gray')
                axes[2, i].axis('off')
                if i == 0:
                    axes[2, i].set_ylabel('GT Gd', fontsize=10)

                axes[3, i].imshow(pred_gd[i, 0].numpy(), cmap='gray')
                if mask is not None:
                    axes[3, i].contour(mask[i, 0].numpy(), colors='red', linewidths=0.5, alpha=0.7)
                axes[3, i].axis('off')
                if i == 0:
                    axes[3, i].set_ylabel('Pred Gd', fontsize=10)
        else:
            images = data['images']
            predicted = data['predicted']

            num_show = min(16, images.shape[0])
            fig, axes = plt.subplots(2, num_show, figsize=(num_show, 2))
            fig.suptitle(f'Worst Batch - Epoch {epoch} (Loss: {data["loss"]:.6f})', fontsize=12)

            for i in range(num_show):
                axes[0, i].imshow(images[i, 0].numpy(), cmap='gray')
                axes[0, i].axis('off')
                if i == 0:
                    axes[0, i].set_ylabel('GT', fontsize=10)

                axes[1, i].imshow(predicted[i, 0].numpy(), cmap='gray')
                if mask is not None:
                    axes[1, i].contour(mask[i, 0].numpy(), colors='red', linewidths=0.5, alpha=0.7)
                axes[1, i].axis('off')
                if i == 0:
                    axes[1, i].set_ylabel('Pred', fontsize=10)

        plt.tight_layout()

        filepath = os.path.join(self.save_dir, f'worst_batch_epoch_{epoch:04d}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')

        self.writer.add_figure('worst_batch', fig, epoch)
        plt.close(fig)

    def _sample_positive_masks(
        self,
        train_dataset: Dataset,
        num_samples: int,
        seg_channel_idx: int,
        return_images: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample slices with positive segmentation masks from dataset.

        Args:
            train_dataset: Dataset to sample from.
            num_samples: Number of samples to get.
            seg_channel_idx: Channel index for segmentation mask.
            return_images: If True, also return ground truth images.

        Returns:
            Segmentation masks [N, 1, H, W], or tuple of (masks, images) if return_images=True.
        """
        seg_masks = []
        gt_images = []
        attempts = 0
        max_attempts = len(train_dataset)

        while len(seg_masks) < num_samples and attempts < max_attempts:
            idx = torch.randint(0, len(train_dataset), (1,)).item()
            data = train_dataset[idx]
            tensor = torch.from_numpy(data).float() if hasattr(data, '__array__') else torch.tensor(data).float()
            seg = tensor[seg_channel_idx:seg_channel_idx + 1, :, :]

            if seg.sum() > 0:
                seg_masks.append(seg)
                if return_images:
                    if seg_channel_idx == 1:  # bravo mode
                        gt_images.append(tensor[0:1, :, :])
                    elif seg_channel_idx == 2:  # dual mode
                        gt_images.append(tensor[0:2, :, :])
            attempts += 1

        if len(seg_masks) < num_samples:
            logger.warning(f"Only found {len(seg_masks)} positive masks for validation")

        if len(seg_masks) == 0:
            raise ValueError("No positive segmentation masks found for validation")

        masks = torch.stack(seg_masks).to(self.device)
        if return_images:
            images = torch.stack(gt_images).to(self.device)
            return masks, images
        return masks

    def _generate_with_intermediate_steps(
        self,
        model: nn.Module,
        model_input: torch.Tensor,
        num_steps: int
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate samples while saving intermediate denoising steps.

        Args:
            model: The diffusion model.
            model_input: Initial noisy input (may include condition channels).
            num_steps: Number of denoising steps.

        Returns:
            Tuple of (final_samples, list_of_intermediate_samples).
        """
        model_config = self.mode.get_model_config()
        out_channels = model_config['out_channels']
        in_channels = model_config['in_channels']

        has_condition = in_channels > out_channels
        if has_condition:
            noisy = model_input[:, :out_channels, :, :]
            condition = model_input[:, out_channels:, :, :]
        else:
            noisy = model_input
            condition = None

        save_at_timesteps = set()
        if self.num_intermediate_steps > 0:
            for frac in [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]:
                save_at_timesteps.add(int(frac * (num_steps - 1)))

        intermediates = []
        current = noisy.clone()

        for step_idx in range(num_steps):
            timestep_val = num_steps - 1 - step_idx
            t = torch.full(
                (current.shape[0],),
                timestep_val,
                device=current.device,
                dtype=torch.long
            )

            if condition is not None:
                full_input = torch.cat([current, condition], dim=1)
            else:
                full_input = current

            with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                pred = model(full_input, t)

            if self.strategy_name == 'rflow':
                dt = 1.0 / num_steps
                current = current + dt * pred
            else:
                current = self.strategy.scheduler.step(pred, timestep_val, current).prev_sample

            if timestep_val in save_at_timesteps:
                intermediates.append(current.clone())

        if not intermediates or not torch.equal(intermediates[-1], current):
            intermediates.append(current.clone())

        return current, intermediates

    def _log_intermediate_steps(
        self,
        epoch: int,
        intermediates: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> None:
        """Log intermediate denoising steps to TensorBoard.

        Args:
            epoch: Current epoch.
            intermediates: List of intermediate samples.
            mask: Optional segmentation mask for overlay.
        """
        if self.writer is None or not intermediates:
            return

        num_steps = len(intermediates)
        fig, axes = plt.subplots(1, num_steps, figsize=(4 * num_steps, 4))

        if num_steps == 1:
            axes = [axes]

        fracs = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
        timestep_labels = [f't={int(frac * (self.num_timesteps - 1))}' for frac in fracs]
        timestep_labels = timestep_labels[:num_steps]

        for i, (intermediate, ax) in enumerate(zip(intermediates, axes)):
            img = intermediate[0, 0].cpu().numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img, cmap='gray')
            if mask is not None:
                ax.contour(mask[0, 0].cpu().numpy(), colors='red', linewidths=0.5, alpha=0.7)
            ax.set_title(timestep_labels[i] if i < len(timestep_labels) else f'Step {i}')
            ax.axis('off')

        plt.tight_layout()
        self.writer.add_figure('denoising_trajectory', fig, epoch)
        plt.close(fig)

        filepath = os.path.join(self.save_dir, f'denoising_trajectory_epoch_{epoch:04d}.png')
        fig2, axes2 = plt.subplots(1, num_steps, figsize=(4 * num_steps, 4))
        if num_steps == 1:
            axes2 = [axes2]
        for i, (intermediate, ax) in enumerate(zip(intermediates, axes2)):
            img = intermediate[0, 0].cpu().numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img, cmap='gray')
            ax.set_title(timestep_labels[i] if i < len(timestep_labels) else f'Step {i}')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig2)

    def generate_samples(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        epoch: int,
        num_samples: int = 4
    ) -> None:
        """Generate and log validation samples to TensorBoard.

        Args:
            model: The diffusion model (or EMA model).
            train_dataset: Dataset for sampling positive masks.
            epoch: Current epoch number.
            num_samples: Number of samples to generate.
        """
        if not self.is_main_process or self.writer is None:
            return

        model.eval()

        try:
            with torch.no_grad():
                model_config = self.mode.get_model_config()
                out_channels = model_config['out_channels']

                seg_masks = None
                gt_images = None
                intermediates = None

                if self.mode_name == 'seg':
                    noise = torch.randn((num_samples, 1, self.image_size, self.image_size), device=self.device)
                    model_input = noise

                elif self.mode_name == 'bravo':
                    need_gt = self.log_ssim or self.log_psnr
                    if need_gt:
                        seg_masks, gt_images = self._sample_positive_masks(
                            train_dataset, num_samples, seg_channel_idx=1, return_images=True
                        )
                    else:
                        seg_masks = self._sample_positive_masks(train_dataset, num_samples, seg_channel_idx=1)
                    noise = torch.randn_like(seg_masks, device=self.device)
                    model_input = torch.cat([noise, seg_masks], dim=1)

                elif self.mode_name == 'dual':
                    need_gt = self.log_ssim or self.log_psnr
                    if need_gt:
                        seg_masks, gt_images = self._sample_positive_masks(
                            train_dataset, num_samples, seg_channel_idx=2, return_images=True
                        )
                    else:
                        seg_masks = self._sample_positive_masks(train_dataset, num_samples, seg_channel_idx=2)
                    noise_pre = torch.randn_like(seg_masks, device=self.device)
                    noise_gd = torch.randn_like(seg_masks, device=self.device)
                    model_input = torch.cat([noise_pre, noise_gd, seg_masks], dim=1)

                else:
                    raise ValueError(f"Unknown mode: {self.mode_name}")

                with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    if self.log_intermediate_steps and self.mode_name != 'seg':
                        samples, intermediates = self._generate_with_intermediate_steps(
                            model, model_input, self.num_timesteps
                        )
                    else:
                        samples = self.strategy.generate(
                            model, model_input, num_steps=self.num_timesteps, device=self.device
                        )

                if intermediates is not None and self.log_intermediate_steps:
                    self._log_intermediate_steps(epoch, intermediates, seg_masks)

                if out_channels == 2:
                    samples_pre = samples[:, 0:1, :, :].float()
                    samples_gd = samples[:, 1:2, :, :].float()

                    samples_pre_norm = torch.clamp(samples_pre, 0, 1)
                    samples_gd_norm = torch.clamp(samples_gd, 0, 1)

                    samples_pre_rgb = samples_pre_norm.repeat(1, 3, 1, 1)
                    samples_gd_rgb = samples_gd_norm.repeat(1, 3, 1, 1)

                    self.writer.add_images('Generated_T1_Pre', samples_pre_rgb, epoch)
                    self.writer.add_images('Generated_T1_Gd', samples_gd_rgb, epoch)

                    if gt_images is not None:
                        gt_pre = gt_images[:, 0:1, :, :]
                        gt_gd = gt_images[:, 1:2, :, :]

                        if self.log_ssim:
                            ssim_pre = self.metrics.compute_ssim(samples_pre_norm, gt_pre)
                            ssim_gd = self.metrics.compute_ssim(samples_gd_norm, gt_gd)
                            self.writer.add_scalar('metrics/ssim_t1_pre', ssim_pre, epoch)
                            self.writer.add_scalar('metrics/ssim_t1_gd', ssim_gd, epoch)

                        if self.log_psnr:
                            psnr_pre = self.metrics.compute_psnr(samples_pre_norm, gt_pre)
                            psnr_gd = self.metrics.compute_psnr(samples_gd_norm, gt_gd)
                            self.writer.add_scalar('metrics/psnr_t1_pre', psnr_pre, epoch)
                            self.writer.add_scalar('metrics/psnr_t1_gd', psnr_gd, epoch)

                    if self.log_boundary_sharpness and seg_masks is not None:
                        sharpness_pre = self.metrics.compute_boundary_sharpness(samples_pre_norm, seg_masks)
                        sharpness_gd = self.metrics.compute_boundary_sharpness(samples_gd_norm, seg_masks)
                        self.writer.add_scalar('metrics/boundary_sharpness_t1_pre', sharpness_pre, epoch)
                        self.writer.add_scalar('metrics/boundary_sharpness_t1_gd', sharpness_gd, epoch)
                else:
                    samples_float = samples.float()
                    samples_normalized = torch.clamp(samples_float, 0, 1)

                    if samples_normalized.dim() == 3:
                        samples_normalized = samples_normalized.unsqueeze(1)

                    samples_rgb = samples_normalized.repeat(1, 3, 1, 1)
                    self.writer.add_images('Generated_Images', samples_rgb, epoch)

                    if gt_images is not None and self.mode_name == 'bravo':
                        if self.log_ssim:
                            ssim_val = self.metrics.compute_ssim(samples_normalized, gt_images)
                            self.writer.add_scalar('metrics/ssim', ssim_val, epoch)

                        if self.log_psnr:
                            psnr_val = self.metrics.compute_psnr(samples_normalized, gt_images)
                            self.writer.add_scalar('metrics/psnr', psnr_val, epoch)

                    if self.log_boundary_sharpness and seg_masks is not None and self.mode_name == 'bravo':
                        sharpness = self.metrics.compute_boundary_sharpness(samples_normalized, seg_masks)
                        self.writer.add_scalar('metrics/boundary_sharpness', sharpness, epoch)

        except Exception as e:
            if self.is_main_process:
                logger.warning(
                    f"Failed to generate validation samples at epoch {epoch}: {e}",
                    exc_info=True
                )
        finally:
            torch.cuda.empty_cache()
