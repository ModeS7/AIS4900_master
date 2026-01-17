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

from medgen.core import ModeType
from .modes import TrainingMode
from .strategies import DiffusionStrategy
from medgen.metrics import MetricsTracker, create_reconstruction_figure
from .spaces import DiffusionSpace, PixelSpace
from .controlnet import ControlNetConditionedUNet, ControlNetGenerationWrapper

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
        space: Optional DiffusionSpace for pixel/latent operations.
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
        space: Optional[DiffusionSpace] = None,
        use_controlnet: bool = False,
        controlnet: Optional[nn.Module] = None,
    ) -> None:
        self.cfg = cfg
        self.strategy = strategy
        self.mode = mode
        self.metrics = metrics
        self.writer = writer
        self.save_dir = save_dir
        self.device = device
        self.is_main_process = is_main_process
        self.space = space if space is not None else PixelSpace()

        # ControlNet support
        self.use_controlnet = use_controlnet
        self.controlnet = controlnet

        self.mode_name: str = cfg.mode.name
        self.strategy_name: str = cfg.strategy.name
        self.image_size: int = cfg.model.image_size
        self.num_timesteps: int = cfg.strategy.num_train_timesteps

        # Logging config
        logging_cfg = cfg.training.get('logging', {})
        self.log_msssim: bool = logging_cfg.get('msssim', True)
        self.log_psnr: bool = logging_cfg.get('psnr', True)
        self.log_lpips: bool = logging_cfg.get('lpips', False)
        self.log_boundary_sharpness: bool = logging_cfg.get('boundary_sharpness', True)
        self.log_intermediate_steps: bool = logging_cfg.get('intermediate_steps', True)
        self.num_intermediate_steps: int = logging_cfg.get('num_intermediate_steps', 5)

    def log_worst_batch(self, epoch: int, data: Dict[str, Any]) -> None:
        """Save visualization of the worst (highest loss) batch.

        Uses shared create_reconstruction_figure for consistent visualization.

        Args:
            epoch: Current epoch number.
            data: Dictionary with 'original', 'generated', 'mask', 'timesteps', 'loss' keys.
        """
        if self.writer is None:
            return

        timesteps = data.get('timesteps')
        avg_timestep = timesteps.float().mean().item() if timesteps is not None else 0

        # Decode from latent space to pixel space for visualization
        original = data['original']
        generated = data['generated']
        if self.space.scale_factor > 1:
            original = self.space.decode(original)
            generated = self.space.decode(generated)

        title = f'Worst Validation Batch - Epoch {epoch} | Loss: {data["loss"]:.6f} | Avg t: {avg_timestep:.0f}'

        fig = create_reconstruction_figure(
            original=original,
            generated=generated,
            title=title,
            timesteps=timesteps,
            mask=data.get('mask'),
            max_samples=8,
        )

        filepath = os.path.join(self.save_dir, f'worst_val_batch_epoch_{epoch:04d}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')

        self.writer.add_figure('Validation/worst_batch', fig, epoch)
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

            # Handle tuple (images, seg) format from extract_slices_dual
            if isinstance(data, tuple):
                images, seg_arr = data
                if isinstance(images, np.ndarray):
                    images = torch.from_numpy(images).float()
                else:
                    images = images.float()
                if isinstance(seg_arr, np.ndarray):
                    seg_arr = torch.from_numpy(seg_arr).float()
                else:
                    seg_arr = seg_arr.float()
                # Combine into [C, H, W] format with seg as last channel
                tensor = torch.cat([images, seg_arr], dim=0)
            else:
                tensor = torch.from_numpy(data).float() if hasattr(data, '__array__') else torch.tensor(data).float()

            seg = tensor[seg_channel_idx:seg_channel_idx + 1, :, :]

            if seg.sum() > 0:
                seg_masks.append(seg)
                if return_images and seg_channel_idx > 0:
                    # Image channels are all channels before seg
                    # seg_channel_idx=1 -> 1 image channel, seg_channel_idx=2 -> 2 image channels
                    gt_images.append(tensor[0:seg_channel_idx, :, :])
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
        num_steps: int,
        mode_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate samples while saving intermediate denoising steps.

        Args:
            model: The diffusion model.
            model_input: Initial noisy input (may include condition channels).
            num_steps: Number of denoising steps.
            mode_id: Optional mode ID tensor for multi-modality models.

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
                if mode_id is not None:
                    pred = model(full_input, t, mode_id=mode_id)
                else:
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
                mode_id = None  # For multi-modality modes

                if self.mode_name == ModeType.SEG:
                    noise = torch.randn((num_samples, 1, self.image_size, self.image_size), device=self.device)
                    model_input = noise

                elif self.mode_name == ModeType.BRAVO:
                    need_gt = self.log_msssim or self.log_psnr or self.log_lpips
                    if need_gt:
                        seg_masks, gt_images = self._sample_positive_masks(
                            train_dataset, num_samples, seg_channel_idx=1, return_images=True
                        )
                    else:
                        seg_masks = self._sample_positive_masks(train_dataset, num_samples, seg_channel_idx=1)

                    if self.use_controlnet and self.space.scale_factor > 1:
                        # ControlNet + latent: noise directly in latent space, seg_masks in pixel space
                        latent_channels = self.space.latent_channels
                        latent_size = self.image_size // self.space.scale_factor
                        noise = torch.randn((num_samples, latent_channels, latent_size, latent_size), device=self.device)
                        model_input = noise
                    else:
                        noise = torch.randn_like(seg_masks, device=self.device)
                        model_input = torch.cat([noise, seg_masks], dim=1)

                elif self.mode_name == ModeType.DUAL:
                    need_gt = self.log_msssim or self.log_psnr or self.log_lpips
                    if need_gt:
                        seg_masks, gt_images = self._sample_positive_masks(
                            train_dataset, num_samples, seg_channel_idx=2, return_images=True
                        )
                    else:
                        seg_masks = self._sample_positive_masks(train_dataset, num_samples, seg_channel_idx=2)

                    if self.use_controlnet and self.space.scale_factor > 1:
                        # ControlNet + latent: noise directly in latent space (2 modalities)
                        latent_channels = self.space.latent_channels * 2  # 2 modalities
                        latent_size = self.image_size // self.space.scale_factor
                        noise = torch.randn((num_samples, latent_channels, latent_size, latent_size), device=self.device)
                        model_input = noise
                    else:
                        noise_pre = torch.randn_like(seg_masks, device=self.device)
                        noise_gd = torch.randn_like(seg_masks, device=self.device)
                        model_input = torch.cat([noise_pre, noise_gd, seg_masks], dim=1)

                elif self.mode_name in (ModeType.MULTI, ModeType.MULTI_MODALITY):
                    # Multi-modality mode: single channel output with mode_id conditioning
                    # For visualization, generate samples for bravo modality (mode_id=0)
                    need_gt = self.log_msssim or self.log_psnr or self.log_lpips
                    if need_gt:
                        seg_masks, gt_images = self._sample_positive_masks(
                            train_dataset, num_samples, seg_channel_idx=1, return_images=True
                        )
                    else:
                        seg_masks = self._sample_positive_masks(train_dataset, num_samples, seg_channel_idx=1)

                    if self.use_controlnet and self.space.scale_factor > 1:
                        # ControlNet + latent: noise directly in latent space
                        latent_channels = self.space.latent_channels
                        latent_size = self.image_size // self.space.scale_factor
                        noise = torch.randn((num_samples, latent_channels, latent_size, latent_size), device=self.device)
                        model_input = noise
                    else:
                        noise = torch.randn_like(seg_masks, device=self.device)
                        model_input = torch.cat([noise, seg_masks], dim=1)
                    # Use bravo modality (mode_id=0) for visualization
                    mode_id = torch.zeros(num_samples, dtype=torch.long, device=self.device)

                else:
                    raise ValueError(f"Unknown mode: {self.mode_name}")

                # For ControlNet, wrap model to bind conditioning
                gen_model = model
                if self.use_controlnet and seg_masks is not None:
                    gen_model = ControlNetGenerationWrapper(model, seg_masks)

                with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    if self.log_intermediate_steps and self.mode_name != ModeType.SEG:
                        samples, intermediates = self._generate_with_intermediate_steps(
                            gen_model, model_input, self.num_timesteps, mode_id=mode_id
                        )
                    else:
                        samples = self.strategy.generate(
                            gen_model, model_input, num_steps=self.num_timesteps, device=self.device,
                            mode_id=mode_id
                        )

                # Decode from latent space to pixel space for visualization
                if self.space.scale_factor > 1:
                    samples = self.space.decode(samples)
                    if intermediates is not None:
                        intermediates = [self.space.decode(inter) for inter in intermediates]

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

                    # Note: Validation metrics (MS-SSIM, PSNR, LPIPS) are logged via
                    # the unified metrics system in trainer.py using Validation/ prefix
                else:
                    samples_float = samples.float()
                    samples_normalized = torch.clamp(samples_float, 0, 1)

                    if samples_normalized.dim() == 3:
                        samples_normalized = samples_normalized.unsqueeze(1)

                    samples_rgb = samples_normalized.repeat(1, 3, 1, 1)
                    self.writer.add_images('Generated_Images', samples_rgb, epoch)

                    # Note: Validation metrics (MS-SSIM, PSNR, LPIPS) are logged via
                    # the unified metrics system in trainer.py using Validation/ prefix

        except Exception as e:
            if self.is_main_process:
                logger.warning(
                    f"Failed to generate validation samples at epoch {epoch}: {e}",
                    exc_info=True
                )
        finally:
            # Reset model to train mode (was set to eval at start)
            model.train()
            torch.cuda.empty_cache()
