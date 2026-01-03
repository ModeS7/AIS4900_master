"""
3D VAE trainer module for training volumetric autoencoders.

This module provides the VAE3DTrainer class which inherits from BaseCompression3DTrainer
and implements 3D VAE-specific functionality:
- KL divergence regularization
- 3D AutoencoderKL model creation
- Gradient checkpointing for memory efficiency
"""
import itertools
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Disable MONAI MetaTensor tracking BEFORE importing MONAI modules
from monai.data import set_track_meta
set_track_meta(False)

from monai.networks.nets import AutoencoderKL

from .compression_trainer import BaseCompression3DTrainer
from .metrics import (
    compute_lpips_3d,
    compute_msssim,
    compute_psnr,
    RegionalMetricsTracker3D,
)
from .tracking import create_worst_batch_figure_3d
from .utils import create_epoch_iterator, get_vram_usage

logger = logging.getLogger(__name__)


class CheckpointedAutoencoder(nn.Module):
    """Wrapper that applies gradient checkpointing to MONAI AutoencoderKL.

    MONAI's AutoencoderKL doesn't have built-in gradient checkpointing.
    This wrapper uses torch.utils.checkpoint to trade compute for memory,
    reducing activation memory by ~50% for 3D volumes.

    Args:
        model: The underlying AutoencoderKL model.
    """

    def __init__(self, model: AutoencoderKL):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with gradient checkpointing."""
        def encode_fn(x):
            h = self.model.encoder(x)
            z_mu = self.model.quant_conv_mu(h)
            z_log_var = self.model.quant_conv_log_sigma(h)
            return z_mu, z_log_var

        z_mu, z_log_var = grad_checkpoint(encode_fn, x, use_reentrant=False)
        z = self.model.sampling(z_mu, z_log_var)

        def decode_fn(z):
            z_post = self.model.post_quant_conv(z)
            return self.model.decoder(z_post)

        reconstruction = grad_checkpoint(decode_fn, z, use_reentrant=False)
        return reconstruction, z_mu, z_log_var

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space (for inference, no checkpointing)."""
        return self.model.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to output (for inference, no checkpointing)."""
        return self.model.decode(z)

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation for diffusion model."""
        return self.model.encode_stage_2_inputs(x)

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        """Decode diffusion model outputs."""
        return self.model.decode_stage_2_outputs(z)


def log_vae_3d_epoch_summary(
    epoch: int,
    total_epochs: int,
    avg_losses: Dict[str, float],
    val_metrics: Dict[str, float],
    elapsed_time: float,
) -> None:
    """Log 3D VAE epoch completion summary."""
    timestamp = time.strftime("%H:%M:%S")
    epoch_pct = ((epoch + 1) / total_epochs) * 100

    val_gen = f"(v:{val_metrics.get('gen', 0):.4f})" if val_metrics else ""
    val_l1 = f"(v:{val_metrics.get('l1', 0):.4f})" if val_metrics else ""
    psnr_str = f"PSNR: {val_metrics.get('psnr', 0):.2f}" if val_metrics.get('psnr') else ""

    logger.info(
        f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
        f"G: {avg_losses['gen']:.4f}{val_gen} | "
        f"L1: {avg_losses['recon']:.4f}{val_l1} | "
        f"D: {avg_losses['disc']:.4f} | "
        f"{psnr_str} | "
        f"Time: {elapsed_time:.1f}s"
    )


class VAE3DTrainer(BaseCompression3DTrainer):
    """3D AutoencoderKL trainer with KL divergence regularization.

    Inherits from BaseCompression3DTrainer and adds:
    - KL divergence loss
    - 3D AutoencoderKL model creation
    - Gradient checkpointing for memory efficiency

    Args:
        cfg: Hydra configuration object.

    Example:
        >>> trainer = VAE3DTrainer(cfg)
        >>> trainer.setup_model()
        >>> trainer.train(train_loader, train_dataset, val_loader)
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize 3D VAE trainer.

        Args:
            cfg: Hydra configuration object.
        """
        super().__init__(cfg)

        # ─────────────────────────────────────────────────────────────────────
        # VAE-specific config
        # ─────────────────────────────────────────────────────────────────────
        self.kl_weight: float = cfg.vae_3d.get('kl_weight', 1e-6)
        self.latent_channels: int = cfg.vae_3d.latent_channels
        self.vae_channels: Tuple[int, ...] = tuple(cfg.vae_3d.channels)
        self.attention_levels: Tuple[bool, ...] = tuple(cfg.vae_3d.attention_levels)
        self.num_res_blocks: int = cfg.vae_3d.get('num_res_blocks', 2)

    def _get_disc_lr(self, cfg: DictConfig) -> float:
        """Get discriminator LR from vae_3d config."""
        return cfg.vae_3d.get('disc_lr', 1e-4)

    def _get_perceptual_weight(self, cfg: DictConfig) -> float:
        """Get perceptual weight from vae_3d config."""
        return cfg.vae_3d.get('perceptual_weight', 0.001)

    def _get_adv_weight(self, cfg: DictConfig) -> float:
        """Get adversarial weight from vae_3d config."""
        return cfg.vae_3d.get('adv_weight', 0.01)

    def _get_disable_gan(self, cfg: DictConfig) -> bool:
        """Determine if GAN is disabled from vae_3d config."""
        return cfg.vae_3d.get('disable_gan', False)

    def _create_fallback_save_dir(self) -> str:
        """Create fallback save directory for 3D VAE."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = self.cfg.training.get('name', '')
        return os.path.join(
            self.cfg.paths.model_dir, "compression_3d", self.cfg.mode.name,
            f"{exp_name}{self.volume_height}x{self.volume_depth}_{timestamp}"
        )

    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize 3D VAE model, discriminator, optimizers, and loss functions.

        Args:
            pretrained_checkpoint: Optional path to checkpoint for loading
                pretrained weights.
        """
        n_channels = self.cfg.mode.get('in_channels', 1)

        # Create 3D AutoencoderKL
        base_model = AutoencoderKL(
            spatial_dims=3,
            in_channels=n_channels,
            out_channels=n_channels,
            channels=self.vae_channels,
            attention_levels=self.attention_levels,
            latent_channels=self.latent_channels,
            num_res_blocks=self.num_res_blocks,
            norm_num_groups=32,
            with_encoder_nonlocal_attn=False,  # Disabled for 3D (memory)
            with_decoder_nonlocal_attn=False,
        ).to(self.device)

        # Load pretrained weights BEFORE wrapping
        if pretrained_checkpoint:
            self._load_pretrained_weights_base(base_model, pretrained_checkpoint)

        # Wrap with gradient checkpointing for memory efficiency
        if self.gradient_checkpointing:
            raw_model = CheckpointedAutoencoder(base_model)
            if self.is_main_process:
                logger.info("Gradient checkpointing enabled (CheckpointedAutoencoder wrapper)")
        else:
            raw_model = base_model

        # Create 3D discriminator if GAN enabled
        raw_disc = None
        if not self.disable_gan:
            raw_disc = self._create_discriminator(n_channels, spatial_dims=3)

        # Wrap models with DDP/compile
        self._wrap_models(raw_model, raw_disc)

        # Setup perceptual and adversarial loss
        if self.use_2_5d_perceptual and self.perceptual_weight > 0:
            self.perceptual_loss_fn = self._create_perceptual_loss(spatial_dims=2)  # 2D for 2.5D
        if not self.disable_gan:
            from monai.losses import PatchAdversarialLoss
            self.adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

        # Setup optimizers and schedulers
        self._setup_optimizers(n_channels)

        # Setup EMA (WARNING: doubles memory for 3D models)
        self._setup_ema()

        # Save metadata
        if self.is_main_process:
            self._save_metadata()

        # Log model info
        if self.is_main_process:
            vae_params = sum(p.numel() for p in self.model_raw.parameters())
            logger.info(f"3D VAE initialized: {vae_params / 1e6:.1f}M parameters")
            logger.info(f"  Channels: {self.vae_channels}")
            logger.info(f"  Latent channels: {self.latent_channels}")
            if not self.disable_gan:
                disc_params = sum(p.numel() for p in self.discriminator_raw.parameters())
                logger.info(f"3D Discriminator: {disc_params / 1e6:.1f}M parameters")
            else:
                logger.info("GAN disabled")
            logger.info(f"Volume: {self.volume_width}x{self.volume_height}x{self.volume_depth}")

            # Compute latent shape
            n_downsamples = len(self.vae_channels) - 1
            latent_h = self.volume_height // (2 ** n_downsamples)
            latent_w = self.volume_width // (2 ** n_downsamples)
            latent_d = self.volume_depth // (2 ** n_downsamples)
            logger.info(f"Latent shape: [{self.latent_channels}, {latent_d}, {latent_h}, {latent_w}]")

    def _load_pretrained_weights_base(
        self,
        base_model: nn.Module,
        checkpoint_path: str,
    ) -> None:
        """Load pretrained weights into base model (before checkpointing wrapper)."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Remove 'model.' prefix if present (from CheckpointedAutoencoder)
                if any(k.startswith('model.') for k in state_dict.keys()):
                    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                base_model.load_state_dict(state_dict)
                if self.is_main_process:
                    logger.info(f"Loaded 3D VAE weights from {checkpoint_path}")
        except FileNotFoundError:
            if self.is_main_process:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")

    def _save_metadata(self) -> None:
        """Save training configuration and 3D VAE config."""
        os.makedirs(self.save_dir, exist_ok=True)

        # Save full config
        config_path = os.path.join(self.save_dir, 'config.yaml')
        OmegaConf.save(self.cfg, config_path)

        n_channels = self.cfg.mode.get('in_channels', 1)
        metadata = {
            'type': 'vae_3d',
            'vae_config': self._get_model_config(),
            'volume': {
                'height': self.volume_height,
                'width': self.volume_width,
                'depth': self.volume_depth,
            },
            'n_epochs': self.n_epochs,
        }

        metadata_path = os.path.join(self.save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _compute_kl_loss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss.

        Args:
            mean: Mean of latent distribution.
            logvar: Log variance of latent distribution.

        Returns:
            KL divergence loss (scalar).
        """
        kl = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).mean()
        return kl

    def _prepare_batch(self, batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare batch for 3D VAE training.

        Args:
            batch: Input batch.

        Returns:
            Tuple of (images, mask).
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

        images = images.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        return images, mask

    def train_step(self, batch: Any) -> Dict[str, float]:
        """Execute 3D VAE training step with KL loss.

        Args:
            batch: Input batch.

        Returns:
            Dict with losses: 'gen', 'disc', 'recon', 'perc', 'kl', 'adv'.
        """
        images, _ = self._prepare_batch(batch)
        grad_clip = self.cfg.training.get('gradient_clip_norm', 1.0)

        d_loss = torch.tensor(0.0, device=self.device)
        adv_loss = torch.tensor(0.0, device=self.device)

        # ==================== Generator Step ====================
        self.optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            reconstruction, mean, logvar = self.model(images)

            # L1 reconstruction loss
            l1_loss = torch.nn.functional.l1_loss(reconstruction, images)

            # KL divergence loss
            kl_loss = self._compute_kl_loss(mean, logvar)

            # Perceptual loss (2.5D)
            if self.perceptual_weight > 0 and self.perceptual_loss_fn is not None:
                p_loss = self._compute_2_5d_perceptual_loss(reconstruction, images)
            else:
                p_loss = torch.tensor(0.0, device=self.device)

            # Adversarial loss
            if not self.disable_gan:
                adv_loss = self._compute_adversarial_loss(reconstruction)

            # Total generator loss
            g_loss = (
                l1_loss
                + self.perceptual_weight * p_loss
                + self.kl_weight * kl_loss
                + self.adv_weight * adv_loss
            )

        g_loss.backward()

        # Gradient clipping
        grad_norm_g = 0.0
        if grad_clip > 0:
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                self.model_raw.parameters(), max_norm=grad_clip
            ).item()

        self.optimizer.step()

        # Track gradient norm
        if self.log_grad_norm:
            self._grad_norm_tracker.update(grad_norm_g)

        # Update EMA
        self._update_ema()

        # ==================== Discriminator Step ====================
        if not self.disable_gan:
            d_loss = self._train_discriminator_step(images, reconstruction.detach())

        return {
            'gen': g_loss.item(),
            'disc': d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss,
            'recon': l1_loss.item(),
            'perc': p_loss.item() if isinstance(p_loss, torch.Tensor) else p_loss,
            'kl': kl_loss.item(),
            'adv': adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
        }

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train 3D VAE for one epoch.

        Args:
            data_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dict with average losses.
        """
        self.model.train()
        if not self.disable_gan and self.discriminator is not None:
            self.discriminator.train()

        epoch_losses = {'gen': 0, 'disc': 0, 'recon': 0, 'perc': 0, 'kl': 0, 'adv': 0}

        disable_pbar = not self.is_main_process or self.is_cluster
        total = self.limit_train_batches if self.limit_train_batches else len(data_loader)
        iterator = itertools.islice(data_loader, self.limit_train_batches) if self.limit_train_batches else data_loader
        pbar = tqdm(iterator, desc=f"Epoch {epoch}", disable=disable_pbar, total=total)

        for step, batch in enumerate(pbar):
            losses = self.train_step(batch)

            for key in epoch_losses:
                epoch_losses[key] += losses[key]

            if not disable_pbar:
                if not self.disable_gan:
                    pbar.set_postfix(
                        G=f"{losses['gen']:.4f}",
                        D=f"{losses['disc']:.4f}",
                        L1=f"{losses['recon']:.4f}",
                    )
                else:
                    pbar.set_postfix(
                        G=f"{losses['gen']:.4f}",
                        L1=f"{losses['recon']:.4f}",
                        KL=f"{losses['kl']:.4f}",
                    )

        # Average losses
        n_batches = self.limit_train_batches if self.limit_train_batches else len(data_loader)
        avg_losses = {key: val / n_batches for key, val in epoch_losses.items()}

        # Log training metrics (single-GPU only)
        if self.writer is not None and self.is_main_process and not self.use_multi_gpu:
            self.writer.add_scalar('Loss/Generator_train', avg_losses['gen'], epoch)
            self.writer.add_scalar('Loss/L1_train', avg_losses['recon'], epoch)
            self.writer.add_scalar('Loss/Perceptual_train', avg_losses['perc'], epoch)
            self.writer.add_scalar('Loss/KL_train', avg_losses['kl'], epoch)

            if not self.disable_gan:
                self.writer.add_scalar('Loss/Discriminator', avg_losses['disc'], epoch)
                self.writer.add_scalar('Loss/Adversarial', avg_losses['adv'], epoch)

        return avg_losses

    def _forward_for_validation(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """3D VAE forward pass for validation.

        Returns:
            Tuple of (reconstruction, weighted_kl_loss).
        """
        reconstruction, mean, logvar = model(images)
        kl_loss = self._compute_kl_loss(mean, logvar)
        return reconstruction, self.kl_weight * kl_loss

    def _get_model_config(self) -> Dict[str, Any]:
        """Get 3D VAE model configuration for checkpoint."""
        n_channels = self.cfg.mode.get('in_channels', 1)
        return {
            'in_channels': n_channels,
            'out_channels': n_channels,
            'latent_channels': self.latent_channels,
            'channels': list(self.vae_channels),
            'attention_levels': list(self.attention_levels),
            'num_res_blocks': self.num_res_blocks,
            'spatial_dims': 3,
        }

    def _log_validation_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float],
        worst_batch_data: Optional[Dict[str, Any]],
        regional_tracker: Optional[RegionalMetricsTracker3D],
        log_figures: bool,
    ) -> None:
        """Log 3D VAE validation metrics including KL loss."""
        if self.writer is None:
            return

        # Call base class method first
        super()._log_validation_metrics(epoch, metrics, worst_batch_data, regional_tracker, log_figures)

        # Add VAE-specific KL loss logging
        if 'reg' in metrics and self.kl_weight > 0:
            self.writer.add_scalar('Loss/KL_val', metrics['reg'] / self.kl_weight, epoch)

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
        elapsed_time: float,
    ) -> None:
        """Log 3D VAE epoch summary."""
        log_vae_3d_epoch_summary(epoch, total_epochs, avg_losses, val_metrics, elapsed_time)

    def evaluate_test_set(
        self,
        test_loader: DataLoader,
        checkpoint_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate 3D VAE on test set.

        Args:
            test_loader: Test data loader.
            checkpoint_name: Checkpoint to load ("best", "latest", or None).

        Returns:
            Dict with test metrics.
        """
        if not self.is_main_process:
            return {}

        # Load checkpoint if specified
        if checkpoint_name is not None:
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{checkpoint_name}.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model_raw.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded {checkpoint_name} checkpoint for test evaluation")
            else:
                logger.warning(f"Checkpoint {checkpoint_path} not found")
                checkpoint_name = "current"

        label = checkpoint_name or "current"
        logger.info("=" * 60)
        logger.info(f"EVALUATING ON TEST SET ({label.upper()} MODEL)")
        logger.info("=" * 60)

        model_to_use = self._get_model_for_eval() if checkpoint_name is None else self.model_raw
        model_to_use.eval()

        # Accumulators
        total_l1 = 0.0
        total_msssim = 0.0
        total_psnr = 0.0
        total_lpips = 0.0
        n_batches = 0
        n_samples = 0

        # Regional tracker
        regional_tracker = None
        if self.log_regional_losses:
            regional_tracker = self._create_regional_tracker()

        # Worst batch tracking
        worst_loss = 0.0
        worst_batch_data = None

        # Sample collection
        sample_inputs = []
        sample_outputs = []
        max_vis_samples = 4

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test evaluation", ncols=100, disable=self.is_cluster):
                images, mask = self._prepare_batch(batch)
                batch_size = images.shape[0]

                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    reconstructed, _, _ = model_to_use(images)

                # Compute metrics
                l1_loss = torch.abs(reconstructed - images).mean().item()
                total_l1 += l1_loss
                if self.log_msssim:
                    total_msssim += compute_msssim(reconstructed.float(), images.float(), spatial_dims=3)
                total_psnr += compute_psnr(reconstructed, images)
                total_lpips += compute_lpips_3d(reconstructed.float(), images.float(), device=self.device)

                # Regional tracking
                if regional_tracker is not None and mask is not None:
                    regional_tracker.update(reconstructed.float(), images.float(), mask)

                # Track worst batch
                if l1_loss > worst_loss:
                    worst_loss = l1_loss
                    worst_batch_data = {
                        'original': images.cpu(),
                        'generated': reconstructed.float().cpu(),
                        'loss': l1_loss,
                    }

                n_batches += 1
                n_samples += batch_size

                # Collect samples
                if len(sample_inputs) < max_vis_samples:
                    remaining = max_vis_samples - len(sample_inputs)
                    sample_inputs.append(images[:remaining].cpu())
                    sample_outputs.append(reconstructed[:remaining].float().cpu())

        # Compute averages
        metrics = {
            'l1': total_l1 / n_batches,
            'psnr': total_psnr / n_batches,
            'lpips': total_lpips / n_batches,
            'n_samples': n_samples,
        }
        if self.log_msssim:
            metrics['msssim'] = total_msssim / n_batches

        # Log results
        logger.info(f"Test Results - {label} ({n_samples} samples):")
        logger.info(f"  L1 Loss: {metrics['l1']:.6f}")
        if 'msssim' in metrics:
            logger.info(f"  MS-SSIM: {metrics['msssim']:.4f}")
        logger.info(f"  PSNR:    {metrics['psnr']:.2f} dB")
        logger.info(f"  LPIPS:   {metrics['lpips']:.4f}")

        # Save results
        results_path = os.path.join(self.save_dir, f'test_results_{label}.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Log to TensorBoard
        tb_prefix = f'test_{label}'
        if self.writer is not None:
            self.writer.add_scalar(f'{tb_prefix}/L1', metrics['l1'], 0)
            self.writer.add_scalar(f'{tb_prefix}/PSNR', metrics['psnr'], 0)
            self.writer.add_scalar(f'{tb_prefix}/LPIPS', metrics['lpips'], 0)
            if 'msssim' in metrics:
                self.writer.add_scalar(f'{tb_prefix}/MS-SSIM', metrics['msssim'], 0)

            if regional_tracker is not None:
                regional_tracker.log_to_tensorboard(self.writer, 0, prefix=f'{tb_prefix}_regional')

            # Worst batch figure
            if worst_batch_data is not None:
                fig = create_worst_batch_figure_3d(
                    original=worst_batch_data['original'],
                    generated=worst_batch_data['generated'],
                    loss=worst_batch_data['loss'],
                )
                self.writer.add_figure(f'{tb_prefix}/worst_batch', fig, 0)
                plt.close(fig)

                # Also save as PNG file
                fig = create_worst_batch_figure_3d(
                    original=worst_batch_data['original'],
                    generated=worst_batch_data['generated'],
                    loss=worst_batch_data['loss'],
                )
                fig_path = os.path.join(self.save_dir, f'test_worst_batch_{label}.png')
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Test worst batch saved to: {fig_path}")

        model_to_use.train()
        return metrics
