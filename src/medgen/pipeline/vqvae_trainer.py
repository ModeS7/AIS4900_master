"""
VQ-VAE trainer module for training vector-quantized autoencoders.

This module provides the VQVAETrainer class which inherits from BaseCompressionTrainer
and implements VQ-VAE-specific functionality:
- Vector quantization with discrete latent codes
- VQ loss (commitment + codebook loss)
- VQVAE model creation
"""
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from monai.networks.nets import VQVAE

from .compression_trainer import BaseCompressionTrainer
from .metrics import (
    compute_lpips,
    compute_msssim,
    compute_psnr,
    create_reconstruction_figure,
    RegionalMetricsTracker,
)
from .tracking import create_worst_batch_figure
from .utils import create_epoch_iterator, get_vram_usage

logger = logging.getLogger(__name__)


def log_vqvae_epoch_summary(
    epoch: int,
    total_epochs: int,
    avg_losses: Dict[str, float],
    val_metrics: Dict[str, float],
    elapsed_time: float,
) -> None:
    """Log VQ-VAE epoch completion summary."""
    timestamp = time.strftime("%H:%M:%S")
    epoch_pct = ((epoch + 1) / total_epochs) * 100

    val_gen = f"(v:{val_metrics.get('gen', 0):.4f})" if val_metrics else ""
    val_l1 = f"(v:{val_metrics.get('l1', 0):.4f})" if val_metrics else ""
    msssim_str = f"MS-SSIM: {val_metrics.get('msssim', 0):.3f}" if val_metrics.get('msssim') else ""

    logger.info(
        f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
        f"G: {avg_losses['gen']:.4f}{val_gen} | "
        f"L1: {avg_losses['recon']:.4f}{val_l1} | "
        f"VQ: {avg_losses['vq']:.4f} | "
        f"D: {avg_losses['disc']:.4f} | "
        f"{msssim_str} | "
        f"Time: {elapsed_time:.1f}s"
    )


class VQVAETrainer(BaseCompressionTrainer):
    """VQ-VAE trainer with discrete latent space.

    Inherits from BaseCompressionTrainer and adds:
    - Vector quantization (VQ) loss
    - VQVAE model creation from MONAI
    - VQ-specific forward pass returning (reconstruction, vq_loss)

    Args:
        cfg: Hydra configuration object.

    Example:
        >>> trainer = VQVAETrainer(cfg)
        >>> trainer.setup_model()
        >>> trainer.train(train_loader, train_dataset, val_loader)
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize VQ-VAE trainer.

        Args:
            cfg: Hydra configuration object.
        """
        super().__init__(cfg)

        # ─────────────────────────────────────────────────────────────────────
        # VQ-VAE-specific config
        # ─────────────────────────────────────────────────────────────────────
        self.num_embeddings: int = cfg.vqvae.get('num_embeddings', 512)
        self.embedding_dim: int = cfg.vqvae.get('embedding_dim', 64)
        self.commitment_cost: float = cfg.vqvae.get('commitment_cost', 0.25)
        self.decay: float = cfg.vqvae.get('decay', 0.99)
        self.epsilon: float = cfg.vqvae.get('epsilon', 1e-5)

        # Architecture config
        self.channels: Tuple[int, ...] = tuple(cfg.vqvae.get('channels', [96, 96, 192]))
        self.num_res_layers: int = cfg.vqvae.get('num_res_layers', 3)
        self.num_res_channels: Tuple[int, ...] = tuple(
            cfg.vqvae.get('num_res_channels', [96, 96, 192])
        )
        self.downsample_parameters: Tuple[Tuple[int, ...], ...] = tuple(
            tuple(p) for p in cfg.vqvae.get('downsample_parameters', [[2, 4, 1, 1]] * 3)
        )
        self.upsample_parameters: Tuple[Tuple[int, ...], ...] = tuple(
            tuple(p) for p in cfg.vqvae.get('upsample_parameters', [[2, 4, 1, 1, 0]] * 3)
        )

        self.image_size: int = cfg.model.image_size

    def _create_fallback_save_dir(self) -> str:
        """Create fallback save directory for VQ-VAE."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = self.cfg.training.get('name', '')
        mode_name = self.cfg.mode.get('name', 'dual')
        run_name = f"{exp_name}{self.image_size}_{timestamp}"
        return os.path.join(self.cfg.paths.model_dir, 'vqvae_2d', mode_name, run_name)

    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize VQ-VAE model, discriminator, optimizers, and loss functions.

        Args:
            pretrained_checkpoint: Optional path to checkpoint for loading
                pretrained weights.
        """
        n_channels = self.cfg.mode.get('in_channels', 1)

        # Create VQVAE
        raw_model = VQVAE(
            spatial_dims=2,
            in_channels=n_channels,
            out_channels=n_channels,
            channels=self.channels,
            num_res_layers=self.num_res_layers,
            num_res_channels=self.num_res_channels,
            downsample_parameters=self.downsample_parameters,
            upsample_parameters=self.upsample_parameters,
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            commitment_cost=self.commitment_cost,
            decay=self.decay,
            epsilon=self.epsilon,
            ddp_sync=self.use_multi_gpu,
        ).to(self.device)

        # Create discriminator if GAN enabled
        raw_disc = None
        if not self.disable_gan:
            raw_disc = self._create_discriminator(n_channels, spatial_dims=2)

        # Load pretrained weights if provided
        if pretrained_checkpoint:
            self._load_pretrained_weights(raw_model, raw_disc, pretrained_checkpoint)

        # Wrap models with DDP/compile
        self._wrap_models(raw_model, raw_disc)

        # Setup perceptual and adversarial loss
        self.perceptual_loss_fn = self._create_perceptual_loss(spatial_dims=2)
        if not self.disable_gan:
            from monai.losses import PatchAdversarialLoss
            self.adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

        # Setup optimizers and schedulers
        self._setup_optimizers(n_channels)

        # Setup EMA
        self._setup_ema()

        # Save metadata
        if self.is_main_process:
            self._save_metadata()

        # Log model info
        if self.is_main_process:
            vqvae_params = sum(p.numel() for p in self.model_raw.parameters())
            logger.info(f"VQ-VAE initialized: {vqvae_params / 1e6:.1f}M parameters")
            logger.info(f"  Codebook: {self.num_embeddings} embeddings x {self.embedding_dim} dim")
            logger.info(f"  Channels: {self.channels}")
            if not self.disable_gan:
                disc_params = sum(p.numel() for p in self.discriminator_raw.parameters())
                logger.info(f"Discriminator initialized: {disc_params / 1e6:.1f}M parameters")
            else:
                logger.info("GAN disabled - discriminator not created")

            # Compute latent size
            n_downsamples = len(self.downsample_parameters)
            latent_size = self.image_size // (2 ** n_downsamples)
            logger.info(f"Latent shape: [{self.embedding_dim}, {latent_size}, {latent_size}]")
            logger.info(f"Loss weights - Perceptual: {self.perceptual_weight}, Adv: {self.adv_weight}")
            logger.info(f"VQ params - Commitment: {self.commitment_cost}, Decay: {self.decay}")

    def _load_pretrained_weights(
        self,
        raw_model: nn.Module,
        raw_disc: Optional[nn.Module],
        checkpoint_path: str,
    ) -> None:
        """Load pretrained weights from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                raw_model.load_state_dict(checkpoint['model_state_dict'])
                if self.is_main_process:
                    logger.info(f"Loaded VQ-VAE weights from {checkpoint_path}")
            if 'discriminator_state_dict' in checkpoint and raw_disc is not None:
                raw_disc.load_state_dict(checkpoint['discriminator_state_dict'])
                if self.is_main_process:
                    logger.info(f"Loaded discriminator weights from {checkpoint_path}")
        except FileNotFoundError:
            if self.is_main_process:
                logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")

    def _save_metadata(self) -> None:
        """Save training configuration and VQ-VAE config."""
        os.makedirs(self.save_dir, exist_ok=True)

        # Save full config
        config_path = os.path.join(self.save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        # Save VQ-VAE config separately
        n_channels = self.cfg.mode.get('in_channels', 1)
        vqvae_config = self._get_model_config()

        metadata_path = os.path.join(self.save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'type': 'vqvae',
                'vqvae_config': vqvae_config,
                'image_size': self.image_size,
                'n_epochs': self.n_epochs,
            }, f, indent=2)

    def _prepare_batch(self, batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare batch for VQ-VAE training.

        Handles multiple batch formats:
        - Tuple of (images, mask)
        - Dict with image keys
        - Single tensor

        Args:
            batch: Input batch.

        Returns:
            Tuple of (images, mask).
        """
        # Handle tuple of (image, seg)
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            images, mask = batch
            if hasattr(images, 'as_tensor'):
                images = images.as_tensor()
            if hasattr(mask, 'as_tensor'):
                mask = mask.as_tensor()
            return images.to(self.device), mask.to(self.device)

        # Handle dict batches
        if isinstance(batch, dict):
            image_keys = self.cfg.mode.get('image_keys', ['t1_pre', 't1_gd'])
            tensors = []
            for key in image_keys:
                if key in batch:
                    tensors.append(batch[key].to(self.device))
            images = torch.cat(tensors, dim=1)
            mask = batch['seg'].to(self.device) if 'seg' in batch else None
            return images, mask

        # Handle tensor input
        if hasattr(batch, 'as_tensor'):
            tensor = batch.as_tensor().to(self.device)
        else:
            tensor = batch.to(self.device)

        # Check if seg is stacked as last channel
        n_image_channels = self.cfg.mode.get('in_channels', 2)
        if tensor.shape[1] > n_image_channels:
            images = tensor[:, :n_image_channels, :, :]
            mask = tensor[:, n_image_channels:n_image_channels + 1, :, :]
            return images, mask

        return tensor, None

    def train_step(self, batch: Any) -> Dict[str, float]:
        """Execute VQ-VAE training step.

        Args:
            batch: Input batch.

        Returns:
            Dict with losses: 'gen', 'disc', 'recon', 'perc', 'vq', 'adv'.
        """
        images, mask = self._prepare_batch(batch)
        grad_clip = self.cfg.training.get('gradient_clip_norm', 1.0)

        d_loss = torch.tensor(0.0, device=self.device)
        adv_loss = torch.tensor(0.0, device=self.device)

        # ==================== Discriminator Step ====================
        if not self.disable_gan:
            with torch.no_grad():
                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    reconstruction_for_d, _ = self.model(images)

            d_loss = self._train_discriminator_step(images, reconstruction_for_d)

        # ==================== Generator Step ====================
        self.optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            # VQVAE forward returns (reconstruction, vq_loss)
            reconstruction, vq_loss = self.model(images)

            # L1 reconstruction loss
            l1_loss = torch.abs(reconstruction - images).mean()

            # Perceptual loss
            p_loss = self._compute_perceptual_loss(reconstruction, images)

            # Adversarial loss
            if not self.disable_gan:
                adv_loss = self._compute_adversarial_loss(reconstruction)

            # Total generator loss
            # Note: vq_loss already includes commitment_cost weighting from the model
            g_loss = (
                l1_loss
                + self.perceptual_weight * p_loss
                + vq_loss
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

        return {
            'gen': g_loss.item(),
            'disc': d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss,
            'recon': l1_loss.item(),
            'perc': p_loss.item(),
            'vq': vq_loss.item(),
            'adv': adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
        }

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train VQ-VAE for one epoch.

        Args:
            data_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dict with average losses.
        """
        self.model.train()
        if not self.disable_gan and self.discriminator is not None:
            self.discriminator.train()

        epoch_losses = {'gen': 0, 'disc': 0, 'recon': 0, 'perc': 0, 'vq': 0, 'adv': 0}

        epoch_iter = create_epoch_iterator(
            data_loader, epoch, self.is_cluster, self.is_main_process,
            limit_batches=self.limit_train_batches
        )

        for step, batch in enumerate(epoch_iter):
            losses = self.train_step(batch)

            for key in epoch_losses:
                epoch_losses[key] += losses[key]

            if hasattr(epoch_iter, 'set_postfix'):
                epoch_iter.set_postfix(
                    G=f"{epoch_losses['gen'] / (step + 1):.4f}",
                    VQ=f"{epoch_losses['vq'] / (step + 1):.4f}",
                    D=f"{epoch_losses['disc'] / (step + 1):.4f}"
                )

            if epoch == 1 and step == 0 and self.is_main_process:
                logger.info(get_vram_usage(self.device))

        # Average losses
        n_batches = self.limit_train_batches if self.limit_train_batches else len(data_loader)
        avg_losses = {key: val / n_batches for key, val in epoch_losses.items()}

        # Log training metrics (single-GPU only)
        if self.writer is not None and self.is_main_process and not self.use_multi_gpu:
            self.writer.add_scalar('Loss/Generator_train', avg_losses['gen'], epoch)
            self.writer.add_scalar('Loss/L1_train', avg_losses['recon'], epoch)
            self.writer.add_scalar('Loss/Perceptual_train', avg_losses['perc'], epoch)
            self.writer.add_scalar('Loss/VQ_train', avg_losses['vq'], epoch)

            if not self.disable_gan:
                self.writer.add_scalar('Loss/Discriminator', avg_losses['disc'], epoch)
                self.writer.add_scalar('Loss/Adversarial', avg_losses['adv'], epoch)

        return avg_losses

    def _forward_for_validation(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """VQ-VAE forward pass for validation.

        Returns:
            Tuple of (reconstruction, vq_loss).
        """
        reconstruction, vq_loss = model(images)
        return reconstruction, vq_loss

    def _get_model_config(self) -> Dict[str, Any]:
        """Get VQ-VAE model configuration for checkpoint."""
        n_channels = self.cfg.mode.get('in_channels', 1)
        return {
            'in_channels': n_channels,
            'out_channels': n_channels,
            'channels': list(self.channels),
            'num_res_layers': self.num_res_layers,
            'num_res_channels': list(self.num_res_channels),
            'downsample_parameters': [list(p) for p in self.downsample_parameters],
            'upsample_parameters': [list(p) for p in self.upsample_parameters],
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embedding_dim,
            'commitment_cost': self.commitment_cost,
            'decay': self.decay,
            'epsilon': self.epsilon,
        }

    def _log_validation_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float],
        worst_batch_data: Optional[Dict[str, Any]],
        regional_tracker: Optional[RegionalMetricsTracker],
        log_figures: bool,
    ) -> None:
        """Log VQ-VAE validation metrics including VQ loss."""
        if self.writer is None:
            return

        # Call base class method first
        super()._log_validation_metrics(epoch, metrics, worst_batch_data, regional_tracker, log_figures)

        # Add VQ-VAE-specific VQ loss logging
        if 'reg' in metrics:
            # Log as VQ_val (reg is the VQ loss for VQVAE)
            self.writer.add_scalar('Loss/VQ_val', metrics['reg'], epoch)

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        avg_losses: Dict[str, float],
        val_metrics: Dict[str, float],
        elapsed_time: float,
    ) -> None:
        """Log VQ-VAE epoch summary."""
        log_vqvae_epoch_summary(epoch, total_epochs, avg_losses, val_metrics, elapsed_time)

    def evaluate_test_set(
        self,
        test_loader: DataLoader,
        checkpoint_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate VQ-VAE on test set.

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

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test evaluation", ncols=100, disable=self.is_cluster):
                images, mask = self._prepare_batch(batch)
                batch_size = images.shape[0]

                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    reconstructed, _ = model_to_use(images)

                # Compute metrics
                l1_loss = torch.abs(reconstructed - images).mean().item()
                total_l1 += l1_loss
                if self.log_msssim:
                    total_msssim += compute_msssim(reconstructed, images)
                total_psnr += compute_psnr(reconstructed, images)
                total_lpips += compute_lpips(reconstructed, images, device=self.device)

                # Regional tracking
                if regional_tracker is not None and mask is not None:
                    regional_tracker.update(reconstructed, images, mask)

                # Track worst batch
                if l1_loss > worst_loss:
                    worst_loss = l1_loss
                    worst_batch_data = self._capture_worst_batch(
                        images, reconstructed, l1_loss,
                        torch.tensor(l1_loss), torch.tensor(0.0), torch.tensor(0.0)
                    )

                n_batches += 1
                n_samples += batch_size

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

            # Worst batch
            if worst_batch_data is not None:
                fig = self._create_worst_batch_figure(worst_batch_data)
                self.writer.add_figure(f'{tb_prefix}/worst_batch', fig, 0)
                plt.close(fig)

                # Also save as PNG file
                fig = self._create_worst_batch_figure(worst_batch_data)
                fig_path = os.path.join(self.save_dir, f'test_worst_batch_{label}.png')
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Test worst batch saved to: {fig_path}")

        model_to_use.train()
        return metrics
