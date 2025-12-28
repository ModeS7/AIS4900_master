"""
3D VAE trainer module for training volumetric autoencoders.

Based on MONAI Generative proven configuration for 3D medical imaging.
Key differences from 2D:
- spatial_dims=3
- Smaller channels (memory)
- No attention layers
- 2.5D perceptual loss option
- Gradient checkpointing
"""
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from monai.losses import PatchAdversarialLoss
from monai.networks.nets import AutoencoderKL, PatchDiscriminator

from medgen.core import (
    setup_distributed,
    create_warmup_cosine_scheduler,
    wrap_model_for_training,
)
from .losses import PerceptualLoss
from .utils import (
    get_vram_usage,
    create_epoch_iterator,
    save_full_checkpoint,
)
from .metrics import compute_psnr
from .tracking import GradientNormTracker

logger = logging.getLogger(__name__)


def log_vae_3d_epoch_summary(
    epoch: int,
    total_epochs: int,
    avg_losses: Dict[str, float],
    val_metrics: Dict[str, float],
    elapsed_time: float
) -> None:
    """Log 3D VAE epoch completion summary."""
    timestamp = time.strftime("%H:%M:%S")
    epoch_pct = ((epoch + 1) / total_epochs) * 100

    if val_metrics:
        val_gen = f"(v:{val_metrics.get('gen', 0):.4f})"
        val_l1 = f"(v:{val_metrics.get('l1', 0):.4f})"
        psnr_str = f"PSNR: {val_metrics.get('psnr', 0):.2f}"
    else:
        val_gen = ""
        val_l1 = ""
        psnr_str = ""

    logger.info(
        f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
        f"G: {avg_losses['gen']:.4f}{val_gen} | "
        f"L1: {avg_losses['recon']:.4f}{val_l1} | "
        f"D: {avg_losses['disc']:.4f} | "
        f"{psnr_str} | "
        f"Time: {elapsed_time:.1f}s"
    )


class VAE3DTrainer:
    """3D AutoencoderKL trainer for volumetric medical imaging.

    Based on MONAI Generative proven configuration.

    Args:
        cfg: Hydra configuration object.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # Extract config values
        self.n_epochs: int = cfg.training.epochs
        self.batch_size: int = cfg.training.batch_size
        self.learning_rate: float = cfg.training.get('learning_rate', 5e-5)
        self.disc_lr: float = cfg.vae_3d.get('disc_lr', 1e-4)
        self.warmup_epochs: int = cfg.training.warmup_epochs
        self.val_interval: int = cfg.training.val_interval
        self.use_multi_gpu: bool = cfg.training.get('use_multi_gpu', False)
        self.use_ema: bool = cfg.training.get('use_ema', False)

        # Volume dimensions
        self.volume_depth: int = cfg.volume.get('depth', 160)
        self.volume_height: int = cfg.volume.get('height', 256)
        self.volume_width: int = cfg.volume.get('width', 256)

        # Loss weights
        self.perceptual_weight: float = cfg.vae_3d.get('perceptual_weight', 0.001)
        self.kl_weight: float = cfg.vae_3d.get('kl_weight', 1e-6)
        self.adv_weight: float = cfg.vae_3d.get('adv_weight', 0.01)

        # 3D specific options
        self.use_2_5d_perceptual: bool = cfg.vae_3d.get('use_2_5d_perceptual', True)
        self.perceptual_slice_fraction: float = cfg.vae_3d.get('perceptual_slice_fraction', 0.25)
        self.gradient_checkpointing: bool = cfg.training.get('gradient_checkpointing', True)

        # Discriminator config
        self.disc_num_layers: int = cfg.vae_3d.get('disc_num_layers', 3)
        self.disc_num_channels: int = cfg.vae_3d.get('disc_num_channels', 64)

        # VAE architecture
        self.latent_channels: int = cfg.vae_3d.latent_channels
        self.vae_channels: tuple = tuple(cfg.vae_3d.channels)
        self.attention_levels: tuple = tuple(cfg.vae_3d.attention_levels)
        self.num_res_blocks: int = cfg.vae_3d.get('num_res_blocks', 2)

        # Gradient clipping
        self.gradient_clip_norm: float = cfg.training.get('gradient_clip_norm', 1.0)

        # Logging options
        logging_cfg = cfg.training.get('logging', {})
        self.log_grad_norm: bool = logging_cfg.get('grad_norm', True)
        self.log_psnr: bool = logging_cfg.get('psnr', True)

        # Precision
        self.weight_dtype = torch.bfloat16

        # Setup device
        if self.use_multi_gpu:
            self.rank, self.local_rank, self.world_size, self.device = self._setup_distributed()
            self.is_main_process: bool = (self.rank == 0)
        else:
            self.device: torch.device = torch.device("cuda")
            self.is_main_process = True
            self.rank: int = 0
            self.world_size: int = 1

        # Initialize directories
        if self.is_main_process:
            try:
                from hydra.core.hydra_config import HydraConfig
                self.save_dir = HydraConfig.get().runtime.output_dir
            except (ImportError, ValueError, AttributeError):
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                exp_name = cfg.training.get('name', '')
                self.save_dir = os.path.join(
                    cfg.paths.model_dir, "compression_3d", cfg.mode.name,
                    f"{exp_name}{self.volume_height}x{self.volume_depth}_{timestamp}"
                )
            os.makedirs(self.save_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "tensorboard"))
        else:
            self.save_dir = None
            self.writer = None

        # Initialize tracking
        self._grad_norm_tracker = GradientNormTracker()
        self.best_loss = float('inf')

        # Model placeholders
        self.model = None
        self.model_raw = None
        self.discriminator = None
        self.discriminator_raw = None
        self.optimizer_g = None
        self.optimizer_d = None
        self.lr_scheduler_g = None
        self.lr_scheduler_d = None
        self.perceptual_loss = None
        self.adv_loss = None
        self.ema = None
        self.val_loader = None

    def _setup_distributed(self):
        """Setup distributed training."""
        return setup_distributed()

    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize 3D VAE model, discriminator, optimizers, and loss functions."""
        n_channels = self.cfg.mode.get('in_channels', 1)

        # Create 3D AutoencoderKL
        raw_model = AutoencoderKL(
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

        # Enable gradient checkpointing for memory efficiency
        if self.gradient_checkpointing:
            if hasattr(raw_model, 'gradient_checkpointing_enable'):
                raw_model.gradient_checkpointing_enable()

        # Create 3D PatchDiscriminator
        raw_disc = PatchDiscriminator(
            spatial_dims=3,
            in_channels=n_channels,
            channels=self.disc_num_channels,
            num_layers_d=self.disc_num_layers,
        ).to(self.device)

        # Load pretrained weights if provided
        if pretrained_checkpoint:
            try:
                checkpoint = torch.load(pretrained_checkpoint, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    raw_model.load_state_dict(checkpoint['model_state_dict'])
                    if self.is_main_process:
                        logger.info(f"Loaded 3D VAE weights from {pretrained_checkpoint}")
            except FileNotFoundError:
                if self.is_main_process:
                    logger.warning(f"Checkpoint not found: {pretrained_checkpoint}")

        # Wrap models
        use_compile = self.cfg.training.get('use_compile', True)
        self.model, self.model_raw = wrap_model_for_training(
            raw_model,
            use_multi_gpu=self.use_multi_gpu,
            local_rank=self.local_rank if self.use_multi_gpu else 0,
            use_compile=use_compile,
            is_main_process=self.is_main_process,
        )

        self.discriminator, self.discriminator_raw = wrap_model_for_training(
            raw_disc,
            use_multi_gpu=self.use_multi_gpu,
            local_rank=self.local_rank if self.use_multi_gpu else 0,
            use_compile=use_compile,
            is_main_process=False,
        )

        # Create optimizers
        self.optimizer_g = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.optimizer_d = AdamW(self.discriminator.parameters(), lr=self.disc_lr)

        # Create schedulers
        self.lr_scheduler_g = create_warmup_cosine_scheduler(
            self.optimizer_g, self.warmup_epochs, self.n_epochs
        )
        self.lr_scheduler_d = create_warmup_cosine_scheduler(
            self.optimizer_d, self.warmup_epochs, self.n_epochs
        )

        # Loss functions
        # 2.5D perceptual loss: compute on sampled slices
        if self.use_2_5d_perceptual and self.perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss(device=self.device)
        else:
            self.perceptual_loss = None

        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")

        # Save config
        if self.is_main_process:
            config_path = os.path.join(self.save_dir, "config.yaml")
            OmegaConf.save(self.cfg, config_path)
            logger.info(f"Config saved to: {config_path}")

            # Log model info
            vae_params = sum(p.numel() for p in self.model_raw.parameters())
            disc_params = sum(p.numel() for p in self.discriminator_raw.parameters())
            logger.info(f"3D VAE initialized: {vae_params / 1e6:.1f}M parameters")
            logger.info(f"  Channels: {self.vae_channels}")
            logger.info(f"  Latent channels: {self.latent_channels}")
            logger.info(f"3D Discriminator: {disc_params / 1e6:.1f}M parameters")
            logger.info(f"Volume: {self.volume_width}x{self.volume_height}x{self.volume_depth}")

            # Compute latent shape
            n_downsamples = len(self.vae_channels) - 1
            latent_h = self.volume_height // (2 ** n_downsamples)
            latent_w = self.volume_width // (2 ** n_downsamples)
            latent_d = self.volume_depth // (2 ** n_downsamples)
            logger.info(f"Latent shape: [{self.latent_channels}, {latent_d}, {latent_h}, {latent_w}]")

    def _compute_kl_loss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss."""
        kl = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).mean()
        return kl

    def _compute_2_5d_perceptual_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss on sampled 2D slices (2.5D approach)."""
        if self.perceptual_loss is None:
            return torch.tensor(0.0, device=self.device)

        # Sample slices from depth dimension
        # reconstruction/target shape: [B, C, D, H, W]
        depth = reconstruction.shape[2]
        n_slices = max(1, int(depth * self.perceptual_slice_fraction))
        slice_indices = torch.linspace(0, depth - 1, n_slices).long()

        total_loss = 0.0
        for idx in slice_indices:
            # Extract 2D slice: [B, C, H, W]
            recon_slice = reconstruction[:, :, idx, :, :]
            target_slice = target[:, :, idx, :, :]

            # Expand to 3 channels for perceptual network
            if recon_slice.shape[1] == 1:
                recon_slice = recon_slice.repeat(1, 3, 1, 1)
                target_slice = target_slice.repeat(1, 3, 1, 1)

            total_loss += self.perceptual_loss(recon_slice, target_slice)

        return total_loss / n_slices

    def _prepare_batch(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare batch for training."""
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

    def train_step(self, batch, epoch: int) -> Dict[str, float]:
        """Execute single training step."""
        images, _ = self._prepare_batch(batch)

        # ========== Generator step ==========
        self.optimizer_g.zero_grad()

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            reconstruction, mean, logvar = self.model(images)

            # L1 reconstruction loss
            l1_loss = torch.nn.functional.l1_loss(reconstruction, images)

            # KL divergence
            kl_loss = self._compute_kl_loss(mean, logvar)

            # Perceptual loss (2.5D)
            if self.perceptual_weight > 0 and self.perceptual_loss is not None:
                p_loss = self._compute_2_5d_perceptual_loss(reconstruction, images)
            else:
                p_loss = torch.tensor(0.0, device=self.device)

            # Adversarial loss
            disc_fake = self.discriminator(reconstruction)
            adv_loss = self.adv_loss(disc_fake, target_is_real=True, for_discriminator=False)

            # Total generator loss
            g_loss = (
                l1_loss +
                self.perceptual_weight * p_loss +
                self.kl_weight * kl_loss +
                self.adv_weight * adv_loss
            )

        g_loss.backward()
        if self.gradient_clip_norm > 0:
            grad_norm_g = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        else:
            grad_norm_g = 0.0

        # Track gradient norms
        if self.log_grad_norm and isinstance(grad_norm_g, torch.Tensor):
            self._grad_norm_tracker.update(grad_norm_g.item())

        self.optimizer_g.step()

        # ========== Discriminator step ==========
        self.optimizer_d.zero_grad()

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            disc_real = self.discriminator(images)
            disc_fake = self.discriminator(reconstruction.detach())

            d_loss_real = self.adv_loss(disc_real, target_is_real=True, for_discriminator=True)
            d_loss_fake = self.adv_loss(disc_fake, target_is_real=False, for_discriminator=True)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        d_loss.backward()
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip_norm)
        self.optimizer_d.step()

        return {
            'gen': g_loss.item(),
            'disc': d_loss.item(),
            'recon': l1_loss.item(),
            'perc': p_loss.item() if isinstance(p_loss, torch.Tensor) else p_loss,
            'kl': kl_loss.item(),
            'adv': adv_loss.item(),
        }

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.discriminator.train()

        total_losses = {'gen': 0, 'disc': 0, 'recon': 0, 'perc': 0, 'kl': 0, 'adv': 0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not self.is_main_process)
        for batch in pbar:
            losses = self.train_step(batch, epoch)

            for key in total_losses:
                total_losses[key] += losses[key]
            n_batches += 1

            pbar.set_postfix({
                'G': f"{losses['gen']:.4f}",
                'D': f"{losses['disc']:.4f}",
                'L1': f"{losses['recon']:.4f}",
            })

        # Average losses
        return {k: v / n_batches for k, v in total_losses.items()}

    def compute_validation_losses(self, epoch: int) -> Dict[str, float]:
        """Compute validation metrics."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_l1 = 0.0
        total_psnr = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images, _ = self._prepare_batch(batch)

                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    reconstruction, mean, logvar = self.model(images)

                total_l1 += torch.nn.functional.l1_loss(reconstruction, images).item()

                if self.log_psnr:
                    # Compute PSNR per slice and average
                    for d in range(reconstruction.shape[2]):
                        total_psnr += compute_psnr(
                            reconstruction[:, :, d, :, :],
                            images[:, :, d, :, :]
                        )
                    total_psnr /= reconstruction.shape[2]

                n_batches += 1

        self.model.train()

        metrics = {
            'l1': total_l1 / n_batches if n_batches > 0 else 0,
            'gen': total_l1 / n_batches if n_batches > 0 else 0,
        }

        if self.log_psnr and n_batches > 0:
            metrics['psnr'] = total_psnr / n_batches

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Validation/L1', metrics['l1'], epoch)
            if 'psnr' in metrics:
                self.writer.add_scalar('Validation/PSNR', metrics['psnr'], epoch)

        return metrics

    def _save_checkpoint(self, epoch: int, name: str) -> None:
        """Save checkpoint."""
        if not self.is_main_process:
            return

        save_full_checkpoint(
            model=self.model_raw,
            optimizer=self.optimizer_g,
            epoch=epoch,
            save_dir=self.save_dir,
            filename=f"checkpoint_{name}.pt",
            scheduler=self.lr_scheduler_g,
        )

    def _log_grad_norms(self, epoch: int) -> None:
        """Log gradient norms to TensorBoard."""
        if not self.log_grad_norm or self.writer is None:
            return

        if self._grad_norm_tracker.count > 0:
            self.writer.add_scalar('training/grad_norm_g_avg', self._grad_norm_tracker.get_avg(), epoch)
            self.writer.add_scalar('training/grad_norm_g_max', self._grad_norm_tracker.get_max(), epoch)
        self._grad_norm_tracker.reset()

    def train(
        self,
        train_loader: DataLoader,
        train_dataset: Dataset,
        val_loader: Optional[DataLoader] = None,
        start_epoch: int = 0,
        max_epochs: Optional[int] = None,
    ) -> int:
        """Execute training loop."""
        n_epochs = max_epochs if max_epochs is not None else self.n_epochs
        self.val_loader = val_loader
        total_start = time.time()

        avg_losses = {'gen': float('inf'), 'disc': float('inf'), 'recon': float('inf'),
                      'perc': float('inf'), 'kl': float('inf'), 'adv': float('inf')}

        if start_epoch > 0 and self.is_main_process:
            logger.info(f"Resuming training from epoch {start_epoch + 1}")

        last_epoch = start_epoch
        try:
            for epoch in range(start_epoch, n_epochs):
                last_epoch = epoch
                epoch_start = time.time()

                if self.use_multi_gpu and hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)

                avg_losses = self.train_epoch(train_loader, epoch)
                epoch_time = time.time() - epoch_start

                # Step schedulers
                if self.lr_scheduler_g is not None:
                    self.lr_scheduler_g.step()
                if self.lr_scheduler_d is not None:
                    self.lr_scheduler_d.step()

                if self.is_main_process:
                    val_metrics = self.compute_validation_losses(epoch)
                    log_vae_3d_epoch_summary(epoch, n_epochs, avg_losses, val_metrics, epoch_time)

                    # TensorBoard logging
                    if self.writer is not None:
                        self.writer.add_scalar('Loss/Generator_train', avg_losses['gen'], epoch)
                        self.writer.add_scalar('Loss/Discriminator', avg_losses['disc'], epoch)
                        self.writer.add_scalar('Loss/L1_train', avg_losses['recon'], epoch)
                        self.writer.add_scalar('Loss/Perceptual_train', avg_losses['perc'], epoch)
                        self.writer.add_scalar('Loss/KL_train', avg_losses['kl'], epoch)
                        self.writer.add_scalar('Loss/Adversarial', avg_losses['adv'], epoch)

                        if self.lr_scheduler_g is not None:
                            self.writer.add_scalar('LR/Generator', self.lr_scheduler_g.get_last_lr()[0], epoch)

                        self._log_grad_norms(epoch)

                    # Save checkpoints
                    is_val_epoch = (epoch + 1) % self.val_interval == 0
                    if is_val_epoch or (epoch + 1) == n_epochs:
                        self._save_checkpoint(epoch, "latest")

                        val_loss = val_metrics.get('gen', avg_losses['gen'])
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            self._save_checkpoint(epoch, "best")
                            logger.info(f"New best model saved (val loss: {val_loss:.6f})")

        finally:
            total_time = time.time() - total_start

            if self.is_main_process:
                logger.info(f"Training completed! Total time: {total_time:.1f}s ({total_time / 3600:.1f}h)")

            if self.use_multi_gpu:
                try:
                    dist.destroy_process_group()
                except Exception as e:
                    logger.warning(f"Error destroying process group: {e}")

        return last_epoch

    def close_writer(self) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
