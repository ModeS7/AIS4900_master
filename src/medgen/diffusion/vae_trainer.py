"""
VAE trainer module for training autoencoders.

This module provides the VAETrainer class for training AutoencoderKL models
with the same infrastructure as DiffusionTrainer: TensorBoard logging,
checkpoint management, multi-GPU support, and metrics tracking.
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
from ema_pytorch import EMA
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
from .quality_metrics import compute_ssim, compute_psnr, compute_lpips
from .worst_batch import create_worst_batch_figure
from .metrics import create_reconstruction_figure
from .utils import (
    get_vram_usage,
    GradientNormTracker,
    FLOPsTracker,
    create_epoch_iterator,
    save_full_checkpoint,
)

logger = logging.getLogger(__name__)


def log_vae_epoch_summary(
    epoch: int,
    total_epochs: int,
    avg_losses: Dict[str, float],
    val_metrics: Dict[str, float],
    elapsed_time: float
) -> None:
    """Log VAE epoch completion summary with train and validation metrics.

    Args:
        epoch: Current epoch number (0-indexed).
        total_epochs: Total number of epochs.
        avg_losses: Dict with 'gen', 'disc', 'recon', 'perc', 'kl', 'adv' training losses.
        val_metrics: Dict with 'gen', 'l1', 'ssim', 'psnr' validation metrics (can be empty).
        elapsed_time: Time taken for the epoch in seconds.
    """
    timestamp = time.strftime("%H:%M:%S")
    epoch_pct = ((epoch + 1) / total_epochs) * 100

    # Format validation metrics if available
    if val_metrics:
        val_gen = f"(v:{val_metrics.get('gen', 0):.4f})"
        val_l1 = f"(v:{val_metrics.get('l1', 0):.4f})"
        ssim_str = f"SSIM: {val_metrics.get('ssim', 0):.3f}"
    else:
        val_gen = ""
        val_l1 = ""
        ssim_str = ""

    print(
        f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
        f"G: {avg_losses['gen']:.4f}{val_gen} | "
        f"L1: {avg_losses['recon']:.4f}{val_l1} | "
        f"D: {avg_losses['disc']:.4f} | "
        f"{ssim_str} | "
        f"Time: {elapsed_time:.1f}s"
    )


class VAETrainer:
    """AutoencoderKL trainer with full feature parity to DiffusionTrainer.

    Supports training VAEs for latent diffusion models with:
    - TensorBoard logging
    - Checkpoint management
    - Multi-GPU support (DDP)
    - EMA weights
    - Gradient clipping
    - Learning rate scheduling

    Args:
        cfg: Hydra configuration object containing all settings.

    Example:
        >>> trainer = VAETrainer(cfg)
        >>> trainer.setup_model()
        >>> trainer.train(train_loader, train_dataset)
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # Extract config values
        self.n_epochs: int = cfg.training.epochs
        self.batch_size: int = cfg.training.batch_size
        self.image_size: int = cfg.model.image_size
        self.learning_rate: float = cfg.training.get('learning_rate', 1e-5)
        self.disc_lr: float = cfg.vae.get('disc_lr', 5e-5)
        self.warmup_epochs: int = cfg.training.warmup_epochs
        self.val_interval: int = cfg.training.val_interval
        self.use_multi_gpu: bool = cfg.training.get('use_multi_gpu', False)
        self.use_ema: bool = cfg.training.get('use_ema', True)
        self.ema_decay: float = cfg.training.ema.get('decay', 0.999)

        # Loss weights (MONAI defaults)
        self.perceptual_weight: float = cfg.vae.get('perceptual_weight', 0.002)
        self.kl_weight: float = cfg.vae.get('kl_weight', 1e-8)
        self.adv_weight: float = cfg.vae.get('adv_weight', 0.005)

        # Staged training options (for progressive training)
        # disable_gan: Skip discriminator entirely (faster, more stable for early training)
        # use_constant_lr: No scheduler, use constant learning rate
        progressive_cfg = cfg.get('progressive', {})
        self.disable_gan: bool = progressive_cfg.get('disable_gan', False)
        self.use_constant_lr: bool = progressive_cfg.get('use_constant_lr', False)

        # Discriminator config
        self.disc_num_layers: int = cfg.vae.get('disc_num_layers', 3)
        self.disc_num_channels: int = cfg.vae.get('disc_num_channels', 64)

        # torch.compile option (default: True)
        self.use_compile: bool = cfg.training.get('use_compile', True)

        # VAE architecture config
        self.latent_channels: int = cfg.vae.latent_channels
        self.vae_channels: tuple = tuple(cfg.vae.channels)
        self.attention_levels: tuple = tuple(cfg.vae.attention_levels)
        self.num_res_blocks: int = cfg.vae.get('num_res_blocks', 2)

        # Determine if running on cluster
        self.is_cluster: bool = (cfg.paths.name == "cluster")

        # Setup device and distributed training
        if self.use_multi_gpu:
            self.rank, self.local_rank, self.world_size, self.device = self._setup_distributed()
            self.is_main_process: bool = (self.rank == 0)
        else:
            self.device: torch.device = torch.device("cuda")
            self.is_main_process = True
            self.rank: int = 0
            self.world_size: int = 1

        # Initialize logging and save directories
        if self.is_main_process:
            # Check for explicit save_dir override (used by progressive training)
            if hasattr(cfg, 'save_dir_override') and cfg.save_dir_override:
                self.save_dir = cfg.save_dir_override
            else:
                try:
                    from hydra.core.hydra_config import HydraConfig
                    self.save_dir = HydraConfig.get().runtime.output_dir
                except (ImportError, ValueError, AttributeError):
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    # Optional experiment name prefix from config (include underscore in value: "exp1_")
                    exp_name = cfg.training.get('name', '')
                    mode_name = cfg.mode.get('name', 'dual')
                    self.run_name = f"{exp_name}{self.image_size}_{timestamp}"
                    # Structure: runs/vae_2d/{mode}/{run_name}
                    self.save_dir = os.path.join(cfg.paths.model_dir, 'vae_2d', mode_name, self.run_name)

            tensorboard_dir = os.path.join(self.save_dir, "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer: Optional[SummaryWriter] = SummaryWriter(tensorboard_dir)
            self.best_loss: float = float('inf')
        else:
            self.writer = None
            self.run_name = ""
            self.save_dir = ""
            self.best_loss = float('inf')

        # Initialize model components (set during setup_model)
        self.model: Optional[nn.Module] = None
        self.model_raw: Optional[nn.Module] = None
        self.discriminator: Optional[nn.Module] = None
        self.discriminator_raw: Optional[nn.Module] = None
        self.ema: Optional[EMA] = None
        self.optimizer_g: Optional[AdamW] = None  # Generator optimizer
        self.optimizer_d: Optional[AdamW] = None  # Discriminator optimizer
        self.lr_scheduler_g: Optional[LRScheduler] = None
        self.lr_scheduler_d: Optional[LRScheduler] = None
        self.perceptual_loss_fn: Optional[nn.Module] = None
        self.adv_loss_fn: Optional[PatchAdversarialLoss] = None

        # Logging config
        logging_cfg = cfg.training.get('logging', {})
        self.log_grad_norm: bool = logging_cfg.get('grad_norm', True)
        self.log_flops: bool = logging_cfg.get('flops', True)
        self.log_lpips: bool = logging_cfg.get('lpips', True)
        self.log_regional_losses: bool = logging_cfg.get('regional_losses', True)

        # Gradient norm tracking (per epoch)
        self._grad_tracker_g = GradientNormTracker()
        self._grad_tracker_d = GradientNormTracker()

        # Regional loss tracking (tumor vs background)
        self._tumor_loss_sum: float = 0.0
        self._bg_loss_sum: float = 0.0
        self._regional_count: int = 0

        # FLOPs tracking
        self._flops_tracker = FLOPsTracker()

        # Validation loader (set in train())
        self.val_loader: Optional[DataLoader] = None

    def _setup_distributed(self) -> Tuple[int, int, int, torch.device]:
        """Setup distributed training with dynamic port allocation."""
        return setup_distributed()

    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize VAE model, discriminator, optimizers, and loss functions.

        Args:
            pretrained_checkpoint: Optional path to checkpoint for loading pretrained weights.
                Used for progressive training to transfer weights between resolutions.
        """
        # For VAE, in_channels must equal out_channels (autoencoder)
        # Use in_channels from mode config (total channels to encode)
        n_channels = self.cfg.mode.get('in_channels', 1)

        # Create AutoencoderKL (Generator)
        raw_model = AutoencoderKL(
            spatial_dims=2,
            in_channels=n_channels,
            out_channels=n_channels,
            channels=self.vae_channels,
            attention_levels=self.attention_levels,
            latent_channels=self.latent_channels,
            num_res_blocks=self.num_res_blocks,
            norm_num_groups=32,
            with_encoder_nonlocal_attn=True,
            with_decoder_nonlocal_attn=True,
        ).to(self.device)

        # Create PatchDiscriminator (only if GAN is enabled)
        raw_disc = None
        if not self.disable_gan:
            raw_disc = PatchDiscriminator(
                spatial_dims=2,
                in_channels=n_channels,
                channels=self.disc_num_channels,
                num_layers_d=self.disc_num_layers,
            ).to(self.device)

        # Load pretrained weights if provided (for progressive training)
        if pretrained_checkpoint:
            try:
                checkpoint = torch.load(pretrained_checkpoint, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    raw_model.load_state_dict(checkpoint['model_state_dict'])
                    if self.is_main_process:
                        logger.info(f"Loaded VAE weights from {pretrained_checkpoint}")
                if 'discriminator_state_dict' in checkpoint and raw_disc is not None:
                    raw_disc.load_state_dict(checkpoint['discriminator_state_dict'])
                    if self.is_main_process:
                        logger.info(f"Loaded discriminator weights from {pretrained_checkpoint}")
            except FileNotFoundError:
                if self.is_main_process:
                    logger.warning(f"Pretrained checkpoint not found: {pretrained_checkpoint}")

        # Wrap VAE model with DDP and/or torch.compile
        self.model, self.model_raw = wrap_model_for_training(
            raw_model,
            use_multi_gpu=self.use_multi_gpu,
            local_rank=self.local_rank if self.use_multi_gpu else 0,
            use_compile=self.use_compile,
            compile_mode="default",
            is_main_process=self.is_main_process,
        )

        # Wrap discriminator if GAN is enabled
        if raw_disc is not None:
            self.discriminator, self.discriminator_raw = wrap_model_for_training(
                raw_disc,
                use_multi_gpu=self.use_multi_gpu,
                local_rank=self.local_rank if self.use_multi_gpu else 0,
                use_compile=self.use_compile,
                compile_mode="default",
                is_main_process=False,  # Suppress duplicate logging
            )

        # Setup perceptual loss (RadImageNet for 2D medical images)
        # Uses shared wrapper that handles multi-channel inputs
        cache_dir = getattr(self.cfg.paths, 'cache_dir', None)
        self.perceptual_loss_fn = PerceptualLoss(
            spatial_dims=2,
            network_type="radimagenet_resnet50",
            cache_dir=cache_dir,
            pretrained=True,
            device=self.device,
            use_compile=self.use_compile,
        )

        # Setup adversarial loss (only if GAN is enabled)
        if not self.disable_gan:
            self.adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

        # Setup generator optimizer
        self.optimizer_g = AdamW(self.model_raw.parameters(), lr=self.learning_rate)

        # Setup discriminator optimizer (only if GAN is enabled)
        if not self.disable_gan:
            self.optimizer_d = AdamW(self.discriminator_raw.parameters(), lr=self.disc_lr)

        # Setup LR schedulers (only if not using constant LR)
        if not self.use_constant_lr:
            self.lr_scheduler_g = create_warmup_cosine_scheduler(
                self.optimizer_g,
                warmup_epochs=self.warmup_epochs,
                total_epochs=self.n_epochs,
            )
            if not self.disable_gan:
                self.lr_scheduler_d = create_warmup_cosine_scheduler(
                    self.optimizer_d,
                    warmup_epochs=self.warmup_epochs,
                    total_epochs=self.n_epochs,
                )
        else:
            if self.is_main_process:
                logger.info(f"Using constant LR: {self.learning_rate} (scheduler disabled)")

        # Create EMA wrapper if enabled (for generator only)
        if self.use_ema:
            self.ema = EMA(
                self.model_raw,
                beta=self.ema_decay,
                update_after_step=self.cfg.training.ema.get('update_after_step', 100),
                update_every=self.cfg.training.ema.get('update_every', 10),
            )
            if self.is_main_process:
                logger.info(f"EMA enabled with decay={self.ema_decay}")

        # Save metadata
        if self.is_main_process:
            self._save_metadata()

        # Log model info
        if self.is_main_process:
            vae_params = sum(p.numel() for p in self.model_raw.parameters())
            logger.info(f"VAE initialized: {vae_params / 1e6:.1f}M parameters")
            if not self.disable_gan:
                disc_params = sum(p.numel() for p in self.discriminator_raw.parameters())
                logger.info(f"Discriminator initialized: {disc_params / 1e6:.1f}M parameters")
            else:
                logger.info("GAN disabled - discriminator not created")
            logger.info(f"Latent shape: [{self.latent_channels}, {self.image_size // 8}, {self.image_size // 8}]")
            logger.info(f"Loss weights - Perceptual: {self.perceptual_weight}, KL: {self.kl_weight}, Adv: {self.adv_weight}")

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> int:
        """Load checkpoint to resume training.

        Loads model weights, discriminator (if GAN enabled), optimizers, schedulers,
        and EMA state from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file (.pt).
            load_optimizer: Whether to load optimizer and scheduler states.
                Set to False when loading for inference or fine-tuning with new optimizer.

        Returns:
            Epoch number from the checkpoint (0-indexed).

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            RuntimeError: If checkpoint is incompatible with current config.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load VAE model weights
        self.model_raw.load_state_dict(checkpoint['model_state_dict'])
        if self.is_main_process:
            logger.info(f"Loaded VAE weights from {checkpoint_path}")

        # Load discriminator weights (if GAN enabled and checkpoint has them)
        if not self.disable_gan and self.discriminator_raw is not None:
            if 'discriminator_state_dict' in checkpoint:
                self.discriminator_raw.load_state_dict(checkpoint['discriminator_state_dict'])
                if self.is_main_process:
                    logger.info("Loaded discriminator weights from checkpoint")
            else:
                if self.is_main_process:
                    logger.warning("Checkpoint has no discriminator weights - using fresh discriminator")

        # Load optimizer states
        if load_optimizer:
            if 'optimizer_g_state_dict' in checkpoint and self.optimizer_g is not None:
                self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
                if self.is_main_process:
                    logger.info("Loaded generator optimizer state")

            if not self.disable_gan and self.optimizer_d is not None:
                if 'optimizer_d_state_dict' in checkpoint:
                    self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
                    if self.is_main_process:
                        logger.info("Loaded discriminator optimizer state")

            # Load scheduler states (only if not using constant LR)
            if not self.use_constant_lr:
                if 'scheduler_g_state_dict' in checkpoint and self.lr_scheduler_g is not None:
                    self.lr_scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
                    if self.is_main_process:
                        logger.info("Loaded generator scheduler state")

                if not self.disable_gan and self.lr_scheduler_d is not None:
                    if 'scheduler_d_state_dict' in checkpoint:
                        self.lr_scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
                        if self.is_main_process:
                            logger.info("Loaded discriminator scheduler state")

        # Load EMA state
        if self.use_ema and self.ema is not None:
            if 'ema_state_dict' in checkpoint:
                self.ema.load_state_dict(checkpoint['ema_state_dict'])
                if self.is_main_process:
                    logger.info("Loaded EMA state from checkpoint")
            else:
                if self.is_main_process:
                    logger.warning("Checkpoint has no EMA state - EMA will start fresh")

        epoch = checkpoint.get('epoch', 0)
        if self.is_main_process:
            logger.info(f"Resuming from epoch {epoch + 1}")

        return epoch

    def _save_metadata(self) -> None:
        """Save training configuration to metadata.json."""
        os.makedirs(self.save_dir, exist_ok=True)

        config_path = os.path.join(self.save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        # Save VAE config separately for easy loading
        # For VAE, in_channels == out_channels (autoencoder)
        n_channels = self.cfg.mode.get('in_channels', 1)
        vae_config = {
            'in_channels': n_channels,
            'out_channels': n_channels,
            'latent_channels': self.latent_channels,
            'channels': list(self.vae_channels),
            'attention_levels': list(self.attention_levels),
            'num_res_blocks': self.num_res_blocks,
            'norm_num_groups': 32,
            'with_encoder_nonlocal_attn': True,
            'with_decoder_nonlocal_attn': True,
        }

        metadata = {
            'type': 'vae',
            'epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'image_size': self.image_size,
            'learning_rate': self.learning_rate,
            'perceptual_weight': self.perceptual_weight,
            'kl_weight': self.kl_weight,
            'warmup_epochs': self.warmup_epochs,
            'val_interval': self.val_interval,
            'multi_gpu': self.use_multi_gpu,
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay if self.use_ema else None,
            'vae_config': vae_config,
            'created_at': datetime.now().isoformat(),
        }

        metadata_path = os.path.join(self.save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Config saved to: {config_path}")

    def _update_metadata_final(self, final_loss: float, final_recon: float, total_time: float) -> None:
        """Update metadata.json with final training results."""
        metadata_path = os.path.join(self.save_dir, 'metadata.json')

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        metadata['results'] = {
            'final_loss': final_loss,
            'final_recon_loss': final_recon,
            'best_loss': self.best_loss,
            'total_time_seconds': total_time,
            'total_time_hours': total_time / 3600,
            'completed_at': datetime.now().isoformat(),
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _update_ema(self) -> None:
        """Update EMA model weights."""
        if self.ema is not None:
            self.ema.update()

    def _compute_kl_loss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss.

        Args:
            mean: Latent mean [B, C, H, W].
            logvar: Latent log variance [B, C, H, W].

        Returns:
            KL divergence loss (scalar).
        """
        # KL divergence: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        # Normalize by batch size and spatial dimensions
        kl = kl / mean.numel()
        return kl

    def _track_regional_losses(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> None:
        """Track L1 loss for tumor vs background regions.

        Args:
            reconstruction: Reconstructed images [B, C, H, W].
            target: Original images [B, C, H, W].
            mask: Binary segmentation mask [B, 1, H, W].
        """
        with torch.no_grad():
            abs_error = torch.abs(reconstruction - target)

            # Expand mask to match number of channels
            tumor_mask = (mask > 0.5).float().expand_as(reconstruction)
            bg_mask = 1.0 - tumor_mask

            # Compute L1 for each region
            tumor_pixels = tumor_mask.sum()
            bg_pixels = bg_mask.sum()

            if tumor_pixels > 0:
                tumor_loss = (abs_error * tumor_mask).sum() / tumor_pixels
                self._tumor_loss_sum += tumor_loss.item()

            if bg_pixels > 0:
                bg_loss = (abs_error * bg_mask).sum() / bg_pixels
                self._bg_loss_sum += bg_loss.item()

            self._regional_count += 1

    def _reset_regional_loss_tracking(self) -> None:
        """Reset regional loss accumulators for a new epoch."""
        self._tumor_loss_sum = 0.0
        self._bg_loss_sum = 0.0
        self._regional_count = 0

    def _log_regional_losses(self, epoch: int) -> None:
        """Log regional loss statistics to TensorBoard."""
        if not self.log_regional_losses or self._regional_count == 0:
            return

        avg_tumor = self._tumor_loss_sum / self._regional_count
        avg_bg = self._bg_loss_sum / self._regional_count
        ratio = avg_tumor / (avg_bg + 1e-8)

        self.writer.add_scalar('Loss/tumor_region', avg_tumor, epoch)
        self.writer.add_scalar('Loss/background_region', avg_bg, epoch)
        self.writer.add_scalar('Loss/tumor_bg_ratio', ratio, epoch)

    def _measure_model_flops(self, sample_batch: torch.Tensor, steps_per_epoch: int) -> None:
        """Measure FLOPs for VAE forward pass using FLOPsTracker.

        Should be called once at the start of training with a sample batch.

        Args:
            sample_batch: Sample input tensor [B, C, H, W].
            steps_per_epoch: Number of training steps per epoch.
        """
        if not self.log_flops:
            return
        self._flops_tracker.measure(
            model=self.model_raw,
            sample_input=sample_batch[:1],  # Single sample
            steps_per_epoch=steps_per_epoch,
            timesteps=None,  # VAE has no timesteps
            is_main_process=self.is_main_process,
        )

    def _prepare_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare batch tensor from dataloader output.

        For VAE dual mode, the batch may contain seg as 3rd channel for
        regional metrics tracking, but it's NOT used in VAE reconstruction.

        Args:
            batch: Input batch - either a dict of tensors or a single tensor.

        Returns:
            Tuple of (images, mask):
            - images: Tensor [B, C, H, W] for VAE input (2 channels for dual)
            - mask: Optional segmentation mask [B, 1, H, W] for regional metrics
        """
        if isinstance(batch, dict):
            # Stack all images into single tensor [B, C, H, W]
            image_keys = self.cfg.mode.get('image_keys', ['t1_pre', 't1_gd'])
            tensors = []
            for key in image_keys:
                if key in batch:
                    tensors.append(batch[key].to(self.device))
            images = torch.cat(tensors, dim=1)
            mask = batch['seg'].to(self.device) if 'seg' in batch else None
            return images, mask
        elif hasattr(batch, 'as_tensor'):
            tensor = batch.as_tensor().to(self.device)
        else:
            tensor = batch.to(self.device)

        # For tensor input: check if seg is stacked as last channel
        # Dual mode: 2 channels (t1_pre, t1_gd) -> 3 channels if seg included
        n_image_channels = self.cfg.mode.get('in_channels', 2)
        if tensor.shape[1] > n_image_channels:
            # Last channel is seg, extract it separately
            images = tensor[:, :n_image_channels, :, :]
            mask = tensor[:, n_image_channels:n_image_channels + 1, :, :]
            return images, mask
        else:
            return tensor, None

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step with optional GAN training.

        Training follows MONAI's approach with proper gradient handling:
        1. Train discriminator on real vs fake (only if GAN enabled)
        2. Train generator with L1 + perceptual + KL + (optional) adversarial loss

        Args:
            batch: Input batch - either a dict of tensors (from dual dataloader)
                   or a single tensor [B, C, H, W].

        Returns:
            Dict with 'gen', 'disc', 'recon', 'perc', 'kl', 'adv' losses.
        """
        images, mask = self._prepare_batch(batch)
        grad_clip = self.cfg.training.get('gradient_clip_norm', 1.0)

        d_loss = torch.tensor(0.0, device=self.device)
        adv_loss = torch.tensor(0.0, device=self.device)

        # ==================== Discriminator Step (if GAN enabled) ====================
        if not self.disable_gan:
            # Forward through generator (no grad needed for D step)
            with torch.no_grad():
                with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    reconstruction_for_d, _, _ = self.model(images)

            self.optimizer_d.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                # Real images -> discriminator should output 1
                logits_real = self.discriminator(images.contiguous())
                # Fake images -> discriminator should output 0
                logits_fake = self.discriminator(reconstruction_for_d.contiguous())

                d_loss = 0.5 * (
                    self.adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                    + self.adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                )

            d_loss.backward()
            grad_norm_d = 0.0
            if grad_clip > 0:
                grad_norm_d = torch.nn.utils.clip_grad_norm_(
                    self.discriminator_raw.parameters(), max_norm=grad_clip
                ).item()
            self.optimizer_d.step()

            # Track discriminator gradient norm
            if self.log_grad_norm:
                self._grad_tracker_d.update(grad_norm_d)

        # ==================== Generator Step ====================
        self.optimizer_g.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=torch.bfloat16):
            # Fresh forward pass for generator gradients
            reconstruction, mean, logvar = self.model(images)

            # L1 reconstruction loss (MONAI uses L1, not MSE)
            l1_loss = torch.abs(reconstruction - images).mean()

            # Track regional losses (tumor vs background) if mask available
            if mask is not None and self.log_regional_losses:
                self._track_regional_losses(reconstruction, images, mask)

            # Perceptual loss (wrapper handles multi-channel inputs)
            p_loss = self.perceptual_loss_fn(reconstruction, images)

            # KL divergence loss
            kl_loss = self._compute_kl_loss(mean, logvar)

            # Adversarial loss (only if GAN enabled)
            if not self.disable_gan:
                logits_fake_for_g = self.discriminator(reconstruction.contiguous())
                adv_loss = self.adv_loss_fn(
                    logits_fake_for_g, target_is_real=True, for_discriminator=False
                )

            # Total generator loss
            g_loss = (
                l1_loss
                + self.perceptual_weight * p_loss
                + self.kl_weight * kl_loss
                + self.adv_weight * adv_loss
            )

        g_loss.backward()
        grad_norm_g = 0.0
        if grad_clip > 0:
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                self.model_raw.parameters(), max_norm=grad_clip
            ).item()
        self.optimizer_g.step()

        # Track generator gradient norm
        if self.log_grad_norm:
            self._grad_tracker_g.update(grad_norm_g)

        if self.use_ema:
            self._update_ema()

        return {
            'gen': g_loss.item(),
            'disc': d_loss.item() if not self.disable_gan else 0.0,
            'recon': l1_loss.item(),
            'perc': p_loss.item(),
            'kl': kl_loss.item(),
            'adv': adv_loss.item(),
        }

    def _reset_grad_norm_tracking(self) -> None:
        """Reset gradient norm tracking for a new epoch."""
        self._grad_tracker_g.reset()
        self._grad_tracker_d.reset()

    def _log_grad_norms(self, epoch: int) -> None:
        """Log gradient norm statistics to TensorBoard."""
        if not self.log_grad_norm:
            return
        self._grad_tracker_g.log(self.writer, epoch, prefix='training/grad_norm_g')
        if self.discriminator is not None:
            self._grad_tracker_d.log(self.writer, epoch, prefix='training/grad_norm_d')

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch.

        Args:
            data_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dict with average losses: 'gen', 'disc', 'recon', 'perc', 'kl', 'adv'.
        """
        self.model.train()
        if not self.disable_gan and self.discriminator is not None:
            self.discriminator.train()

        # Reset tracking for this epoch
        self._reset_grad_norm_tracking()
        self._reset_regional_loss_tracking()

        epoch_losses = {'gen': 0, 'disc': 0, 'recon': 0, 'perc': 0, 'kl': 0, 'adv': 0}

        epoch_iter = create_epoch_iterator(
            data_loader, epoch, self.is_cluster, self.is_main_process
        )

        for step, batch in enumerate(epoch_iter):
            losses = self.train_step(batch)

            for key in epoch_losses:
                epoch_losses[key] += losses[key]

            # Update progress bar if tqdm instance (has set_postfix method)
            if hasattr(epoch_iter, 'set_postfix'):
                epoch_iter.set_postfix(
                    G=f"{epoch_losses['gen'] / (step + 1):.4f}",
                    D=f"{epoch_losses['disc'] / (step + 1):.4f}"
                )

            if epoch == 1 and step == 0 and self.is_main_process:
                logger.info(get_vram_usage(self.device))

        # Average losses
        n_batches = len(data_loader)
        return {key: val / n_batches for key, val in epoch_losses.items()}

    def compute_validation_losses(self, epoch: int) -> Dict[str, float]:
        """Compute losses and metrics on full validation set (every epoch).

        Runs inference on entire val_loader and computes:
        - L1, Perceptual, KL, Generator losses (for train/val comparison)
        - SSIM, PSNR, LPIPS quality metrics

        Args:
            epoch: Current epoch number.

        Returns:
            Dict with validation metrics: 'l1', 'perc', 'kl', 'gen', 'ssim', 'psnr', 'lpips'.
        """
        if self.val_loader is None:
            return {}

        model_to_use = self.ema.ema_model if self.ema is not None else self.model_raw
        model_to_use.eval()

        # Accumulators
        total_l1 = 0.0
        total_perc = 0.0
        total_kl = 0.0
        total_gen = 0.0
        total_ssim = 0.0
        total_psnr = 0.0
        total_lpips = 0.0
        n_batches = 0

        # Worst batch tracking
        worst_loss = 0.0
        worst_batch_data = None

        # Mark CUDA graph step boundary to prevent tensor caching issues
        # when perceptual loss (compiled with torch.compile) is called during validation
        if self.use_compile:
            torch.compiler.cudagraph_mark_step_begin()

        with torch.no_grad():
            for batch in self.val_loader:
                images, _ = self._prepare_batch(batch)  # mask unused in validation metrics

                with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    reconstructed, mean, logvar = model_to_use(images)

                    # Compute losses (same as training)
                    l1_loss = torch.abs(reconstructed - images).mean()
                    p_loss = self.perceptual_loss_fn(reconstructed, images)
                    kl_loss = self._compute_kl_loss(mean, logvar)
                    g_loss = l1_loss + self.perceptual_weight * p_loss + self.kl_weight * kl_loss

                loss_val = g_loss.item()
                total_l1 += l1_loss.item()
                total_perc += p_loss.item()
                total_kl += kl_loss.item()
                total_gen += loss_val

                # Track worst batch
                if loss_val > worst_loss:
                    worst_loss = loss_val
                    # Convert to dict for dual mode (triggers dual-channel figure)
                    n_channels = self.cfg.mode.get('in_channels', 1)
                    if n_channels == 2:
                        image_keys = self.cfg.mode.get('image_keys', ['t1_pre', 't1_gd'])
                        orig_dict = {image_keys[0]: images[:, 0:1].cpu(), image_keys[1]: images[:, 1:2].cpu()}
                        gen_dict = {image_keys[0]: reconstructed[:, 0:1].float().cpu(), image_keys[1]: reconstructed[:, 1:2].float().cpu()}
                    else:
                        orig_dict = images.cpu()
                        gen_dict = reconstructed.float().cpu()
                    worst_batch_data = {
                        'original': orig_dict,
                        'generated': gen_dict,
                        'loss': loss_val,
                        'loss_breakdown': {'L1': l1_loss.item(), 'Perc': p_loss.item(), 'KL': kl_loss.item()},
                    }

                # Quality metrics
                total_ssim += compute_ssim(reconstructed, images)
                total_psnr += compute_psnr(reconstructed, images)
                if self.log_lpips:
                    total_lpips += compute_lpips(reconstructed, images, self.device)

                n_batches += 1

        model_to_use.train()

        # Compute averages
        metrics = {
            'l1': total_l1 / n_batches,
            'perc': total_perc / n_batches,
            'kl': total_kl / n_batches,
            'gen': total_gen / n_batches,
            'ssim': total_ssim / n_batches,
            'psnr': total_psnr / n_batches,
        }

        if self.log_lpips:
            metrics['lpips'] = total_lpips / n_batches

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Loss/L1_val', metrics['l1'], epoch)
            self.writer.add_scalar('Loss/Perceptual_val', metrics['perc'], epoch)
            self.writer.add_scalar('Loss/KL_val', metrics['kl'], epoch)
            self.writer.add_scalar('Loss/Generator_val', metrics['gen'], epoch)
            self.writer.add_scalar('Validation/SSIM', metrics['ssim'], epoch)
            self.writer.add_scalar('Validation/PSNR', metrics['psnr'], epoch)
            if 'lpips' in metrics:
                self.writer.add_scalar('Validation/LPIPS', metrics['lpips'], epoch)

            # Log worst batch figure
            if worst_batch_data is not None:
                fig = create_worst_batch_figure(
                    original=worst_batch_data['original'],
                    generated=worst_batch_data['generated'],
                    loss=worst_batch_data['loss'],
                    loss_breakdown=worst_batch_data['loss_breakdown'],
                )
                self.writer.add_figure('Validation/worst_batch', fig, epoch)
                plt.close(fig)

        return metrics

    def _save_vae_checkpoint(self, epoch: int, filename: str) -> str:
        """Save VAE checkpoint with config for easy loading.

        Args:
            epoch: Current epoch number.
            filename: Checkpoint filename (without extension).

        Returns:
            Path to saved checkpoint (.pt).
        """
        # Include VAE config in checkpoint for easy reconstruction
        n_channels = self.cfg.mode.get('in_channels', 1)
        model_config = {
            'in_channels': n_channels,
            'out_channels': n_channels,
            'latent_channels': self.latent_channels,
            'channels': list(self.vae_channels),
            'attention_levels': list(self.attention_levels),
            'num_res_blocks': self.num_res_blocks,
            'norm_num_groups': 32,
            'with_encoder_nonlocal_attn': True,
            'with_decoder_nonlocal_attn': True,
        }

        # Build extra state for VAE-specific components
        extra_state = {
            'disable_gan': self.disable_gan,
            'use_constant_lr': self.use_constant_lr,
        }

        # Add discriminator state if GAN is enabled
        if not self.disable_gan and self.discriminator_raw is not None:
            extra_state['discriminator_state_dict'] = self.discriminator_raw.state_dict()
            extra_state['disc_config'] = {
                'in_channels': n_channels,
                'channels': self.disc_num_channels,
                'num_layers_d': self.disc_num_layers,
            }
            if self.optimizer_d is not None:
                extra_state['optimizer_d_state_dict'] = self.optimizer_d.state_dict()
            if not self.use_constant_lr and self.lr_scheduler_d is not None:
                extra_state['scheduler_d_state_dict'] = self.lr_scheduler_d.state_dict()

        # Use subprocess-isolated compression to avoid heap corruption
        return save_full_checkpoint(
            model=self.model_raw,
            optimizer=self.optimizer_g,
            epoch=epoch,
            save_dir=self.save_dir,
            filename=filename,
            model_config=model_config,
            scheduler=self.lr_scheduler_g if not self.use_constant_lr else None,
            ema=self.ema,
            extra_state=extra_state,
        )

    def train(
        self,
        train_loader: DataLoader,
        train_dataset: Dataset,
        val_loader: Optional[DataLoader] = None,
        start_epoch: int = 0
    ) -> None:
        """Execute the main training loop.

        Args:
            train_loader: Training data loader.
            train_dataset: Training dataset (for validation sampling if no val_loader).
            val_loader: Optional validation data loader. If provided, used for
                validation metrics instead of sampling from train_dataset.
            start_epoch: Epoch to start from (for resuming training).
        """
        self.val_loader = val_loader
        total_start = time.time()

        # Measure FLOPs on first batch (once at start of training)
        if self.log_flops:
            try:
                first_batch = next(iter(train_loader))
                sample_images, _ = self._prepare_batch(first_batch)
                self._measure_model_flops(sample_images, len(train_loader))
            except Exception as e:
                if self.is_main_process:
                    logger.warning(f"Could not measure FLOPs: {e}")

        avg_losses = {'gen': float('inf'), 'disc': float('inf'), 'recon': float('inf'),
                      'perc': float('inf'), 'kl': float('inf'), 'adv': float('inf')}

        if start_epoch > 0 and self.is_main_process:
            logger.info(f"Resuming training from epoch {start_epoch + 1}")

        try:
            for epoch in range(start_epoch, self.n_epochs):
                epoch_start = time.time()

                if self.use_multi_gpu and hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)

                avg_losses = self.train_epoch(train_loader, epoch)

                if self.use_multi_gpu:
                    loss_tensor = torch.tensor(
                        [avg_losses['gen'], avg_losses['disc'], avg_losses['recon'],
                         avg_losses['perc'], avg_losses['kl'], avg_losses['adv']],
                        device=self.device
                    )
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    loss_vals = (loss_tensor / self.world_size).cpu().numpy()
                    avg_losses = dict(zip(['gen', 'disc', 'recon', 'perc', 'kl', 'adv'], loss_vals))

                epoch_time = time.time() - epoch_start

                # Step schedulers (only if not using constant LR)
                if not self.use_constant_lr:
                    if self.lr_scheduler_g is not None:
                        self.lr_scheduler_g.step()
                    if not self.disable_gan and self.lr_scheduler_d is not None:
                        self.lr_scheduler_d.step()

                if self.is_main_process:
                    # Compute validation metrics every epoch
                    val_metrics = self.compute_validation_losses(epoch)

                    # Log epoch summary with train and val metrics
                    log_vae_epoch_summary(epoch, self.n_epochs, avg_losses, val_metrics, epoch_time)

                    if self.writer is not None:
                        # Training losses (with _train suffix)
                        self.writer.add_scalar('Loss/Generator_train', avg_losses['gen'], epoch)
                        self.writer.add_scalar('Loss/Discriminator', avg_losses['disc'], epoch)
                        self.writer.add_scalar('Loss/L1_train', avg_losses['recon'], epoch)
                        self.writer.add_scalar('Loss/Perceptual_train', avg_losses['perc'], epoch)
                        self.writer.add_scalar('Loss/KL_train', avg_losses['kl'], epoch)
                        self.writer.add_scalar('Loss/Adversarial', avg_losses['adv'], epoch)

                        # Log learning rates
                        if not self.use_constant_lr and self.lr_scheduler_g is not None:
                            self.writer.add_scalar('LR/Generator', self.lr_scheduler_g.get_last_lr()[0], epoch)
                        else:
                            self.writer.add_scalar('LR/Generator', self.learning_rate, epoch)

                        if not self.disable_gan:
                            if not self.use_constant_lr and self.lr_scheduler_d is not None:
                                self.writer.add_scalar('LR/Discriminator', self.lr_scheduler_d.get_last_lr()[0], epoch)
                            else:
                                self.writer.add_scalar('LR/Discriminator', self.disc_lr, epoch)

                        # Log gradient norms
                        self._log_grad_norms(epoch)

                        # Log regional losses (tumor vs background)
                        self._log_regional_losses(epoch)

                        # Log FLOPs per epoch
                        self._flops_tracker.log_epoch(self.writer, epoch)

                    is_val_epoch = (epoch + 1) % self.val_interval == 0

                    if is_val_epoch or (epoch + 1) == self.n_epochs:
                        # Save latest checkpoint (for resuming)
                        self._save_vae_checkpoint(epoch, "latest")

                        # Save best checkpoint if validation loss improved
                        val_gen_loss = val_metrics.get('gen', avg_losses['gen'])
                        if val_gen_loss < self.best_loss:
                            self.best_loss = val_gen_loss
                            self._save_vae_checkpoint(epoch, "best")
                            logger.info(f"New best model saved (val G loss: {val_gen_loss:.6f})")

        finally:
            total_time = time.time() - total_start

            if self.is_main_process:
                logger.info(f"Training completed! Total time: {total_time:.1f}s ({total_time / 3600:.1f}h)")
                self._update_metadata_final(avg_losses['gen'], avg_losses['recon'], total_time)
                # Note: writer is NOT closed here - call close_writer() after test evaluation

            if self.use_multi_gpu:
                try:
                    dist.destroy_process_group()
                except Exception as e:
                    logger.warning(f"Error destroying process group: {e}")

    def evaluate_test_set(
        self,
        test_loader: DataLoader,
        checkpoint_name: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate VAE reconstruction on test set.

        Runs inference on the entire test set and computes metrics:
        - L1 reconstruction loss
        - SSIM (Structural Similarity Index)
        - PSNR (Peak Signal-to-Noise Ratio)
        - LPIPS (Learned Perceptual Image Patch Similarity) if enabled

        Results are saved to test_results_{checkpoint_name}.json and logged to TensorBoard.

        Args:
            test_loader: DataLoader for test set.
            checkpoint_name: Name of checkpoint to load ("best", "latest", or None
                for current model state).

        Returns:
            Dict with test metrics: 'l1', 'ssim', 'psnr', 'lpips', 'n_samples'.
        """
        if not self.is_main_process:
            return {}

        # Load checkpoint if specified
        if checkpoint_name is not None:
            checkpoint_path = os.path.join(self.save_dir, f"{checkpoint_name}.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model_raw.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded {checkpoint_name} checkpoint for test evaluation")
            else:
                logger.warning(f"Checkpoint {checkpoint_path} not found, using current model state")
                checkpoint_name = "current"

        label = checkpoint_name or "current"
        logger.info("=" * 60)
        logger.info(f"EVALUATING ON TEST SET ({label.upper()} MODEL)")
        logger.info("=" * 60)

        # Use EMA model if available and no checkpoint loaded, otherwise raw model
        if checkpoint_name is None and self.ema is not None:
            model_to_use = self.ema.ema_model
        else:
            model_to_use = self.model_raw
        model_to_use.eval()

        # Accumulators for metrics
        total_l1 = 0.0
        total_ssim = 0.0
        total_psnr = 0.0
        total_lpips = 0.0
        n_batches = 0
        n_samples = 0

        # Worst batch tracking
        worst_loss = 0.0
        worst_batch_data = None

        # Store samples for visualization
        sample_inputs = []
        sample_outputs = []
        max_vis_samples = 16

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test evaluation", ncols=100):
                images, _ = self._prepare_batch(batch)
                batch_size = images.shape[0]

                with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    reconstructed, _, _ = model_to_use(images)

                # Compute metrics
                l1_loss = torch.abs(reconstructed - images).mean().item()
                total_l1 += l1_loss
                total_ssim += compute_ssim(reconstructed, images)
                total_psnr += compute_psnr(reconstructed, images)
                if self.log_lpips:
                    total_lpips += compute_lpips(reconstructed, images, self.device)

                # Track worst batch
                if l1_loss > worst_loss:
                    worst_loss = l1_loss
                    # Convert to dict for dual mode (triggers dual-channel figure)
                    n_channels = self.cfg.mode.get('in_channels', 1)
                    if n_channels == 2:
                        image_keys = self.cfg.mode.get('image_keys', ['t1_pre', 't1_gd'])
                        orig_dict = {image_keys[0]: images[:, 0:1].cpu(), image_keys[1]: images[:, 1:2].cpu()}
                        gen_dict = {image_keys[0]: reconstructed[:, 0:1].float().cpu(), image_keys[1]: reconstructed[:, 1:2].float().cpu()}
                    else:
                        orig_dict = images.cpu()
                        gen_dict = reconstructed.float().cpu()
                    worst_batch_data = {
                        'original': orig_dict,
                        'generated': gen_dict,
                        'loss': l1_loss,
                    }

                n_batches += 1
                n_samples += batch_size

                # Collect samples for visualization
                if len(sample_inputs) < max_vis_samples:
                    remaining = max_vis_samples - len(sample_inputs)
                    sample_inputs.append(images[:remaining].cpu())
                    sample_outputs.append(reconstructed[:remaining].cpu())

        # Compute averages
        metrics = {
            'l1': total_l1 / n_batches,
            'ssim': total_ssim / n_batches,
            'psnr': total_psnr / n_batches,
            'n_samples': n_samples,
        }

        if self.log_lpips:
            metrics['lpips'] = total_lpips / n_batches

        # Log results
        logger.info(f"Test Results - {label} ({n_samples} samples):")
        logger.info(f"  L1 Loss: {metrics['l1']:.6f}")
        logger.info(f"  SSIM:    {metrics['ssim']:.4f}")
        logger.info(f"  PSNR:    {metrics['psnr']:.2f} dB")
        if 'lpips' in metrics:
            logger.info(f"  LPIPS:   {metrics['lpips']:.4f}")

        # Save results to JSON (with checkpoint name suffix)
        results_path = os.path.join(self.save_dir, f'test_results_{label}.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Test results saved to: {results_path}")

        # Log to TensorBoard (with checkpoint name prefix)
        tb_prefix = f'test_{label}'
        if self.writer is not None:
            self.writer.add_scalar(f'{tb_prefix}/L1', metrics['l1'], 0)
            self.writer.add_scalar(f'{tb_prefix}/SSIM', metrics['ssim'], 0)
            self.writer.add_scalar(f'{tb_prefix}/PSNR', metrics['psnr'], 0)
            if 'lpips' in metrics:
                self.writer.add_scalar(f'{tb_prefix}/LPIPS', metrics['lpips'], 0)

            # Create and save visualization
            if sample_inputs:
                all_inputs = torch.cat(sample_inputs, dim=0)[:max_vis_samples]
                all_outputs = torch.cat(sample_outputs, dim=0)[:max_vis_samples]
                fig = self._create_test_reconstruction_figure(all_inputs, all_outputs, metrics, label)
                self.writer.add_figure(f'{tb_prefix}/reconstructions', fig, 0)
                plt.close(fig)

                # Also save as image file
                fig_path = os.path.join(self.save_dir, f'test_reconstructions_{label}.png')
                fig = self._create_test_reconstruction_figure(all_inputs, all_outputs, metrics, label)
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Test reconstructions saved to: {fig_path}")

            # Log worst batch figure
            if worst_batch_data is not None:
                fig = create_worst_batch_figure(
                    original=worst_batch_data['original'],
                    generated=worst_batch_data['generated'],
                    loss=worst_batch_data['loss'],
                )
                self.writer.add_figure(f'{tb_prefix}/worst_batch', fig, 0)
                plt.close(fig)

        model_to_use.train()
        return metrics

    def _create_test_reconstruction_figure(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        metrics: Dict[str, float],
        label: str = "test"
    ) -> plt.Figure:
        """Create test set reconstruction comparison figure.

        Uses shared create_reconstruction_figure for consistent visualization.

        Args:
            original: Original images [N, C, H, W].
            reconstructed: Reconstructed images [N, C, H, W].
            metrics: Dict with test metrics for title.
            label: Checkpoint label for title (e.g., "best", "latest").

        Returns:
            Matplotlib figure.
        """
        title = f"Test Results ({label})"
        display_metrics = {'SSIM': metrics['ssim'], 'PSNR': metrics['psnr']}
        if 'lpips' in metrics:
            display_metrics['LPIPS'] = metrics['lpips']

        return create_reconstruction_figure(
            original=original,
            generated=reconstructed,
            title=title,
            max_samples=8,
            metrics=display_metrics,
        )

    def close_writer(self) -> None:
        """Close TensorBoard writer. Call after all logging is complete."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
