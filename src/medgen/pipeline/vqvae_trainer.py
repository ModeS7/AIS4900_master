"""
VQ-VAE trainer module for training vector-quantized autoencoders.

This module provides the VQVAETrainer class for training VQVAE models
with discrete latent spaces using vector quantization instead of KL regularization.
Supports the same infrastructure as VAETrainer: TensorBoard logging,
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
from monai.networks.nets import VQVAE, PatchDiscriminator

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
from .metrics import (
    create_reconstruction_figure,
    RegionalMetricsTracker,
    compute_msssim,
    compute_psnr,
    compute_lpips,
    reset_msssim_nan_warning,
)
from .tracking import (
    GradientNormTracker,
    FLOPsTracker,
    create_worst_batch_figure,
)

logger = logging.getLogger(__name__)


def log_vqvae_epoch_summary(
    epoch: int,
    total_epochs: int,
    avg_losses: Dict[str, float],
    val_metrics: Dict[str, float],
    elapsed_time: float
) -> None:
    """Log VQ-VAE epoch completion summary with train and validation metrics.

    Args:
        epoch: Current epoch number (0-indexed).
        total_epochs: Total number of epochs.
        avg_losses: Dict with 'gen', 'disc', 'recon', 'perc', 'vq', 'adv' training losses.
        val_metrics: Dict with 'gen', 'l1', 'msssim', 'psnr' validation metrics (can be empty).
        elapsed_time: Time taken for the epoch in seconds.
    """
    timestamp = time.strftime("%H:%M:%S")
    epoch_pct = ((epoch + 1) / total_epochs) * 100

    # Format validation metrics if available
    if val_metrics:
        val_gen = f"(v:{val_metrics.get('gen', 0):.4f})"
        val_l1 = f"(v:{val_metrics.get('l1', 0):.4f})"
        msssim_str = f"MS-SSIM: {val_metrics.get('msssim', 0):.3f}"
    else:
        val_gen = ""
        val_l1 = ""
        msssim_str = ""

    logger.info(
        f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
        f"G: {avg_losses['gen']:.4f}{val_gen} | "
        f"L1: {avg_losses['recon']:.4f}{val_l1} | "
        f"VQ: {avg_losses['vq']:.4f} | "
        f"D: {avg_losses['disc']:.4f} | "
        f"{msssim_str} | "
        f"Time: {elapsed_time:.1f}s"
    )


class VQVAETrainer:
    """VQ-VAE trainer with full feature parity to VAETrainer.

    Supports training VQ-VAEs for latent diffusion models with:
    - Discrete latent space via vector quantization
    - TensorBoard logging
    - Checkpoint management
    - Multi-GPU support (DDP)
    - EMA weights
    - Gradient clipping
    - Learning rate scheduling

    Key differences from VAETrainer:
    - Uses VQVAE instead of AutoencoderKL
    - Forward returns (reconstruction, vq_loss) instead of (reconstruction, mean, logvar)
    - VQ loss computed internally by quantizer (commitment + codebook loss)
    - No KL divergence

    Args:
        cfg: Hydra configuration object containing all settings.

    Example:
        >>> trainer = VQVAETrainer(cfg)
        >>> trainer.setup_model()
        >>> trainer.train(train_loader, train_dataset)
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # Extract config values
        self.n_epochs: int = cfg.training.epochs
        self.batch_size: int = cfg.training.batch_size
        self.image_size: int = cfg.model.image_size
        self.learning_rate: float = cfg.training.get('learning_rate', 1e-4)
        self.disc_lr: float = cfg.vqvae.get('disc_lr', 5e-4)
        self.warmup_epochs: int = cfg.training.warmup_epochs

        # Compute val_interval: num_validations takes priority over val_interval
        num_validations = cfg.training.get('num_validations', None)
        if num_validations and num_validations > 0:
            self.val_interval: int = max(1, self.n_epochs // num_validations)
        else:
            self.val_interval: int = cfg.training.val_interval
        self.use_multi_gpu: bool = cfg.training.get('use_multi_gpu', False)
        self.use_ema: bool = cfg.training.get('use_ema', True)
        self.ema_decay: float = cfg.training.ema.get('decay', 0.999)

        # Loss weights (from vqvae/default.yaml)
        self.perceptual_weight: float = cfg.vqvae.get('perceptual_weight', 0.1)
        self.adv_weight: float = cfg.vqvae.get('adv_weight', 0.1)

        # VQ-VAE specific config
        self.num_embeddings: int = cfg.vqvae.get('num_embeddings', 512)
        self.embedding_dim: int = cfg.vqvae.get('embedding_dim', 64)
        self.commitment_cost: float = cfg.vqvae.get('commitment_cost', 0.25)
        self.decay: float = cfg.vqvae.get('decay', 0.99)
        self.epsilon: float = cfg.vqvae.get('epsilon', 1e-5)

        # Architecture config
        self.channels: tuple = tuple(cfg.vqvae.get('channels', [96, 96, 192]))
        self.num_res_layers: int = cfg.vqvae.get('num_res_layers', 3)
        self.num_res_channels: tuple = tuple(cfg.vqvae.get('num_res_channels', [96, 96, 192]))
        self.downsample_parameters: tuple = tuple(
            tuple(p) for p in cfg.vqvae.get('downsample_parameters', [[2, 4, 1, 1]] * 3)
        )
        self.upsample_parameters: tuple = tuple(
            tuple(p) for p in cfg.vqvae.get('upsample_parameters', [[2, 4, 1, 1, 0]] * 3)
        )

        # GAN options
        self.disable_gan: bool = cfg.vqvae.get('disable_gan', False)
        self.use_constant_lr: bool = cfg.get('progressive', {}).get('use_constant_lr', False)

        # Discriminator config
        self.disc_num_layers: int = cfg.vqvae.get('disc_num_layers', 3)
        self.disc_num_channels: int = cfg.vqvae.get('disc_num_channels', 64)

        # torch.compile option (default: True)
        self.use_compile: bool = cfg.training.get('use_compile', True)

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
                    exp_name = cfg.training.get('name', '')
                    mode_name = cfg.mode.get('name', 'dual')
                    self.run_name = f"{exp_name}{self.image_size}_{timestamp}"
                    self.save_dir = os.path.join(cfg.paths.model_dir, 'vqvae_2d', mode_name, self.run_name)

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
        self.optimizer_g: Optional[AdamW] = None
        self.optimizer_d: Optional[AdamW] = None
        self.lr_scheduler_g: Optional[LRScheduler] = None
        self.lr_scheduler_d: Optional[LRScheduler] = None
        self.perceptual_loss_fn: Optional[nn.Module] = None
        self.adv_loss_fn: Optional[PatchAdversarialLoss] = None

        # Logging config
        logging_cfg = cfg.training.get('logging', {})
        self.log_grad_norm: bool = logging_cfg.get('grad_norm', True)
        self.log_flops: bool = logging_cfg.get('flops', True)
        self.log_msssim: bool = logging_cfg.get('msssim', True)
        self.log_regional_losses: bool = logging_cfg.get('regional_losses', True)

        # Gradient norm tracking
        self._grad_tracker_g = GradientNormTracker()
        self._grad_tracker_d = GradientNormTracker()

        # FLOPs tracking
        self._flops_tracker = FLOPsTracker()

        # Validation loader (set in train())
        self.val_loader: Optional[DataLoader] = None

    def _setup_distributed(self) -> Tuple[int, int, int, torch.device]:
        """Setup distributed training with dynamic port allocation."""
        return setup_distributed()

    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize VQ-VAE model, discriminator, optimizers, and loss functions.

        Args:
            pretrained_checkpoint: Optional path to checkpoint for loading pretrained weights.
        """
        # For VQ-VAE, in_channels must equal out_channels (autoencoder)
        n_channels = self.cfg.mode.get('in_channels', 1)

        # Create VQVAE model
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

        # Create PatchDiscriminator (only if GAN is enabled)
        raw_disc = None
        if not self.disable_gan:
            raw_disc = PatchDiscriminator(
                spatial_dims=2,
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
                        logger.info(f"Loaded VQ-VAE weights from {pretrained_checkpoint}")
                if 'discriminator_state_dict' in checkpoint and raw_disc is not None:
                    raw_disc.load_state_dict(checkpoint['discriminator_state_dict'])
                    if self.is_main_process:
                        logger.info(f"Loaded discriminator weights from {pretrained_checkpoint}")
            except FileNotFoundError:
                if self.is_main_process:
                    logger.warning(f"Pretrained checkpoint not found: {pretrained_checkpoint}")

        # Wrap VQVAE model with DDP and/or torch.compile
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

        # Setup perceptual loss
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

        # Create EMA wrapper if enabled
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
            vqvae_params = sum(p.numel() for p in self.model_raw.parameters())
            logger.info(f"VQ-VAE initialized: {vqvae_params / 1e6:.1f}M parameters")
            logger.info(f"  Codebook: {self.num_embeddings} embeddings x {self.embedding_dim} dim")
            logger.info(f"  Channels: {self.channels}")
            if not self.disable_gan:
                disc_params = sum(p.numel() for p in self.discriminator_raw.parameters())
                logger.info(f"Discriminator initialized: {disc_params / 1e6:.1f}M parameters")
            else:
                logger.info("GAN disabled - discriminator not created")

            # Compute latent size based on downsampling
            n_downsamples = len(self.downsample_parameters)
            latent_size = self.image_size // (2 ** n_downsamples)
            logger.info(f"Latent shape: [{self.embedding_dim}, {latent_size}, {latent_size}]")
            logger.info(f"Loss weights - Perceptual: {self.perceptual_weight}, Adv: {self.adv_weight}")
            logger.info(f"VQ params - Commitment: {self.commitment_cost}, Decay: {self.decay}")

    def _save_metadata(self) -> None:
        """Save training configuration to metadata.json."""
        os.makedirs(self.save_dir, exist_ok=True)

        config_path = os.path.join(self.save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        # Save VQVAE config separately for easy loading
        n_channels = self.cfg.mode.get('in_channels', 1)
        vqvae_config = {
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

        metadata = {
            'type': 'vqvae',
            'epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'image_size': self.image_size,
            'learning_rate': self.learning_rate,
            'perceptual_weight': self.perceptual_weight,
            'adv_weight': self.adv_weight,
            'warmup_epochs': self.warmup_epochs,
            'val_interval': self.val_interval,
            'multi_gpu': self.use_multi_gpu,
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay if self.use_ema else None,
            'vqvae_config': vqvae_config,
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
            'final_loss': float(final_loss),
            'final_recon_loss': float(final_recon),
            'best_loss': float(self.best_loss),
            'total_time_seconds': float(total_time),
            'total_time_hours': float(total_time / 3600),
            'completed_at': datetime.now().isoformat(),
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _update_ema(self) -> None:
        """Update EMA model weights."""
        if self.ema is not None:
            self.ema.update()

    def _prepare_batch(
        self, batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare batch tensor from dataloader output.

        Args:
            batch: Input batch - can be:
                - Dict of tensors (keyed by modality)
                - Tuple of (image, seg) tensors
                - Single tensor (with optional seg as extra channel)

        Returns:
            Tuple of (images, mask):
            - images: Tensor [B, C, H, W] for VQ-VAE input
            - mask: Optional segmentation mask [B, 1, H, W] for regional metrics
        """
        # Handle tuple of (image, seg) from multi-modality loaders
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            images, mask = batch
            if hasattr(images, 'as_tensor'):
                images = images.as_tensor()
            if hasattr(mask, 'as_tensor'):
                mask = mask.as_tensor()
            return images.to(self.device), mask.to(self.device)

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
        n_image_channels = self.cfg.mode.get('in_channels', 2)
        if tensor.shape[1] > n_image_channels:
            images = tensor[:, :n_image_channels, :, :]
            mask = tensor[:, n_image_channels:n_image_channels + 1, :, :]
            return images, mask
        else:
            return tensor, None

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step with optional GAN training.

        Training follows VQ-VAE approach:
        1. Train discriminator on real vs fake (only if GAN enabled)
        2. Train generator with L1 + perceptual + VQ + (optional) adversarial loss

        Args:
            batch: Input batch.

        Returns:
            Dict with 'gen', 'disc', 'recon', 'perc', 'vq', 'adv' losses.
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
                    reconstruction_for_d, _ = self.model(images)

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

            if self.log_grad_norm:
                self._grad_tracker_d.update(grad_norm_d)

        # ==================== Generator Step ====================
        self.optimizer_g.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=torch.bfloat16):
            # VQVAE forward returns (reconstruction, vq_loss)
            reconstruction, vq_loss = self.model(images)

            # L1 reconstruction loss
            l1_loss = torch.abs(reconstruction - images).mean()

            # Perceptual loss
            p_loss = self.perceptual_loss_fn(reconstruction, images)

            # Adversarial loss (only if GAN enabled)
            if not self.disable_gan:
                logits_fake_for_g = self.discriminator(reconstruction.contiguous())
                adv_loss = self.adv_loss_fn(
                    logits_fake_for_g, target_is_real=True, for_discriminator=False
                )

            # Total generator loss
            # Note: vq_loss already includes commitment_cost weighting from the model
            g_loss = (
                l1_loss
                + self.perceptual_weight * p_loss
                + vq_loss
                + self.adv_weight * adv_loss
            )

        g_loss.backward()
        grad_norm_g = 0.0
        if grad_clip > 0:
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                self.model_raw.parameters(), max_norm=grad_clip
            ).item()
        self.optimizer_g.step()

        if self.log_grad_norm:
            self._grad_tracker_g.update(grad_norm_g)

        if self.use_ema:
            self._update_ema()

        return {
            'gen': g_loss.item(),
            'disc': d_loss.item() if not self.disable_gan else 0.0,
            'recon': l1_loss.item(),
            'perc': p_loss.item(),
            'vq': vq_loss.item(),
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
            Dict with average losses: 'gen', 'disc', 'recon', 'perc', 'vq', 'adv'.
        """
        self.model.train()
        if not self.disable_gan and self.discriminator is not None:
            self.discriminator.train()

        self._reset_grad_norm_tracking()

        epoch_losses = {'gen': 0, 'disc': 0, 'recon': 0, 'perc': 0, 'vq': 0, 'adv': 0}

        epoch_iter = create_epoch_iterator(
            data_loader, epoch, self.is_cluster, self.is_main_process
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

        n_batches = len(data_loader)
        return {key: val / n_batches for key, val in epoch_losses.items()}

    def compute_validation_losses(self, epoch: int) -> Dict[str, float]:
        """Compute losses and metrics on full validation set.

        Args:
            epoch: Current epoch number.

        Returns:
            Dict with validation metrics.
        """
        if self.val_loader is None:
            return {}

        model_to_use = self.ema.ema_model if self.ema is not None else self.model_raw
        model_to_use.eval()

        total_l1 = 0.0
        total_perc = 0.0
        total_vq = 0.0
        total_gen = 0.0
        total_msssim = 0.0
        total_psnr = 0.0
        total_lpips = 0.0
        n_batches = 0

        worst_loss = 0.0
        worst_batch_data = None

        regional_tracker = None
        if self.log_regional_losses:
            regional_tracker = RegionalMetricsTracker(
                image_size=self.cfg.model.image_size,
                fov_mm=self.cfg.paths.get('fov_mm', 240.0),
                loss_fn='l1',
                device=self.device,
            )

        if self.use_compile:
            torch.compiler.cudagraph_mark_step_begin()

        reset_msssim_nan_warning()

        with torch.no_grad():
            for batch in self.val_loader:
                images, mask = self._prepare_batch(batch)

                with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    reconstructed, vq_loss = model_to_use(images)

                    l1_loss = torch.abs(reconstructed - images).mean()
                    p_loss = self.perceptual_loss_fn(reconstructed, images)
                    g_loss = l1_loss + self.perceptual_weight * p_loss + vq_loss

                loss_val = g_loss.item()
                total_l1 += l1_loss.item()
                total_perc += p_loss.item()
                total_vq += vq_loss.item()
                total_gen += loss_val

                # Track worst batch
                if loss_val > worst_loss:
                    worst_loss = loss_val
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
                        'loss_breakdown': {'L1': l1_loss.item(), 'Perc': p_loss.item(), 'VQ': vq_loss.item()},
                    }

                if self.log_msssim:
                    total_msssim += compute_msssim(reconstructed, images)
                total_psnr += compute_psnr(reconstructed, images)
                total_lpips += compute_lpips(reconstructed, images, device=self.device)

                if regional_tracker is not None and mask is not None:
                    regional_tracker.update(reconstructed, images, mask)

                n_batches += 1

        model_to_use.train()

        metrics = {
            'l1': total_l1 / n_batches,
            'perc': total_perc / n_batches,
            'vq': total_vq / n_batches,
            'gen': total_gen / n_batches,
            'psnr': total_psnr / n_batches,
            'lpips': total_lpips / n_batches,
        }

        if self.log_msssim:
            metrics['msssim'] = total_msssim / n_batches

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Loss/L1_val', metrics['l1'], epoch)
            self.writer.add_scalar('Loss/Perceptual_val', metrics['perc'], epoch)
            self.writer.add_scalar('Loss/VQ_val', metrics['vq'], epoch)
            self.writer.add_scalar('Loss/Generator_val', metrics['gen'], epoch)
            self.writer.add_scalar('Validation/PSNR', metrics['psnr'], epoch)
            self.writer.add_scalar('Validation/LPIPS', metrics['lpips'], epoch)
            if 'msssim' in metrics:
                self.writer.add_scalar('Validation/MS-SSIM', metrics['msssim'], epoch)

            if worst_batch_data is not None:
                fig = create_worst_batch_figure(
                    original=worst_batch_data['original'],
                    generated=worst_batch_data['generated'],
                    loss=worst_batch_data['loss'],
                    loss_breakdown=worst_batch_data['loss_breakdown'],
                )
                self.writer.add_figure('Validation/worst_batch', fig, epoch)
                plt.close(fig)

            if regional_tracker is not None:
                regional_tracker.log_to_tensorboard(self.writer, epoch, prefix='regional')

        return metrics

    def _save_vqvae_checkpoint(self, epoch: int, filename: str) -> str:
        """Save VQ-VAE checkpoint with config for easy loading.

        Args:
            epoch: Current epoch number.
            filename: Checkpoint filename (without extension).

        Returns:
            Path to saved checkpoint (.pt).
        """
        n_channels = self.cfg.mode.get('in_channels', 1)
        model_config = {
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

        extra_state = {
            'disable_gan': self.disable_gan,
            'use_constant_lr': self.use_constant_lr,
        }

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

    def _compute_per_modality_validation(self, epoch: int) -> None:
        """Compute and log validation metrics for each modality separately.

        For multi-modality training, this logs PSNR, LPIPS, MS-SSIM and regional
        metrics for each modality (bravo, t1_pre, t1_gd) to compare with
        single-modality experiments.

        Args:
            epoch: Current epoch number.
        """
        if not hasattr(self, 'per_modality_val_loaders') or not self.per_modality_val_loaders:
            return

        model_to_use = self.ema.ema_model if self.ema is not None else self.model_raw
        model_to_use.eval()

        for modality, loader in self.per_modality_val_loaders.items():
            total_psnr = 0.0
            total_lpips = 0.0
            total_msssim = 0.0
            n_batches = 0

            # Initialize regional tracker for this modality
            regional_tracker = None
            if self.log_regional_losses:
                regional_tracker = RegionalMetricsTracker(
                    image_size=self.cfg.model.image_size,
                    fov_mm=self.cfg.paths.get('fov_mm', 240.0),
                    loss_fn='l1',
                    device=self.device,
                )

            with torch.no_grad():
                for batch in loader:
                    images, mask = self._prepare_batch(batch)

                    with torch.amp.autocast('cuda', enabled=True, dtype=self.weight_dtype):
                        reconstructed, _ = model_to_use(images)

                    # Compute metrics
                    total_psnr += compute_psnr(reconstructed, images)
                    total_lpips += compute_lpips(reconstructed, images, device=self.device)
                    if self.log_msssim:
                        total_msssim += compute_msssim(reconstructed, images)

                    # Regional tracking (tumor vs background)
                    if regional_tracker is not None and mask is not None:
                        regional_tracker.update(reconstructed, images, mask)

                    n_batches += 1

            # Compute averages and log
            if n_batches > 0 and self.writer is not None:
                avg_psnr = total_psnr / n_batches
                avg_lpips = total_lpips / n_batches
                self.writer.add_scalar(f'Validation/PSNR_{modality}', avg_psnr, epoch)
                self.writer.add_scalar(f'Validation/LPIPS_{modality}', avg_lpips, epoch)
                if self.log_msssim:
                    avg_msssim = total_msssim / n_batches
                    self.writer.add_scalar(f'Validation/MS-SSIM_{modality}', avg_msssim, epoch)

                # Log regional metrics for this modality
                if regional_tracker is not None:
                    regional_tracker.log_to_tensorboard(
                        self.writer, epoch, prefix=f'regional_{modality}'
                    )

        model_to_use.train()

    def train(
        self,
        train_loader: DataLoader,
        train_dataset: Dataset,
        val_loader: Optional[DataLoader] = None,
        start_epoch: int = 0,
        max_epochs: Optional[int] = None,
        early_stop_fn: Optional[callable] = None,
        per_modality_val_loaders: Optional[Dict[str, DataLoader]] = None,
    ) -> int:
        """Execute the main training loop.

        Args:
            train_loader: Training data loader.
            train_dataset: Training dataset.
            val_loader: Optional validation data loader.
            start_epoch: Epoch to start from (for resuming training).
            max_epochs: Maximum epochs to train.
            early_stop_fn: Optional callback for early stopping.
            per_modality_val_loaders: Optional dict mapping modality names to
                separate validation loaders for per-modality metric tracking.

        Returns:
            The last completed epoch number.
        """
        n_epochs = max_epochs if max_epochs is not None else self.n_epochs
        self.val_loader = val_loader
        self.per_modality_val_loaders = per_modality_val_loaders
        total_start = time.time()

        # Measure FLOPs on first batch
        if self.log_flops:
            try:
                first_batch = next(iter(train_loader))
                sample_images, _ = self._prepare_batch(first_batch)
                self._flops_tracker.measure(
                    model=self.model_raw,
                    sample_input=sample_images[:1],
                    steps_per_epoch=len(train_loader),
                    timesteps=None,
                    is_main_process=self.is_main_process,
                )
            except Exception as e:
                if self.is_main_process:
                    logger.warning(f"Could not measure FLOPs: {e}")

        avg_losses = {'gen': float('inf'), 'disc': float('inf'), 'recon': float('inf'),
                      'perc': float('inf'), 'vq': float('inf'), 'adv': float('inf')}

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

                if self.use_multi_gpu:
                    loss_tensor = torch.tensor(
                        [avg_losses['gen'], avg_losses['disc'], avg_losses['recon'],
                         avg_losses['perc'], avg_losses['vq'], avg_losses['adv']],
                        device=self.device
                    )
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    loss_vals = (loss_tensor / self.world_size).cpu().numpy()
                    avg_losses = dict(zip(['gen', 'disc', 'recon', 'perc', 'vq', 'adv'], loss_vals))

                epoch_time = time.time() - epoch_start

                if not self.use_constant_lr:
                    if self.lr_scheduler_g is not None:
                        self.lr_scheduler_g.step()
                    if not self.disable_gan and self.lr_scheduler_d is not None:
                        self.lr_scheduler_d.step()

                if self.is_main_process:
                    val_metrics = self.compute_validation_losses(epoch)
                    self._compute_per_modality_validation(epoch)
                    log_vqvae_epoch_summary(epoch, n_epochs, avg_losses, val_metrics, epoch_time)

                    if self.writer is not None:
                        self.writer.add_scalar('Loss/Generator_train', avg_losses['gen'], epoch)
                        self.writer.add_scalar('Loss/Discriminator', avg_losses['disc'], epoch)
                        self.writer.add_scalar('Loss/L1_train', avg_losses['recon'], epoch)
                        self.writer.add_scalar('Loss/Perceptual_train', avg_losses['perc'], epoch)
                        self.writer.add_scalar('Loss/VQ_train', avg_losses['vq'], epoch)
                        self.writer.add_scalar('Loss/Adversarial', avg_losses['adv'], epoch)

                        if not self.use_constant_lr and self.lr_scheduler_g is not None:
                            self.writer.add_scalar('LR/Generator', self.lr_scheduler_g.get_last_lr()[0], epoch)
                        else:
                            self.writer.add_scalar('LR/Generator', self.learning_rate, epoch)

                        if not self.disable_gan:
                            if not self.use_constant_lr and self.lr_scheduler_d is not None:
                                self.writer.add_scalar('LR/Discriminator', self.lr_scheduler_d.get_last_lr()[0], epoch)
                            else:
                                self.writer.add_scalar('LR/Discriminator', self.disc_lr, epoch)

                        self._log_grad_norms(epoch)
                        self._flops_tracker.log_epoch(self.writer, epoch)

                    is_val_epoch = (epoch + 1) % self.val_interval == 0

                    if is_val_epoch or (epoch + 1) == n_epochs:
                        self._save_vqvae_checkpoint(epoch, "latest")

                        val_gen_loss = val_metrics.get('gen', avg_losses['gen'])
                        if val_gen_loss < self.best_loss:
                            self.best_loss = val_gen_loss
                            self._save_vqvae_checkpoint(epoch, "best")
                            logger.info(f"New best model saved (val G loss: {val_gen_loss:.6f})")

                    if early_stop_fn is not None:
                        if early_stop_fn(epoch, avg_losses, val_metrics):
                            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                            break

        finally:
            total_time = time.time() - total_start

            if self.is_main_process:
                logger.info(f"Training completed! Total time: {total_time:.1f}s ({total_time / 3600:.1f}h)")
                self._update_metadata_final(avg_losses['gen'], avg_losses['recon'], total_time)

            if self.use_multi_gpu:
                try:
                    dist.destroy_process_group()
                except Exception as e:
                    logger.warning(f"Error destroying process group: {e}")

        return last_epoch

    def close_writer(self) -> None:
        """Close TensorBoard writer. Call after all logging is complete."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
