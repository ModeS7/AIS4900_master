"""
DC-AE (Deep Compression Autoencoder) trainer module.

This module provides the DCAETrainer class for training DC-AE models
for high-compression 2D MRI slice encoding (32× or 64× spatial compression).

Based on MIT HAN Lab's DC-AE: https://arxiv.org/abs/2410.10733

Key differences from VAETrainer:
- Deterministic encoder (no KL divergence)
- Higher compression (32× or 64× vs 4-8×)
- Uses diffusers AutoencoderDC instead of MONAI AutoencoderKL
- Supports pretrained ImageNet weights for fine-tuning
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
import torch
import torch.nn as nn
from diffusers import AutoencoderDC
from ema_pytorch import EMA
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from monai.losses import PatchAdversarialLoss
from monai.networks.nets import PatchDiscriminator

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
    compute_msssim,
    compute_psnr,
    compute_lpips,
    RegionalMetricsTracker,
    reset_msssim_nan_warning,
)
from .tracking import (
    GradientNormTracker,
    FLOPsTracker,
    create_worst_batch_figure,
)

logger = logging.getLogger(__name__)


def log_dcae_epoch_summary(
    epoch: int,
    total_epochs: int,
    avg_losses: Dict[str, float],
    val_metrics: Dict[str, float],
    elapsed_time: float
) -> None:
    """Log DC-AE epoch completion summary."""
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


class DCAETrainer:
    """DC-AE trainer for high-compression 2D MRI encoding.

    Uses diffusers AutoencoderDC with:
    - L1 + Perceptual loss (no KL - deterministic encoder)
    - Optional GAN training (Phase 3)
    - Pretrained ImageNet weights support

    Args:
        cfg: Hydra configuration object containing all settings.

    Example:
        >>> trainer = DCAETrainer(cfg)
        >>> trainer.setup_model()
        >>> trainer.train(train_loader, train_dataset)
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # Extract config values
        self.n_epochs: int = cfg.training.epochs
        self.batch_size: int = cfg.training.batch_size
        self.image_size: int = cfg.dcae.get('image_size', 256)
        self.learning_rate: float = cfg.training.get('learning_rate', 1e-4)
        self.disc_lr: float = cfg.dcae.get('disc_lr', 5e-4)
        self.warmup_epochs: int = cfg.training.warmup_epochs
        self.val_interval: int = cfg.training.val_interval
        self.use_multi_gpu: bool = cfg.training.get('use_multi_gpu', False)
        self.use_ema: bool = cfg.training.get('use_ema', False)
        self.ema_decay: float = cfg.training.ema.get('decay', 0.999) if self.use_ema else 0.999

        # Loss weights (no KL for DC-AE - deterministic)
        self.l1_weight: float = cfg.dcae.get('l1_weight', 1.0)
        self.perceptual_weight: float = cfg.dcae.get('perceptual_weight', 0.1)
        self.adv_weight: float = cfg.dcae.get('adv_weight', 0.0)

        # Training phase (1=no GAN, 3=with GAN)
        self.training_phase: int = cfg.training.get('phase', 1)
        self.disable_gan: bool = (self.adv_weight == 0.0) or (self.training_phase == 1)

        # DC-AE architecture config
        self.latent_channels: int = cfg.dcae.latent_channels
        self.compression_ratio: int = cfg.dcae.compression_ratio
        self.scaling_factor: float = cfg.dcae.get('scaling_factor', 1.0)

        # Pretrained model path (from HuggingFace or null)
        self.pretrained: Optional[str] = cfg.dcae.get('pretrained', None)

        # torch.compile option
        self.use_compile: bool = cfg.training.get('use_compile', True)

        # Precision config
        precision_cfg = cfg.training.get('precision', {})
        dtype_str = precision_cfg.get('dtype', 'bf16')
        self.weight_dtype: torch.dtype = {
            'bf16': torch.bfloat16,
            'fp16': torch.float16,
            'fp32': torch.float32,
        }.get(dtype_str, torch.bfloat16)

        # Cluster mode
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
            try:
                from hydra.core.hydra_config import HydraConfig
                self.save_dir = HydraConfig.get().runtime.output_dir
            except (ImportError, ValueError, AttributeError):
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                exp_name = cfg.training.get('name', '')
                mode_name = cfg.mode.get('name', 'multi_modality')
                self.save_dir = os.path.join(
                    cfg.paths.model_dir, 'dcae', mode_name,
                    f"{exp_name}{self.image_size}_{timestamp}"
                )

            tensorboard_dir = os.path.join(self.save_dir, "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer: Optional[SummaryWriter] = SummaryWriter(tensorboard_dir)
            self.best_loss: float = float('inf')
        else:
            self.writer = None
            self.save_dir = ""
            self.best_loss = float('inf')

        # Initialize model components
        self.model: Optional[nn.Module] = None
        self.model_raw: Optional[AutoencoderDC] = None
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
        self.log_msssim: bool = logging_cfg.get('msssim', True)
        self.log_psnr: bool = logging_cfg.get('psnr', True)
        self.log_lpips: bool = logging_cfg.get('lpips', True)
        self.log_regional_losses: bool = logging_cfg.get('regional_losses', True)
        self.log_flops: bool = logging_cfg.get('flops', True)

        # Gradient norm tracking
        self._grad_tracker_g = GradientNormTracker()
        self._grad_tracker_d = GradientNormTracker()

        # FLOPs tracking
        self._flops_tracker: Optional[FLOPsTracker] = None
        self._flops_logged: bool = False

        # Validation loader
        self.val_loader: Optional[DataLoader] = None
        self.per_modality_val_loaders: Optional[Dict[str, DataLoader]] = None

    def _setup_distributed(self) -> Tuple[int, int, int, torch.device]:
        """Setup distributed training."""
        return setup_distributed()

    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize DC-AE model, discriminator, optimizers, and loss functions.

        Args:
            pretrained_checkpoint: Optional path to checkpoint for resuming training.
        """
        n_channels = self.cfg.mode.get('in_channels', 1)

        # Create AutoencoderDC
        if self.pretrained:
            # Load pretrained from HuggingFace
            if self.is_main_process:
                logger.info(f"Loading pretrained DC-AE from: {self.pretrained}")

            raw_model = AutoencoderDC.from_pretrained(
                self.pretrained,
                torch_dtype=torch.float32,  # Load in fp32, cast later
            )

            # Modify input layer for grayscale (1 channel) if pretrained was 3 channels
            if raw_model.encoder.conv_in.in_channels != n_channels:
                if self.is_main_process:
                    logger.info(f"Replacing conv_in: {raw_model.encoder.conv_in.in_channels} -> {n_channels} channels")
                old_conv = raw_model.encoder.conv_in
                new_conv = nn.Conv2d(
                    n_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                )
                # Initialize with mean of RGB weights for grayscale
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                    if old_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias)
                raw_model.encoder.conv_in = new_conv

            # Modify output layer for grayscale
            if raw_model.decoder.conv_out.conv.out_channels != n_channels:
                if self.is_main_process:
                    logger.info(f"Replacing conv_out: {raw_model.decoder.conv_out.conv.out_channels} -> {n_channels} channels")
                old_conv = raw_model.decoder.conv_out.conv
                new_conv = nn.Conv2d(
                    old_conv.in_channels, n_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                )
                with torch.no_grad():
                    # Average the output weights
                    new_conv.weight.copy_(old_conv.weight.mean(dim=0, keepdim=True))
                    if old_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias[:n_channels])
                raw_model.decoder.conv_out.conv = new_conv

            raw_model = raw_model.to(self.device)
        else:
            # Train from scratch
            if self.is_main_process:
                logger.info("Creating DC-AE from scratch")

            raw_model = AutoencoderDC(
                in_channels=n_channels,
                latent_channels=self.latent_channels,
                encoder_block_out_channels=tuple(self.cfg.dcae.encoder_block_out_channels),
                decoder_block_out_channels=tuple(self.cfg.dcae.decoder_block_out_channels),
                encoder_layers_per_block=tuple(self.cfg.dcae.encoder_layers_per_block),
                decoder_layers_per_block=tuple(self.cfg.dcae.decoder_layers_per_block),
                encoder_qkv_multiscales=tuple(tuple(x) for x in self.cfg.dcae.encoder_qkv_multiscales),
                decoder_qkv_multiscales=tuple(tuple(x) for x in self.cfg.dcae.decoder_qkv_multiscales),
                encoder_block_types=self.cfg.dcae.encoder_block_types,
                decoder_block_types=self.cfg.dcae.decoder_block_types,
                downsample_block_type=self.cfg.dcae.downsample_block_type,
                upsample_block_type=self.cfg.dcae.upsample_block_type,
                encoder_out_shortcut=self.cfg.dcae.encoder_out_shortcut,
                decoder_in_shortcut=self.cfg.dcae.decoder_in_shortcut,
                scaling_factor=self.scaling_factor,
            ).to(self.device)

        # Create PatchDiscriminator (only if GAN enabled)
        raw_disc = None
        if not self.disable_gan:
            raw_disc = PatchDiscriminator(
                spatial_dims=2,
                in_channels=n_channels,
                channels=64,
                num_layers_d=3,
            ).to(self.device)

        # Wrap for training (DDP, compile, etc.)
        self.model_raw = raw_model
        self.model, _ = wrap_model_for_training(
            raw_model,
            use_multi_gpu=self.use_multi_gpu,
            local_rank=getattr(self, 'local_rank', 0),
            use_compile=self.use_compile,
            is_main_process=self.is_main_process,
        )

        if raw_disc is not None:
            self.discriminator_raw = raw_disc
            self.discriminator, _ = wrap_model_for_training(
                raw_disc,
                use_multi_gpu=self.use_multi_gpu,
                local_rank=getattr(self, 'local_rank', 0),
                use_compile=self.use_compile,
                is_main_process=self.is_main_process,
            )

        # Create optimizers
        self.optimizer_g = AdamW(self.model_raw.parameters(), lr=self.learning_rate)

        if not self.disable_gan and self.discriminator_raw is not None:
            self.optimizer_d = AdamW(self.discriminator_raw.parameters(), lr=self.disc_lr)

        # Create schedulers
        self.lr_scheduler_g = create_warmup_cosine_scheduler(
            self.optimizer_g, self.warmup_epochs, self.n_epochs
        )

        if not self.disable_gan and self.optimizer_d is not None:
            self.lr_scheduler_d = create_warmup_cosine_scheduler(
                self.optimizer_d, self.warmup_epochs, self.n_epochs
            )

        # Perceptual loss
        if self.perceptual_weight > 0:
            self.perceptual_loss_fn = PerceptualLoss(
                device=self.device,
                use_compile=self.use_compile,
            )

        # Adversarial loss
        if not self.disable_gan:
            self.adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

        # EMA
        if self.use_ema:
            self.ema = EMA(
                self.model_raw,
                beta=self.ema_decay,
                update_after_step=100,
                update_every=10,
            )

        # Save config
        if self.is_main_process:
            self._save_metadata()

        # Log model info
        if self.is_main_process:
            n_params = sum(p.numel() for p in self.model_raw.parameters()) / 1e6
            logger.info(f"DC-AE parameters: {n_params:.2f}M")
            logger.info(f"Compression: {self.compression_ratio}× | Latent channels: {self.latent_channels}")
            logger.info(f"GAN: {'Disabled' if self.disable_gan else 'Enabled'}")

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> int:
        """Load checkpoint to resume training."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model_raw.load_state_dict(checkpoint['model_state_dict'])
        if self.is_main_process:
            logger.info(f"Loaded DC-AE weights from {checkpoint_path}")

        if not self.disable_gan and self.discriminator_raw is not None:
            if 'discriminator_state_dict' in checkpoint:
                self.discriminator_raw.load_state_dict(checkpoint['discriminator_state_dict'])

        if load_optimizer:
            if 'optimizer_g_state_dict' in checkpoint and self.optimizer_g is not None:
                self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])

            if 'scheduler_g_state_dict' in checkpoint and self.lr_scheduler_g is not None:
                self.lr_scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])

        if self.use_ema and self.ema is not None:
            if 'ema_state_dict' in checkpoint:
                self.ema.load_state_dict(checkpoint['ema_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        if self.is_main_process:
            logger.info(f"Resuming from epoch {epoch + 1}")

        return epoch

    def _save_metadata(self) -> None:
        """Save training configuration."""
        os.makedirs(self.save_dir, exist_ok=True)

        config_path = os.path.join(self.save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        metadata = {
            'type': 'dcae',
            'compression_ratio': self.compression_ratio,
            'latent_channels': self.latent_channels,
            'scaling_factor': self.scaling_factor,
            'pretrained': self.pretrained,
            'epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'image_size': self.image_size,
            'learning_rate': self.learning_rate,
            'l1_weight': self.l1_weight,
            'perceptual_weight': self.perceptual_weight,
            'adv_weight': self.adv_weight,
            'training_phase': self.training_phase,
            'created_at': datetime.now().isoformat(),
        }

        metadata_path = os.path.join(self.save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Config saved to: {config_path}")

    def _prepare_batch(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare batch tensor and optional mask from dataloader output.

        Args:
            batch: Batch from dataloader (dict, tuple, or tensor).

        Returns:
            Tuple of (images, mask) where mask may be None.
        """
        mask = None
        if isinstance(batch, dict):
            images = batch.get('image', batch.get('images'))
            mask = batch.get('mask', batch.get('seg'))
        elif isinstance(batch, (tuple, list)):
            images = batch[0]
            if len(batch) > 1:
                mask = batch[1]
        else:
            images = batch

        images = images.to(self.device, dtype=self.weight_dtype)
        if mask is not None:
            mask = mask.to(self.device)

        return images, mask

    def train_step(self, batch) -> Dict[str, float]:
        """Execute single training step."""
        images, _ = self._prepare_batch(batch)  # mask not used in training

        # ========================
        # Generator (Encoder-Decoder) Update
        # ========================
        self.optimizer_g.zero_grad()

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            # DC-AE forward: deterministic encoding (no sampling)
            latent = self.model.encode(images, return_dict=False)[0]
            reconstruction = self.model.decode(latent, return_dict=False)[0]

            # L1 reconstruction loss
            l1_loss = torch.nn.functional.l1_loss(reconstruction, images)

            # Perceptual loss
            if self.perceptual_weight > 0 and self.perceptual_loss_fn is not None:
                p_loss = self.perceptual_loss_fn(reconstruction.float(), images.float())
            else:
                p_loss = torch.tensor(0.0, device=self.device)

            # Adversarial loss (generator wants discriminator to output 1)
            if not self.disable_gan and self.discriminator is not None:
                logits_fake = self.discriminator(reconstruction)
                adv_loss = self.adv_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                adv_loss = torch.tensor(0.0, device=self.device)

            # Total generator loss
            g_loss = (
                self.l1_weight * l1_loss +
                self.perceptual_weight * p_loss +
                self.adv_weight * adv_loss
            )

        # Backward and step
        g_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model_raw.parameters(), 1.0)
        self._grad_tracker_g.update(grad_norm)
        self.optimizer_g.step()

        # Update EMA
        if self.ema is not None:
            self.ema.update()

        # ========================
        # Discriminator Update
        # ========================
        d_loss = torch.tensor(0.0, device=self.device)

        if not self.disable_gan and self.discriminator is not None:
            self.optimizer_d.zero_grad()

            with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                logits_real = self.discriminator(images)
                loss_real = self.adv_loss_fn(logits_real, target_is_real=True, for_discriminator=True)

                logits_fake = self.discriminator(reconstruction.detach())
                loss_fake = self.adv_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)

                d_loss = (loss_real + loss_fake) * 0.5

            d_loss.backward()
            grad_norm_d = torch.nn.utils.clip_grad_norm_(self.discriminator_raw.parameters(), 1.0)
            self._grad_tracker_d.update(grad_norm_d)
            self.optimizer_d.step()

        return {
            'gen': g_loss.item(),
            'disc': d_loss.item(),
            'recon': l1_loss.item(),
            'perc': p_loss.item(),
            'adv': adv_loss.item(),
        }

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        if self.discriminator is not None:
            self.discriminator.train()

        total_losses = {'gen': 0.0, 'disc': 0.0, 'recon': 0.0, 'perc': 0.0, 'adv': 0.0}
        n_batches = 0

        # Reset gradient trackers
        self._grad_tracker_g.reset()
        self._grad_tracker_d.reset()

        disable_pbar = not self.is_main_process or self.is_cluster
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=disable_pbar)

        for batch in pbar:
            losses = self.train_step(batch)

            for k, v in losses.items():
                total_losses[k] += v
            n_batches += 1

            if not disable_pbar:
                pbar.set_postfix({
                    'G': f"{losses['gen']:.4f}",
                    'L1': f"{losses['recon']:.4f}",
                })

        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}

        # Step schedulers
        if self.lr_scheduler_g is not None:
            self.lr_scheduler_g.step()
        if self.lr_scheduler_d is not None:
            self.lr_scheduler_d.step()

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Loss/Generator_train', avg_losses['gen'], epoch)
            self.writer.add_scalar('Loss/L1_train', avg_losses['recon'], epoch)
            self.writer.add_scalar('Loss/Perceptual_train', avg_losses['perc'], epoch)

            if not self.disable_gan:
                self.writer.add_scalar('Loss/Discriminator', avg_losses['disc'], epoch)
                self.writer.add_scalar('Loss/Adversarial', avg_losses['adv'], epoch)

            self.writer.add_scalar('LR/Generator', self.optimizer_g.param_groups[0]['lr'], epoch)

            if not self.disable_gan and self.optimizer_d is not None:
                self.writer.add_scalar('LR/Discriminator', self.optimizer_d.param_groups[0]['lr'], epoch)

            if self.log_grad_norm:
                avg_norm, max_norm = self._grad_tracker_g.compute()
                self.writer.add_scalar('training/grad_norm_g_avg', avg_norm, epoch)
                self.writer.add_scalar('training/grad_norm_g_max', max_norm, epoch)

                if not self.disable_gan:
                    avg_norm_d, max_norm_d = self._grad_tracker_d.compute()
                    self.writer.add_scalar('training/grad_norm_d_avg', avg_norm_d, epoch)
                    self.writer.add_scalar('training/grad_norm_d_max', max_norm_d, epoch)

        return avg_losses

    def compute_validation_losses(self, epoch: int) -> Dict[str, float]:
        """Compute validation metrics including regional tracking."""
        if self.val_loader is None:
            return {}

        model_to_eval = self.ema.ema_model if self.ema is not None else self.model
        model_to_eval.eval()

        total_l1 = 0.0
        total_perc = 0.0
        total_gen = 0.0
        total_psnr = 0.0
        total_lpips = 0.0
        total_msssim = 0.0
        n_batches = 0

        worst_loss = 0.0
        worst_batch_data = None

        # Initialize regional tracker for validation (if enabled)
        regional_tracker = None
        if self.log_regional_losses:
            regional_tracker = RegionalMetricsTracker(
                image_size=self.cfg.model.image_size,
                fov_mm=self.cfg.paths.get('fov_mm', 240.0),
                loss_fn='l1',  # DC-AE uses L1
                device=self.device,
            )

        # Mark CUDA graph step boundary for torch.compile
        if self.use_compile:
            torch.compiler.cudagraph_mark_step_begin()

        with torch.no_grad():
            for batch in self.val_loader:
                images, mask = self._prepare_batch(batch)

                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    latent = model_to_eval.encode(images, return_dict=False)[0]
                    reconstruction = model_to_eval.decode(latent, return_dict=False)[0]

                    l1_loss = torch.nn.functional.l1_loss(reconstruction, images)

                    if self.perceptual_weight > 0 and self.perceptual_loss_fn is not None:
                        p_loss = self.perceptual_loss_fn(reconstruction.float(), images.float())
                    else:
                        p_loss = torch.tensor(0.0, device=self.device)

                    g_loss = self.l1_weight * l1_loss + self.perceptual_weight * p_loss

                loss_val = g_loss.item()
                total_l1 += l1_loss.item()
                total_perc += p_loss.item()
                total_gen += loss_val

                # Regional tracking (tumor vs background)
                if regional_tracker is not None and mask is not None:
                    regional_tracker.update(reconstruction, images, mask)

                # Track worst batch
                if loss_val > worst_loss:
                    worst_loss = loss_val
                    worst_batch_data = {
                        'original': images.cpu(),
                        'generated': reconstruction.float().cpu(),
                        'loss': loss_val,
                        'loss_breakdown': {'L1': l1_loss.item(), 'Perc': p_loss.item()},
                    }

                # Quality metrics
                if self.log_psnr:
                    total_psnr += compute_psnr(reconstruction, images)

                if self.log_lpips:
                    total_lpips += compute_lpips(reconstruction.float(), images.float(), device=self.device)

                if self.log_msssim:
                    total_msssim += compute_msssim(reconstruction.float(), images.float())

                n_batches += 1

        self.model.train()

        if n_batches == 0:
            return {}

        metrics = {
            'l1': total_l1 / n_batches,
            'perc': total_perc / n_batches,
            'gen': total_gen / n_batches,
        }

        if self.log_psnr:
            metrics['psnr'] = total_psnr / n_batches
        if self.log_lpips:
            metrics['lpips'] = total_lpips / n_batches
        if self.log_msssim:
            metrics['msssim'] = total_msssim / n_batches

        # TensorBoard logging
        if self.writer is not None:
            self.writer.add_scalar('Loss/L1_val', metrics['l1'], epoch)
            self.writer.add_scalar('Loss/Perceptual_val', metrics['perc'], epoch)
            self.writer.add_scalar('Loss/Generator_val', metrics['gen'], epoch)

            if 'psnr' in metrics:
                self.writer.add_scalar('Validation/PSNR', metrics['psnr'], epoch)
            if 'lpips' in metrics:
                self.writer.add_scalar('Validation/LPIPS', metrics['lpips'], epoch)
            if 'msssim' in metrics:
                self.writer.add_scalar('Validation/MS-SSIM', metrics['msssim'], epoch)

            # Log regional metrics (tumor vs background)
            if regional_tracker is not None:
                regional_tracker.log_to_tensorboard(self.writer, epoch, prefix='regional')

            # Worst batch figure
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

        model_to_eval = self.ema.ema_model if self.ema is not None else self.model
        model_to_eval.eval()

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

                    with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                        latent = model_to_eval.encode(images, return_dict=False)[0]
                        reconstruction = model_to_eval.decode(latent, return_dict=False)[0]

                    # Compute metrics
                    if self.log_psnr:
                        total_psnr += compute_psnr(reconstruction, images)
                    if self.log_lpips:
                        total_lpips += compute_lpips(reconstruction, images, device=self.device)
                    if self.log_msssim:
                        total_msssim += compute_msssim(reconstruction, images)

                    # Regional tracking (tumor vs background)
                    if regional_tracker is not None and mask is not None:
                        regional_tracker.update(reconstruction, images, mask)

                    n_batches += 1

            # Compute averages and log
            if n_batches > 0 and self.writer is not None:
                if self.log_psnr:
                    avg_psnr = total_psnr / n_batches
                    self.writer.add_scalar(f'Validation/PSNR_{modality}', avg_psnr, epoch)
                if self.log_lpips:
                    avg_lpips = total_lpips / n_batches
                    self.writer.add_scalar(f'Validation/LPIPS_{modality}', avg_lpips, epoch)
                if self.log_msssim:
                    avg_msssim = total_msssim / n_batches
                    self.writer.add_scalar(f'Validation/MS-SSIM_{modality}', avg_msssim, epoch)

                # Log regional metrics for this modality
                if regional_tracker is not None:
                    regional_tracker.log_to_tensorboard(
                        self.writer, epoch, prefix=f'regional_{modality}'
                    )

        model_to_eval.train()

    def _measure_model_flops(self, sample_images: torch.Tensor, n_batches: int) -> None:
        """Measure and log model FLOPs."""
        if self._flops_logged or not self.is_main_process:
            return

        try:
            from .tracking import measure_model_flops

            # Wrap model for FLOPs measurement
            def forward_fn(x):
                latent = self.model_raw.encode(x, return_dict=False)[0]
                return self.model_raw.decode(latent, return_dict=False)[0]

            gflops = measure_model_flops(forward_fn, sample_images[:1])
            if gflops > 0 and self.writer is not None:
                self.writer.add_scalar('model/gflops_per_sample', gflops, 0)
                logger.info(f"Model FLOPs: {gflops:.2f} GFLOPs/sample")
            self._flops_logged = True
        except Exception as e:
            logger.warning(f"Could not measure FLOPs: {e}")

    def train(
        self,
        train_loader: DataLoader,
        train_dataset,
        val_loader: Optional[DataLoader] = None,
        start_epoch: int = 0,
        per_modality_val_loaders: Optional[Dict[str, DataLoader]] = None,
    ) -> None:
        """Main training loop.

        Args:
            train_loader: Training data loader.
            train_dataset: Training dataset.
            val_loader: Optional validation data loader.
            start_epoch: Epoch to start from (for resuming training).
            per_modality_val_loaders: Optional dict mapping modality names to
                separate validation loaders for per-modality metric tracking.
                e.g., {'bravo': loader, 't1_pre': loader, 't1_gd': loader}
        """
        self.val_loader = val_loader
        self.per_modality_val_loaders = per_modality_val_loaders
        training_start = time.time()

        if self.is_main_process:
            logger.info(f"Starting DC-AE training for {self.n_epochs} epochs")
            logger.info(f"Batch size: {self.batch_size}, LR: {self.learning_rate}")
            logger.info(get_vram_usage(self.device))

        # Measure FLOPs on first batch (once at start of training)
        if self.log_flops:
            try:
                first_batch = next(iter(train_loader))
                sample_images, _ = self._prepare_batch(first_batch)
                self._measure_model_flops(sample_images, len(train_loader))
            except Exception as e:
                if self.is_main_process:
                    logger.warning(f"Could not measure FLOPs: {e}")

        for epoch in range(start_epoch, self.n_epochs):
            epoch_start = time.time()

            # Train epoch
            avg_losses = self.train_epoch(train_loader, epoch)

            # Validation
            val_metrics = {}
            if (epoch + 1) % self.val_interval == 0 or epoch == self.n_epochs - 1:
                val_metrics = self.compute_validation_losses(epoch)

                # Per-modality validation (if loaders provided)
                self._compute_per_modality_validation(epoch)

            # Epoch summary
            elapsed = time.time() - epoch_start
            if self.is_main_process:
                log_dcae_epoch_summary(epoch, self.n_epochs, avg_losses, val_metrics, elapsed)

            # Save checkpoints
            if self.is_main_process:
                val_loss = val_metrics.get('gen', avg_losses['gen'])

                # Save best
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self._save_checkpoint(epoch, 'best')

                # Save latest
                self._save_checkpoint(epoch, 'latest')

        # Training complete
        total_time = time.time() - training_start
        if self.is_main_process:
            logger.info(f"Training complete in {total_time/3600:.2f} hours")
            if self.writer is not None:
                self.writer.close()

    def _save_checkpoint(self, epoch: int, name: str) -> None:
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model_raw.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'scheduler_g_state_dict': self.lr_scheduler_g.state_dict() if self.lr_scheduler_g else None,
            'best_loss': self.best_loss,
        }

        if not self.disable_gan and self.discriminator_raw is not None:
            checkpoint['discriminator_state_dict'] = self.discriminator_raw.state_dict()
            if self.optimizer_d is not None:
                checkpoint['optimizer_d_state_dict'] = self.optimizer_d.state_dict()

        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        path = os.path.join(self.save_dir, f'{name}.pt')
        torch.save(checkpoint, path)

    def evaluate_test(
        self,
        test_loader: DataLoader,
        checkpoint_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate model on test set with comprehensive metrics.

        Args:
            test_loader: Test data loader.
            checkpoint_name: Which checkpoint to load ('best', 'latest', or None for current).

        Returns:
            Dict with test metrics.
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
        total_msssim = 0.0
        total_psnr = 0.0
        total_lpips = 0.0
        n_batches = 0
        n_samples = 0

        # Initialize regional tracker for test evaluation (if enabled)
        regional_tracker = None
        if self.log_regional_losses:
            regional_tracker = RegionalMetricsTracker(
                image_size=self.cfg.model.image_size,
                fov_mm=self.cfg.paths.get('fov_mm', 240.0),
                loss_fn='l1',
                device=self.device,
            )

        # Worst batch tracking
        worst_loss = 0.0
        worst_batch_data = None

        # Store samples for visualization
        sample_inputs = []
        sample_outputs = []
        max_vis_samples = 16

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test evaluation", ncols=100, disable=self.is_cluster):
                images, mask = self._prepare_batch(batch)
                batch_size = images.shape[0]

                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    latent = model_to_use.encode(images, return_dict=False)[0]
                    reconstruction = model_to_use.decode(latent, return_dict=False)[0]

                # Compute metrics
                l1_loss = torch.abs(reconstruction - images).mean().item()
                total_l1 += l1_loss
                if self.log_msssim:
                    total_msssim += compute_msssim(reconstruction, images)
                if self.log_psnr:
                    total_psnr += compute_psnr(reconstruction, images)
                if self.log_lpips:
                    total_lpips += compute_lpips(reconstruction, images, device=self.device)

                # Regional tracking (tumor vs background)
                if regional_tracker is not None and mask is not None:
                    regional_tracker.update(reconstruction, images, mask)

                # Track worst batch
                if l1_loss > worst_loss:
                    worst_loss = l1_loss
                    worst_batch_data = {
                        'original': images.cpu(),
                        'generated': reconstruction.float().cpu(),
                        'loss': l1_loss,
                    }

                n_batches += 1
                n_samples += batch_size

                # Collect samples for visualization
                if len(sample_inputs) < max_vis_samples:
                    remaining = max_vis_samples - len(sample_inputs)
                    sample_inputs.append(images[:remaining].cpu())
                    sample_outputs.append(reconstruction[:remaining].cpu())

        # Compute averages
        metrics = {
            'l1': total_l1 / n_batches,
            'n_samples': n_samples,
        }
        if self.log_psnr:
            metrics['psnr'] = total_psnr / n_batches
        if self.log_lpips:
            metrics['lpips'] = total_lpips / n_batches
        if self.log_msssim:
            metrics['msssim'] = total_msssim / n_batches

        # Log results
        logger.info(f"Test Results - {label} ({n_samples} samples):")
        logger.info(f"  L1 Loss: {metrics['l1']:.6f}")
        if 'msssim' in metrics:
            logger.info(f"  MS-SSIM: {metrics['msssim']:.4f}")
        if 'psnr' in metrics:
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
            if 'psnr' in metrics:
                self.writer.add_scalar(f'{tb_prefix}/PSNR', metrics['psnr'], 0)
            if 'lpips' in metrics:
                self.writer.add_scalar(f'{tb_prefix}/LPIPS', metrics['lpips'], 0)
            if 'msssim' in metrics:
                self.writer.add_scalar(f'{tb_prefix}/MS-SSIM', metrics['msssim'], 0)

            # Log regional metrics (tumor vs background)
            if regional_tracker is not None:
                regional_tracker.log_to_tensorboard(self.writer, 0, prefix=f'{tb_prefix}_regional')

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

        Args:
            original: Original images [N, C, H, W].
            reconstructed: Reconstructed images [N, C, H, W].
            metrics: Dict with test metrics for title.
            label: Checkpoint label for title (e.g., "best", "latest").

        Returns:
            Matplotlib figure.
        """
        return create_reconstruction_figure(
            original=original,
            reconstructed=reconstructed,
            title=f"DC-AE Test Reconstructions ({label})",
            metrics=metrics,
        )
