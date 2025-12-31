"""
3D VAE trainer module for training volumetric autoencoders.

Based on MONAI Generative proven configuration for 3D medical imaging.
Key differences from 2D:
- spatial_dims=3
- Smaller channels (memory)
- No attention layers
- 2.5D perceptual loss option
- Gradient checkpointing (proper implementation via torch.utils.checkpoint)

Memory optimizations:
- Disable MONAI MetaTensor tracking (fixes torch.compile recompilations)
- Gradient checkpointing reduces activation memory by ~50%
- Use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for fragmentation
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
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ema_pytorch import EMA

# Disable MONAI MetaTensor tracking BEFORE importing MONAI modules
# This fixes torch.compile recompilation issues (32+ recompiles -> 1)
from monai.data import set_track_meta
set_track_meta(False)

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
from .metrics import (
    compute_psnr,
    compute_lpips_3d,
    compute_msssim,
    RegionalMetricsTracker3D,
)
from .tracking import GradientNormTracker, FLOPsTracker, create_worst_batch_figure_3d

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
        """Forward pass with gradient checkpointing.

        Checkpoints encoder and decoder separately to save activation memory.
        """
        # Checkpoint encoder (saves most memory for 3D)
        def encode_fn(x):
            h = self.model.encoder(x)
            z_mu = self.model.quant_conv_mu(h)
            z_log_var = self.model.quant_conv_log_sigma(h)
            return z_mu, z_log_var

        z_mu, z_log_var = grad_checkpoint(encode_fn, x, use_reentrant=False)

        # Sample from latent distribution (not checkpointed - cheap)
        z = self.model.sampling(z_mu, z_log_var)

        # Checkpoint decoder
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
        # Compute val_interval: num_validations takes priority over val_interval
        num_validations = cfg.training.get('num_validations', None)
        if num_validations and num_validations > 0:
            self.val_interval: int = max(1, self.n_epochs // num_validations)
        else:
            self.val_interval: int = cfg.training.val_interval
        self.use_multi_gpu: bool = cfg.training.get('use_multi_gpu', False)
        self.use_ema: bool = cfg.training.get('use_ema', False)
        self.ema_decay: float = cfg.training.ema.get('decay', 0.999) if self.use_ema else 0.999

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
        self.disable_gan: bool = cfg.vae_3d.get('disable_gan', False)

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
        self.log_lpips: bool = logging_cfg.get('lpips', True)
        self.log_msssim: bool = logging_cfg.get('msssim', True)
        self.log_regional_losses: bool = logging_cfg.get('regional_losses', True)
        self.log_flops: bool = logging_cfg.get('flops', True)
        self.fov_mm: float = cfg.paths.get('fov_mm', 240.0)

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

        # Cluster mode (disable progress bars)
        self.is_cluster: bool = (cfg.paths.name == "cluster")

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
        self._grad_norm_tracker_d = GradientNormTracker()
        self._flops_tracker = FLOPsTracker()
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

        # Load pretrained weights BEFORE wrapping (load into base_model)
        if pretrained_checkpoint:
            try:
                checkpoint = torch.load(pretrained_checkpoint, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    # Handle both wrapped and unwrapped checkpoint formats
                    state_dict = checkpoint['model_state_dict']
                    # Remove 'model.' prefix if present (from CheckpointedAutoencoder)
                    if any(k.startswith('model.') for k in state_dict.keys()):
                        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                    base_model.load_state_dict(state_dict)
                    if self.is_main_process:
                        logger.info(f"Loaded 3D VAE weights from {pretrained_checkpoint}")
            except FileNotFoundError:
                if self.is_main_process:
                    logger.warning(f"Checkpoint not found: {pretrained_checkpoint}")

        # Wrap with gradient checkpointing for memory efficiency (~50% reduction)
        if self.gradient_checkpointing:
            raw_model = CheckpointedAutoencoder(base_model)
            if self.is_main_process:
                logger.info("Gradient checkpointing enabled (CheckpointedAutoencoder wrapper)")
        else:
            raw_model = base_model

        # Create 3D PatchDiscriminator (if GAN enabled)
        raw_disc = None
        if not self.disable_gan:
            raw_disc = PatchDiscriminator(
                spatial_dims=3,
                in_channels=n_channels,
                channels=self.disc_num_channels,
                num_layers_d=self.disc_num_layers,
            ).to(self.device)

        # Wrap models
        use_compile = self.cfg.training.get('use_compile', True)
        self.model, self.model_raw = wrap_model_for_training(
            raw_model,
            use_multi_gpu=self.use_multi_gpu,
            local_rank=self.local_rank if self.use_multi_gpu else 0,
            use_compile=use_compile,
            is_main_process=self.is_main_process,
        )

        if raw_disc is not None:
            self.discriminator, self.discriminator_raw = wrap_model_for_training(
                raw_disc,
                use_multi_gpu=self.use_multi_gpu,
                local_rank=self.local_rank if self.use_multi_gpu else 0,
                use_compile=use_compile,
                is_main_process=False,
            )

        # Create optimizers
        self.optimizer_g = AdamW(self.model.parameters(), lr=self.learning_rate)
        if not self.disable_gan:
            self.optimizer_d = AdamW(self.discriminator.parameters(), lr=self.disc_lr)

        # Create schedulers
        self.lr_scheduler_g = create_warmup_cosine_scheduler(
            self.optimizer_g, self.warmup_epochs, self.n_epochs
        )
        if not self.disable_gan:
            self.lr_scheduler_d = create_warmup_cosine_scheduler(
                self.optimizer_d, self.warmup_epochs, self.n_epochs
            )

        # Loss functions
        # 2.5D perceptual loss: compute on sampled slices
        if self.use_2_5d_perceptual and self.perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss(device=self.device)
        else:
            self.perceptual_loss = None

        if not self.disable_gan:
            self.adv_loss = PatchAdversarialLoss(criterion="least_squares")

        # Create EMA wrapper if enabled (for generator only)
        # WARNING: EMA doubles memory usage for 3D models
        if self.use_ema:
            self.ema = EMA(
                self.model_raw,
                beta=self.ema_decay,
                update_after_step=self.cfg.training.ema.get('update_after_step', 100),
                update_every=self.cfg.training.ema.get('update_every', 10),
            )
            if self.is_main_process:
                logger.warning(f"EMA enabled with decay={self.ema_decay} - this doubles memory for 3D models!")

        # Save config
        if self.is_main_process:
            config_path = os.path.join(self.save_dir, "config.yaml")
            OmegaConf.save(self.cfg, config_path)
            logger.info(f"Config saved to: {config_path}")

            # Log model info
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

    def _measure_model_flops(self, sample_batch: torch.Tensor, steps_per_epoch: int) -> None:
        """Measure FLOPs for 3D VAE forward pass.

        Should be called once at the start of training with a sample batch.

        Args:
            sample_batch: Sample input tensor [B, C, D, H, W].
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

    def train_step(self, batch) -> Dict[str, float]:
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

            # Adversarial loss (only if GAN enabled)
            if not self.disable_gan:
                disc_fake = self.discriminator(reconstruction)
                adv_loss = self.adv_loss(disc_fake, target_is_real=True, for_discriminator=False)
            else:
                adv_loss = torch.tensor(0.0, device=self.device)

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

        # ========== Discriminator step (only if GAN enabled) ==========
        d_loss = torch.tensor(0.0, device=self.device)
        if not self.disable_gan:
            self.optimizer_d.zero_grad()

            with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                disc_real = self.discriminator(images)
                disc_fake = self.discriminator(reconstruction.detach())

                d_loss_real = self.adv_loss(disc_real, target_is_real=True, for_discriminator=True)
                d_loss_fake = self.adv_loss(disc_fake, target_is_real=False, for_discriminator=True)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)

            d_loss.backward()
            if self.gradient_clip_norm > 0:
                grad_norm_d = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip_norm)
                if self.log_grad_norm and isinstance(grad_norm_d, torch.Tensor):
                    self._grad_norm_tracker_d.update(grad_norm_d.item())
            self.optimizer_d.step()

        # Update EMA
        if self.ema is not None:
            self.ema.update()

        return {
            'gen': g_loss.item(),
            'disc': d_loss.item(),
            'recon': l1_loss.item(),
            'perc': p_loss.item(),
            'kl': kl_loss.item(),
            'adv': adv_loss.item(),
        }

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        if not self.disable_gan:
            self.discriminator.train()

        total_losses = {'gen': 0, 'disc': 0, 'recon': 0, 'perc': 0, 'kl': 0, 'adv': 0}
        n_batches = 0

        # Disable progress bar on cluster (too much log output)
        disable_pbar = not self.is_main_process or self.is_cluster
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=disable_pbar)
        for batch in pbar:
            losses = self.train_step(batch)

            for key in total_losses:
                total_losses[key] += losses[key]
            n_batches += 1

            # Progress bar shows disc loss only if GAN enabled
            if not disable_pbar:
                if not self.disable_gan:
                    pbar.set_postfix({
                        'G': f"{losses['gen']:.4f}",
                        'D': f"{losses['disc']:.4f}",
                        'L1': f"{losses['recon']:.4f}",
                    })
                else:
                    pbar.set_postfix({
                        'G': f"{losses['gen']:.4f}",
                        'L1': f"{losses['recon']:.4f}",
                        'KL': f"{losses['kl']:.4f}",
                    })

        # Average losses
        return {k: v / n_batches for k, v in total_losses.items()}

    def compute_validation_losses(self, epoch: int) -> Dict[str, float]:
        """Compute comprehensive validation metrics.

        Includes:
        - All loss components (L1, Perceptual, KL, Generator)
        - Quality metrics (PSNR, LPIPS, MS-SSIM)
        - Regional tumor metrics (tumor/background loss, size categories)
        - Worst batch visualization (8 worst slices from worst volume)
        """
        if self.val_loader is None:
            return {}

        # Use EMA model for validation if available
        model_to_eval = self.ema.ema_model if self.ema is not None else self.model
        model_to_eval.eval()

        # Loss accumulators
        total_l1 = 0.0
        total_perc = 0.0
        total_kl = 0.0
        total_gen = 0.0

        # Quality metric accumulators
        total_psnr = 0.0
        total_lpips = 0.0
        total_msssim = 0.0
        n_batches = 0

        # Worst batch tracking
        worst_loss = 0.0
        worst_batch_data = None

        # Regional tracker
        regional_tracker = None
        if self.log_regional_losses:
            regional_tracker = RegionalMetricsTracker3D(
                volume_size=(self.volume_height, self.volume_width, self.volume_depth),
                fov_mm=self.fov_mm,
                loss_fn='l1',
                device=self.device,
            )

        with torch.no_grad():
            for batch in self.val_loader:
                images, mask = self._prepare_batch(batch)

                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    reconstruction, mean, logvar = model_to_eval(images)

                    # Compute all loss components
                    l1_loss = torch.nn.functional.l1_loss(reconstruction, images)

                    if self.perceptual_weight > 0 and self.perceptual_loss is not None:
                        p_loss = self._compute_2_5d_perceptual_loss(reconstruction, images)
                    else:
                        p_loss = torch.tensor(0.0, device=self.device)

                    kl_loss = self._compute_kl_loss(mean, logvar)

                    # Total generator loss (matching training)
                    g_loss = l1_loss + self.perceptual_weight * p_loss + self.kl_weight * kl_loss

                loss_val = g_loss.item()
                total_l1 += l1_loss.item()
                total_perc += p_loss.item()
                total_kl += kl_loss.item()
                total_gen += loss_val

                # Track worst batch
                if loss_val > worst_loss:
                    worst_loss = loss_val
                    worst_batch_data = {
                        'original': images.cpu(),
                        'generated': reconstruction.float().cpu(),
                        'loss': loss_val,
                        'loss_breakdown': {
                            'L1': l1_loss.item(),
                            'Perc': p_loss.item(),
                            'KL': kl_loss.item(),
                        },
                    }

                # Quality metrics
                if self.log_psnr:
                    total_psnr += compute_psnr(reconstruction, images)

                if self.log_lpips:
                    total_lpips += compute_lpips_3d(reconstruction.float(), images.float(), device=self.device)

                if self.log_msssim:
                    total_msssim += compute_msssim(reconstruction.float(), images.float(), spatial_dims=3)

                # Regional metrics
                if regional_tracker is not None and mask is not None:
                    regional_tracker.update(reconstruction.float(), images.float(), mask)

                n_batches += 1

        self.model.train()

        if n_batches == 0:
            return {}

        # Compute averages
        metrics = {
            'l1': total_l1 / n_batches,
            'perc': total_perc / n_batches,
            'kl': total_kl / n_batches,
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
            # Validation losses
            self.writer.add_scalar('Loss/L1_val', metrics['l1'], epoch)
            self.writer.add_scalar('Loss/Perceptual_val', metrics['perc'], epoch)
            self.writer.add_scalar('Loss/KL_val', metrics['kl'], epoch)
            self.writer.add_scalar('Loss/Generator_val', metrics['gen'], epoch)

            # Quality metrics
            if 'psnr' in metrics:
                self.writer.add_scalar('Validation/PSNR', metrics['psnr'], epoch)
            if 'lpips' in metrics:
                self.writer.add_scalar('Validation/LPIPS', metrics['lpips'], epoch)
            if 'msssim' in metrics:
                self.writer.add_scalar('Validation/MS-SSIM', metrics['msssim'], epoch)

            # Worst batch figure (8 worst slices from worst volume)
            if worst_batch_data is not None:
                fig = create_worst_batch_figure_3d(
                    original=worst_batch_data['original'][:1],  # [1, C, D, H, W]
                    generated=worst_batch_data['generated'][:1],
                    loss=worst_batch_data['loss'],
                    loss_breakdown=worst_batch_data['loss_breakdown'],
                    num_slices=8,
                )
                self.writer.add_figure('Validation/worst_batch', fig, epoch)
                plt.close(fig)

            # Regional metrics
            if regional_tracker is not None:
                regional_tracker.log_to_tensorboard(self.writer, epoch, prefix='regional')

        return metrics

    def _save_checkpoint(self, epoch: int, name: str) -> None:
        """Save checkpoint with full model config for reproducibility."""
        if not self.is_main_process:
            return

        # Include 3D VAE config in checkpoint for easy reconstruction
        n_channels = self.cfg.mode.get('in_channels', 1)
        model_config = {
            'in_channels': n_channels,
            'out_channels': n_channels,
            'latent_channels': self.latent_channels,
            'channels': list(self.vae_channels),
            'attention_levels': list(self.attention_levels),
            'num_res_blocks': self.num_res_blocks,
            'spatial_dims': 3,
        }

        # Build extra state for VAE-specific components
        extra_state = {
            'disable_gan': self.disable_gan,
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
            if self.lr_scheduler_d is not None:
                extra_state['scheduler_d_state_dict'] = self.lr_scheduler_d.state_dict()

        save_full_checkpoint(
            model=self.model_raw,
            optimizer=self.optimizer_g,
            epoch=epoch,
            save_dir=self.save_dir,
            filename=f"checkpoint_{name}.pt",
            model_config=model_config,
            scheduler=self.lr_scheduler_g,
            ema=self.ema,
            extra_state=extra_state,
        )

    def _log_grad_norms(self, epoch: int) -> None:
        """Log gradient norms to TensorBoard."""
        if not self.log_grad_norm or self.writer is None:
            return

        if self._grad_norm_tracker.count > 0:
            self.writer.add_scalar('training/grad_norm_g_avg', self._grad_norm_tracker.get_avg(), epoch)
            self.writer.add_scalar('training/grad_norm_g_max', self._grad_norm_tracker.get_max(), epoch)
        self._grad_norm_tracker.reset()

        # Discriminator grad norms (only if GAN enabled)
        if not self.disable_gan and self._grad_norm_tracker_d.count > 0:
            self.writer.add_scalar('training/grad_norm_d_avg', self._grad_norm_tracker_d.get_avg(), epoch)
            self.writer.add_scalar('training/grad_norm_d_max', self._grad_norm_tracker_d.get_max(), epoch)
        self._grad_norm_tracker_d.reset()

    def train(
        self,
        train_loader: DataLoader,
        train_dataset: Dataset,
        val_loader: Optional[DataLoader] = None,
        start_epoch: int = 0,
        max_epochs: Optional[int] = None,
        per_modality_val_loaders: Optional[Dict[str, DataLoader]] = None,
    ) -> int:
        """Execute training loop.

        Args:
            train_loader: Training data loader.
            train_dataset: Training dataset.
            val_loader: Optional validation data loader.
            start_epoch: Epoch to start from (for resuming training).
            max_epochs: Maximum epochs to train (overrides self.n_epochs if provided).
            per_modality_val_loaders: Optional dict mapping modality names to
                separate validation loaders for per-modality metric tracking.
                e.g., {'bravo': loader, 't1_pre': loader, 't1_gd': loader}

        Returns:
            The last completed epoch number.
        """
        n_epochs = max_epochs if max_epochs is not None else self.n_epochs
        self.val_loader = val_loader
        self.per_modality_val_loaders = per_modality_val_loaders
        total_start = time.time()

        avg_losses = {'gen': float('inf'), 'disc': float('inf'), 'recon': float('inf'),
                      'perc': float('inf'), 'kl': float('inf'), 'adv': float('inf')}

        if start_epoch > 0 and self.is_main_process:
            logger.info(f"Resuming training from epoch {start_epoch + 1}")

        # Measure FLOPs once at start
        if self.is_main_process and self.log_flops:
            sample_batch = next(iter(train_loader))
            sample_images, _ = self._prepare_batch(sample_batch)
            self._measure_model_flops(sample_images, len(train_loader))

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

                    # Per-modality validation (if loaders provided)
                    self._compute_per_modality_validation(epoch)

                    log_vae_3d_epoch_summary(epoch, n_epochs, avg_losses, val_metrics, epoch_time)

                    # TensorBoard logging
                    if self.writer is not None:
                        self.writer.add_scalar('Loss/Generator_train', avg_losses['gen'], epoch)
                        self.writer.add_scalar('Loss/L1_train', avg_losses['recon'], epoch)
                        self.writer.add_scalar('Loss/Perceptual_train', avg_losses['perc'], epoch)
                        self.writer.add_scalar('Loss/KL_train', avg_losses['kl'], epoch)

                        # Discriminator/Adversarial losses (only if GAN enabled)
                        if not self.disable_gan:
                            self.writer.add_scalar('Loss/Discriminator', avg_losses['disc'], epoch)
                            self.writer.add_scalar('Loss/Adversarial', avg_losses['adv'], epoch)

                        if self.lr_scheduler_g is not None:
                            self.writer.add_scalar('LR/Generator', self.lr_scheduler_g.get_last_lr()[0], epoch)
                        if not self.disable_gan and self.lr_scheduler_d is not None:
                            self.writer.add_scalar('LR/Discriminator', self.lr_scheduler_d.get_last_lr()[0], epoch)

                        self._log_grad_norms(epoch)

                        # Log FLOPs
                        self._flops_tracker.log_epoch(self.writer, epoch)

                    # Save checkpoints every epoch
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

        model_to_eval = self.model_raw
        model_to_eval.eval()

        for modality, loader in self.per_modality_val_loaders.items():
            total_psnr = 0.0
            total_lpips = 0.0
            total_msssim = 0.0
            n_batches = 0

            # Initialize regional tracker for this modality
            regional_tracker = None
            if self.log_regional_losses:
                regional_tracker = RegionalMetricsTracker3D(
                    loss_fn='l1',
                    device=self.device,
                )

            with torch.no_grad():
                for batch in loader:
                    images, mask = self._prepare_batch(batch)

                    with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                        reconstruction, _, _ = model_to_eval(images)

                    # Compute metrics
                    if self.log_psnr:
                        total_psnr += compute_psnr(reconstruction, images)
                    if self.log_lpips:
                        total_lpips += compute_lpips_3d(reconstruction, images, device=self.device)
                    if self.log_msssim:
                        total_msssim += compute_msssim(reconstruction, images)

                    # Regional tracking (tumor vs background)
                    if regional_tracker is not None and mask is not None:
                        regional_tracker.update(reconstruction.float(), images.float(), mask)

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
            regional_tracker = RegionalMetricsTracker3D(
                loss_fn='l1',
                device=self.device,
            )

        # Worst batch tracking
        worst_loss = 0.0
        worst_batch_data = None

        # Store samples for visualization
        sample_inputs = []
        sample_outputs = []
        max_vis_samples = 4  # Fewer samples for 3D (memory)

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test evaluation", ncols=100, disable=self.is_cluster):
                images, mask = self._prepare_batch(batch)
                batch_size = images.shape[0]

                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    reconstruction, _, _ = model_to_use(images)

                # Compute metrics
                l1_loss = torch.abs(reconstruction - images).mean().item()
                total_l1 += l1_loss
                if self.log_msssim:
                    total_msssim += compute_msssim(reconstruction, images)
                if self.log_psnr:
                    total_psnr += compute_psnr(reconstruction, images)
                if self.log_lpips:
                    total_lpips += compute_lpips_3d(reconstruction, images, device=self.device)

                # Regional tracking (tumor vs background)
                if regional_tracker is not None and mask is not None:
                    regional_tracker.update(reconstruction.float(), images.float(), mask)

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
                    sample_outputs.append(reconstruction[:remaining].float().cpu())

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

            # Log worst batch figure
            if worst_batch_data is not None:
                fig = create_worst_batch_figure_3d(
                    original=worst_batch_data['original'],
                    generated=worst_batch_data['generated'],
                    loss=worst_batch_data['loss'],
                )
                self.writer.add_figure(f'{tb_prefix}/worst_batch', fig, 0)
                plt.close(fig)

            # Create and save reconstruction visualization
            if sample_inputs:
                all_inputs = torch.cat(sample_inputs, dim=0)[:max_vis_samples]
                all_outputs = torch.cat(sample_outputs, dim=0)[:max_vis_samples]
                fig = self._create_test_reconstruction_figure_3d(all_inputs, all_outputs, metrics, label)
                self.writer.add_figure(f'{tb_prefix}/reconstructions', fig, 0)
                plt.close(fig)

                # Also save as image file
                fig_path = os.path.join(self.save_dir, f'test_reconstructions_{label}.png')
                fig = self._create_test_reconstruction_figure_3d(all_inputs, all_outputs, metrics, label)
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Test reconstructions saved to: {fig_path}")

        model_to_use.train()
        return metrics

    def _create_test_reconstruction_figure_3d(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        metrics: Dict[str, float],
        label: str = "test",
        num_slices: int = 5,
    ) -> plt.Figure:
        """Create test set reconstruction comparison figure for 3D volumes.

        Shows central slices from multiple samples in a grid.
        Layout: samples × (num_slices × 2) with original/reconstructed pairs.

        Args:
            original: Original volumes [N, C, D, H, W].
            reconstructed: Reconstructed volumes [N, C, D, H, W].
            metrics: Dict with test metrics for title.
            label: Checkpoint label for title (e.g., "best", "latest").
            num_slices: Number of slices to show per sample.

        Returns:
            Matplotlib figure.
        """
        n_samples = min(original.shape[0], 4)  # Max 4 samples
        depth = original.shape[2]

        # Select evenly spaced slices
        slice_indices = np.linspace(depth // 4, 3 * depth // 4, num_slices, dtype=int)

        # Create figure: n_samples rows × (num_slices * 2) columns
        fig, axes = plt.subplots(n_samples * 2, num_slices, figsize=(2 * num_slices, 4 * n_samples))
        if n_samples == 1 and num_slices == 1:
            axes = np.array([[axes]])
        elif n_samples == 1:
            axes = axes.reshape(2, num_slices)
        elif num_slices == 1:
            axes = axes.reshape(n_samples * 2, 1)

        for sample_idx in range(n_samples):
            orig_np = original[sample_idx, 0].cpu().float().numpy()  # [D, H, W]
            recon_np = reconstructed[sample_idx, 0].cpu().float().numpy()

            for col, slice_idx in enumerate(slice_indices):
                # Original row
                row_orig = sample_idx * 2
                axes[row_orig, col].imshow(np.clip(orig_np[slice_idx], 0, 1), cmap='gray', vmin=0, vmax=1)
                axes[row_orig, col].axis('off')
                if col == 0:
                    axes[row_orig, col].set_ylabel(f'Sample {sample_idx}\nOriginal', fontsize=8)
                if sample_idx == 0:
                    axes[row_orig, col].set_title(f'Slice {slice_idx}', fontsize=8)

                # Reconstructed row
                row_recon = sample_idx * 2 + 1
                axes[row_recon, col].imshow(np.clip(recon_np[slice_idx], 0, 1), cmap='gray', vmin=0, vmax=1)
                axes[row_recon, col].axis('off')
                if col == 0:
                    axes[row_recon, col].set_ylabel('Recon', fontsize=8)

        # Build title with metrics
        title_parts = [f"3D VAE Test Reconstructions ({label})"]
        metric_strs = []
        if 'psnr' in metrics:
            metric_strs.append(f"PSNR: {metrics['psnr']:.2f}")
        if 'lpips' in metrics:
            metric_strs.append(f"LPIPS: {metrics['lpips']:.4f}")
        if 'msssim' in metrics:
            metric_strs.append(f"MS-SSIM: {metrics['msssim']:.4f}")
        if metric_strs:
            title_parts.append(f"({', '.join(metric_strs)})")

        fig.suptitle(" ".join(title_parts), fontsize=10)
        plt.tight_layout()
        return fig
