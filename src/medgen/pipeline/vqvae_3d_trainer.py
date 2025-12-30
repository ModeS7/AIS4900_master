"""
3D VQ-VAE trainer module for training volumetric vector-quantized autoencoders.

Key advantages over 3D KL-VAE:
- Lower memory (no mu/logvar branches, simpler backward pass)
- Discrete codebook (cleaner latent space)

Based on MONAI Generative VQ-VAE implementation.
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
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Disable MONAI MetaTensor tracking (fixes torch.compile issues)
from monai.data import set_track_meta
set_track_meta(False)

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


class CheckpointedVQVAE(nn.Module):
    """Wrapper that applies gradient checkpointing to MONAI VQVAE.

    Reduces activation memory by ~50% for 3D volumes.
    """

    def __init__(self, model: VQVAE):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with gradient checkpointing."""
        # Checkpoint encoder
        def encode_fn(x):
            return self.model.encode(x)

        quantized, vq_loss = grad_checkpoint(encode_fn, x, use_reentrant=False)

        # Checkpoint decoder
        def decode_fn(z):
            return self.model.decode(z)

        reconstruction = grad_checkpoint(decode_fn, quantized, use_reentrant=False)

        return reconstruction, vq_loss

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode without checkpointing (for inference)."""
        return self.model.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode without checkpointing (for inference)."""
        return self.model.decode(z)

    def index_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Get codebook indices for input."""
        return self.model.index_quantize(x)

    def decode_samples(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from codebook indices."""
        return self.model.decode_samples(indices)


def log_vqvae_3d_epoch_summary(
    epoch: int,
    total_epochs: int,
    avg_losses: Dict[str, float],
    val_metrics: Dict[str, float],
    elapsed_time: float
) -> None:
    """Log 3D VQ-VAE epoch summary."""
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
        f"VQ: {avg_losses['vq']:.4f} | "
        f"D: {avg_losses['disc']:.4f} | "
        f"{psnr_str} | "
        f"Time: {elapsed_time:.1f}s"
    )


class VQVAE3DTrainer:
    """3D VQ-VAE trainer for volumetric medical imaging.

    Lower memory than KL-VAE due to:
    - No mu/logvar branches
    - Simpler backward pass through codebook lookup
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # Training config
        self.n_epochs: int = cfg.training.epochs
        self.batch_size: int = cfg.training.batch_size
        self.learning_rate: float = cfg.training.get('learning_rate', 5e-5)
        self.disc_lr: float = cfg.vqvae_3d.get('disc_lr', 5e-4)
        self.warmup_epochs: int = cfg.training.warmup_epochs
        self.val_interval: int = cfg.training.val_interval
        self.use_multi_gpu: bool = cfg.training.get('use_multi_gpu', False)

        # Volume dimensions
        self.volume_depth: int = cfg.volume.get('depth', 160)
        self.volume_height: int = cfg.volume.get('height', 256)
        self.volume_width: int = cfg.volume.get('width', 256)

        # Loss weights
        self.perceptual_weight: float = cfg.vqvae_3d.get('perceptual_weight', 0.002)
        self.adv_weight: float = cfg.vqvae_3d.get('adv_weight', 0.005)

        # VQ-VAE specific
        self.num_embeddings: int = cfg.vqvae_3d.get('num_embeddings', 512)
        self.embedding_dim: int = cfg.vqvae_3d.get('embedding_dim', 3)
        self.commitment_cost: float = cfg.vqvae_3d.get('commitment_cost', 0.25)
        self.decay: float = cfg.vqvae_3d.get('decay', 0.99)
        self.epsilon: float = cfg.vqvae_3d.get('epsilon', 1e-5)

        # Architecture
        self.channels: tuple = tuple(cfg.vqvae_3d.get('channels', [64, 128]))
        self.num_res_layers: int = cfg.vqvae_3d.get('num_res_layers', 2)
        self.num_res_channels: tuple = tuple(cfg.vqvae_3d.get('num_res_channels', [64, 128]))
        self.downsample_parameters: tuple = tuple(
            tuple(p) for p in cfg.vqvae_3d.get('downsample_parameters', [[2, 4, 1, 1]] * 2)
        )
        self.upsample_parameters: tuple = tuple(
            tuple(p) for p in cfg.vqvae_3d.get('upsample_parameters', [[2, 4, 1, 1, 0]] * 2)
        )

        # 3D specific options
        self.use_2_5d_perceptual: bool = cfg.vqvae_3d.get('use_2_5d_perceptual', True)
        self.perceptual_slice_fraction: float = cfg.vqvae_3d.get('perceptual_slice_fraction', 0.25)
        self.gradient_checkpointing: bool = cfg.training.get('gradient_checkpointing', True)
        self.disable_gan: bool = cfg.vqvae_3d.get('disable_gan', False)

        # Discriminator
        self.disc_num_layers: int = cfg.vqvae_3d.get('disc_num_layers', 3)
        self.disc_num_channels: int = cfg.vqvae_3d.get('disc_num_channels', 64)

        # Gradient clipping
        self.gradient_clip_norm: float = cfg.training.get('gradient_clip_norm', 1.0)

        # Logging
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

        # Device setup
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
                    cfg.paths.model_dir, "vqvae_3d", cfg.mode.name,
                    f"{exp_name}{self.volume_height}x{self.volume_depth}_{timestamp}"
                )
            os.makedirs(self.save_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "tensorboard"))
        else:
            self.save_dir = None
            self.writer = None

        # Tracking
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
        self.val_loader = None

    def _setup_distributed(self):
        return setup_distributed()

    def setup_model(self, pretrained_checkpoint: Optional[str] = None) -> None:
        """Initialize 3D VQ-VAE model."""
        n_channels = self.cfg.mode.get('in_channels', 1)

        # Create 3D VQVAE
        base_model = VQVAE(
            spatial_dims=3,
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

        # Load pretrained weights
        if pretrained_checkpoint:
            try:
                checkpoint = torch.load(pretrained_checkpoint, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    if any(k.startswith('model.') for k in state_dict.keys()):
                        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                    base_model.load_state_dict(state_dict)
                    if self.is_main_process:
                        logger.info(f"Loaded 3D VQ-VAE weights from {pretrained_checkpoint}")
            except FileNotFoundError:
                if self.is_main_process:
                    logger.warning(f"Checkpoint not found: {pretrained_checkpoint}")

        # Wrap with gradient checkpointing
        if self.gradient_checkpointing:
            raw_model = CheckpointedVQVAE(base_model)
            if self.is_main_process:
                logger.info("Gradient checkpointing enabled")
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
        use_compile = self.cfg.training.get('use_compile', False)
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

        # Optimizers
        self.optimizer_g = AdamW(self.model.parameters(), lr=self.learning_rate)
        if not self.disable_gan:
            self.optimizer_d = AdamW(self.discriminator.parameters(), lr=self.disc_lr)

        # Schedulers
        self.lr_scheduler_g = create_warmup_cosine_scheduler(
            self.optimizer_g, self.warmup_epochs, self.n_epochs
        )
        if not self.disable_gan:
            self.lr_scheduler_d = create_warmup_cosine_scheduler(
                self.optimizer_d, self.warmup_epochs, self.n_epochs
            )

        # Loss functions
        if self.use_2_5d_perceptual and self.perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss(device=self.device)

        if not self.disable_gan:
            self.adv_loss = PatchAdversarialLoss(criterion="least_squares")

        # Save config
        if self.is_main_process:
            config_path = os.path.join(self.save_dir, "config.yaml")
            OmegaConf.save(self.cfg, config_path)
            logger.info(f"Config saved to: {config_path}")

            # Log model info
            vqvae_params = sum(p.numel() for p in self.model_raw.parameters())
            logger.info(f"3D VQ-VAE: {vqvae_params / 1e6:.1f}M parameters")
            logger.info(f"  Channels: {self.channels}")
            logger.info(f"  Codebook: {self.num_embeddings} x {self.embedding_dim}")
            if not self.disable_gan:
                disc_params = sum(p.numel() for p in self.discriminator_raw.parameters())
                logger.info(f"3D Discriminator: {disc_params / 1e6:.1f}M parameters")
            else:
                logger.info("GAN disabled")
            logger.info(f"Volume: {self.volume_width}x{self.volume_height}x{self.volume_depth}")

            # Latent shape
            n_downsamples = len(self.channels)
            latent_h = self.volume_height // (2 ** n_downsamples)
            latent_w = self.volume_width // (2 ** n_downsamples)
            latent_d = self.volume_depth // (2 ** n_downsamples)
            logger.info(f"Latent shape: [{self.embedding_dim}, {latent_d}, {latent_h}, {latent_w}]")

    def _compute_2_5d_perceptual_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss on sampled 2D slices."""
        if self.perceptual_loss is None:
            return torch.tensor(0.0, device=self.device)

        depth = reconstruction.shape[2]
        n_slices = max(1, int(depth * self.perceptual_slice_fraction))
        slice_indices = torch.linspace(0, depth - 1, n_slices).long()

        total_loss = 0.0
        for idx in slice_indices:
            recon_slice = reconstruction[:, :, idx, :, :]
            target_slice = target[:, :, idx, :, :]

            if recon_slice.shape[1] == 1:
                recon_slice = recon_slice.repeat(1, 3, 1, 1)
                target_slice = target_slice.repeat(1, 3, 1, 1)

            total_loss += self.perceptual_loss(recon_slice, target_slice)

        return total_loss / n_slices

    def _measure_model_flops(self, sample_batch: torch.Tensor, steps_per_epoch: int) -> None:
        """Measure FLOPs for 3D VQ-VAE forward pass.

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
            timesteps=None,  # VQ-VAE has no timesteps
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

        d_loss = torch.tensor(0.0, device=self.device)
        adv_loss = torch.tensor(0.0, device=self.device)

        # ========== Discriminator step ==========
        if not self.disable_gan:
            with torch.no_grad():
                with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                    reconstruction_for_d, _ = self.model(images)

            self.optimizer_d.zero_grad()

            with autocast('cuda', enabled=True, dtype=self.weight_dtype):
                disc_real = self.discriminator(images)
                disc_fake = self.discriminator(reconstruction_for_d)

                d_loss_real = self.adv_loss(disc_real, target_is_real=True, for_discriminator=True)
                d_loss_fake = self.adv_loss(disc_fake, target_is_real=False, for_discriminator=True)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)

            d_loss.backward()
            if self.gradient_clip_norm > 0:
                grad_norm_d = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip_norm)
                if self.log_grad_norm and isinstance(grad_norm_d, torch.Tensor):
                    self._grad_norm_tracker_d.update(grad_norm_d.item())
            self.optimizer_d.step()

        # ========== Generator step ==========
        self.optimizer_g.zero_grad()

        with autocast('cuda', enabled=True, dtype=self.weight_dtype):
            reconstruction, vq_loss = self.model(images)

            # L1 reconstruction loss
            l1_loss = torch.nn.functional.l1_loss(reconstruction, images)

            # Perceptual loss (2.5D)
            if self.perceptual_weight > 0 and self.perceptual_loss is not None:
                p_loss = self._compute_2_5d_perceptual_loss(reconstruction, images)
            else:
                p_loss = torch.tensor(0.0, device=self.device)

            # Adversarial loss
            if not self.disable_gan:
                disc_fake = self.discriminator(reconstruction)
                adv_loss = self.adv_loss(disc_fake, target_is_real=True, for_discriminator=False)

            # Total generator loss
            g_loss = (
                l1_loss +
                self.perceptual_weight * p_loss +
                vq_loss +
                self.adv_weight * adv_loss
            )

        g_loss.backward()
        if self.gradient_clip_norm > 0:
            grad_norm_g = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            if self.log_grad_norm and isinstance(grad_norm_g, torch.Tensor):
                self._grad_norm_tracker.update(grad_norm_g.item())

        self.optimizer_g.step()

        return {
            'gen': g_loss.item(),
            'disc': d_loss.item() if not self.disable_gan else 0.0,
            'recon': l1_loss.item(),
            'perc': p_loss.item() if isinstance(p_loss, torch.Tensor) else p_loss,
            'vq': vq_loss.item(),
            'adv': adv_loss.item() if not self.disable_gan else 0.0,
        }

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        if not self.disable_gan and self.discriminator is not None:
            self.discriminator.train()

        total_losses = {'gen': 0, 'disc': 0, 'recon': 0, 'perc': 0, 'vq': 0, 'adv': 0}
        n_batches = 0

        # Disable progress bar on cluster (too much log output)
        disable_pbar = not self.is_main_process or self.is_cluster
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=disable_pbar)
        for batch in pbar:
            losses = self.train_step(batch)

            for key in total_losses:
                total_losses[key] += losses[key]
            n_batches += 1

            if not disable_pbar:
                pbar.set_postfix({
                    'G': f"{losses['gen']:.4f}",
                    'VQ': f"{losses['vq']:.4f}",
                    'L1': f"{losses['recon']:.4f}",
                })

            # Log VRAM on first batch
            if epoch == 0 and n_batches == 1 and self.is_main_process:
                logger.info(get_vram_usage(self.device))

        return {k: v / n_batches for k, v in total_losses.items()}

    def compute_validation_losses(self, epoch: int) -> Dict[str, float]:
        """Compute comprehensive validation metrics.

        Includes:
        - All loss components (L1, Perceptual, VQ, Generator)
        - Quality metrics (PSNR, LPIPS, MS-SSIM)
        - Regional tumor metrics (tumor/background loss, size categories)
        - Worst batch visualization (8 worst slices from worst volume)
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        # Loss accumulators
        total_l1 = 0.0
        total_perc = 0.0
        total_vq = 0.0
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
                    reconstruction, vq_loss = self.model(images)

                    # Compute all loss components
                    l1_loss = torch.nn.functional.l1_loss(reconstruction, images)

                    if self.perceptual_weight > 0 and self.perceptual_loss is not None:
                        p_loss = self._compute_2_5d_perceptual_loss(reconstruction, images)
                    else:
                        p_loss = torch.tensor(0.0, device=self.device)

                    # Total generator loss (matching training)
                    g_loss = l1_loss + self.perceptual_weight * p_loss + vq_loss

                loss_val = g_loss.item()
                total_l1 += l1_loss.item()
                total_perc += p_loss.item()
                total_vq += vq_loss.item()
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
                            'Perc': total_perc,
                            'VQ': vq_loss.item(),
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
            'vq': total_vq / n_batches,
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
            self.writer.add_scalar('Loss/VQ_val', metrics['vq'], epoch)
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
                # Use first volume from batch for visualization
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
        """Save checkpoint."""
        if not self.is_main_process:
            return

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

        save_full_checkpoint(
            model=self.model_raw,
            optimizer=self.optimizer_g,
            epoch=epoch,
            save_dir=self.save_dir,
            filename=name,
            model_config=model_config,
            scheduler=self.lr_scheduler_g,
        )

    def _log_grad_norms(self, epoch: int) -> None:
        """Log gradient norms."""
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
    ) -> int:
        """Execute training loop."""
        n_epochs = max_epochs if max_epochs is not None else self.n_epochs
        self.val_loader = val_loader
        total_start = time.time()

        avg_losses = {'gen': float('inf'), 'disc': float('inf'), 'recon': float('inf'),
                      'perc': float('inf'), 'vq': float('inf'), 'adv': float('inf')}

        if start_epoch > 0 and self.is_main_process:
            logger.info(f"Resuming from epoch {start_epoch + 1}")

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
                if not self.disable_gan and self.lr_scheduler_d is not None:
                    self.lr_scheduler_d.step()

                if self.is_main_process:
                    val_metrics = self.compute_validation_losses(epoch)
                    log_vqvae_3d_epoch_summary(epoch, n_epochs, avg_losses, val_metrics, epoch_time)

                    # TensorBoard
                    if self.writer is not None:
                        self.writer.add_scalar('Loss/Generator_train', avg_losses['gen'], epoch)
                        self.writer.add_scalar('Loss/Discriminator', avg_losses['disc'], epoch)
                        self.writer.add_scalar('Loss/L1_train', avg_losses['recon'], epoch)
                        self.writer.add_scalar('Loss/Perceptual_train', avg_losses['perc'], epoch)
                        self.writer.add_scalar('Loss/VQ_train', avg_losses['vq'], epoch)
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
                        logger.info(f"New best model (val loss: {val_loss:.6f})")

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
