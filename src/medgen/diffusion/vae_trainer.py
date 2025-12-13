"""
VAE trainer module for training autoencoders.

This module provides the VAETrainer class for training AutoencoderKL models
with the same infrastructure as DiffusionTrainer: TensorBoard logging,
checkpoint management, multi-GPU support, and metrics tracking.
"""
import glob
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from monai.losses import PerceptualLoss, PatchAdversarialLoss
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from skimage.metrics import structural_similarity as ssim_skimage

from .utils import get_vram_usage, save_checkpoint, save_model_only

logger = logging.getLogger(__name__)


def log_vae_epoch_summary(
    epoch: int,
    total_epochs: int,
    avg_losses: Dict[str, float],
    elapsed_time: float
) -> None:
    """Log VAE epoch completion summary.

    Args:
        epoch: Current epoch number (0-indexed).
        total_epochs: Total number of epochs.
        avg_losses: Dict with 'gen', 'disc', 'recon', 'kl', 'adv' losses.
        elapsed_time: Time taken for the epoch in seconds.
    """
    timestamp = time.strftime("%H:%M:%S")
    epoch_pct = ((epoch + 1) / total_epochs) * 100

    print(
        f"[{timestamp}] Epoch {epoch + 1:3d}/{total_epochs} ({epoch_pct:5.1f}%) | "
        f"G: {avg_losses['gen']:.4f} | D: {avg_losses['disc']:.4f} | "
        f"L1: {avg_losses['recon']:.4f} | Perc: {avg_losses['perc']:.4f} | "
        f"KL: {avg_losses['kl']:.4f} | Adv: {avg_losses['adv']:.4f} | "
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

        # Discriminator config
        self.disc_num_layers: int = cfg.vae.get('disc_num_layers', 3)
        self.disc_num_channels: int = cfg.vae.get('disc_num_channels', 64)

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
            try:
                from hydra.core.hydra_config import HydraConfig
                self.save_dir = HydraConfig.get().runtime.output_dir
            except (ImportError, ValueError, AttributeError):
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                self.run_name = f"vae_{self.image_size}_{timestamp}"
                self.save_dir = os.path.join(cfg.paths.model_dir, "vae", self.run_name)

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

    def _setup_distributed(self) -> Tuple[int, int, int, torch.device]:
        """Setup distributed training with dynamic port allocation."""
        if 'SLURM_PROCID' in os.environ:
            rank = int(os.environ['SLURM_PROCID'])
            local_rank = int(os.environ['SLURM_LOCALID'])

            if 'SLURM_NTASKS' in os.environ:
                world_size = int(os.environ['SLURM_NTASKS'])
            else:
                nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
                tasks_per_node = int(os.environ.get('SLURM_NTASKS_PER_NODE', 1))
                world_size = nodes * tasks_per_node

            if 'SLURM_JOB_NODELIST' in os.environ:
                nodelist = os.environ['SLURM_JOB_NODELIST']
                master_addr = nodelist.split(',')[0].split('[')[0]
                os.environ['MASTER_ADDR'] = master_addr
            else:
                os.environ['MASTER_ADDR'] = os.environ.get('SLURM_LAUNCH_NODE_IPADDR', 'localhost')

            if 'SLURM_JOB_ID' in os.environ:
                job_id = int(os.environ['SLURM_JOB_ID'])
                port = 12000 + (job_id % 53000)
                os.environ['MASTER_PORT'] = str(port)
                if rank == 0:
                    logger.info(f"Using dynamic port: {port} (from SLURM_JOB_ID: {job_id})")
            else:
                os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
        else:
            rank = int(os.environ.get('RANK', 0))
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))

        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

        return rank, local_rank, world_size, device

    def setup_model(self) -> None:
        """Initialize VAE model, discriminator, optimizers, and loss functions."""
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

        # Create PatchDiscriminator
        raw_disc = PatchDiscriminator(
            spatial_dims=2,
            in_channels=n_channels,
            channels=self.disc_num_channels,
            num_layers_d=self.disc_num_layers,
        ).to(self.device)

        if self.use_multi_gpu:
            self.model_raw = raw_model
            self.discriminator_raw = raw_disc
            ddp_model = DDP(
                raw_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
            ddp_disc = DDP(
                raw_disc,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
            self.model = torch.compile(ddp_model, mode="reduce-overhead")
            self.discriminator = torch.compile(ddp_disc, mode="reduce-overhead")
            if self.is_main_process:
                logger.info("Multi-GPU: Compiled DDP VAE + Discriminator")
        else:
            self.model_raw = raw_model
            self.discriminator_raw = raw_disc
            self.model = torch.compile(raw_model, mode="default")
            self.discriminator = torch.compile(raw_disc, mode="default")

        # Setup perceptual loss (RadImageNet for 2D medical images)
        perceptual_loss = PerceptualLoss(
            spatial_dims=2,
            network_type="radimagenet_resnet50",
            cache_dir=self.cfg.paths.cache_dir,
            pretrained=True,
        ).to(self.device)
        self.perceptual_loss_fn = torch.compile(perceptual_loss, mode="reduce-overhead")

        # Setup adversarial loss
        self.adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")

        # Setup generator optimizer
        self.optimizer_g = AdamW(self.model_raw.parameters(), lr=self.learning_rate)

        # Setup discriminator optimizer
        self.optimizer_d = AdamW(self.discriminator_raw.parameters(), lr=self.disc_lr)

        # Warmup + Cosine scheduler for generator
        warmup_scheduler_g = LinearLR(
            self.optimizer_g,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_epochs
        )
        cosine_scheduler_g = CosineAnnealingLR(
            self.optimizer_g,
            T_max=self.n_epochs - self.warmup_epochs,
            eta_min=1e-6
        )
        self.lr_scheduler_g = SequentialLR(
            self.optimizer_g,
            schedulers=[warmup_scheduler_g, cosine_scheduler_g],
            milestones=[self.warmup_epochs]
        )

        # Warmup + Cosine scheduler for discriminator
        warmup_scheduler_d = LinearLR(
            self.optimizer_d,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_epochs
        )
        cosine_scheduler_d = CosineAnnealingLR(
            self.optimizer_d,
            T_max=self.n_epochs - self.warmup_epochs,
            eta_min=1e-6
        )
        self.lr_scheduler_d = SequentialLR(
            self.optimizer_d,
            schedulers=[warmup_scheduler_d, cosine_scheduler_d],
            milestones=[self.warmup_epochs]
        )

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
            disc_params = sum(p.numel() for p in self.discriminator_raw.parameters())
            logger.info(f"VAE initialized: {vae_params / 1e6:.1f}M parameters")
            logger.info(f"Discriminator initialized: {disc_params / 1e6:.1f}M parameters")
            logger.info(f"Latent shape: [{self.latent_channels}, {self.image_size // 8}, {self.image_size // 8}]")
            logger.info(f"Loss weights - Perceptual: {self.perceptual_weight}, KL: {self.kl_weight}, Adv: {self.adv_weight}")

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

    def _cleanup_old_checkpoints(self, keep_n: int = 3) -> None:
        """Keep only the N most recent epoch checkpoints."""
        pattern = os.path.join(self.save_dir, "epoch_*.pt")
        checkpoints = glob.glob(pattern)

        if len(checkpoints) <= keep_n:
            return

        def get_epoch_num(path: str) -> int:
            basename = os.path.basename(path)
            return int(basename.split('_')[1].split('.')[0])

        checkpoints.sort(key=get_epoch_num)

        for old_ckpt in checkpoints[:-keep_n]:
            try:
                os.remove(old_ckpt)
                logger.debug(f"Removed old checkpoint: {old_ckpt}")
            except OSError as e:
                logger.warning(f"Failed to remove checkpoint {old_ckpt}: {e}")

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

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare batch tensor from dataloader output.

        Args:
            batch: Input batch - either a dict of tensors or a single tensor.

        Returns:
            Stacked tensor [B, C, H, W].
        """
        if isinstance(batch, dict):
            # Stack all images into single tensor [B, C, H, W]
            # Order: image keys first, then seg if present
            image_keys = self.cfg.mode.get('image_keys', ['t1_pre', 't1_gd'])
            tensors = []
            for key in image_keys:
                if key in batch:
                    tensors.append(batch[key].to(self.device))
            if 'seg' in batch:
                tensors.append(batch['seg'].to(self.device))
            return torch.cat(tensors, dim=1)
        elif hasattr(batch, 'as_tensor'):
            return batch.as_tensor().to(self.device)
        else:
            return batch.to(self.device)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step with GAN training.

        Training follows MONAI's approach with proper gradient handling:
        1. Train discriminator on real vs fake (detached generator output)
        2. Train generator with L1 + perceptual + KL + adversarial loss

        Args:
            batch: Input batch - either a dict of tensors (from dual dataloader)
                   or a single tensor [B, C, H, W].

        Returns:
            Dict with 'gen', 'disc', 'recon', 'perc', 'kl', 'adv' losses.
        """
        images = self._prepare_batch(batch)
        grad_clip = self.cfg.training.get('gradient_clip_norm', 1.0)

        # ==================== Discriminator Step ====================
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
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.discriminator_raw.parameters(), max_norm=grad_clip)
        self.optimizer_d.step()

        # ==================== Generator Step ====================
        self.optimizer_g.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=torch.bfloat16):
            # Fresh forward pass for generator gradients
            reconstruction, mean, logvar = self.model(images)

            # L1 reconstruction loss (MONAI uses L1, not MSE)
            l1_loss = torch.abs(reconstruction - images).mean()

            # Perceptual loss (cast to float32 for pretrained network)
            p_loss = self.perceptual_loss_fn(reconstruction.float(), images.float())

            # KL divergence loss
            kl_loss = self._compute_kl_loss(mean, logvar)

            # Adversarial loss (generator wants discriminator to output 1 for fakes)
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
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model_raw.parameters(), max_norm=grad_clip)
        self.optimizer_g.step()

        if self.use_ema:
            self._update_ema()

        return {
            'gen': g_loss.item(),
            'disc': d_loss.item(),
            'recon': l1_loss.item(),
            'perc': p_loss.item(),
            'kl': kl_loss.item(),
            'adv': adv_loss.item(),
        }

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch.

        Args:
            data_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dict with average losses: 'gen', 'disc', 'recon', 'perc', 'kl', 'adv'.
        """
        self.model.train()
        self.discriminator.train()

        epoch_losses = {'gen': 0, 'disc': 0, 'recon': 0, 'perc': 0, 'kl': 0, 'adv': 0}

        use_progress_bars = (not self.is_cluster) and self.is_main_process

        if use_progress_bars:
            progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), ncols=100)
            progress_bar.set_description(f"Epoch {epoch}")
            steps_iter = progress_bar
        else:
            steps_iter = enumerate(data_loader)

        for step, batch in steps_iter:
            losses = self.train_step(batch)

            for key in epoch_losses:
                epoch_losses[key] += losses[key]

            if use_progress_bars:
                progress_bar.set_postfix(
                    G=f"{epoch_losses['gen'] / (step + 1):.4f}",
                    D=f"{epoch_losses['disc'] / (step + 1):.4f}"
                )

            if epoch == 1 and step == 0 and self.is_main_process:
                logger.info(get_vram_usage(self.device))

        # Average losses
        n_batches = len(data_loader)
        return {key: val / n_batches for key, val in epoch_losses.items()}

    def _compute_ssim(self, generated: torch.Tensor, reference: torch.Tensor) -> float:
        """Compute SSIM between generated and reference images.

        Args:
            generated: Generated images [B, C, H, W].
            reference: Reference images [B, C, H, W].

        Returns:
            Average SSIM across batch (using first channel only).
        """
        ssim_values = []
        gen_np = generated.cpu().float().numpy()
        ref_np = reference.cpu().float().numpy()

        for i in range(gen_np.shape[0]):
            gen_img = np.clip(gen_np[i, 0], 0, 1)
            ref_img = np.clip(ref_np[i, 0], 0, 1)
            ssim_val = ssim_skimage(gen_img, ref_img, data_range=1.0)
            ssim_values.append(ssim_val)

        return float(np.mean(ssim_values))

    def _compute_psnr(self, generated: torch.Tensor, reference: torch.Tensor) -> float:
        """Compute PSNR between generated and reference images.

        Args:
            generated: Generated images [B, C, H, W].
            reference: Reference images [B, C, H, W].

        Returns:
            Average PSNR across batch.
        """
        gen_np = np.clip(generated.cpu().float().numpy(), 0, 1)
        ref_np = np.clip(reference.cpu().float().numpy(), 0, 1)

        mse = np.mean((gen_np - ref_np) ** 2)
        if mse < 1e-10:
            return 100.0

        psnr = 10 * np.log10(1.0 / mse)
        return float(psnr)

    def _generate_validation_samples(self, dataset: Dataset, epoch: int) -> None:
        """Generate reconstruction samples for visualization.

        Args:
            dataset: Training dataset.
            epoch: Current epoch number.
        """
        if self.writer is None:
            return

        model_to_use = self.ema.ema_model if self.ema is not None else self.model_raw
        model_to_use.eval()

        # Sample random images
        n_samples = min(8, len(dataset))
        indices = torch.randperm(len(dataset))[:n_samples]

        samples = []
        image_keys = self.cfg.mode.get('image_keys', ['t1_pre', 't1_gd'])
        for idx in indices:
            sample = dataset[idx]
            if isinstance(sample, dict):
                # Stack all channels from dict
                tensors = []
                for key in image_keys:
                    if key in sample:
                        val = sample[key]
                        if not isinstance(val, torch.Tensor):
                            val = torch.from_numpy(np.array(val))
                        tensors.append(val)
                if 'seg' in sample:
                    val = sample['seg']
                    if not isinstance(val, torch.Tensor):
                        val = torch.from_numpy(np.array(val))
                    tensors.append(val)
                sample = torch.cat(tensors, dim=0)
            elif hasattr(sample, 'as_tensor'):
                sample = sample.as_tensor()
            elif isinstance(sample, np.ndarray):
                sample = torch.from_numpy(sample)
            samples.append(sample)

        samples = torch.stack(samples).to(self.device)

        # Reconstruct
        with torch.no_grad():
            with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                reconstructed, _, _ = model_to_use(samples)

        # Compute metrics
        ssim = self._compute_ssim(reconstructed, samples)
        psnr = self._compute_psnr(reconstructed, samples)

        self.writer.add_scalar('validation/SSIM', ssim, epoch)
        self.writer.add_scalar('validation/PSNR', psnr, epoch)

        # Create visualization figure
        fig = self._create_reconstruction_figure(samples, reconstructed)
        self.writer.add_figure('validation/reconstructions', fig, epoch)
        plt.close(fig)

        if self.is_main_process:
            logger.info(f"Validation - SSIM: {ssim:.4f}, PSNR: {psnr:.2f} dB")

        model_to_use.train()

    def _create_reconstruction_figure(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> plt.Figure:
        """Create side-by-side reconstruction comparison figure.

        Args:
            original: Original images [B, C, H, W].
            reconstructed: Reconstructed images [B, C, H, W].

        Returns:
            Matplotlib figure.
        """
        n_samples = min(8, original.shape[0])
        n_channels = original.shape[1]

        # For multi-channel, show first channel only
        orig_np = original[:n_samples, 0].cpu().numpy()
        recon_np = reconstructed[:n_samples, 0].cpu().float().numpy()
        diff_np = np.abs(orig_np - recon_np)

        fig, axes = plt.subplots(3, n_samples, figsize=(2 * n_samples, 6))

        for i in range(n_samples):
            # Original
            axes[0, i].imshow(orig_np[i], cmap='gray', vmin=0, vmax=1)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)

            # Reconstructed
            axes[1, i].imshow(recon_np[i], cmap='gray', vmin=0, vmax=1)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)

            # Difference
            axes[2, i].imshow(diff_np[i], cmap='hot', vmin=0, vmax=0.2)
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_title('Difference', fontsize=10)

        plt.tight_layout()
        return fig

    def _save_vae_checkpoint(self, epoch: int, filename: str) -> str:
        """Save VAE checkpoint with config for easy loading.

        Args:
            epoch: Current epoch number.
            filename: Checkpoint filename (without extension).

        Returns:
            Path to saved checkpoint.
        """
        os.makedirs(self.save_dir, exist_ok=True)

        # Include VAE config in checkpoint for easy reconstruction
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

        disc_config = {
            'in_channels': n_channels,
            'channels': self.disc_num_channels,
            'num_layers_d': self.disc_num_layers,
        }

        checkpoint = {
            'model_state_dict': self.model_raw.state_dict(),
            'discriminator_state_dict': self.discriminator_raw.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.lr_scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.lr_scheduler_d.state_dict(),
            'epoch': epoch,
            'config': vae_config,
            'disc_config': disc_config,
        }

        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        save_path = os.path.join(self.save_dir, f"{filename}.pt")
        torch.save(checkpoint, save_path)
        return save_path

    def train(self, train_loader: DataLoader, train_dataset: Dataset) -> None:
        """Execute the main training loop.

        Args:
            train_loader: Training data loader.
            train_dataset: Training dataset (for validation sampling).
        """
        total_start = time.time()

        avg_losses = {'gen': float('inf'), 'disc': float('inf'), 'recon': float('inf'),
                      'perc': float('inf'), 'kl': float('inf'), 'adv': float('inf')}

        try:
            for epoch in range(self.n_epochs):
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

                # Step both schedulers
                self.lr_scheduler_g.step()
                self.lr_scheduler_d.step()

                if self.is_main_process:
                    log_vae_epoch_summary(epoch, self.n_epochs, avg_losses, epoch_time)

                    if self.writer is not None:
                        self.writer.add_scalar('Loss/Generator', avg_losses['gen'], epoch)
                        self.writer.add_scalar('Loss/Discriminator', avg_losses['disc'], epoch)
                        self.writer.add_scalar('Loss/L1', avg_losses['recon'], epoch)
                        self.writer.add_scalar('Loss/Perceptual', avg_losses['perc'], epoch)
                        self.writer.add_scalar('Loss/KL', avg_losses['kl'], epoch)
                        self.writer.add_scalar('Loss/Adversarial', avg_losses['adv'], epoch)
                        self.writer.add_scalar('LR/Generator', self.lr_scheduler_g.get_last_lr()[0], epoch)
                        self.writer.add_scalar('LR/Discriminator', self.lr_scheduler_d.get_last_lr()[0], epoch)

                    is_val_epoch = (epoch + 1) % self.val_interval == 0

                    if is_val_epoch or (epoch + 1) == self.n_epochs:
                        self._generate_validation_samples(train_dataset, epoch)

                        self._save_vae_checkpoint(epoch, f"epoch_{epoch:04d}")
                        self._cleanup_old_checkpoints(keep_n=3)

                        self._save_vae_checkpoint(epoch, "latest")

                        if avg_losses['gen'] < self.best_loss:
                            self.best_loss = avg_losses['gen']
                            self._save_vae_checkpoint(epoch, "best")
                            logger.info(f"New best model saved (G loss: {avg_losses['gen']:.6f})")

        finally:
            total_time = time.time() - total_start

            if self.is_main_process:
                logger.info(f"Training completed! Total time: {total_time:.1f}s ({total_time / 3600:.1f}h)")
                self._update_metadata_final(avg_losses['gen'], avg_losses['recon'], total_time)

                if self.writer is not None:
                    self.writer.close()

            if self.use_multi_gpu:
                try:
                    dist.destroy_process_group()
                except Exception as e:
                    logger.warning(f"Error destroying process group: {e}")
