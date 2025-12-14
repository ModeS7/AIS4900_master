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

from monai.losses import PatchAdversarialLoss
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from skimage.metrics import structural_similarity as ssim_skimage

from .losses import PerceptualLoss
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
        self.log_worst_batch: bool = logging_cfg.get('worst_batch', False)

        # Gradient norm tracking (per epoch)
        self._grad_norm_g_sum: float = 0.0
        self._grad_norm_g_max: float = 0.0
        self._grad_norm_d_sum: float = 0.0
        self._grad_norm_d_max: float = 0.0
        self._grad_norm_count: int = 0

        # FLOPs tracking
        self._flops_measured: bool = False
        self._forward_flops: int = 0
        self._total_flops: int = 0

        # Worst batch tracking
        self._worst_batch_loss: float = 0.0
        self._worst_batch_data: Optional[Dict[str, Any]] = None

        # LPIPS model (initialized lazily in setup_model)
        self._lpips_model: Optional[Any] = None

        # Validation loader (set in train())
        self.val_loader: Optional[DataLoader] = None

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
        if pretrained_checkpoint and os.path.exists(pretrained_checkpoint):
            checkpoint = torch.load(pretrained_checkpoint, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                raw_model.load_state_dict(checkpoint['model_state_dict'])
                if self.is_main_process:
                    logger.info(f"Loaded VAE weights from {pretrained_checkpoint}")
            if 'discriminator_state_dict' in checkpoint and raw_disc is not None:
                raw_disc.load_state_dict(checkpoint['discriminator_state_dict'])
                if self.is_main_process:
                    logger.info(f"Loaded discriminator weights from {pretrained_checkpoint}")

        if self.use_multi_gpu:
            self.model_raw = raw_model
            ddp_model = DDP(
                raw_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
            if self.use_compile:
                self.model = torch.compile(ddp_model, mode="reduce-overhead")
            else:
                self.model = ddp_model

            if raw_disc is not None:
                self.discriminator_raw = raw_disc
                ddp_disc = DDP(
                    raw_disc,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                    gradient_as_bucket_view=True,
                )
                if self.use_compile:
                    self.discriminator = torch.compile(ddp_disc, mode="reduce-overhead")
                else:
                    self.discriminator = ddp_disc
            if self.is_main_process:
                msg = f"Multi-GPU: {'Compiled ' if self.use_compile else ''}DDP VAE"
                if not self.disable_gan:
                    msg += " + Discriminator"
                logger.info(msg)
        else:
            self.model_raw = raw_model
            if self.use_compile:
                self.model = torch.compile(raw_model, mode="default")
            else:
                self.model = raw_model
            if raw_disc is not None:
                self.discriminator_raw = raw_disc
                if self.use_compile:
                    self.discriminator = torch.compile(raw_disc, mode="default")
                else:
                    self.discriminator = raw_disc

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

            # Warmup + Cosine scheduler for discriminator (only if GAN is enabled)
            if not self.disable_gan:
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

        # Initialize LPIPS model for validation metrics (if enabled)
        if self.log_lpips:
            try:
                import lpips
                self._lpips_model = lpips.LPIPS(net='alex', verbose=False).to(self.device)
                self._lpips_model.eval()
                for param in self._lpips_model.parameters():
                    param.requires_grad = False
                if self.is_main_process:
                    logger.info("LPIPS metric initialized (AlexNet)")
            except ImportError:
                if self.is_main_process:
                    logger.warning("lpips package not installed - LPIPS metric disabled")
                self.log_lpips = False

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
            checkpoint_path: Path to the checkpoint file.
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

    def _measure_model_flops(self, sample_batch: torch.Tensor) -> None:
        """Measure FLOPs for VAE forward pass using torch.profiler.

        Should be called once at the start of training with a sample batch.
        The measured FLOPs are stored and logged to TensorBoard.

        Args:
            sample_batch: Sample input tensor [B, C, H, W].
        """
        if not self.log_flops or self._flops_measured:
            return

        self.model_raw.eval()
        with torch.no_grad():
            try:
                from torch.profiler import profile, ProfilerActivity

                # Run profiler to count FLOPs
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    with_flops=True,
                ) as prof:
                    _ = self.model_raw(sample_batch)

                # Sum all FLOPs from profiler events
                total_flops = sum(e.flops for e in prof.key_averages() if e.flops > 0)

                self._forward_flops = total_flops
                self._flops_measured = True

                if self.is_main_process and total_flops > 0:
                    batch_size = sample_batch.shape[0]
                    flops_per_sample = total_flops / batch_size
                    gflops_per_sample = flops_per_sample / 1e9
                    # VAE has encoder + decoder, training has forward + backward ≈ 3x
                    logger.info(
                        f"VAE FLOPs measured: {gflops_per_sample:.2f} GFLOPs/sample "
                        f"(forward), ~{gflops_per_sample * 3:.2f} GFLOPs/sample (train step)"
                    )

                    if self.writer is not None:
                        self.writer.add_scalar('model/gflops_per_sample', gflops_per_sample, 0)

            except Exception as e:
                if self.is_main_process:
                    logger.warning(f"Failed to measure FLOPs: {e}")

        self.model_raw.train()

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
        images = self._prepare_batch(batch)
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
                self._grad_norm_d_sum += grad_norm_d
                self._grad_norm_d_max = max(self._grad_norm_d_max, grad_norm_d)

        # ==================== Generator Step ====================
        self.optimizer_g.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=torch.bfloat16):
            # Fresh forward pass for generator gradients
            reconstruction, mean, logvar = self.model(images)

            # L1 reconstruction loss (MONAI uses L1, not MSE)
            l1_loss = torch.abs(reconstruction - images).mean()

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
            self._grad_norm_g_sum += grad_norm_g
            self._grad_norm_g_max = max(self._grad_norm_g_max, grad_norm_g)
            self._grad_norm_count += 1

        # Track worst batch (for debugging high loss samples)
        g_loss_val = g_loss.item()
        if self.log_worst_batch and g_loss_val > self._worst_batch_loss:
            self._worst_batch_loss = g_loss_val
            self._worst_batch_data = {
                'input': images.detach(),
                'output': reconstruction.detach(),
                'loss': g_loss_val,
                'l1_loss': l1_loss.item(),
                'perceptual_loss': p_loss.item(),
            }

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
        self._grad_norm_g_sum = 0.0
        self._grad_norm_g_max = 0.0
        self._grad_norm_d_sum = 0.0
        self._grad_norm_d_max = 0.0
        self._grad_norm_count = 0

    def _log_grad_norms(self, epoch: int) -> None:
        """Log gradient norm statistics to TensorBoard."""
        if not self.log_grad_norm or self._grad_norm_count == 0 or self.writer is None:
            return

        avg_grad_norm_g = self._grad_norm_g_sum / self._grad_norm_count
        self.writer.add_scalar('training/grad_norm_g_avg', avg_grad_norm_g, epoch)
        self.writer.add_scalar('training/grad_norm_g_max', self._grad_norm_g_max, epoch)

        if not self.disable_gan:
            avg_grad_norm_d = self._grad_norm_d_sum / self._grad_norm_count
            self.writer.add_scalar('training/grad_norm_d_avg', avg_grad_norm_d, epoch)
            self.writer.add_scalar('training/grad_norm_d_max', self._grad_norm_d_max, epoch)

    def _log_worst_batch(self, epoch: int) -> None:
        """Log worst batch visualization to TensorBoard."""
        if not self.log_worst_batch or self._worst_batch_data is None or self.writer is None:
            return

        data = self._worst_batch_data
        input_imgs = data['input'].cpu()
        output_imgs = data['output'].cpu()

        # Create visualization figure
        n_samples = min(4, input_imgs.shape[0])
        fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))

        for i in range(n_samples):
            # Input
            if input_imgs.shape[1] == 1:
                axes[0, i].imshow(input_imgs[i, 0].numpy(), cmap='gray')
            else:
                axes[0, i].imshow(input_imgs[i, 0].numpy(), cmap='gray')
            axes[0, i].set_title(f'Input {i}')
            axes[0, i].axis('off')

            # Output
            if output_imgs.shape[1] == 1:
                axes[1, i].imshow(output_imgs[i, 0].numpy(), cmap='gray')
            else:
                axes[1, i].imshow(output_imgs[i, 0].numpy(), cmap='gray')
            axes[1, i].set_title(f'Recon {i}')
            axes[1, i].axis('off')

        fig.suptitle(f"Worst Batch - Loss: {data['loss']:.4f} (L1: {data['l1_loss']:.4f}, Perc: {data['perceptual_loss']:.4f})")
        plt.tight_layout()

        self.writer.add_figure('training/worst_batch', fig, epoch)
        plt.close(fig)

        # Log scalar
        self.writer.add_scalar('training/worst_batch_loss', data['loss'], epoch)

        # Reset for next epoch
        self._worst_batch_loss = 0.0
        self._worst_batch_data = None

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

        # Reset gradient norm tracking for this epoch
        self._reset_grad_norm_tracking()

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

    def _compute_lpips(self, generated: torch.Tensor, reference: torch.Tensor) -> float:
        """Compute LPIPS (perceptual similarity) between generated and reference images.

        Args:
            generated: Generated images [B, C, H, W].
            reference: Reference images [B, C, H, W].

        Returns:
            Average LPIPS across batch (lower is better, 0 = identical).
        """
        if self._lpips_model is None:
            return 0.0

        # LPIPS expects images in [-1, 1] range and RGB (3 channels)
        # For grayscale, replicate to 3 channels
        gen = generated.float()
        ref = reference.float()

        # Normalize to [-1, 1]
        gen = gen * 2.0 - 1.0
        ref = ref * 2.0 - 1.0

        # Handle different channel counts (LPIPS expects 3-channel RGB)
        num_channels = gen.shape[1]

        with torch.no_grad():
            if num_channels == 1:
                # Single channel: replicate to 3
                gen = gen.repeat(1, 3, 1, 1)
                ref = ref.repeat(1, 3, 1, 1)
                lpips_values = self._lpips_model(gen, ref)
            elif num_channels <= 3:
                # 2-3 channels: compute per-channel LPIPS and average
                lpips_per_channel = []
                for ch in range(num_channels):
                    ch_gen = gen[:, ch:ch+1].repeat(1, 3, 1, 1)
                    ch_ref = ref[:, ch:ch+1].repeat(1, 3, 1, 1)
                    lpips_per_channel.append(self._lpips_model(ch_gen, ch_ref))
                lpips_values = torch.stack(lpips_per_channel).mean(dim=0)
            else:
                # >3 channels: use first 3 channels computed per-channel
                lpips_per_channel = []
                for ch in range(3):
                    ch_gen = gen[:, ch:ch+1].repeat(1, 3, 1, 1)
                    ch_ref = ref[:, ch:ch+1].repeat(1, 3, 1, 1)
                    lpips_per_channel.append(self._lpips_model(ch_gen, ch_ref))
                lpips_values = torch.stack(lpips_per_channel).mean(dim=0)

        return float(lpips_values.mean().item())

    def _generate_validation_samples(self, dataset: Dataset, epoch: int) -> None:
        """Generate reconstruction samples and compute validation metrics.

        Uses val_loader if provided, otherwise samples from training dataset.

        Args:
            dataset: Training dataset (used if no val_loader).
            epoch: Current epoch number.
        """
        if self.writer is None:
            return

        model_to_use = self.ema.ema_model if self.ema is not None else self.model_raw
        model_to_use.eval()

        image_keys = self.cfg.mode.get('image_keys', ['t1_pre', 't1_gd'])

        # Get validation samples from val_loader or dataset
        if self.val_loader is not None:
            # Use validation loader
            val_batch = next(iter(self.val_loader))
            samples = self._prepare_batch(val_batch)
            # Limit to 8 samples for visualization
            samples = samples[:8]
        else:
            # Sample random images from training dataset
            n_samples = min(8, len(dataset))
            indices = torch.randperm(len(dataset))[:n_samples]

            samples_list = []
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
                samples_list.append(sample)

            samples = torch.stack(samples_list).to(self.device)

        # Reconstruct
        with torch.no_grad():
            with autocast('cuda', enabled=True, dtype=torch.bfloat16):
                reconstructed, _, _ = model_to_use(samples)

        # Compute metrics
        ssim = self._compute_ssim(reconstructed, samples)
        psnr = self._compute_psnr(reconstructed, samples)

        self.writer.add_scalar('validation/SSIM', ssim, epoch)
        self.writer.add_scalar('validation/PSNR', psnr, epoch)

        # Compute LPIPS if enabled
        lpips_val = 0.0
        if self.log_lpips and self._lpips_model is not None:
            lpips_val = self._compute_lpips(reconstructed, samples)
            self.writer.add_scalar('validation/LPIPS', lpips_val, epoch)

        # Create visualization figure
        fig = self._create_reconstruction_figure(samples, reconstructed)
        self.writer.add_figure('validation/reconstructions', fig, epoch)
        plt.close(fig)

        if self.is_main_process:
            msg = f"Validation - SSIM: {ssim:.4f}, PSNR: {psnr:.2f} dB"
            if self.log_lpips:
                msg += f", LPIPS: {lpips_val:.4f}"
            logger.info(msg)

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
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'epoch': epoch,
            'config': vae_config,
            'disable_gan': self.disable_gan,
            'use_constant_lr': self.use_constant_lr,
        }

        # Add discriminator state if GAN is enabled
        if not self.disable_gan and self.discriminator_raw is not None:
            checkpoint['discriminator_state_dict'] = self.discriminator_raw.state_dict()
            checkpoint['disc_config'] = disc_config
            if self.optimizer_d is not None:
                checkpoint['optimizer_d_state_dict'] = self.optimizer_d.state_dict()

        # Add scheduler states if not using constant LR
        if not self.use_constant_lr:
            if self.lr_scheduler_g is not None:
                checkpoint['scheduler_g_state_dict'] = self.lr_scheduler_g.state_dict()
            if not self.disable_gan and self.lr_scheduler_d is not None:
                checkpoint['scheduler_d_state_dict'] = self.lr_scheduler_d.state_dict()

        # Add EMA state if enabled
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        save_path = os.path.join(self.save_dir, f"{filename}.pt")
        torch.save(checkpoint, save_path)
        return save_path

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
        if self.log_flops and not self._flops_measured:
            try:
                first_batch = next(iter(train_loader))
                sample_images = self._prepare_batch(first_batch)
                self._measure_model_flops(sample_images)
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
                    log_vae_epoch_summary(epoch, self.n_epochs, avg_losses, epoch_time)

                    if self.writer is not None:
                        self.writer.add_scalar('Loss/Generator', avg_losses['gen'], epoch)
                        self.writer.add_scalar('Loss/Discriminator', avg_losses['disc'], epoch)
                        self.writer.add_scalar('Loss/L1', avg_losses['recon'], epoch)
                        self.writer.add_scalar('Loss/Perceptual', avg_losses['perc'], epoch)
                        self.writer.add_scalar('Loss/KL', avg_losses['kl'], epoch)
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

                    is_val_epoch = (epoch + 1) % self.val_interval == 0

                    # Log worst batch at validation intervals
                    if is_val_epoch and self.log_worst_batch:
                        self._log_worst_batch(epoch)

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
