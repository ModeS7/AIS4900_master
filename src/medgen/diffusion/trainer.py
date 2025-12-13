"""
Diffusion model trainer module.

This module provides the DiffusionTrainer class for training diffusion models
with various strategies (DDPM, Rectified Flow) and modes (segmentation,
conditional single, conditional dual).
"""
import glob
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch._dynamo.config
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

from monai.losses import PerceptualLoss
from monai.networks.nets import DiffusionModelUNet

from skimage.metrics import structural_similarity as ssim_skimage
from scipy import ndimage

from .modes import ConditionalDualMode, ConditionalSingleMode, SegmentationMode, TrainingMode
from .strategies import DDPMStrategy, RFlowStrategy, DiffusionStrategy
from .utils import get_vram_usage, log_epoch_summary, save_checkpoint, save_model_only

logger = logging.getLogger(__name__)


class DiffusionTrainer:
    """Unified diffusion model trainer composing strategy and mode.

    This trainer supports multiple diffusion strategies (DDPM, Rectified Flow)
    and training modes (segmentation, conditional single, conditional dual).
    It handles distributed training, mixed precision, and checkpoint management.

    Args:
        cfg: Hydra configuration object containing all settings.

    Example:
        >>> trainer = DiffusionTrainer(cfg)
        >>> trainer.setup_model(train_dataset)
        >>> trainer.train(train_loader, train_dataset)
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # Extract config values
        self.strategy_name: str = cfg.strategy.name
        self.mode_name: str = cfg.mode.name
        self.n_epochs: int = cfg.training.epochs
        self.batch_size: int = cfg.training.batch_size
        self.image_size: int = cfg.model.image_size
        self.learning_rate: float = cfg.training.learning_rate
        self.perceptual_weight: float = cfg.training.perceptual_weight
        self.num_timesteps: int = cfg.strategy.num_train_timesteps
        self.warmup_epochs: int = cfg.training.warmup_epochs
        self.val_interval: int = cfg.training.val_interval
        self.use_multi_gpu: bool = cfg.training.use_multi_gpu
        self.use_ema: bool = cfg.training.use_ema
        self.ema_decay: float = cfg.training.ema.decay
        self.use_min_snr: bool = cfg.training.use_min_snr
        self.min_snr_gamma: float = cfg.training.min_snr_gamma

        # Logging config (with defaults for backward compatibility)
        logging_cfg = cfg.training.get('logging', {})
        self.log_grad_norm: bool = logging_cfg.get('grad_norm', True)
        self.log_timestep_losses: bool = logging_cfg.get('timestep_losses', True)
        self.log_regional_losses: bool = logging_cfg.get('regional_losses', True)
        self.log_timestep_region: bool = logging_cfg.get('timestep_region_losses', True)
        self.log_ssim: bool = logging_cfg.get('ssim', True)
        self.log_psnr: bool = logging_cfg.get('psnr', True)
        self.log_boundary_sharpness: bool = logging_cfg.get('boundary_sharpness', True)
        self.log_intermediate_steps: bool = logging_cfg.get('intermediate_steps', True)
        self.log_worst_batch: bool = logging_cfg.get('worst_batch', True)
        self.num_intermediate_steps: int = logging_cfg.get('num_intermediate_steps', 5)

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

        # Initialize strategy and mode
        self.strategy = self._create_strategy(self.strategy_name)
        self.mode = self._create_mode(self.mode_name)
        self.scheduler = self.strategy.setup_scheduler(self.num_timesteps, self.image_size)

        # Initialize logging and save directories
        # Hydra manages the run directory via hydra.run.dir
        if self.is_main_process:
            # Get Hydra output directory (or use fallback)
            try:
                from hydra.core.hydra_config import HydraConfig
                self.save_dir = HydraConfig.get().runtime.output_dir
            except (ImportError, ValueError, AttributeError):
                # Fallback for non-Hydra runs (e.g., direct script execution)
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                self.run_name = f"{self.strategy_name}_{self.image_size}_{timestamp}"
                self.save_dir = os.path.join(cfg.paths.model_dir, self.mode_name, self.run_name)

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
        self.ema: Optional[EMA] = None  # EMA wrapper from ema-pytorch
        self.optimizer: Optional[AdamW] = None
        self.lr_scheduler: Optional[LRScheduler] = None
        self.perceptual_loss_fn: Optional[nn.Module] = None

        # Timestep loss tracking (for analysis of which timesteps are hardest)
        # GPU accumulators - initialized lazily on first use
        self.num_timestep_bins: int = 10
        self._timestep_accum_initialized: bool = False
        self.timestep_loss_sum: Optional[torch.Tensor] = None  # [num_bins]
        self.timestep_loss_count: Optional[torch.Tensor] = None  # [num_bins]

        # Regional loss tracking (tumor vs background) - GPU accumulators for efficiency
        # Initialized lazily on first use to ensure correct device
        self._regional_accum_initialized: bool = False
        self.tumor_loss_sum: Optional[torch.Tensor] = None
        self.tumor_loss_count: Optional[torch.Tensor] = None
        self.bg_loss_sum: Optional[torch.Tensor] = None
        self.bg_loss_count: Optional[torch.Tensor] = None

        # Tumor size loss tracking based on clinical definitions
        # Clinical thresholds (diameter in mm): tiny <10, small 10-20, medium 20-30, large >30
        # Convert to percentage of image area based on resolution
        self.tumor_size_thresholds = self._compute_tumor_size_thresholds()
        # GPU accumulators per size category (initialized lazily)
        self.tumor_size_loss_sum: Dict[str, Optional[torch.Tensor]] = {
            size: None for size in self.tumor_size_thresholds.keys()
        }
        self.tumor_size_loss_count: Dict[str, Optional[torch.Tensor]] = {
            size: None for size in self.tumor_size_thresholds.keys()
        }

        # Gradient norm tracking - GPU accumulators
        self.grad_norm_sum: Optional[torch.Tensor] = None
        self.grad_norm_max: Optional[torch.Tensor] = None
        self.grad_norm_count: int = 0

        # Worst batch tracking
        self.worst_batch_loss: float = 0.0
        self.worst_batch_data: Optional[Dict[str, Any]] = None

        # 2D timestep-region loss tracking - GPU accumulators [num_bins, 2] (tumor, background)
        self._timestep_region_accum_initialized: bool = False
        self.timestep_region_loss_sum: Optional[torch.Tensor] = None  # [num_bins, 2]
        self.timestep_region_loss_count: Optional[torch.Tensor] = None  # [num_bins, 2]

    def _compute_tumor_size_thresholds(self) -> Dict[str, Tuple[float, float]]:
        """Compute tumor size thresholds based on image resolution.

        Clinical definitions (diameter):
            - tiny:   <10mm  (often non-measurable per RANO-BM)
            - small:  10-20mm (small metastases, SRS alone)
            - medium: 20-30mm (SRS candidates)
            - large:  >30mm  (often surgical)

        FOV is read from config (dataset-specific).
        For other resolutions, scales proportionally assuming same FOV.
        """
        # Get FOV from config (default 240mm for BrainMetShare)
        fov_mm = self.cfg.paths.get('fov_mm', 240.0)
        mm_per_pixel = fov_mm / self.image_size
        total_pixels = self.image_size ** 2

        # Clinical diameter thresholds in mm
        diameter_thresholds = {
            'tiny': (0, 10),       # <10mm
            'small': (10, 20),     # 10-20mm
            'medium': (20, 30),    # 20-30mm
            'large': (30, 150),    # >30mm (cap at brain width)
        }

        # Convert diameter (mm) to area percentage
        thresholds = {}
        for size_name, (d_low, d_high) in diameter_thresholds.items():
            # Diameter in pixels
            d_low_px = d_low / mm_per_pixel
            d_high_px = d_high / mm_per_pixel
            # Approximate circular area
            area_low = 3.14159 * (d_low_px / 2) ** 2
            area_high = 3.14159 * (d_high_px / 2) ** 2
            # Convert to percentage
            pct_low = area_low / total_pixels
            pct_high = area_high / total_pixels
            thresholds[size_name] = (pct_low, pct_high)

        if self.is_main_process:
            logger.info(f"Tumor size thresholds for {self.image_size}px ({mm_per_pixel:.2f} mm/px):")
            for name, (low, high) in thresholds.items():
                logger.info(f"  {name}: {low*100:.3f}% - {high*100:.3f}%")

        return thresholds

    def _create_strategy(self, strategy: str) -> DiffusionStrategy:
        """Create a diffusion strategy instance."""
        strategies: Dict[str, type] = {
            'ddpm': DDPMStrategy,
            'rflow': RFlowStrategy
        }
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")
        return strategies[strategy]()

    def _create_mode(self, mode: str) -> TrainingMode:
        """Create a training mode instance."""
        modes: Dict[str, type] = {
            'seg': SegmentationMode,
            'bravo': ConditionalSingleMode,
            'dual': ConditionalDualMode
        }
        if mode not in modes:
            raise ValueError(f"Unknown mode: {mode}. Choose from {list(modes.keys())}")
        return modes[mode]()

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

            # Use SLURM_JOB_ID to generate unique port (prevents conflicts)
            if 'SLURM_JOB_ID' in os.environ:
                job_id = int(os.environ['SLURM_JOB_ID'])
                # Map job IDs to ports 12000-64999 (avoiding privileged <1024 and high >65535)
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

    def setup_model(self, train_dataset: Dataset) -> None:
        """Initialize model, optimizer, and loss functions."""
        model_input_channels = self.mode.get_model_config()

        # Get model architecture from config
        channels = tuple(self.cfg.model.channels)
        attention_levels = tuple(self.cfg.model.attention_levels)
        num_res_blocks = self.cfg.model.num_res_blocks
        num_head_channels = self.cfg.model.num_head_channels

        raw_model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=model_input_channels['in_channels'],
            out_channels=model_input_channels['out_channels'],
            channels=channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels
        ).to(self.device)

        if self.use_multi_gpu:
            self.model_raw = raw_model

            if self.mode_name == 'dual' and self.image_size == 256:
                torch._dynamo.config.optimize_ddp = False
                if self.is_main_process:
                    logger.info("Disabled DDPOptimizer for dual 256 (compilation workaround)")

            ddp_model = DDP(
                raw_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=False
            )

            self.model = torch.compile(ddp_model, mode="reduce-overhead")

            if self.is_main_process:
                logger.info("Multi-GPU: Compiled DDP wrapper with mode='reduce-overhead'")
        else:
            self.model_raw = raw_model
            self.model = torch.compile(raw_model, mode="default")

        # Setup perceptual loss
        perceptual_loss = PerceptualLoss(
            spatial_dims=2,
            network_type="radimagenet_resnet50",
            cache_dir=self.cfg.paths.cache_dir,
            pretrained=True,
        ).to(self.device)

        self.perceptual_loss_fn = torch.compile(perceptual_loss, mode="reduce-overhead")

        # Setup optimizer
        self.optimizer = AdamW(self.model_raw.parameters(), lr=self.learning_rate)

        # Warmup + Cosine scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.n_epochs - self.warmup_epochs,
            eta_min=1e-6
        )
        self.lr_scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs]
        )

        # Create EMA wrapper if enabled
        if self.use_ema:
            self.ema = EMA(
                self.model_raw,
                beta=self.ema_decay,
                update_after_step=self.cfg.training.ema.update_after_step,
                update_every=self.cfg.training.ema.update_every,
            )
            if self.is_main_process:
                logger.info(f"EMA enabled with decay={self.ema_decay}")

        # Save metadata
        if self.is_main_process:
            self._save_metadata()

    def _save_metadata(self) -> None:
        """Save training configuration to metadata.json."""
        os.makedirs(self.save_dir, exist_ok=True)

        # Save full Hydra config as YAML
        config_path = os.path.join(self.save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        # Save metadata JSON for quick reference
        metadata = {
            'strategy': self.strategy_name,
            'mode': self.mode_name,
            'epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'image_size': self.image_size,
            'learning_rate': self.learning_rate,
            'perceptual_weight': self.perceptual_weight,
            'num_timesteps': self.num_timesteps,
            'warmup_epochs': self.warmup_epochs,
            'val_interval': self.val_interval,
            'multi_gpu': self.use_multi_gpu,
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay if self.use_ema else None,
            'use_min_snr': self.use_min_snr,
            'min_snr_gamma': self.min_snr_gamma if self.use_min_snr else None,
            'model': {
                'channels': list(self.cfg.model.channels),
                'attention_levels': list(self.cfg.model.attention_levels),
                'num_res_blocks': self.cfg.model.num_res_blocks,
                'num_head_channels': self.cfg.model.num_head_channels,
            },
            'created_at': datetime.now().isoformat(),
        }

        metadata_path = os.path.join(self.save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Config saved to: {config_path}")

    def _update_metadata_final(self, final_loss: float, final_mse: float, total_time: float) -> None:
        """Update metadata.json with final training results."""
        metadata_path = os.path.join(self.save_dir, 'metadata.json')

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        metadata['results'] = {
            'final_loss': final_loss,
            'final_mse': final_mse,
            'best_loss': self.best_loss,
            'total_time_seconds': total_time,
            'total_time_hours': total_time / 3600,
            'completed_at': datetime.now().isoformat(),
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _cleanup_old_checkpoints(self, keep_n: int = 3) -> None:
        """Keep only the N most recent epoch checkpoints.

        Args:
            keep_n: Number of most recent epoch checkpoints to keep. Default 3.
        """
        pattern = os.path.join(self.save_dir, "epoch_*.pt")
        checkpoints = glob.glob(pattern)

        if len(checkpoints) <= keep_n:
            return

        # Sort by epoch number extracted from filename (not lexicographic)
        def get_epoch_num(path: str) -> int:
            # Extract epoch number from "epoch_0123.pt"
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

    def _compute_snr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute Min-SNR loss weights for given timesteps."""
        if self.strategy_name == 'ddpm':
            alphas_cumprod = self.scheduler.alphas_cumprod.to(timesteps.device)
            alpha_bar = alphas_cumprod[timesteps]
            snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
        else:
            t_normalized = timesteps.float() / self.num_timesteps
            snr = (1.0 - t_normalized) / (t_normalized + 1e-8)

        snr_clipped = torch.clamp(snr, max=self.min_snr_gamma)
        weights = snr_clipped / (snr + 1e-8)

        return weights

    def _compute_regional_losses(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[float, float]:
        """Compute MSE loss separately for tumor and background regions.

        Args:
            predicted: Predicted clean image [B, C, H, W]
            target: Ground truth image [B, C, H, W]
            mask: Binary segmentation mask [B, 1, H, W]

        Returns:
            Tuple of (tumor_loss, background_loss)
        """
        # Expand mask to match image channels
        if predicted.shape[1] > 1:
            mask = mask.expand(-1, predicted.shape[1], -1, -1)

        # Compute per-pixel squared error
        sq_error = (predicted - target) ** 2

        # Tumor region loss (where mask > 0)
        tumor_mask = (mask > 0.5).float()
        tumor_pixels = tumor_mask.sum() + 1e-8
        tumor_loss = (sq_error * tumor_mask).sum() / tumor_pixels

        # Background region loss (where mask = 0)
        bg_mask = 1.0 - tumor_mask
        bg_pixels = bg_mask.sum() + 1e-8
        bg_loss = (sq_error * bg_mask).sum() / bg_pixels

        return tumor_loss.item(), bg_loss.item()

    def _get_tumor_size_category_single(self, mask: torch.Tensor) -> str:
        """Categorize tumor size for a single sample based on clinical definitions.

        Args:
            mask: Binary segmentation mask [1, 1, H, W] (single sample)

        Returns:
            Size category string: 'tiny', 'small', 'medium', or 'large'
        """
        total_pixels = mask.shape[2] * mask.shape[3]
        tumor_pixels = (mask > 0.5).float().sum().item()
        tumor_ratio = tumor_pixels / total_pixels

        for size_name, (low, high) in self.tumor_size_thresholds.items():
            if low <= tumor_ratio < high:
                return size_name

        return 'large'  # Fallback

    def _init_regional_accumulators(self, device: torch.device) -> None:
        """Initialize GPU accumulators for regional loss tracking."""
        self.tumor_loss_sum = torch.tensor(0.0, device=device)
        self.tumor_loss_count = torch.tensor(0, device=device, dtype=torch.long)
        self.bg_loss_sum = torch.tensor(0.0, device=device)
        self.bg_loss_count = torch.tensor(0, device=device, dtype=torch.long)

        for size in self.tumor_size_thresholds.keys():
            self.tumor_size_loss_sum[size] = torch.tensor(0.0, device=device)
            self.tumor_size_loss_count[size] = torch.tensor(0, device=device, dtype=torch.long)

        self._regional_accum_initialized = True

    def _track_regional_losses(
        self,
        predicted_clean: Union[torch.Tensor, Dict[str, torch.Tensor]],
        images: Union[torch.Tensor, Dict[str, torch.Tensor]],
        mask: Optional[torch.Tensor]
    ) -> None:
        """Vectorized tracking of losses by region and tumor size.

        Computes all samples in parallel on GPU, accumulates without CPU transfer.

        Args:
            predicted_clean: Model's prediction of clean image(s)
            images: Ground truth image(s)
            mask: Segmentation mask (None for seg mode)
        """
        if mask is None:
            return

        # Lazy initialization of GPU accumulators
        if not self._regional_accum_initialized:
            self._init_regional_accumulators(mask.device)

        # Handle dict (dual mode) vs tensor (single mode)
        if isinstance(predicted_clean, dict):
            pred = torch.cat(list(predicted_clean.values()), dim=1)
            img = torch.cat(list(images.values()), dim=1)
        else:
            pred, img = predicted_clean, images

        # Vectorized computation for entire batch
        sq_error = (pred - img) ** 2  # [B, C, H, W]

        # Create masks and expand to match channels
        tumor_mask = (mask > 0.5).float()  # [B, 1, H, W]
        if sq_error.shape[1] > 1:
            tumor_mask_expanded = tumor_mask.expand_as(sq_error)
        else:
            tumor_mask_expanded = tumor_mask
        bg_mask_expanded = 1.0 - tumor_mask_expanded

        # Per-sample tumor pixel counts (for filtering and size categorization)
        tumor_pixels = tumor_mask.sum(dim=(1, 2, 3))  # [B]
        total_pixels = mask.shape[2] * mask.shape[3]

        # Per-sample regional losses (vectorized)
        tumor_pixels_safe = tumor_pixels.clamp(min=1)
        bg_pixels = total_pixels - tumor_pixels
        bg_pixels_safe = bg_pixels.clamp(min=1)

        # Sum over C, H, W dims, keep batch dim
        tumor_loss_per_sample = (sq_error * tumor_mask_expanded).sum(dim=(1, 2, 3)) / tumor_pixels_safe
        bg_loss_per_sample = (sq_error * bg_mask_expanded).sum(dim=(1, 2, 3)) / bg_pixels_safe

        # Filter samples with actual tumors (at least 10 pixels)
        has_tumor = tumor_pixels > 10
        valid_tumor_losses = tumor_loss_per_sample[has_tumor]
        valid_bg_losses = bg_loss_per_sample[has_tumor]
        valid_tumor_pixels = tumor_pixels[has_tumor]

        # Accumulate regional losses on GPU (no CPU transfer)
        if valid_tumor_losses.numel() > 0:
            self.tumor_loss_sum += valid_tumor_losses.sum()
            self.tumor_loss_count += valid_tumor_losses.numel()
            self.bg_loss_sum += valid_bg_losses.sum()
            self.bg_loss_count += valid_bg_losses.numel()

            # Categorize by tumor size and accumulate (no .any() sync - sum of empty = 0)
            tumor_ratios = valid_tumor_pixels / total_pixels
            for size_name, (low, high) in self.tumor_size_thresholds.items():
                size_mask = (tumor_ratios >= low) & (tumor_ratios < high)
                masked_losses = valid_tumor_losses[size_mask]
                self.tumor_size_loss_sum[size_name] += masked_losses.sum()
                self.tumor_size_loss_count[size_name] += size_mask.sum()

    def _track_timestep_region_loss(
        self,
        timesteps: torch.Tensor,
        predicted_clean: Union[torch.Tensor, Dict[str, torch.Tensor]],
        images: Union[torch.Tensor, Dict[str, torch.Tensor]],
        mask: torch.Tensor
    ) -> None:
        """Vectorized 2D loss tracking by timestep bin AND region - no CPU sync.

        Args:
            timesteps: Diffusion timesteps for current batch [B]
            predicted_clean: Model's prediction of clean image(s)
            images: Ground truth image(s)
            mask: Segmentation mask
        """
        # Lazy initialization of 2D GPU accumulators
        if not self._timestep_region_accum_initialized:
            device = mask.device
            self.timestep_region_loss_sum = torch.zeros(self.num_timestep_bins, 2, device=device)
            self.timestep_region_loss_count = torch.zeros(self.num_timestep_bins, 2, device=device, dtype=torch.long)
            self._timestep_region_accum_initialized = True

        # Vectorized computation (same as _track_regional_losses)
        if isinstance(predicted_clean, dict):
            pred = torch.cat(list(predicted_clean.values()), dim=1)
            img = torch.cat(list(images.values()), dim=1)
        else:
            pred, img = predicted_clean, images

        sq_error = (pred - img) ** 2
        tumor_mask = (mask > 0.5).float()
        if sq_error.shape[1] > 1:
            tumor_mask_expanded = tumor_mask.expand_as(sq_error)
        else:
            tumor_mask_expanded = tumor_mask
        bg_mask_expanded = 1.0 - tumor_mask_expanded

        tumor_pixels = tumor_mask.sum(dim=(1, 2, 3))  # [B]
        total_pixels = mask.shape[2] * mask.shape[3]

        tumor_pixels_safe = tumor_pixels.clamp(min=1)
        bg_pixels_safe = (total_pixels - tumor_pixels).clamp(min=1)

        tumor_loss = (sq_error * tumor_mask_expanded).sum(dim=(1, 2, 3)) / tumor_pixels_safe  # [B]
        bg_loss = (sq_error * bg_mask_expanded).sum(dim=(1, 2, 3)) / bg_pixels_safe  # [B]

        # Filter samples with tumors
        has_tumor = tumor_pixels > 10
        valid_tumor = tumor_loss[has_tumor]
        valid_bg = bg_loss[has_tumor]
        valid_timesteps = timesteps[has_tumor]

        if valid_tumor.numel() > 0:
            bin_size = self.num_timesteps // self.num_timestep_bins
            bin_indices = (valid_timesteps // bin_size).clamp(max=self.num_timestep_bins - 1).long()

            # Scatter to 2D tensor (no sync)
            self.timestep_region_loss_sum[:, 0].scatter_add_(0, bin_indices, valid_tumor)
            self.timestep_region_loss_sum[:, 1].scatter_add_(0, bin_indices, valid_bg)
            ones = torch.ones_like(bin_indices)
            self.timestep_region_loss_count[:, 0].scatter_add_(0, bin_indices, ones)
            self.timestep_region_loss_count[:, 1].scatter_add_(0, bin_indices, ones)

    def train_step(self, batch: torch.Tensor) -> Tuple[float, float, float]:
        """Execute a single training step."""
        self.optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=torch.bfloat16):
            prepared = self.mode.prepare_batch(batch, self.device)
            images = prepared['images']
            labels_dict = {'labels': prepared.get('labels')}

            if isinstance(images, dict):
                noise = {key: torch.randn_like(img).to(self.device) for key, img in images.items()}
            else:
                noise = torch.randn_like(images).to(self.device)

            timesteps = self.strategy.sample_timesteps(images)
            noisy_images = self.strategy.add_noise(images, noise, timesteps)
            model_input = self.mode.format_model_input(noisy_images, labels_dict)

            prediction = self.strategy.predict_noise_or_velocity(self.model, model_input, timesteps)
            mse_loss, predicted_clean = self.strategy.compute_loss(prediction, images, noise, noisy_images, timesteps)

            if self.use_min_snr:
                snr_weights = self._compute_snr_weights(timesteps)
                weight_mean = snr_weights.mean()
                mse_loss = mse_loss * weight_mean

            if isinstance(predicted_clean, dict):
                p_losses = [
                    self.perceptual_loss_fn(pred.float(), images[key].float())
                    for key, pred in predicted_clean.items()
                ]
                p_loss = sum(p_losses) / len(p_losses)  # Average, not sum
            else:
                p_loss = self.perceptual_loss_fn(predicted_clean.float(), images.float())

            total_loss = mse_loss + self.perceptual_weight * p_loss

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model_raw.parameters(), max_norm=self.cfg.training.gradient_clip_norm
        )
        self.optimizer.step()

        if self.use_ema:
            self._update_ema()

        # Track gradient norm (GPU accumulator - no sync)
        if self.log_grad_norm:
            if self.grad_norm_sum is None:
                self.grad_norm_sum = torch.tensor(0.0, device=self.device)
                self.grad_norm_max = torch.tensor(0.0, device=self.device)
            self.grad_norm_sum = self.grad_norm_sum + grad_norm
            self.grad_norm_max = torch.maximum(self.grad_norm_max, grad_norm)
            self.grad_norm_count += 1

        # Track timestep losses (per-sample)
        if self.log_timestep_losses:
            with torch.no_grad():
                self._track_timestep_loss_batch(timesteps, predicted_clean, images)

        # Track regional losses for conditional modes (only if enabled)
        mask = labels_dict.get('labels')
        if self.mode.is_conditional and mask is not None:
            with torch.no_grad():
                if self.log_regional_losses:
                    self._track_regional_losses(predicted_clean, images, mask)

                # Track 2D timestep-region losses
                if self.log_timestep_region:
                    self._track_timestep_region_loss(timesteps, predicted_clean, images, mask)

        # Track worst batch
        loss_val = total_loss.item()
        if self.log_worst_batch and loss_val > self.worst_batch_loss:
            self.worst_batch_loss = loss_val
            self.worst_batch_data = {
                'images': images.detach().cpu() if not isinstance(images, dict) else {k: v.detach().cpu() for k, v in images.items()},
                'mask': mask.detach().cpu() if mask is not None else None,
                'predicted': predicted_clean.detach().cpu() if not isinstance(predicted_clean, dict) else {k: v.detach().cpu() for k, v in predicted_clean.items()},
                'loss': loss_val,
            }

        return total_loss.item(), mse_loss.item(), p_loss.item()

    def _track_timestep_loss_batch(
        self,
        timesteps: torch.Tensor,
        predicted_clean: Union[torch.Tensor, Dict[str, torch.Tensor]],
        images: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> None:
        """Vectorized timestep loss tracking - no CPU sync.

        Args:
            timesteps: Timesteps for each sample in batch [B]
            predicted_clean: Predicted clean images
            images: Ground truth images
        """
        # Lazy initialization of GPU accumulators
        if not self._timestep_accum_initialized:
            device = timesteps.device
            self.timestep_loss_sum = torch.zeros(self.num_timestep_bins, device=device)
            self.timestep_loss_count = torch.zeros(self.num_timestep_bins, device=device, dtype=torch.long)
            self._timestep_accum_initialized = True

        # Vectorized MSE per sample
        if isinstance(predicted_clean, dict):
            pred = torch.cat(list(predicted_clean.values()), dim=1)
            img = torch.cat(list(images.values()), dim=1)
        else:
            pred, img = predicted_clean, images

        mse_per_sample = ((pred - img) ** 2).mean(dim=(1, 2, 3))  # [B]

        # Compute bin indices (all on GPU)
        bin_size = self.num_timesteps // self.num_timestep_bins
        bin_indices = (timesteps // bin_size).clamp(max=self.num_timestep_bins - 1).long()

        # Accumulate with scatter_add (no sync)
        self.timestep_loss_sum.scatter_add_(0, bin_indices, mse_per_sample)
        ones = torch.ones_like(bin_indices)
        self.timestep_loss_count.scatter_add_(0, bin_indices, ones)

    def _log_timestep_losses(self, epoch: int) -> None:
        """Save timestep loss distribution to JSON file."""
        if not self._timestep_accum_initialized or self.timestep_loss_sum is None:
            return

        # Single CPU transfer at epoch end
        counts = self.timestep_loss_count.cpu()
        total_count = counts.sum().item()
        if total_count == 0:
            return

        # Compute averages per bin
        sums = self.timestep_loss_sum.cpu()
        bin_size = self.num_timesteps // self.num_timestep_bins

        epoch_data = {}
        for bin_idx in range(self.num_timestep_bins):
            bin_start = bin_idx * bin_size
            bin_end = (bin_idx + 1) * bin_size - 1
            bin_label = f"{bin_start:04d}-{bin_end:04d}"
            count = counts[bin_idx].item()
            if count > 0:
                epoch_data[bin_label] = (sums[bin_idx] / count).item()
            else:
                epoch_data[bin_label] = 0.0

        filepath = os.path.join(self.save_dir, 'timestep_losses.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                all_data = json.load(f)
        else:
            all_data = {}

        all_data[str(epoch)] = epoch_data

        with open(filepath, 'w') as f:
            json.dump(all_data, f, indent=2)

        # Reset GPU accumulators for next epoch
        device = self.timestep_loss_sum.device
        self.timestep_loss_sum = torch.zeros(self.num_timestep_bins, device=device)
        self.timestep_loss_count = torch.zeros(self.num_timestep_bins, device=device, dtype=torch.long)

    def _log_regional_losses(self, epoch: int) -> None:
        """Save regional loss data (tumor vs background, by size) to JSON and TensorBoard."""
        if not self._regional_accum_initialized or self.tumor_loss_count is None:
            return

        # Single CPU transfer at epoch end
        tumor_count = self.tumor_loss_count.item()
        if tumor_count == 0:
            return  # No data to log (probably seg mode or no tumors)

        avg_tumor_loss = (self.tumor_loss_sum / tumor_count).item()
        avg_bg_loss = (self.bg_loss_sum / self.bg_loss_count).item()
        tumor_bg_ratio = avg_tumor_loss / (avg_bg_loss + 1e-8)

        # Compute averages by tumor size (single transfer per category)
        size_losses = {}
        for size_name in self.tumor_size_thresholds.keys():
            count = self.tumor_size_loss_count[size_name].item()
            if count > 0:
                size_losses[size_name] = (self.tumor_size_loss_sum[size_name] / count).item()
            else:
                size_losses[size_name] = 0.0

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('loss/tumor_region', avg_tumor_loss, epoch)
            self.writer.add_scalar('loss/background_region', avg_bg_loss, epoch)
            self.writer.add_scalar('loss/tumor_bg_ratio', tumor_bg_ratio, epoch)

            for size_name, loss_val in size_losses.items():
                self.writer.add_scalar(f'loss/tumor_size_{size_name}', loss_val, epoch)

        # Save to JSON file
        epoch_data = {
            'tumor': avg_tumor_loss,
            'background': avg_bg_loss,
            'tumor_bg_ratio': tumor_bg_ratio,
            'by_size': size_losses,
        }

        filepath = os.path.join(self.save_dir, 'regional_losses.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                all_data = json.load(f)
        else:
            all_data = {}

        all_data[str(epoch)] = epoch_data

        with open(filepath, 'w') as f:
            json.dump(all_data, f, indent=2)

        # Reset GPU accumulators for next epoch (keep on same device)
        device = self.tumor_loss_sum.device
        self.tumor_loss_sum = torch.tensor(0.0, device=device)
        self.tumor_loss_count = torch.tensor(0, device=device, dtype=torch.long)
        self.bg_loss_sum = torch.tensor(0.0, device=device)
        self.bg_loss_count = torch.tensor(0, device=device, dtype=torch.long)
        for size in self.tumor_size_thresholds.keys():
            self.tumor_size_loss_sum[size] = torch.tensor(0.0, device=device)
            self.tumor_size_loss_count[size] = torch.tensor(0, device=device, dtype=torch.long)

    def _log_grad_norms(self, epoch: int) -> None:
        """Log gradient norm statistics to TensorBoard."""
        if self.grad_norm_count == 0 or self.writer is None or self.grad_norm_sum is None:
            return

        # Single CPU transfer at epoch end
        avg_grad_norm = (self.grad_norm_sum / self.grad_norm_count).item()
        max_grad_norm = self.grad_norm_max.item()

        self.writer.add_scalar('training/grad_norm_avg', avg_grad_norm, epoch)
        self.writer.add_scalar('training/grad_norm_max', max_grad_norm, epoch)

        # Reset GPU accumulators for next epoch
        self.grad_norm_sum = torch.tensor(0.0, device=self.device)
        self.grad_norm_max = torch.tensor(0.0, device=self.device)
        self.grad_norm_count = 0

    def _log_worst_batch(self, epoch: int) -> None:
        """Save visualization of the worst (highest loss) batch from this epoch.

        Layout:
        - Dual mode: 4×8 grid (8 samples) - rows: GT_pre, Pred_pre, GT_gd, Pred_gd
        - Single mode: 2×16 grid (16 samples) - rows: GT, Pred
        """
        if self.worst_batch_data is None or self.writer is None:
            return

        data = self.worst_batch_data
        mask = data['mask']
        is_dual = isinstance(data['images'], dict)

        if is_dual:
            # Dual mode: 4 rows × 8 samples
            keys = list(data['images'].keys())
            images_pre = data['images'][keys[0]]
            images_gd = data['images'][keys[1]]
            pred_pre = data['predicted'][keys[0]]
            pred_gd = data['predicted'][keys[1]]

            num_show = min(8, images_pre.shape[0])
            fig, axes = plt.subplots(4, num_show, figsize=(2 * num_show, 8))
            fig.suptitle(f'Worst Batch - Epoch {epoch} (Loss: {data["loss"]:.6f})', fontsize=12)

            for i in range(num_show):
                # Row 0: GT T1_pre
                axes[0, i].imshow(images_pre[i, 0].numpy(), cmap='gray')
                axes[0, i].axis('off')
                if i == 0:
                    axes[0, i].set_ylabel('GT Pre', fontsize=10)

                # Row 1: Pred T1_pre
                axes[1, i].imshow(pred_pre[i, 0].numpy(), cmap='gray')
                if mask is not None:
                    axes[1, i].contour(mask[i, 0].numpy(), colors='red', linewidths=0.5, alpha=0.7)
                axes[1, i].axis('off')
                if i == 0:
                    axes[1, i].set_ylabel('Pred Pre', fontsize=10)

                # Row 2: GT T1_gd
                axes[2, i].imshow(images_gd[i, 0].numpy(), cmap='gray')
                axes[2, i].axis('off')
                if i == 0:
                    axes[2, i].set_ylabel('GT Gd', fontsize=10)

                # Row 3: Pred T1_gd
                axes[3, i].imshow(pred_gd[i, 0].numpy(), cmap='gray')
                if mask is not None:
                    axes[3, i].contour(mask[i, 0].numpy(), colors='red', linewidths=0.5, alpha=0.7)
                axes[3, i].axis('off')
                if i == 0:
                    axes[3, i].set_ylabel('Pred Gd', fontsize=10)

        else:
            # Single mode (seg/bravo): 2 rows × 16 samples
            images = data['images']
            predicted = data['predicted']

            num_show = min(16, images.shape[0])
            fig, axes = plt.subplots(2, num_show, figsize=(num_show, 2))
            fig.suptitle(f'Worst Batch - Epoch {epoch} (Loss: {data["loss"]:.6f})', fontsize=12)

            for i in range(num_show):
                # Row 0: Ground truth
                axes[0, i].imshow(images[i, 0].numpy(), cmap='gray')
                axes[0, i].axis('off')
                if i == 0:
                    axes[0, i].set_ylabel('GT', fontsize=10)

                # Row 1: Prediction
                axes[1, i].imshow(predicted[i, 0].numpy(), cmap='gray')
                if mask is not None:
                    axes[1, i].contour(mask[i, 0].numpy(), colors='red', linewidths=0.5, alpha=0.7)
                axes[1, i].axis('off')
                if i == 0:
                    axes[1, i].set_ylabel('Pred', fontsize=10)

        plt.tight_layout()

        # Save to file
        filepath = os.path.join(self.save_dir, f'worst_batch_epoch_{epoch:04d}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')

        # Add to TensorBoard (same figure)
        self.writer.add_figure('worst_batch', fig, epoch)
        plt.close(fig)

        # Reset for next epoch
        self.worst_batch_loss = 0.0
        self.worst_batch_data = None

    def _log_timestep_region_losses(self, epoch: int) -> None:
        """Log 2D heatmap of loss by timestep bin and region."""
        if self.writer is None:
            return

        if not self._timestep_region_accum_initialized or self.timestep_region_loss_sum is None:
            return

        # Single CPU transfer at epoch end
        counts = self.timestep_region_loss_count.cpu()
        total_count = counts.sum().item()
        if total_count == 0:
            return

        sums = self.timestep_region_loss_sum.cpu()
        bin_size = self.num_timesteps // self.num_timestep_bins

        # Build 2D array: rows = timestep bins, cols = [tumor, background]
        heatmap_data = np.zeros((self.num_timestep_bins, 2))
        labels_timestep = []

        for bin_idx in range(self.num_timestep_bins):
            bin_start = bin_idx * bin_size
            bin_end = (bin_idx + 1) * bin_size - 1
            labels_timestep.append(f'{bin_start}-{bin_end}')

            for col_idx in range(2):
                count = counts[bin_idx, col_idx].item()
                if count > 0:
                    heatmap_data[bin_idx, col_idx] = (sums[bin_idx, col_idx] / count).item()
                else:
                    heatmap_data[bin_idx, col_idx] = 0.0

        # Create heatmap figure
        fig, ax = plt.subplots(figsize=(6, 10))
        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Tumor', 'Background'])
        ax.set_yticks(range(self.num_timestep_bins))
        ax.set_yticklabels(labels_timestep)
        ax.set_xlabel('Region')
        ax.set_ylabel('Timestep Range')
        ax.set_title(f'Loss by Timestep & Region (Epoch {epoch})')
        plt.colorbar(im, ax=ax, label='MSE Loss')

        # Add text annotations
        for i in range(self.num_timestep_bins):
            for j in range(2):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.4f}',
                               ha='center', va='center', color='black', fontsize=8)

        plt.tight_layout()
        self.writer.add_figure('loss/timestep_region_heatmap', fig, epoch)
        plt.close(fig)

        # Save to JSON
        epoch_data = {}
        for bin_idx in range(self.num_timestep_bins):
            bin_start = bin_idx * bin_size
            bin_label = f'{bin_start:04d}'
            epoch_data[bin_label] = {
                'tumor': float(heatmap_data[bin_idx, 0]),
                'background': float(heatmap_data[bin_idx, 1]),
            }

        filepath = os.path.join(self.save_dir, 'timestep_region_losses.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                all_data = json.load(f)
        else:
            all_data = {}

        all_data[str(epoch)] = epoch_data

        with open(filepath, 'w') as f:
            json.dump(all_data, f, indent=2)

        # Reset GPU accumulators for next epoch
        device = self.timestep_region_loss_sum.device
        self.timestep_region_loss_sum = torch.zeros(self.num_timestep_bins, 2, device=device)
        self.timestep_region_loss_count = torch.zeros(self.num_timestep_bins, 2, device=device, dtype=torch.long)

    def _compute_ssim(self, generated: torch.Tensor, reference: torch.Tensor) -> float:
        """Compute SSIM between generated and reference images.

        Args:
            generated: Generated images [B, 1, H, W]
            reference: Reference images [B, 1, H, W]

        Returns:
            Average SSIM across batch
        """
        ssim_values = []
        gen_np = generated.cpu().numpy()
        ref_np = reference.cpu().numpy()

        for i in range(gen_np.shape[0]):
            gen_img = gen_np[i, 0]
            ref_img = ref_np[i, 0]
            # Normalize to 0-1 range
            gen_img = np.clip(gen_img, 0, 1)
            ref_img = np.clip(ref_img, 0, 1)
            ssim_val = ssim_skimage(gen_img, ref_img, data_range=1.0)
            ssim_values.append(ssim_val)

        return float(np.mean(ssim_values))

    def _compute_psnr(self, generated: torch.Tensor, reference: torch.Tensor) -> float:
        """Compute PSNR between generated and reference images.

        Args:
            generated: Generated images [B, 1, H, W]
            reference: Reference images [B, 1, H, W]

        Returns:
            Average PSNR across batch
        """
        gen_np = np.clip(generated.cpu().numpy(), 0, 1)
        ref_np = np.clip(reference.cpu().numpy(), 0, 1)

        mse = np.mean((gen_np - ref_np) ** 2)
        if mse < 1e-10:
            return 100.0  # Essentially identical

        psnr = 10 * np.log10(1.0 / mse)
        return float(psnr)

    def _compute_boundary_sharpness(
        self,
        generated: torch.Tensor,
        mask: torch.Tensor,
        dilation_pixels: int = 3
    ) -> float:
        """Compute boundary sharpness in tumor regions.

        Measures gradient magnitude at tumor boundaries. Higher = sharper edges.

        Args:
            generated: Generated images [B, 1, H, W]
            mask: Segmentation masks [B, 1, H, W]
            dilation_pixels: Pixels to dilate mask for boundary region

        Returns:
            Average boundary sharpness
        """
        sharpness_values = []
        gen_np = generated.cpu().numpy()
        mask_np = mask.cpu().numpy()

        for i in range(gen_np.shape[0]):
            img = gen_np[i, 0]
            m = (mask_np[i, 0] > 0.5).astype(np.float32)

            if m.sum() < 10:  # Skip if tumor too small
                continue

            # Create boundary mask: dilated - eroded
            dilated = ndimage.binary_dilation(m, iterations=dilation_pixels)
            eroded = ndimage.binary_erosion(m, iterations=dilation_pixels)
            boundary = (dilated.astype(np.float32) - eroded.astype(np.float32))

            if boundary.sum() < 1:
                continue

            # Compute gradient magnitude
            grad_x = ndimage.sobel(img, axis=0)
            grad_y = ndimage.sobel(img, axis=1)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # Average gradient at boundary
            boundary_grad = (grad_mag * boundary).sum() / (boundary.sum() + 1e-8)
            sharpness_values.append(boundary_grad)

        return float(np.mean(sharpness_values)) if sharpness_values else 0.0

    def _generate_with_intermediate_steps(
        self,
        model: nn.Module,
        model_input: torch.Tensor,
        num_steps: int
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate samples while saving intermediate denoising steps.

        Args:
            model: The diffusion model
            model_input: Initial noisy input (may include condition channels)
            num_steps: Number of denoising steps

        Returns:
            Tuple of (final_samples, list_of_intermediate_samples)
        """
        # Determine output channels based on mode
        model_config = self.mode.get_model_config()
        out_channels = model_config['out_channels']
        in_channels = model_config['in_channels']

        # Separate noisy channels from condition channels
        # For bravo: in=2 (noise+mask), out=1 (image)
        # For dual: in=3 (noise_pre+noise_gd+mask), out=2 (pre+gd)
        # For seg: in=1, out=1 (no condition)
        has_condition = in_channels > out_channels
        if has_condition:
            noisy = model_input[:, :out_channels, :, :]  # Channels to denoise
            condition = model_input[:, out_channels:, :, :]  # Condition (mask)
        else:
            noisy = model_input
            condition = None

        # Calculate which steps to save (evenly spaced + final)
        save_at_timesteps = set()
        if self.num_intermediate_steps > 0:
            # Save at specific timestep values: 1000, 750, 500, 250, 100, 0 (or scaled)
            for frac in [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]:
                save_at_timesteps.add(int(frac * (num_steps - 1)))

        intermediates = []
        current = noisy.clone()

        for step_idx in range(num_steps):
            timestep_val = num_steps - 1 - step_idx
            t = torch.full(
                (current.shape[0],),
                timestep_val,
                device=current.device,
                dtype=torch.long
            )

            # Re-concatenate condition for model input
            if condition is not None:
                full_input = torch.cat([current, condition], dim=1)
            else:
                full_input = current

            with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                pred = model(full_input, t)

            # Update only the noisy channels
            if self.strategy_name == 'rflow':
                # Euler step: x_{t-dt} = x_t + dt * v (velocity points from noise to clean)
                dt = 1.0 / num_steps
                current = current + dt * pred
            else:
                current = self.scheduler.step(pred, timestep_val, current).prev_sample

            # Save intermediate at key timesteps
            if timestep_val in save_at_timesteps:
                intermediates.append(current.clone())

        # Ensure we have the final result
        if not intermediates or not torch.equal(intermediates[-1], current):
            intermediates.append(current.clone())

        return current, intermediates

    def _log_intermediate_steps(
        self,
        epoch: int,
        intermediates: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> None:
        """Log intermediate denoising steps to TensorBoard.

        Args:
            epoch: Current epoch
            intermediates: List of intermediate samples
            mask: Optional segmentation mask for overlay
        """
        if self.writer is None or not intermediates:
            return

        # Create figure showing progression
        num_steps = len(intermediates)
        fig, axes = plt.subplots(1, num_steps, figsize=(4 * num_steps, 4))

        if num_steps == 1:
            axes = [axes]

        # Generate labels based on the fractions used in _generate_with_intermediate_steps
        fracs = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
        timestep_labels = [f't={int(frac * (self.num_timesteps - 1))}' for frac in fracs]
        # Trim to match actual number of intermediates
        timestep_labels = timestep_labels[:num_steps]

        for i, (intermediate, ax) in enumerate(zip(intermediates, axes)):
            img = intermediate[0, 0].cpu().numpy()  # First sample, first channel
            img = np.clip(img, 0, 1)
            ax.imshow(img, cmap='gray')
            if mask is not None:
                ax.contour(mask[0, 0].cpu().numpy(), colors='red', linewidths=0.5, alpha=0.7)
            ax.set_title(timestep_labels[i] if i < len(timestep_labels) else f'Step {i}')
            ax.axis('off')

        plt.tight_layout()
        self.writer.add_figure('denoising_trajectory', fig, epoch)
        plt.close(fig)

        # Also save to file
        filepath = os.path.join(self.save_dir, f'denoising_trajectory_epoch_{epoch:04d}.png')
        fig2, axes2 = plt.subplots(1, num_steps, figsize=(4 * num_steps, 4))
        if num_steps == 1:
            axes2 = [axes2]
        for i, (intermediate, ax) in enumerate(zip(intermediates, axes2)):
            img = intermediate[0, 0].cpu().numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img, cmap='gray')
            ax.set_title(timestep_labels[i] if i < len(timestep_labels) else f'Step {i}')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig2)

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Tuple[float, float, float]:
        """Train the model for one epoch."""
        self.model.train()
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_perceptual_loss = 0

        use_progress_bars = (not self.is_cluster) and self.is_main_process

        if use_progress_bars:
            progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), ncols=100)
            progress_bar.set_description(f"Epoch {epoch}")
            steps_iter = progress_bar
        else:
            steps_iter = enumerate(data_loader)

        for step, batch in steps_iter:
            loss, mse_loss, p_loss = self.train_step(batch)

            epoch_loss += loss
            epoch_mse_loss += mse_loss
            epoch_perceptual_loss += p_loss

            if use_progress_bars:
                progress_bar.set_postfix(loss=f"{epoch_loss / (step + 1):.6f}")

            if epoch == 1 and step == 0 and self.is_main_process:
                logger.info(get_vram_usage(self.device))

        return epoch_loss / len(data_loader), epoch_mse_loss / len(data_loader), epoch_perceptual_loss / len(data_loader)

    def _sample_positive_masks(
            self, train_dataset: Dataset, num_samples: int, seg_channel_idx: int,
            return_images: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample slices with positive segmentation masks from dataset.

        Args:
            train_dataset: Dataset to sample from
            num_samples: Number of samples to get
            seg_channel_idx: Channel index for segmentation mask
            return_images: If True, also return ground truth images

        Returns:
            Segmentation masks [N, 1, H, W], or tuple of (masks, images) if return_images=True
        """
        seg_masks = []
        gt_images = []
        attempts = 0
        max_attempts = len(train_dataset)

        while len(seg_masks) < num_samples and attempts < max_attempts:
            idx = torch.randint(0, len(train_dataset), (1,)).item()
            data = train_dataset[idx]
            tensor = torch.from_numpy(data).float() if hasattr(data, '__array__') else torch.tensor(data).float()
            seg = tensor[seg_channel_idx:seg_channel_idx + 1, :, :]

            if seg.sum() > 0:
                seg_masks.append(seg)
                if return_images:
                    # Get the image channel (first channel for bravo, 0 and 1 for dual)
                    if seg_channel_idx == 1:  # bravo mode
                        gt_images.append(tensor[0:1, :, :])
                    elif seg_channel_idx == 2:  # dual mode
                        gt_images.append(tensor[0:2, :, :])  # Both T1_pre and T1_gd
            attempts += 1

        if len(seg_masks) < num_samples:
            logger.warning(f"Only found {len(seg_masks)} positive masks for validation")

        if len(seg_masks) == 0:
            raise ValueError("No positive segmentation masks found for validation")

        masks = torch.stack(seg_masks).to(self.device)
        if return_images:
            images = torch.stack(gt_images).to(self.device)
            return masks, images
        return masks

    def generate_validation_samples(
            self, epoch: int, train_dataset: Dataset, num_samples: int = 4
    ) -> None:
        """Generate and log validation samples to TensorBoard."""
        if not self.is_main_process or self.writer is None:
            return

        model_to_use = self.ema.ema_model if self.ema is not None else self.model_raw
        model_to_use.eval()

        try:
            with torch.no_grad():
                model_config = self.mode.get_model_config()
                out_channels = model_config['out_channels']

                seg_masks = None
                gt_images = None
                intermediates = None

                if self.mode_name == 'seg':
                    noise = torch.randn((num_samples, 1, self.image_size, self.image_size), device=self.device)
                    model_input = noise

                elif self.mode_name == 'bravo':
                    # Get masks and ground truth images for metrics
                    need_gt = self.log_ssim or self.log_psnr
                    if need_gt:
                        seg_masks, gt_images = self._sample_positive_masks(
                            train_dataset, num_samples, seg_channel_idx=1, return_images=True
                        )
                    else:
                        seg_masks = self._sample_positive_masks(train_dataset, num_samples, seg_channel_idx=1)
                    noise = torch.randn_like(seg_masks, device=self.device)
                    model_input = torch.cat([noise, seg_masks], dim=1)

                elif self.mode_name == 'dual':
                    need_gt = self.log_ssim or self.log_psnr
                    if need_gt:
                        seg_masks, gt_images = self._sample_positive_masks(
                            train_dataset, num_samples, seg_channel_idx=2, return_images=True
                        )
                    else:
                        seg_masks = self._sample_positive_masks(train_dataset, num_samples, seg_channel_idx=2)
                    noise_pre = torch.randn_like(seg_masks, device=self.device)
                    noise_gd = torch.randn_like(seg_masks, device=self.device)
                    model_input = torch.cat([noise_pre, noise_gd, seg_masks], dim=1)

                else:
                    raise ValueError(f"Unknown mode: {self.mode_name}")

                # Generate samples (with intermediate steps if configured)
                with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    if self.log_intermediate_steps and self.mode_name != 'seg':
                        samples, intermediates = self._generate_with_intermediate_steps(
                            model_to_use, model_input, self.num_timesteps
                        )
                    else:
                        samples = self.strategy.generate(
                            model_to_use, model_input, num_steps=self.num_timesteps, device=self.device
                        )

                # Log intermediate denoising steps
                if intermediates is not None and self.log_intermediate_steps:
                    self._log_intermediate_steps(epoch, intermediates, seg_masks)

                # Process and log generated samples
                if out_channels == 2:
                    samples_pre = samples[:, 0:1, :, :].float()
                    samples_gd = samples[:, 1:2, :, :].float()

                    samples_pre_norm = torch.clamp(samples_pre, 0, 1)
                    samples_gd_norm = torch.clamp(samples_gd, 0, 1)

                    samples_pre_rgb = samples_pre_norm.repeat(1, 3, 1, 1)
                    samples_gd_rgb = samples_gd_norm.repeat(1, 3, 1, 1)

                    self.writer.add_images('Generated_T1_Pre', samples_pre_rgb, epoch)
                    self.writer.add_images('Generated_T1_Gd', samples_gd_rgb, epoch)

                    # Compute metrics for dual mode
                    if gt_images is not None:
                        gt_pre = gt_images[:, 0:1, :, :]
                        gt_gd = gt_images[:, 1:2, :, :]

                        if self.log_ssim:
                            ssim_pre = self._compute_ssim(samples_pre_norm, gt_pre)
                            ssim_gd = self._compute_ssim(samples_gd_norm, gt_gd)
                            self.writer.add_scalar('metrics/ssim_t1_pre', ssim_pre, epoch)
                            self.writer.add_scalar('metrics/ssim_t1_gd', ssim_gd, epoch)

                        if self.log_psnr:
                            psnr_pre = self._compute_psnr(samples_pre_norm, gt_pre)
                            psnr_gd = self._compute_psnr(samples_gd_norm, gt_gd)
                            self.writer.add_scalar('metrics/psnr_t1_pre', psnr_pre, epoch)
                            self.writer.add_scalar('metrics/psnr_t1_gd', psnr_gd, epoch)

                    if self.log_boundary_sharpness and seg_masks is not None:
                        sharpness_pre = self._compute_boundary_sharpness(samples_pre_norm, seg_masks)
                        sharpness_gd = self._compute_boundary_sharpness(samples_gd_norm, seg_masks)
                        self.writer.add_scalar('metrics/boundary_sharpness_t1_pre', sharpness_pre, epoch)
                        self.writer.add_scalar('metrics/boundary_sharpness_t1_gd', sharpness_gd, epoch)
                else:
                    samples_float = samples.float()
                    samples_normalized = torch.clamp(samples_float, 0, 1)

                    if samples_normalized.dim() == 3:
                        samples_normalized = samples_normalized.unsqueeze(1)

                    samples_rgb = samples_normalized.repeat(1, 3, 1, 1)
                    self.writer.add_images('Generated_Images', samples_rgb, epoch)

                    # Compute metrics for bravo mode
                    if gt_images is not None and self.mode_name == 'bravo':
                        if self.log_ssim:
                            ssim_val = self._compute_ssim(samples_normalized, gt_images)
                            self.writer.add_scalar('metrics/ssim', ssim_val, epoch)

                        if self.log_psnr:
                            psnr_val = self._compute_psnr(samples_normalized, gt_images)
                            self.writer.add_scalar('metrics/psnr', psnr_val, epoch)

                    if self.log_boundary_sharpness and seg_masks is not None and self.mode_name == 'bravo':
                        sharpness = self._compute_boundary_sharpness(samples_normalized, seg_masks)
                        self.writer.add_scalar('metrics/boundary_sharpness', sharpness, epoch)

        except Exception as e:
            if self.is_main_process:
                logger.warning(
                    f"Failed to generate validation samples at epoch {epoch}: {e}",
                    exc_info=True
                )
        finally:
            torch.cuda.empty_cache()
            self.model.train()

    def train(self, train_loader: DataLoader, train_dataset: Dataset) -> None:
        """Execute the main training loop."""
        total_start = time.time()

        # Initialize with safe defaults for finally block (in case of early exception)
        avg_loss = float('inf')
        avg_mse = float('inf')

        try:
            for epoch in range(self.n_epochs):
                epoch_start = time.time()

                if self.use_multi_gpu and hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)

                avg_loss, avg_mse, avg_perceptual = self.train_epoch(train_loader, epoch)

                if self.use_multi_gpu:
                    loss_tensor = torch.tensor([avg_loss, avg_mse, avg_perceptual], device=self.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    avg_loss, avg_mse, avg_perceptual = (loss_tensor / self.world_size).cpu().numpy()

                epoch_time = time.time() - epoch_start
                self.lr_scheduler.step()

                if self.is_main_process:
                    log_epoch_summary(epoch, self.n_epochs, (avg_loss, avg_mse, avg_perceptual), epoch_time)

                    if self.writer is not None:
                        self.writer.add_scalar('Loss/Total', avg_loss, epoch)
                        self.writer.add_scalar('Loss/MSE', avg_mse, epoch)
                        self.writer.add_scalar('Loss/Perceptual', avg_perceptual, epoch)
                        self.writer.add_scalar('LR', self.lr_scheduler.get_last_lr()[0], epoch)

                        # Log gradient norms every epoch (lightweight)
                        if self.log_grad_norm:
                            self._log_grad_norms(epoch)

                        if (epoch + 1) % self.val_interval == 0:
                            if self.log_timestep_losses:
                                self._log_timestep_losses(epoch)
                            if self.log_regional_losses:
                                self._log_regional_losses(epoch)
                            if self.log_timestep_region:
                                self._log_timestep_region_losses(epoch)
                            if self.log_worst_batch:
                                self._log_worst_batch(epoch)

                    if (epoch + 1) % self.val_interval == 0 or (epoch + 1) == self.n_epochs:
                        self.generate_validation_samples(epoch, train_dataset)

                        filename = f"epoch_{epoch:04d}"
                        save_model_only(self.model_raw, epoch, self.save_dir, filename, self.ema)
                        self._cleanup_old_checkpoints(keep_n=3)

                        save_checkpoint(
                            self.model_raw, self.optimizer, self.lr_scheduler,
                            epoch, self.save_dir, "latest", self.ema
                        )

                        if avg_loss < self.best_loss:
                            self.best_loss = avg_loss
                            save_checkpoint(
                                self.model_raw, self.optimizer, self.lr_scheduler,
                                epoch, self.save_dir, "best", self.ema
                            )
                            logger.info(f"New best model saved (loss: {avg_loss:.6f})")

        finally:
            # Ensure cleanup always happens, even on exceptions
            total_time = time.time() - total_start

            if self.is_main_process:
                logger.info(f"Training completed! Total time: {total_time:.1f}s ({total_time / 3600:.1f}h)")
                self._update_metadata_final(avg_loss, avg_mse, total_time)

                if self.writer is not None:
                    self.writer.close()

            if self.use_multi_gpu:
                try:
                    dist.destroy_process_group()
                except Exception as e:
                    logger.warning(f"Error destroying process group: {e}")
