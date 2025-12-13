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
from typing import Any, Dict, Optional, Tuple

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

from .modes import ConditionalDualMode, ConditionalSingleMode, SegmentationMode, TrainingMode
from .strategies import DDPMStrategy, RFlowStrategy, DiffusionStrategy
from .metrics import MetricsTracker
from .visualization import ValidationVisualizer
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
        if self.is_main_process:
            try:
                from hydra.core.hydra_config import HydraConfig
                self.save_dir = HydraConfig.get().runtime.output_dir
            except (ImportError, ValueError, AttributeError):
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

        # Initialize metrics tracker
        self.metrics = MetricsTracker(
            cfg=cfg,
            device=self.device,
            writer=self.writer,
            save_dir=self.save_dir,
            is_main_process=self.is_main_process,
            is_conditional=self.mode.is_conditional,
        )
        self.metrics.set_scheduler(self.scheduler)

        # Initialize model components (set during setup_model)
        self.model: Optional[nn.Module] = None
        self.model_raw: Optional[nn.Module] = None
        self.ema: Optional[EMA] = None
        self.optimizer: Optional[AdamW] = None
        self.lr_scheduler: Optional[LRScheduler] = None
        self.perceptual_loss_fn: Optional[nn.Module] = None

        # Visualization helper (initialized in setup_model after strategy is ready)
        self.visualizer: Optional[ValidationVisualizer] = None

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

    def setup_model(self, train_dataset: Dataset) -> None:
        """Initialize model, optimizer, and loss functions."""
        model_input_channels = self.mode.get_model_config()

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

        # Compile fused forward pass
        compile_fused = self.cfg.training.get('compile_fused_forward', True)
        if compile_fused and self.is_main_process:
            logger.info("Compiling fused forward pass")
        self._setup_compiled_forward(compile_fused)

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

        # Initialize visualization helper
        self.visualizer = ValidationVisualizer(
            cfg=self.cfg,
            strategy=self.strategy,
            mode=self.mode,
            metrics=self.metrics,
            writer=self.writer,
            save_dir=self.save_dir,
            device=self.device,
            is_main_process=self.is_main_process,
        )

        # Save metadata
        if self.is_main_process:
            self._save_metadata()

    def _setup_compiled_forward(self, enabled: bool) -> None:
        """Setup compiled forward functions for fused model + loss computation."""
        self._use_compiled_forward = enabled

        if not enabled:
            self._compiled_forward_single = None
            self._compiled_forward_dual = None
            return

        def _forward_single(
            model: nn.Module,
            perceptual_fn: nn.Module,
            model_input: torch.Tensor,
            timesteps: torch.Tensor,
            images: torch.Tensor,
            noise: torch.Tensor,
            noisy_images: torch.Tensor,
            perceptual_weight: float,
            strategy_name: str,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Fused forward: model prediction + MSE loss + perceptual loss."""
            prediction = model(model_input, timesteps)

            if strategy_name == 'rflow':
                t_normalized = timesteps.float() / 1000.0
                t_expanded = t_normalized.view(-1, 1, 1, 1)
                predicted_clean = torch.clamp(noisy_images + t_expanded * prediction, 0, 1)
            else:
                predicted_clean = torch.clamp(noisy_images - prediction, 0, 1)

            if strategy_name == 'rflow':
                target = images - noise
            else:
                target = noise
            mse_loss = ((prediction - target) ** 2).mean()

            p_loss = perceptual_fn(predicted_clean.float(), images.float())
            total_loss = mse_loss + perceptual_weight * p_loss

            return total_loss, mse_loss, p_loss, predicted_clean

        def _forward_dual(
            model: nn.Module,
            perceptual_fn: nn.Module,
            model_input: torch.Tensor,
            timesteps: torch.Tensor,
            images_0: torch.Tensor,
            images_1: torch.Tensor,
            noise_0: torch.Tensor,
            noise_1: torch.Tensor,
            noisy_0: torch.Tensor,
            noisy_1: torch.Tensor,
            perceptual_weight: float,
            strategy_name: str,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Fused forward for dual mode with separate tensor inputs."""
            prediction = model(model_input, timesteps)
            pred_0 = prediction[:, 0:1, :, :]
            pred_1 = prediction[:, 1:2, :, :]

            if strategy_name == 'rflow':
                t_normalized = timesteps.float() / 1000.0
                t_expanded = t_normalized.view(-1, 1, 1, 1)
                clean_0 = torch.clamp(noisy_0 + t_expanded * pred_0, 0, 1)
                clean_1 = torch.clamp(noisy_1 + t_expanded * pred_1, 0, 1)
            else:
                clean_0 = torch.clamp(noisy_0 - pred_0, 0, 1)
                clean_1 = torch.clamp(noisy_1 - pred_1, 0, 1)

            if strategy_name == 'rflow':
                target_0 = images_0 - noise_0
                target_1 = images_1 - noise_1
            else:
                target_0 = noise_0
                target_1 = noise_1

            mse_0 = ((pred_0 - target_0) ** 2).mean()
            mse_1 = ((pred_1 - target_1) ** 2).mean()
            mse_loss = (mse_0 + mse_1) / 2.0

            p_0 = perceptual_fn(clean_0.float(), images_0.float())
            p_1 = perceptual_fn(clean_1.float(), images_1.float())
            p_loss = (p_0 + p_1) / 2.0

            total_loss = mse_loss + perceptual_weight * p_loss

            return total_loss, mse_loss, p_loss, clean_0, clean_1

        self._compiled_forward_single = torch.compile(
            _forward_single, mode="reduce-overhead", fullgraph=True
        )
        self._compiled_forward_dual = torch.compile(
            _forward_dual, mode="reduce-overhead", fullgraph=True
        )

        if self.is_main_process:
            logger.info(f"Compiled fused forward functions for mode: {self.mode_name}")

    def _save_metadata(self) -> None:
        """Save training configuration to metadata.json."""
        os.makedirs(self.save_dir, exist_ok=True)

        config_path = os.path.join(self.save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))

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

            if self._use_compiled_forward and self.mode_name == 'dual':
                keys = list(images.keys())
                total_loss, mse_loss, p_loss, clean_0, clean_1 = self._compiled_forward_dual(
                    self.model,
                    self.perceptual_loss_fn,
                    model_input,
                    timesteps,
                    images[keys[0]],
                    images[keys[1]],
                    noise[keys[0]],
                    noise[keys[1]],
                    noisy_images[keys[0]],
                    noisy_images[keys[1]],
                    self.perceptual_weight,
                    self.strategy_name,
                )
                predicted_clean = {keys[0]: clean_0, keys[1]: clean_1}

                if self.use_min_snr:
                    snr_weights = self.metrics.compute_snr_weights(timesteps)
                    total_loss = total_loss * snr_weights.mean()

            elif self._use_compiled_forward and self.mode_name in ('seg', 'bravo'):
                total_loss, mse_loss, p_loss, predicted_clean = self._compiled_forward_single(
                    self.model,
                    self.perceptual_loss_fn,
                    model_input,
                    timesteps,
                    images,
                    noise,
                    noisy_images,
                    self.perceptual_weight,
                    self.strategy_name,
                )

                if self.use_min_snr:
                    snr_weights = self.metrics.compute_snr_weights(timesteps)
                    total_loss = total_loss * snr_weights.mean()

            else:
                prediction = self.strategy.predict_noise_or_velocity(self.model, model_input, timesteps)
                mse_loss, predicted_clean = self.strategy.compute_loss(prediction, images, noise, noisy_images, timesteps)

                if self.use_min_snr:
                    snr_weights = self.metrics.compute_snr_weights(timesteps)
                    weight_mean = snr_weights.mean()
                    mse_loss = mse_loss * weight_mean

                if isinstance(predicted_clean, dict):
                    p_losses = [
                        self.perceptual_loss_fn(pred.float(), images[key].float())
                        for key, pred in predicted_clean.items()
                    ]
                    p_loss = sum(p_losses) / len(p_losses)
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

        # Track metrics using MetricsTracker
        mask = labels_dict.get('labels')
        with torch.no_grad():
            self.metrics.track_step(
                timesteps=timesteps,
                predicted_clean=predicted_clean,
                images=images,
                mask=mask,
                grad_norm=grad_norm,
                loss=total_loss.item(),
            )

        return total_loss.item(), mse_loss.item(), p_loss.item()

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

    def train(self, train_loader: DataLoader, train_dataset: Dataset) -> None:
        """Execute the main training loop."""
        total_start = time.time()

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

                    # Log metrics (grad norms every epoch, others at val_interval)
                    is_val_epoch = (epoch + 1) % self.val_interval == 0
                    self.metrics.log_epoch(epoch, log_all=is_val_epoch)

                    # Log worst batch at val_interval
                    if is_val_epoch and self.metrics.log_worst_batch:
                        worst_data = self.metrics.get_worst_batch_data()
                        if worst_data is not None:
                            self.visualizer.log_worst_batch(epoch, worst_data)

                    if is_val_epoch or (epoch + 1) == self.n_epochs:
                        model_to_use = self.ema.ema_model if self.ema is not None else self.model_raw
                        self.visualizer.generate_samples(model_to_use, train_dataset, epoch)

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
