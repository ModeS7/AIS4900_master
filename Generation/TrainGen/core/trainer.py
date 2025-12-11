"""
Diffusion model trainer module.

This module provides the DiffusionTrainer class for training diffusion models
with various strategies (DDPM, Rectified Flow) and modes (segmentation,
conditional single, conditional dual).
"""
# TODO: Add one more layer to the network (increase model capacity)
# TODO: Train RFlow on 100 steps (faster inference, compare quality vs 1000 steps)
# TODO: Fix dataset test segmentation masks
# TODO: Implement classifier-free guidance (train with mask dropout)
# TODO: Label distortion dataset (~500 images) for DRaFT training
# TODO: Train distortion classifier as DRaFT reward model
# TODO: Compare noise schedules (cosine/linear/sigmoid) systematically

import glob
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch._dynamo.config
import torch.distributed as dist
from ema_pytorch import EMA
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

from config import PathConfig
from .modes import ConditionalDualMode, ConditionalSingleMode, SegmentationMode
from .strategies import DDPMStrategy, RFlowStrategy
from .utils import get_vram_usage, log_epoch_summary, save_checkpoint, save_model_only

logger = logging.getLogger(__name__)

StrategyType = Literal['ddpm', 'rflow']
ModeType = Literal['seg', 'bravo', 'dual']
ComputeType = Literal['local', 'cluster']


class DiffusionTrainer:
    """Unified diffusion model trainer composing strategy and mode.

    This trainer supports multiple diffusion strategies (DDPM, Rectified Flow)
    and training modes (segmentation, conditional single, conditional dual).
    It handles distributed training, mixed precision, and checkpoint management.

    Args:
        strategy: Diffusion strategy to use. Options: 'ddpm', 'rflow'.
        mode: Training mode. Options: 'seg', 'bravo', 'dual'.
        n_epochs: Number of training epochs.
        batch_size: Training batch size.
        image_size: Input image resolution (assumes square images).
        learning_rate: Initial learning rate for AdamW optimizer.
        perceptual_weight: Weight for perceptual loss component.
        num_timesteps: Number of diffusion timesteps for training.
        warmup_epochs: Number of epochs for linear learning rate warmup.
        val_interval: Epochs between validation and checkpoint saves.
        compute: Compute environment for path configuration.
        use_multi_gpu: Whether to use distributed data parallel training.

    Example:
        >>> trainer = DiffusionTrainer(
        ...     strategy='ddpm',
        ...     mode='bravo',
        ...     n_epochs=500,
        ...     batch_size=16
        ... )
        >>> trainer.setup_model(train_dataset)
        >>> trainer.train(train_loader, train_dataset)
    """

    def __init__(
            self,
            strategy: StrategyType = 'ddpm',
            mode: ModeType = 'seg',
            n_epochs: int = 500,
            batch_size: int = 16,
            image_size: int = 128,
            learning_rate: float = 1e-4,
            perceptual_weight: float = 0.001,
            num_timesteps: int = 1000,
            warmup_epochs: int = 5,
            val_interval: int = 10,
            compute: ComputeType = 'local',
            use_multi_gpu: bool = False,
            use_ema: bool = True,
            ema_decay: float = 0.9999,
            use_min_snr: bool = True,
            min_snr_gamma: float = 5.0
    ) -> None:
        self.strategy_name: StrategyType = strategy
        self.mode_name: ModeType = mode
        self.n_epochs: int = n_epochs
        self.batch_size: int = batch_size
        self.image_size: int = image_size
        self.learning_rate: float = learning_rate
        self.perceptual_weight: float = perceptual_weight
        self.num_timesteps: int = num_timesteps
        self.warmup_epochs: int = warmup_epochs
        self.val_interval: int = val_interval
        self.compute: ComputeType = compute
        self.use_multi_gpu: bool = use_multi_gpu
        self.use_ema: bool = use_ema
        self.ema_decay: float = ema_decay
        self.use_min_snr: bool = use_min_snr
        self.min_snr_gamma: float = min_snr_gamma

        # Setup device and distributed training
        if use_multi_gpu:
            self.rank, self.local_rank, self.world_size, self.device = self._setup_distributed()
            self.is_main_process: bool = (self.rank == 0)
        else:
            self.device: torch.device = torch.device("cuda")
            self.is_main_process = True
            self.rank: int = 0
            self.world_size: int = 1

        # Initialize strategy and mode
        self.strategy = self._create_strategy(strategy)
        self.mode = self._create_mode(mode)
        self.scheduler = self.strategy.setup_scheduler(num_timesteps, image_size)

        # Setup paths using centralized configuration
        self.path_config: PathConfig = PathConfig(compute=compute)

        # Initialize logging and save directories
        # Structure: {model_dir}/{mode}/{strategy}_{size}_{timestamp}/
        #   - tensorboard/  (TensorBoard logs)
        #   - metadata.json, hparams.json
        #   - checkpoints (best.pt, latest.pt, epoch_*.pt)
        if self.is_main_process:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_name: str = f"{strategy}_{image_size}_{timestamp}"
            self.save_dir: str = str(self.path_config.model_dir / mode / self.run_name)
            tensorboard_dir = os.path.join(self.save_dir, "tensorboard")
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
        self.num_timestep_bins: int = 10
        self.timestep_losses: DefaultDict[int, List[float]] = defaultdict(list)

    def _create_strategy(self, strategy: StrategyType) -> Union[DDPMStrategy, RFlowStrategy]:
        """Create a diffusion strategy instance.

        Args:
            strategy: Strategy type identifier.

        Returns:
            Instantiated strategy object.

        Raises:
            ValueError: If strategy is not recognized.
        """
        strategies: Dict[str, type] = {
            'ddpm': DDPMStrategy,
            'rflow': RFlowStrategy
        }
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")
        return strategies[strategy]()

    def _create_mode(
            self, mode: ModeType
    ) -> Union[SegmentationMode, ConditionalSingleMode, ConditionalDualMode]:
        """Create a training mode instance.

        Args:
            mode: Mode type identifier.

        Returns:
            Instantiated mode object.

        Raises:
            ValueError: If mode is not recognized.
        """
        modes: Dict[str, type] = {
            'seg': SegmentationMode,
            'bravo': ConditionalSingleMode,
            'dual': ConditionalDualMode
        }
        if mode not in modes:
            raise ValueError(f"Unknown mode: {mode}. Choose from {list(modes.keys())}")

        return modes[mode]()

    def _setup_distributed(self) -> Tuple[int, int, int, torch.device]:
        """Setup distributed training with dynamic port allocation.

        Configures distributed training for both SLURM cluster environments
        and standard PyTorch distributed setups.

        Returns:
            Tuple containing (rank, local_rank, world_size, device).
        """
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
                # Map job_id to port range 12000-65000
                port = 12000 + (job_id % 53000)
                os.environ['MASTER_PORT'] = str(port)
                if rank == 0:
                    print(f"Using dynamic port: {port} (from SLURM_JOB_ID: {job_id})")
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
        """Initialize model, optimizer, and loss functions.

        Args:
            train_dataset: Training dataset used to determine model configuration.
        """
        model_input_channels = self.mode.get_model_config()

        raw_model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=model_input_channels['in_channels'],
            out_channels=model_input_channels['out_channels'],
            channels=(128, 256, 256),
            attention_levels=(False, True, True),
            num_res_blocks=1,
            num_head_channels=256
        ).to(self.device)

        if self.use_multi_gpu:
            # Store raw model for saving
            self.model_raw = raw_model

            # Disable DDP optimizer for dual 256
            if self.mode_name == 'dual' and self.image_size == 256:
                torch._dynamo.config.optimize_ddp = False
                if self.is_main_process:
                    print("Disabled DDPOptimizer for dual 256 (compilation workaround)")

            # Wrap in DDP
            ddp_model = DDP(
                raw_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=False
            )

            # Compile the DDP wrapper
            self.model = torch.compile(ddp_model, mode="reduce-overhead")

            if self.is_main_process:
                print("Multi-GPU: Compiled DDP wrapper with mode='reduce-overhead'")
        else:
            self.model_raw = raw_model
            self.model = torch.compile(raw_model, mode="default")

        # Setup perceptual loss
        perceptual_loss = PerceptualLoss(
            spatial_dims=2,
            network_type="radimagenet_resnet50",
            cache_dir=str(self.path_config.cache_dir),
            pretrained=True,
        ).to(self.device)

        # Compile perceptual loss
        self.perceptual_loss_fn = torch.compile(perceptual_loss, mode="reduce-overhead")

        # Setup optimizer - always use raw model parameters
        self.optimizer = AdamW(self.model_raw.parameters(), lr=self.learning_rate)

        # Warmup + Cosine scheduler: linear warmup then cosine annealing
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,  # Start at 10% of target LR
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

        # Create EMA wrapper if enabled (using ema-pytorch library)
        if self.use_ema:
            self.ema = EMA(
                self.model_raw,
                beta=self.ema_decay,        # 0.9999
                update_after_step=100,      # warmup: start after 100 steps
                update_every=10,            # update every 10 steps (save compute)
            )
            if self.is_main_process:
                print(f"EMA enabled with decay={self.ema_decay}, warmup=100 steps")

        # Save metadata
        if self.is_main_process:
            self._save_metadata()

    def _save_metadata(self) -> None:
        """Save training configuration to metadata.json."""
        os.makedirs(self.save_dir, exist_ok=True)

        metadata = {
            'run_name': self.run_name,
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
            'compute': self.compute,
            'multi_gpu': self.use_multi_gpu,
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay if self.use_ema else None,
            'use_min_snr': self.use_min_snr,
            'min_snr_gamma': self.min_snr_gamma if self.use_min_snr else None,
            'model': {
                'channels': [128, 256, 256],
                'attention_levels': [False, True, True],
                'num_res_blocks': 1,
                'num_head_channels': 256,
            },
            'created_at': datetime.now().isoformat(),
        }

        metadata_path = os.path.join(self.save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to: {metadata_path}")

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

    def _cleanup_old_checkpoints(self, keep_n: int = 2) -> None:
        """Keep only the N most recent epoch checkpoints.

        Args:
            keep_n: Number of recent epoch checkpoints to keep.
        """
        pattern = os.path.join(self.save_dir, "epoch_*.pt")
        checkpoints = sorted(glob.glob(pattern))

        # Remove all but the last N
        for old_ckpt in checkpoints[:-keep_n]:
            try:
                os.remove(old_ckpt)
            except OSError:
                pass  # Ignore errors if file already removed

    def _update_ema(self) -> None:
        """Update EMA model weights using ema-pytorch library.

        The library handles warmup (update_after_step) and update frequency
        (update_every) internally.
        """
        if self.ema is not None:
            self.ema.update()

    def _compute_snr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute Min-SNR loss weights for given timesteps.

        Min-SNR weighting prevents high-noise timesteps from dominating the loss
        by clamping the weight based on signal-to-noise ratio.

        Reference: Hang et al. "Efficient Diffusion Training via Min-SNR Weighting"

        Args:
            timesteps: Tensor of timestep indices.

        Returns:
            Tensor of weights for each timestep.
        """
        # For DDPM: SNR(t) = alpha_bar(t) / (1 - alpha_bar(t))
        # For RFlow: Use linear interpolation approximation
        if self.strategy_name == 'ddpm':
            alphas_cumprod = self.scheduler.alphas_cumprod.to(timesteps.device)
            alpha_bar = alphas_cumprod[timesteps]
            snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
        else:
            # RFlow: t ranges from 0 to 1, use t as proxy for noise level
            # At t=0: pure data (high SNR), at t=1: pure noise (low SNR)
            t_normalized = timesteps.float() / self.num_timesteps
            # Approximate SNR as (1-t)/t
            snr = (1.0 - t_normalized) / (t_normalized + 1e-8)

        # Min-SNR-gamma weighting: weight = min(SNR, gamma) / SNR
        # This clips the effective weight for low-SNR (high noise) timesteps
        snr_clipped = torch.clamp(snr, max=self.min_snr_gamma)
        weights = snr_clipped / (snr + 1e-8)

        return weights

    def find_optimal_lr(
            self,
            dataloader: DataLoader,
            min_lr: float = 1e-7,
            max_lr: float = 1e-1,
            num_steps: int = 100,
            smoothing: float = 0.05
    ) -> float:
        """Run learning rate finder and return optimal LR.

        Sweeps through learning rates exponentially and finds the point of
        steepest descent in the loss curve, which indicates a good learning rate.

        Args:
            dataloader: DataLoader providing training batches.
            min_lr: Minimum learning rate to test.
            max_lr: Maximum learning rate to test.
            num_steps: Number of LR steps to test.
            smoothing: Exponential smoothing factor for loss.

        Returns:
            Suggested optimal learning rate.
        """
        if not self.is_main_process:
            return self.learning_rate

        print(f"\n{'=' * 50}")
        print("Running Learning Rate Finder")
        print(f"Range: {min_lr:.2e} - {max_lr:.2e}, Steps: {num_steps}")
        print(f"{'=' * 50}\n")

        # Use raw model (uncompiled) for LR finding - faster
        model = self.model_raw
        model.train()

        # Temporary optimizer for LR sweep
        temp_optimizer = AdamW(model.parameters(), lr=min_lr)
        lr_mult = (max_lr / min_lr) ** (1.0 / num_steps)

        learning_rates: List[float] = []
        losses: List[float] = []
        smoothed_loss: Optional[float] = None
        best_loss = float('inf')

        # Save model state to restore after
        model_state = {k: v.clone() for k, v in model.state_dict().items()}

        data_iter = iter(dataloader)
        pbar = tqdm(total=num_steps, desc="LR Finder", ncols=100)

        for step in range(num_steps):
            # Get next batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Forward pass (without compilation for speed)
            temp_optimizer.zero_grad(set_to_none=True)

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

                # Use raw model directly
                prediction = model(model_input, timesteps=timesteps)
                mse_loss, predicted_clean = self.strategy.compute_loss(
                    prediction, images, noise, noisy_images, timesteps
                )

                if isinstance(predicted_clean, dict):
                    p_loss = sum(
                        self.perceptual_loss_fn(pred.float(), images[key].float())
                        for key, pred in predicted_clean.items()
                    )
                else:
                    p_loss = self.perceptual_loss_fn(predicted_clean.float(), images.float())

                total_loss = mse_loss + self.perceptual_weight * p_loss

            total_loss.backward()
            temp_optimizer.step()

            loss_val = total_loss.item()

            # Smooth the loss
            if smoothed_loss is None:
                smoothed_loss = loss_val
            else:
                smoothed_loss = smoothing * loss_val + (1 - smoothing) * smoothed_loss

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Stop if loss explodes
            if step > 10 and smoothed_loss > 4 * best_loss:
                print(f"\nStopping: loss exploded at LR={temp_optimizer.param_groups[0]['lr']:.2e}")
                break

            current_lr = temp_optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            losses.append(smoothed_loss)

            # Update LR
            for param_group in temp_optimizer.param_groups:
                param_group['lr'] *= lr_mult

            pbar.update(1)
            pbar.set_postfix(lr=f"{current_lr:.2e}", loss=f"{smoothed_loss:.4f}")

        pbar.close()

        # Restore model state
        model.load_state_dict(model_state)

        # Find optimal LR (steepest descent)
        if len(losses) > 10:
            gradients = np.gradient(losses)
            min_grad_idx = np.argmin(gradients)
            suggested_lr = learning_rates[min_grad_idx]
            min_loss_idx = np.argmin(losses)
            min_loss_lr = learning_rates[min_loss_idx]
        else:
            suggested_lr = self.learning_rate
            min_loss_lr = self.learning_rate

        # Save plot
        plot_path = self.path_config.base_prefix / 'AIS4005_IP' / 'misc' / f'lr_finder_{self.mode_name}_{self.run_name}.png'
        self._plot_lr_finder(learning_rates, losses, str(plot_path), suggested_lr)

        print(f"\nLR Finder Results:")
        print(f"  Suggested LR (steepest descent): {suggested_lr:.2e}")
        print(f"  LR at minimum loss: {min_loss_lr:.2e}")
        print(f"  Plot saved to: {plot_path}")

        return suggested_lr

    def _plot_lr_finder(
            self,
            learning_rates: List[float],
            losses: List[float],
            save_path: str,
            suggested_lr: float
    ) -> None:
        """Plot learning rate finder results.

        Args:
            learning_rates: List of learning rates tested.
            losses: List of corresponding losses.
            save_path: Path to save the plot.
            suggested_lr: Suggested LR to mark on plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(learning_rates, losses, 'b-', linewidth=2)
        plt.xscale('log')

        plt.axvline(x=suggested_lr, color='r', linestyle='--', linewidth=2,
                    label=f'Suggested LR: {suggested_lr:.2e}')

        if len(losses) > 0:
            min_loss_idx = np.argmin(losses)
            plt.scatter([learning_rates[min_loss_idx]], [losses[min_loss_idx]],
                        color='green', s=100, zorder=5,
                        label=f'Min loss at LR={learning_rates[min_loss_idx]:.2e}')

        plt.xlabel('Learning Rate', fontsize=12)
        plt.ylabel('Loss (smoothed)', fontsize=12)
        plt.title('Learning Rate Finder', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def update_learning_rate(self, new_lr: float) -> None:
        """Update learning rate and recreate optimizer/scheduler.

        Args:
            new_lr: New learning rate to use.
        """
        self.learning_rate = new_lr

        # Recreate optimizer with new LR
        self.optimizer = AdamW(self.model_raw.parameters(), lr=new_lr)

        # Recreate scheduler
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

        if self.is_main_process:
            print(f"Learning rate updated to: {new_lr:.2e}")

    def train_step(self, batch: torch.Tensor) -> Tuple[float, float, float]:
        """Execute a single training step.

        Performs forward pass, loss computation, and backward pass for one batch.

        Args:
            batch: Input batch tensor from the data loader.

        Returns:
            Tuple of (total_loss, mse_loss, perceptual_loss) as floats.
        """
        self.optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=torch.bfloat16):
            # Mode prepares the batch
            prepared = self.mode.prepare_batch(batch, self.device)
            images = prepared['images']  # Can be tensor or dict
            labels_dict = {'labels': prepared.get('labels')}

            # Generate noise (same structure as images)
            if isinstance(images, dict):
                noise = {key: torch.randn_like(img).to(self.device) for key, img in images.items()}
            else:
                noise = torch.randn_like(images).to(self.device)

            # Sample timesteps
            timesteps = self.strategy.sample_timesteps(images)

            # Add noise to images
            noisy_images = self.strategy.add_noise(images, noise, timesteps)

            # Format model input
            model_input = self.mode.format_model_input(noisy_images, labels_dict)

            # Model prediction
            prediction = self.strategy.predict_noise_or_velocity(self.model, model_input, timesteps)

            # Compute loss
            mse_loss, predicted_clean = self.strategy.compute_loss(prediction, images, noise, noisy_images, timesteps)

            # Apply Min-SNR weighting if enabled
            if self.use_min_snr:
                snr_weights = self._compute_snr_weights(timesteps)
                weight_mean = snr_weights.mean()
                mse_loss = mse_loss * weight_mean
                # Debug: print once at start
                if not hasattr(self, '_min_snr_debug_done'):
                    print(f"[DEBUG] Min-SNR enabled, weight mean: {weight_mean.item():.4f}")
                    self._min_snr_debug_done = True

            # Compute perceptual loss
            if isinstance(predicted_clean, dict):
                # Dual-image: average perceptual loss
                p_loss = sum(
                    self.perceptual_loss_fn(pred.float(), images[key].float())
                    for key, pred in predicted_clean.items()
                )
            else:
                # Single-image
                p_loss = self.perceptual_loss_fn(predicted_clean.float(), images.float())

            # Combined loss
            total_loss = mse_loss + self.perceptual_weight * p_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_raw.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update EMA model if enabled
        if self.use_ema:
            self._update_ema()

        # Track loss by timestep bin for analysis
        self._track_timestep_loss(timesteps, mse_loss.item())

        return total_loss.item(), mse_loss.item(), p_loss.item()

    def _track_timestep_loss(self, timesteps: torch.Tensor, mse_loss: float) -> None:
        """Track MSE loss by timestep bin for analysis.

        Args:
            timesteps: Tensor of sampled timesteps for the batch.
            mse_loss: MSE loss value for the batch.
        """
        # Compute bin for each timestep and aggregate
        bin_size = self.num_timesteps // self.num_timestep_bins
        timesteps_cpu = timesteps.cpu().numpy()

        for t in timesteps_cpu:
            bin_idx = min(int(t) // bin_size, self.num_timestep_bins - 1)
            self.timestep_losses[bin_idx].append(mse_loss)

    def _log_timestep_losses(self, epoch: int) -> None:
        """Save timestep loss distribution to JSON file.

        Args:
            epoch: Current epoch number for logging.
        """
        if not self.timestep_losses:
            return

        bin_size = self.num_timesteps // self.num_timestep_bins

        # Build dict of bin labels to average losses
        epoch_data = {}
        for bin_idx in range(self.num_timestep_bins):
            losses = self.timestep_losses.get(bin_idx, [])
            bin_start = bin_idx * bin_size
            bin_end = (bin_idx + 1) * bin_size - 1
            bin_label = f"{bin_start:04d}-{bin_end:04d}"
            epoch_data[bin_label] = float(np.mean(losses)) if losses else 0.0

        # Append to timestep_losses.json
        filepath = os.path.join(self.save_dir, 'timestep_losses.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                all_data = json.load(f)
        else:
            all_data = {}

        all_data[str(epoch)] = epoch_data

        with open(filepath, 'w') as f:
            json.dump(all_data, f, indent=2)

        # Clear tracked losses for next interval
        self.timestep_losses.clear()

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Tuple[float, float, float]:
        """Train the model for one epoch.

        Args:
            data_loader: DataLoader providing training batches.
            epoch: Current epoch number (for logging).

        Returns:
            Tuple of average (total_loss, mse_loss, perceptual_loss) for the epoch.
        """
        self.model.train()
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_perceptual_loss = 0

        use_progress_bars = (self.compute != "cluster") and self.is_main_process

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
                print(f"{get_vram_usage(self.device)}")

        return epoch_loss / len(data_loader), epoch_mse_loss / len(data_loader), epoch_perceptual_loss / len(
            data_loader)

    def _sample_positive_masks(
            self, train_dataset: Dataset, num_samples: int, seg_channel_idx: int
    ) -> torch.Tensor:
        """Sample slices with positive segmentation masks from dataset.

        Args:
            train_dataset: Dataset to sample from.
            num_samples: Number of positive masks to sample.
            seg_channel_idx: Channel index of segmentation mask in data.

        Returns:
            Tensor of segmentation masks with shape [num_samples, 1, H, W].

        Raises:
            ValueError: If no positive masks found in dataset.
        """
        seg_masks = []
        attempts = 0
        max_attempts = len(train_dataset)

        while len(seg_masks) < num_samples and attempts < max_attempts:
            idx = torch.randint(0, len(train_dataset), (1,)).item()
            data = train_dataset[idx]
            tensor = torch.from_numpy(data).float() if hasattr(data, '__array__') else torch.tensor(data).float()
            seg = tensor[seg_channel_idx:seg_channel_idx + 1, :, :]

            if seg.sum() > 0:
                seg_masks.append(seg)
            attempts += 1

        if len(seg_masks) < num_samples:
            print(f"Warning: Only found {len(seg_masks)} positive masks for validation")

        if len(seg_masks) == 0:
            raise ValueError("No positive segmentation masks found for validation")

        return torch.stack(seg_masks).to(self.device)

    def generate_validation_samples(
            self, epoch: int, train_dataset: Dataset, num_samples: int = 4
    ) -> None:
        """Generate and log validation samples to TensorBoard.

        Creates synthetic samples using the current model state and logs them
        to TensorBoard for visual inspection of training progress.

        Args:
            epoch: Current epoch number for logging.
            train_dataset: Training dataset to sample conditioning data from.
            num_samples: Number of samples to generate.
        """
        if not self.is_main_process or self.writer is None:
            return

        # Use EMA model for generation if available, otherwise raw model
        model_to_use = self.ema.ema_model if self.ema is not None else self.model_raw
        model_to_use.eval()

        try:
            with torch.no_grad():
                model_config = self.mode.get_model_config()
                out_channels = model_config['out_channels']

                # Prepare noise and conditioning based on mode
                if self.mode_name == 'seg':
                    # Unconditional: just noise [B, 1, H, W]
                    noise = torch.randn((num_samples, 1, self.image_size, self.image_size), device=self.device)
                    model_input = noise

                elif self.mode_name == 'bravo':
                    # Conditional single: [noise, seg_mask] = [B, 2, H, W]
                    seg_masks = self._sample_positive_masks(train_dataset, num_samples, seg_channel_idx=1)
                    noise = torch.randn_like(seg_masks, device=self.device)
                    model_input = torch.cat([noise, seg_masks], dim=1)  # [B, 2, H, W]

                elif self.mode_name == 'dual':
                    # Conditional dual: [noise_pre, noise_gd, seg_mask] = [B, 3, H, W]
                    seg_masks = self._sample_positive_masks(train_dataset, num_samples, seg_channel_idx=2)
                    noise_pre = torch.randn_like(seg_masks, device=self.device)
                    noise_gd = torch.randn_like(seg_masks, device=self.device)
                    model_input = torch.cat([noise_pre, noise_gd, seg_masks], dim=1)  # [B, 3, H, W]

                else:
                    raise ValueError(f"Unknown mode: {self.mode_name}")

                # Generate samples
                with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    samples = self.strategy.generate(model_to_use, model_input, num_steps=self.num_timesteps, device=self.device)

                # Log based on output channels
                if out_channels == 2:
                    # Dual-image mode: log both channels separately
                    samples_pre = samples[:, 0:1, :, :].float()
                    samples_gd = samples[:, 1:2, :, :].float()

                    # Normalize and convert to RGB for TensorBoard
                    samples_pre_norm = torch.clamp(samples_pre, 0, 1)
                    samples_gd_norm = torch.clamp(samples_gd, 0, 1)

                    # Repeat channels: [B, 1, H, W] -> [B, 3, H, W]
                    samples_pre_rgb = samples_pre_norm.repeat(1, 3, 1, 1)
                    samples_gd_rgb = samples_gd_norm.repeat(1, 3, 1, 1)

                    self.writer.add_images('Generated_T1_Pre', samples_pre_rgb, epoch)
                    self.writer.add_images('Generated_T1_Gd', samples_gd_rgb, epoch)
                else:
                    # Single-image mode: log samples
                    samples_float = samples.float()
                    samples_normalized = torch.clamp(samples_float, 0, 1)

                    # Ensure correct shape [B, 1, H, W]
                    if samples_normalized.dim() == 3:
                        samples_normalized = samples_normalized.unsqueeze(1)

                    # Convert to RGB: [B, 1, H, W] -> [B, 3, H, W]
                    samples_rgb = samples_normalized.repeat(1, 3, 1, 1)

                    # Verify shape before logging
                    assert samples_rgb.dim() == 4, f"Expected 4D tensor, got {samples_rgb.dim()}D"
                    assert samples_rgb.shape[1] == 3, f"Expected 3 channels, got {samples_rgb.shape[1]}"

                    self.writer.add_images('Generated_Images', samples_rgb, epoch)

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
        """Execute the main training loop.

        Runs the complete training process including epoch iteration,
        validation, and checkpointing.

        Args:
            train_loader: DataLoader providing training batches.
            train_dataset: Training dataset for validation sample generation.
        """
        total_start = time.time()

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

                    # Log timestep-wise loss analysis at validation intervals
                    if (epoch + 1) % self.val_interval == 0:
                        self._log_timestep_losses(epoch)

                if (epoch + 1) % self.val_interval == 0 or (epoch + 1) == self.n_epochs:
                    self.generate_validation_samples(epoch, train_dataset)

                    # Save epoch checkpoint (model only - lightweight)
                    filename = f"epoch_{epoch:04d}"
                    save_model_only(self.model_raw, epoch, self.save_dir, filename, self.ema)

                    # Cleanup old epoch checkpoints (keep only 2 most recent)
                    self._cleanup_old_checkpoints(keep_n=2)

                    # Save latest (full state for resuming training)
                    save_checkpoint(
                        self.model_raw, self.optimizer, self.lr_scheduler,
                        epoch, self.save_dir, "latest", self.ema
                    )

                    # Save best model if this is the lowest loss (full state)
                    if avg_loss < self.best_loss:
                        self.best_loss = avg_loss
                        save_checkpoint(
                            self.model_raw, self.optimizer, self.lr_scheduler,
                            epoch, self.save_dir, "best", self.ema
                        )
                        print(f"New best model saved (loss: {avg_loss:.6f})")

        if self.is_main_process:
            total_time = time.time() - total_start
            print(f"\nTraining completed! Total time: {total_time:.1f}s ({total_time / 3600:.1f}h)")

            # Update metadata with final results
            self._update_metadata_final(avg_loss, avg_mse, total_time)

            # Log hyperparameters with final metrics
            if self.writer is not None:
                self._log_hparams(avg_loss, avg_mse)
                self.writer.close()

        if self.use_multi_gpu:
            dist.destroy_process_group()

    def _log_hparams(self, final_loss: float, final_mse: float) -> None:
        """Save hyperparameters and final metrics to JSON.

        Args:
            final_loss: Final total loss value.
            final_mse: Final MSE loss value.
        """
        hparams = {
            'strategy': self.strategy_name,
            'mode': self.mode_name,
            'epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'image_size': self.image_size,
            'learning_rate': self.learning_rate,
            'perceptual_weight': self.perceptual_weight,
            'num_timesteps': self.num_timesteps,
            'warmup_epochs': self.warmup_epochs,
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay if self.use_ema else None,
            'use_min_snr': self.use_min_snr,
            'min_snr_gamma': self.min_snr_gamma if self.use_min_snr else None,
        }

        metrics = {
            'final_loss': float(final_loss),
            'final_mse': float(final_mse),
            'best_loss': float(self.best_loss),
        }

        hparams_path = os.path.join(self.save_dir, 'hparams.json')
        with open(hparams_path, 'w') as f:
            json.dump({'hyperparameters': hparams, 'metrics': metrics}, f, indent=2)