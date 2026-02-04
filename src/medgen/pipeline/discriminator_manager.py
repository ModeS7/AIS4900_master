"""Discriminator management for GAN training.

This module provides:
- DiscriminatorManager: Manages discriminator for GAN training in compression models

Handles discriminator creation, optimizer setup, training steps, and model wrapping.
"""
import logging
from typing import Any

import torch
from monai.losses import PatchAdversarialLoss
from monai.networks.nets import PatchDiscriminator
from torch import nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler

from medgen.core import create_warmup_cosine_scheduler, wrap_model_for_training

logger = logging.getLogger(__name__)


class DiscriminatorManager:
    """Manages discriminator for GAN training in compression models.

    Handles:
    - Discriminator model creation (PatchDiscriminator)
    - Adversarial loss function creation (PatchAdversarialLoss)
    - Optimizer and scheduler setup
    - Model wrapping (DDP, torch.compile)
    - Training step execution
    - State dict for checkpointing
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_layers: int,
        num_channels: int,
        learning_rate: float,
        optimizer_betas: tuple[float, float],
        warmup_epochs: int,
        total_epochs: int,
        device: torch.device,
        enabled: bool = True,
        gradient_clip_norm: float = 1.0,
        is_main_process: bool = True,
    ) -> None:
        """Initialize discriminator manager.

        Args:
            spatial_dims: Spatial dimensions (2 for 2D, 3 for 3D).
            in_channels: Number of input channels.
            num_layers: Number of discriminator layers.
            num_channels: Number of discriminator channels.
            learning_rate: Learning rate for discriminator optimizer.
            optimizer_betas: Betas for Adam optimizer.
            warmup_epochs: Number of warmup epochs for scheduler.
            total_epochs: Total number of training epochs.
            device: Device to place models on.
            enabled: Whether GAN training is enabled.
            gradient_clip_norm: Max gradient norm for clipping.
            is_main_process: Whether this is the main process (for logging).
        """
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.optimizer_betas = optimizer_betas
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.device = device
        self.enabled = enabled
        self.gradient_clip_norm = gradient_clip_norm
        self.is_main_process = is_main_process

        # State
        self.discriminator: nn.Module | None = None
        self.discriminator_raw: nn.Module | None = None
        self.optimizer: AdamW | None = None
        self.lr_scheduler: LRScheduler | None = None
        self.loss_fn: PatchAdversarialLoss | None = None

    def create(self) -> nn.Module | None:
        """Create discriminator model.

        Returns:
            PatchDiscriminator model, or None if GAN is disabled.
        """
        if not self.enabled:
            return None

        self.discriminator_raw = PatchDiscriminator(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            channels=self.num_channels,
            num_layers_d=self.num_layers,
        ).to(self.device)

        self.discriminator = self.discriminator_raw
        return self.discriminator_raw

    def create_loss_fn(self) -> PatchAdversarialLoss | None:
        """Create adversarial loss function.

        Returns:
            PatchAdversarialLoss with least_squares criterion, or None if disabled.
        """
        if not self.enabled:
            return None

        self.loss_fn = PatchAdversarialLoss(criterion="least_squares")
        return self.loss_fn

    def setup_optimizer(self, use_constant_lr: bool = False) -> None:
        """Setup discriminator optimizer and scheduler.

        Args:
            use_constant_lr: If True, skip scheduler creation.
        """
        if not self.enabled or self.discriminator_raw is None:
            return

        self.optimizer = AdamW(
            self.discriminator_raw.parameters(),
            lr=self.learning_rate,
            betas=self.optimizer_betas,
        )

        if not use_constant_lr:
            self.lr_scheduler = create_warmup_cosine_scheduler(
                self.optimizer,
                warmup_epochs=self.warmup_epochs,
                total_epochs=self.total_epochs,
            )

        if self.is_main_process:
            logger.info(f"Discriminator optimizer: lr={self.learning_rate}, betas={self.optimizer_betas}")

    def wrap_model(
        self,
        use_multi_gpu: bool,
        local_rank: int,
        use_compile: bool,
        compile_mode: str,
        weight_dtype: torch.dtype = torch.float32,
        pure_weights: bool = False,
    ) -> None:
        """Wrap discriminator for DDP/compile.

        Args:
            use_multi_gpu: Whether to use DDP.
            local_rank: Local rank for DDP.
            use_compile: Whether to apply torch.compile.
            compile_mode: Compile mode (e.g., "default").
            weight_dtype: Weight dtype for mixed precision.
            pure_weights: Whether to convert weights to target dtype.
        """
        if not self.enabled or self.discriminator_raw is None:
            return

        # Convert weights to target dtype if pure_weights is enabled
        if pure_weights and weight_dtype != torch.float32:
            self.discriminator_raw = self.discriminator_raw.to(weight_dtype)
            if self.is_main_process:
                logger.info(f"Converted discriminator weights to {weight_dtype}")

        self.discriminator, self.discriminator_raw = wrap_model_for_training(
            self.discriminator_raw,
            use_multi_gpu=use_multi_gpu,
            local_rank=local_rank,
            use_compile=use_compile,
            compile_mode=compile_mode,
            is_main_process=False,  # Suppress duplicate logging
        )

    def train_step(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        weight_dtype: torch.dtype,
        grad_norm_tracker: Any | None = None,
        log_grad_norm: bool = True,
    ) -> float:
        """Train discriminator for one step.

        Args:
            real: Real images [B, C, ...].
            fake: Generated/fake images [B, C, ...].
            weight_dtype: Dtype for autocast.
            grad_norm_tracker: Optional gradient norm tracker.
            log_grad_norm: Whether to track gradient norms.

        Returns:
            Discriminator loss value.
        """
        if not self.enabled or self.discriminator is None or self.loss_fn is None:
            return 0.0

        self.optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=True, dtype=weight_dtype):
            # Real images -> discriminator should output 1
            logits_real = self.discriminator(real.contiguous())
            # Fake images -> discriminator should output 0
            # Detach to prevent gradient flow through generator (saves memory)
            logits_fake = self.discriminator(fake.detach().contiguous())

            d_loss = 0.5 * (
                self.loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                + self.loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
            )

        d_loss.backward()

        # Gradient clipping and tracking
        grad_norm_d = 0.0
        if self.gradient_clip_norm > 0:
            grad_norm_d = torch.nn.utils.clip_grad_norm_(
                self.discriminator_raw.parameters(), max_norm=self.gradient_clip_norm
            ).item()

        self.optimizer.step()

        # Track discriminator gradient norm
        if log_grad_norm and grad_norm_tracker is not None:
            grad_norm_tracker.update(grad_norm_d)

        return d_loss.item()

    def compute_generator_loss(
        self,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """Compute adversarial loss for generator.

        Args:
            fake: Generated images [B, C, ...].

        Returns:
            Adversarial loss for generator (wants discriminator to output 1).
        """
        if not self.enabled or self.discriminator is None or self.loss_fn is None:
            return torch.tensor(0.0, device=self.device)

        logits_fake = self.discriminator(fake.contiguous())
        return self.loss_fn(logits_fake, target_is_real=True, for_discriminator=False)

    def on_epoch_end(self) -> None:
        """Step scheduler at epoch end."""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        """Get state for checkpointing.

        Returns:
            State dict containing discriminator, optimizer, and scheduler states.
        """
        state = {}

        if self.discriminator_raw is not None:
            state['discriminator_state_dict'] = self.discriminator_raw.state_dict()

        if self.optimizer is not None:
            state['optimizer_d_state_dict'] = self.optimizer.state_dict()

        if self.lr_scheduler is not None:
            state['scheduler_d_state_dict'] = self.lr_scheduler.state_dict()

        state['disc_config'] = {
            'in_channels': self.in_channels,
            'channels': self.num_channels,
            'num_layers_d': self.num_layers,
        }

        return state

    def load_state_dict(
        self,
        state: dict[str, Any],
        load_optimizer: bool = True,
    ) -> None:
        """Load state from checkpoint.

        Args:
            state: State dict from checkpoint.
            load_optimizer: Whether to load optimizer/scheduler states.
        """
        if 'discriminator_state_dict' in state and self.discriminator_raw is not None:
            self.discriminator_raw.load_state_dict(state['discriminator_state_dict'])
            if self.is_main_process:
                logger.info("Loaded discriminator weights")

        if load_optimizer:
            if 'optimizer_d_state_dict' in state and self.optimizer is not None:
                self.optimizer.load_state_dict(state['optimizer_d_state_dict'])

            if 'scheduler_d_state_dict' in state and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(state['scheduler_d_state_dict'])
