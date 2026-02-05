"""Generation context for decoupling visualization from trainer.

This module provides the GenerationContext dataclass which bundles
scheduler and space properties needed by visualization functions,
avoiding tight coupling to the full DiffusionTrainer object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from medgen.pipeline.trainer import DiffusionTrainer


@dataclass(frozen=True)
class GenerationContext:
    """Immutable context for sample generation.

    Bundles scheduler and space properties needed by visualization functions,
    avoiding tight coupling to the full DiffusionTrainer object.

    Usage:
        ctx = GenerationContext.from_trainer(trainer)
        samples = generate_trajectory(
            model, noise, ctx.num_train_timesteps,
            is_conditional=ctx.is_conditional, latent_channels=ctx.latent_channels
        )

    Attributes:
        num_train_timesteps: Number of training timesteps from scheduler.
        device: Device for tensor operations.
        spatial_dims: Number of spatial dimensions (2 or 3).
        latent_channels: Number of latent channels from space.
        scale_factor: Compression scale factor from space (1 = pixel space).
        is_conditional: Whether the mode requires conditioning.
    """

    num_train_timesteps: int
    device: torch.device
    spatial_dims: int = 2
    latent_channels: int = 1
    scale_factor: int = 1
    is_conditional: bool = False

    @classmethod
    def from_trainer(cls, trainer: 'DiffusionTrainer') -> GenerationContext:
        """Extract generation context from trainer.

        Args:
            trainer: The DiffusionTrainer instance.

        Returns:
            GenerationContext with properties extracted from trainer.
        """
        return cls(
            num_train_timesteps=trainer.scheduler.num_train_timesteps,
            device=trainer.device,
            spatial_dims=trainer.spatial_dims,
            latent_channels=trainer.space.latent_channels,
            scale_factor=trainer.space.scale_factor,
            is_conditional=trainer.mode.is_conditional,
        )

    @classmethod
    def from_scheduler(
        cls,
        scheduler,
        device: torch.device,
        spatial_dims: int = 2,
        latent_channels: int = 1,
        scale_factor: int = 1,
        is_conditional: bool = False,
    ) -> GenerationContext:
        """Create from scheduler directly (useful for testing).

        Args:
            scheduler: Scheduler object with num_train_timesteps attribute.
            device: Device for tensor operations.
            spatial_dims: Number of spatial dimensions (2 or 3).
            latent_channels: Number of latent channels.
            scale_factor: Compression scale factor (1 = pixel space).
            is_conditional: Whether the mode requires conditioning.

        Returns:
            GenerationContext with specified properties.
        """
        return cls(
            num_train_timesteps=scheduler.num_train_timesteps,
            device=device,
            spatial_dims=spatial_dims,
            latent_channels=latent_channels,
            scale_factor=scale_factor,
            is_conditional=is_conditional,
        )
