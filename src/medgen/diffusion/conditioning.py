"""Conditioning context for diffusion generation.

Bundles all conditioning signals into a single dataclass to simplify
function signatures and make it easier to add new conditioning types.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .batch_data import BatchData


@dataclass(frozen=True)
class ConditioningContext:
    """Bundles all conditioning signals for diffusion generation.

    Consolidates 7+ parameters into one immutable object, making function
    signatures cleaner and conditioning easier to extend.

    Attributes:
        omega: ScoreAug parameters [B, 5]. None if no ScoreAug.
        mode_id: Mode ID for multi-modality [B].
        size_bins: Size bin counts [B, num_bins] for FiLM conditioning.
        bin_maps: Spatial bin maps [B, num_bins, ...] for input conditioning.
        image_conditioning: Image tensor (seg mask) for concat conditioning.
        cfg_scale: CFG scale (1.0 = no guidance).
        cfg_scale_end: End scale for dynamic CFG. None = constant.
        latent_channels: Noise channels (1=pixel, 4=latent).

    Example:
        # Create from individual parameters
        ctx = ConditioningContext(
            size_bins=torch.randn(4, 7),
            cfg_scale=2.0,
        )

        # Create empty context for unconditional generation
        ctx = ConditioningContext.empty()

        # Create from BatchData
        ctx = ConditioningContext.from_batch(batch_data, cfg_scale=2.0)

        # Use in generation
        samples = strategy.generate(model, noise, 25, device, conditioning=ctx)
    """

    # Embedding conditioning
    omega: Tensor | None = None
    mode_id: Tensor | None = None
    size_bins: Tensor | None = None

    # Spatial conditioning
    bin_maps: Tensor | None = None
    image_conditioning: Tensor | None = None

    # CFG settings
    cfg_scale: float = 1.0
    cfg_scale_end: float | None = None

    # Space settings
    latent_channels: int = 1

    # --- Factory Methods ---

    @classmethod
    def empty(cls) -> ConditioningContext:
        """Create empty context (unconditional generation)."""
        return cls()

    @classmethod
    def from_batch(
        cls,
        batch_data: BatchData,
        cfg_scale: float = 1.0,
        cfg_scale_end: float | None = None,
        latent_channels: int = 1,
        image_conditioning: Tensor | None = None,
        omega: Tensor | None = None,
    ) -> ConditioningContext:
        """Create context from BatchData.

        Args:
            batch_data: BatchData instance with conditioning tensors.
            cfg_scale: CFG scale (1.0 = no guidance).
            cfg_scale_end: End scale for dynamic CFG.
            latent_channels: Noise channels (1=pixel, 4=latent).
            image_conditioning: Override image conditioning (default: batch_data.labels).
            omega: ScoreAug omega parameters.

        Returns:
            ConditioningContext with fields populated from batch_data.
        """
        return cls(
            omega=omega,
            size_bins=batch_data.size_bins,
            bin_maps=batch_data.bin_maps,
            mode_id=batch_data.mode_id,
            image_conditioning=image_conditioning
            if image_conditioning is not None
            else batch_data.labels,
            cfg_scale=cfg_scale,
            cfg_scale_end=cfg_scale_end,
            latent_channels=latent_channels,
        )

    # --- Properties ---

    @property
    def use_cfg(self) -> bool:
        """Whether CFG is enabled (scale > 1.0)."""
        return self.cfg_scale > 1.0

    @property
    def use_dynamic_cfg(self) -> bool:
        """Whether dynamic CFG is enabled."""
        return self.cfg_scale_end is not None and self.cfg_scale_end != self.cfg_scale

    @property
    def has_embedding_conditioning(self) -> bool:
        """Whether any embedding conditioning is set."""
        return any(
            [
                self.omega is not None,
                self.mode_id is not None,
                self.size_bins is not None,
            ]
        )

    @property
    def has_spatial_conditioning(self) -> bool:
        """Whether spatial conditioning is set."""
        return self.bin_maps is not None or self.image_conditioning is not None

    # --- CFG Helpers ---

    def get_cfg_scale_at_step(self, step_idx: int, total_steps: int) -> float:
        """Get interpolated CFG scale for dynamic CFG.

        Linearly interpolates from cfg_scale (at step 0) to cfg_scale_end
        (at final step).

        Args:
            step_idx: Current step index (0 to total_steps-1).
            total_steps: Total number of sampling steps.

        Returns:
            CFG scale for this step. If dynamic CFG is disabled, returns
            constant cfg_scale.
        """
        if not self.use_dynamic_cfg:
            return self.cfg_scale
        assert self.cfg_scale_end is not None  # guaranteed by use_dynamic_cfg check
        progress = step_idx / max(total_steps - 1, 1)
        return self.cfg_scale + progress * (self.cfg_scale_end - self.cfg_scale)

    def get_uncond_tensors(self) -> dict[str, Tensor | None]:
        """Get zeros tensors for CFG unconditional branch.

        Creates zero-filled tensors matching the shapes of conditioning
        tensors, used for the unconditional forward pass in CFG.

        Returns:
            Dict with keys 'size_bins', 'bin_maps', 'image_conditioning'
            mapped to zero tensors (or None if original was None).
        """
        uncond: dict[str, Tensor | None] = {}
        if self.size_bins is not None:
            uncond['size_bins'] = torch.zeros_like(self.size_bins)
        if self.bin_maps is not None:
            uncond['bin_maps'] = torch.zeros_like(self.bin_maps)
        if self.image_conditioning is not None:
            uncond['image_conditioning'] = torch.zeros_like(self.image_conditioning)
        return uncond

    def to_device(self, device: torch.device) -> ConditioningContext:
        """Move all tensors to device. Returns new instance.

        Args:
            device: Target device.

        Returns:
            New ConditioningContext with tensors on the specified device.
        """

        def _move(t: Tensor | None) -> Tensor | None:
            return t.to(device) if t is not None else None

        return ConditioningContext(
            omega=_move(self.omega),
            mode_id=_move(self.mode_id),
            size_bins=_move(self.size_bins),
            bin_maps=_move(self.bin_maps),
            image_conditioning=_move(self.image_conditioning),
            cfg_scale=self.cfg_scale,
            cfg_scale_end=self.cfg_scale_end,
            latent_channels=self.latent_channels,
        )

    def with_cfg(
        self, cfg_scale: float, cfg_scale_end: float | None = None
    ) -> ConditioningContext:
        """Create new context with different CFG settings.

        Convenience method for adjusting CFG scale without rebuilding
        the entire context.

        Args:
            cfg_scale: New CFG scale.
            cfg_scale_end: New CFG end scale (None for constant).

        Returns:
            New ConditioningContext with updated CFG settings.
        """
        return ConditioningContext(
            omega=self.omega,
            mode_id=self.mode_id,
            size_bins=self.size_bins,
            bin_maps=self.bin_maps,
            image_conditioning=self.image_conditioning,
            cfg_scale=cfg_scale,
            cfg_scale_end=cfg_scale_end,
            latent_channels=self.latent_channels,
        )

    def with_latent_channels(self, latent_channels: int) -> ConditioningContext:
        """Create new context with different latent channel count.

        Args:
            latent_channels: New latent channel count.

        Returns:
            New ConditioningContext with updated latent channels.
        """
        return ConditioningContext(
            omega=self.omega,
            mode_id=self.mode_id,
            size_bins=self.size_bins,
            bin_maps=self.bin_maps,
            image_conditioning=self.image_conditioning,
            cfg_scale=self.cfg_scale,
            cfg_scale_end=self.cfg_scale_end,
            latent_channels=latent_channels,
        )
