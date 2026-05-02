"""Two-model handoff wrapper for diffusion sampling.

Used for low-t-fine-tuned variants (e.g. the exp48 family) where the network
only learned t∈[0, handoff_t]. At inference we use a different model (the
seed/base) for t > handoff_t and the fine-tuned model for t <= handoff_t.

MONAI timestep convention: timesteps∈[0, num_train_timesteps],
timesteps=0 → clean, timesteps=T → noise. So `timesteps > threshold`
means we are in the noisier (high-t) regime.
"""
from __future__ import annotations

import torch
from torch import nn


class HandoffWrapper(nn.Module):
    def __init__(
        self,
        high_t_model: nn.Module,
        low_t_model: nn.Module,
        handoff_t: float,
        num_train_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        self.high_t_model = high_t_model
        self.low_t_model = low_t_model
        if not 0.0 < handoff_t < 1.0:
            raise ValueError(f"handoff_t must be in (0, 1), got {handoff_t}")
        self.handoff_threshold = handoff_t * num_train_timesteps
        self.handoff_t_norm = handoff_t

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, **kwargs):
        t_val = timesteps.flatten()[0].item()
        if t_val > self.handoff_threshold:
            return self.high_t_model(x=x, timesteps=timesteps, **kwargs)
        return self.low_t_model(x=x, timesteps=timesteps, **kwargs)
