"""DiffRS (Diffusion Rejection Sampling) implementation.

Improves diffusion sampling quality without retraining by adding a post-hoc
discriminator that evaluates intermediate samples during generation.

At each denoising step, the discriminator checks if the intermediate sample
looks like real data at that noise level. Bad trajectories are rejected and
retried with new noise.

Reference:
    DiffRS: Diffusion Rejection Sampling (ICML 2024)
    https://github.com/aailabkaist/DiffRS

Architecture:
    - Feature extractor: Frozen UNet encoder (already trained diffusion model)
    - Classification head: Tiny ~500 param head (GroupNorm -> SiLU -> Pool -> Linear)
    - The diffusion model is never modified
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from monai.networks.nets.diffusion_model_unet import get_timestep_embedding
from torch import Tensor

if TYPE_CHECKING:
    from .conditioning import ConditioningContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction from frozen UNet encoder
# ---------------------------------------------------------------------------

def _unwrap_to_unet(model: nn.Module) -> nn.Module:
    """Unwrap model wrappers to get the underlying DiffusionModelUNet.

    Handles: SizeBinModelWrapper, ScoreAugModelWrapper, CombinedModelWrapper,
    ModeEmbedModelWrapper, torch.compile'd models, etc.

    All wrappers store the inner model as ``self.model``.
    """
    unwrapped = model

    # Handle torch.compile wrapper
    if hasattr(unwrapped, '_orig_mod'):
        unwrapped = unwrapped._orig_mod

    # Unwrap custom wrappers (all use self.model)
    while hasattr(unwrapped, 'model') and not hasattr(unwrapped, 'down_blocks'):
        unwrapped = unwrapped.model

    # Final torch.compile check on inner model
    if hasattr(unwrapped, '_orig_mod'):
        unwrapped = unwrapped._orig_mod

    if not hasattr(unwrapped, 'down_blocks'):
        raise ValueError(
            f"Could not unwrap to DiffusionModelUNet. "
            f"Got {type(unwrapped).__name__} without 'down_blocks'."
        )
    return unwrapped


@torch.no_grad()
def extract_encoder_features(
    model: nn.Module, x: Tensor, timesteps: Tensor,
) -> Tensor:
    """Run the UNet encoder + middle_block, return bottleneck features.

    Reuses the already-trained UNet's encoder -- no new parameters.
    The encoder has timestep conditioning via FiLM, so features are
    timestep-aware automatically.

    Args:
        model: Trained diffusion model (or wrapped version).
        x: Noisy input [B, C, ...] at timestep t.
        timesteps: Timestep tensor [B].

    Returns:
        Bottleneck features [B, C_bottleneck, ...].
    """
    unet = _unwrap_to_unet(model)

    # Timestep embedding (same as first part of UNet.forward)
    t_emb = get_timestep_embedding(timesteps, unet.block_out_channels[0])
    t_emb = t_emb.to(x.dtype)
    emb = unet.time_embed(t_emb)

    # Encoder path
    h = unet.conv_in(x)
    for downsample_block in unet.down_blocks:
        h, _ = downsample_block(hidden_states=h, temb=emb)
    h = unet.middle_block(hidden_states=h, temb=emb)

    return h


def get_bottleneck_channels(model: nn.Module) -> int:
    """Get the number of channels at the UNet bottleneck.

    This is the last entry in block_out_channels (e.g., 512).
    """
    unet = _unwrap_to_unet(model)
    return unet.block_out_channels[-1]


# ---------------------------------------------------------------------------
# Discriminator head
# ---------------------------------------------------------------------------

class DiffRSHead(nn.Module):
    """Tiny classification head for DiffRS.

    Takes bottleneck features from the frozen UNet encoder and outputs
    a single logit (real vs generated). ~500 trainable parameters.
    """

    def __init__(self, in_channels: int, spatial_dims: int = 2):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, in_channels), in_channels)
        self.act = nn.SiLU()
        Pool = nn.AdaptiveAvgPool2d if spatial_dims == 2 else nn.AdaptiveAvgPool3d
        self.pool = Pool(1)
        self.linear = nn.Linear(in_channels, 1)

    def forward(self, features: Tensor) -> Tensor:
        """[B, C, ...] -> [B] logits."""
        h = self.act(self.norm(features))
        h = self.pool(h).flatten(1)  # [B, C]
        return self.linear(h).squeeze(-1)  # [B]


# ---------------------------------------------------------------------------
# Combined discriminator (frozen encoder + trainable head)
# ---------------------------------------------------------------------------

class DiffRSDiscriminator:
    """Wraps frozen UNet encoder + trainable head.

    Not an nn.Module -- just a callable container.
    The UNet is never in this object's parameters (stays frozen externally).
    """

    def __init__(self, model: nn.Module, head: DiffRSHead, device: torch.device):
        self.model = model
        self.head = head
        self.device = device

    @torch.no_grad()
    def get_log_ratio(self, x_t: Tensor, timesteps: Tensor) -> Tensor:
        """Compute log-likelihood ratio L_t(x_t) = logit. Shape: [B]."""
        features = extract_encoder_features(self.model, x_t, timesteps)
        return self.head(features)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_diffrs_head(
    checkpoint_path: str, device: torch.device,
) -> DiffRSHead:
    """Load trained DiffRS head from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    head = DiffRSHead(
        in_channels=ckpt['in_channels'],
        spatial_dims=ckpt['spatial_dims'],
    )
    head.load_state_dict(ckpt['head_state_dict'])
    head.to(device).eval()
    return head


# ---------------------------------------------------------------------------
# Adaptive threshold estimation (warmup)
# ---------------------------------------------------------------------------

def estimate_adaptive_thresholds(
    strategy: Any,
    model: nn.Module,
    discriminator: DiffRSDiscriminator,
    sample_shape: tuple[int, ...],
    num_steps: int,
    device: torch.device,
    iter_warmup: int = 10,
    rej_percentile: float = 0.75,
    conditioning: ConditioningContext | None = None,
) -> tuple[Tensor, Tensor]:
    """Estimate per-timestep adaptive thresholds via warmup sampling.

    Runs the base sampler ``iter_warmup`` times, collecting discriminator
    log-ratios at each timestep to build threshold distributions.

    Args:
        strategy: RFlowStrategy instance (with scheduler already set up).
        model: Diffusion model.
        discriminator: DiffRSDiscriminator.
        sample_shape: Shape of a single sample [C, ...] (no batch dim).
        num_steps: Number of sampling steps.
        device: Computation device.
        iter_warmup: Number of warmup trajectories.
        rej_percentile: Quantile for threshold (gamma in paper).
        conditioning: Optional ConditioningContext for guided generation.

    Returns:
        (adaptive, adaptive2): Tensors of shape [num_steps + 1] with
        per-timestep thresholds and consecutive-difference thresholds.
    """
    from .conditioning import ConditioningContext

    if conditioning is None:
        conditioning = ConditioningContext()

    batch_size = sample_shape[0] if len(sample_shape) > 3 else 1

    # Compute input_img_size_numel for scheduler
    if len(sample_shape) == 4:
        input_numel = sample_shape[1] * sample_shape[2] * sample_shape[3]
    elif len(sample_shape) == 3:
        input_numel = sample_shape[1] * sample_shape[2]
    else:
        input_numel = 1
    for d in sample_shape[1:]:
        pass  # already computed above

    # Set up scheduler timesteps
    strategy.scheduler.set_timesteps(
        num_inference_steps=num_steps,
        device=device,
        input_img_size_numel=input_numel,
    )

    t_steps = strategy.scheduler.timesteps
    all_next = torch.cat((
        t_steps[1:],
        torch.tensor([0], dtype=t_steps.dtype, device=device),
    ))
    n_steps = len(t_steps)

    # Collect log-ratios per timestep
    lst_adaptive: list[list[Tensor]] = [[] for _ in range(n_steps + 1)]
    lst_adaptive2: list[list[Tensor]] = [[] for _ in range(n_steps + 1)]

    warmup_done = 0
    # Use batch_size from sample_shape for warmup
    noise = torch.randn(sample_shape, device=device)
    x = noise.clone()
    step_idx = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    log_ratio_prev = torch.zeros(x.shape[0], device=device)

    latent_channels = conditioning.latent_channels

    while warmup_done < iter_warmup:
        # Get current and next timesteps for each sample
        cur_t_idx = step_idx.clamp(max=n_steps - 1)
        t_cur = t_steps[cur_t_idx]
        t_next = all_next[cur_t_idx]

        timesteps_batch = t_cur

        # Compute log ratio at current step
        # Need the noisy sample in the model's input space
        log_ratio = discriminator.get_log_ratio(x, timesteps_batch).cpu()

        for i in range(x.shape[0]):
            si = step_idx[i].item()
            lst_adaptive[si].append(log_ratio[i])
            if si > 0:
                lst_adaptive2[si].append(log_ratio[i] - log_ratio_prev[i])

        log_ratio_prev = log_ratio.clone()

        # Euler denoising step (simplified -- no CFG for warmup)
        with torch.no_grad():
            # Simple velocity prediction
            velocity = model(x=x, timesteps=timesteps_batch)
            # RFlow Euler step: x_next = x + (t_next - t_cur) / T * v ... but
            # use scheduler.step which handles this correctly
            x_denoised = torch.zeros_like(x)
            for i in range(x.shape[0]):
                x_denoised[i:i+1], _ = strategy.scheduler.step(
                    velocity[i:i+1], t_cur[i], x[i:i+1], t_next[i],
                )
            x = x_denoised

        step_idx = step_idx + 1

        # Check for completed trajectories
        done = step_idx >= n_steps
        if done.any():
            # Collect final step log ratio
            final_t = torch.zeros(done.sum(), device=device)
            log_ratio_final = discriminator.get_log_ratio(
                x[done], final_t,
            ).cpu()
            for i, lr in enumerate(log_ratio_final):
                lst_adaptive[n_steps].append(lr)

            # Re-init completed samples
            x[done] = torch.randn_like(x[done])
            step_idx[done] = 0
            log_ratio_prev[done] = 0.0
            warmup_done += 1

    # Compute thresholds as quantiles
    adaptive = torch.zeros(n_steps + 1)
    adaptive2 = torch.zeros(n_steps + 1)

    for k in range(n_steps + 1):
        if lst_adaptive[k]:
            vals = torch.stack(lst_adaptive[k])
            adaptive[k] = max(0.0, torch.quantile(vals.float(), rej_percentile).item())
        if lst_adaptive2[k]:
            vals = torch.stack(lst_adaptive2[k])
            adaptive2[k] = max(0.0, torch.quantile(vals.float(), rej_percentile).item())

    logger.info("DiffRS adaptive thresholds estimated from %d warmup runs", iter_warmup)
    return adaptive.to(device), adaptive2.to(device)


# ---------------------------------------------------------------------------
# DiffRS sampling loop (Algorithm 3 from paper)
# ---------------------------------------------------------------------------

def diffrs_sampling_loop(
    strategy: Any,
    model: nn.Module,
    discriminator: DiffRSDiscriminator,
    noise: Tensor,
    num_steps: int,
    device: torch.device,
    adaptive: Tensor,
    adaptive2: Tensor,
    *,
    conditioning: ConditioningContext | None = None,
    backsteps: int = 1,
    min_backsteps: int = 0,
    max_backsteps: int | None = None,
    max_iter: int = 999999,
) -> Tensor:
    """DiffRS sampling with rejection and backsteps.

    Implements Algorithm 3 from the DiffRS paper, adapted for RFlow/Euler.

    Args:
        strategy: RFlowStrategy with scheduler set up.
        model: Diffusion model.
        discriminator: DiffRSDiscriminator.
        noise: Initial noise [B, C, ...].
        num_steps: Number of Euler steps.
        device: Computation device.
        adaptive: Per-timestep thresholds [num_steps + 1].
        adaptive2: Per-timestep difference thresholds [num_steps + 1].
        conditioning: Optional ConditioningContext.
        backsteps: Number of timesteps to go back on rejection.
        min_backsteps: Don't backstep below this step index.
        max_backsteps: Don't backstep above this step index.
        max_iter: Max NFE per sample before full re-init.

    Returns:
        Generated samples [B, C, ...].
    """
    from .conditioning import ConditioningContext

    if conditioning is None:
        conditioning = ConditioningContext()
    if max_backsteps is None:
        max_backsteps = num_steps

    batch_size = noise.shape[0]

    t_steps = strategy.scheduler.timesteps
    all_next = torch.cat((
        t_steps[1:],
        torch.tensor([0], dtype=t_steps.dtype, device=device),
    ))

    x = noise.clone()
    lst_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    log_ratio_prev = torch.zeros(batch_size, device=device)
    per_sample_nfe = torch.zeros(batch_size, dtype=torch.long, device=device)
    results = torch.zeros_like(x)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    total_finished = 0

    # Main loop: keep going until all samples are done
    while not finished.all():
        active = ~finished
        if not active.any():
            break

        cur_idx = lst_idx[active].clamp(max=num_steps - 1)
        t_cur = t_steps[cur_idx]
        t_next = all_next[cur_idx]
        x_active = x[active]

        # --- Marginal rejection at step 0 ---
        at_zero = lst_idx[active] == 0
        if min_backsteps == 0 and at_zero.any():
            zero_mask = torch.where(active)[0][at_zero]
            _do_marginal_rejection(
                x, lst_idx, log_ratio_prev, zero_mask,
                t_steps, adaptive, discriminator, device,
            )
            # Refresh active state after rejection may have modified x
            cur_idx = lst_idx[active].clamp(max=num_steps - 1)
            t_cur = t_steps[cur_idx]
            t_next = all_next[cur_idx]
            x_active = x[active]

        # --- Euler denoising step ---
        timesteps_batch = t_cur
        with torch.no_grad():
            velocity = model(x=x_active, timesteps=timesteps_batch)
        per_sample_nfe[active] += 1

        x_next = torch.zeros_like(x_active)
        for i in range(x_active.shape[0]):
            x_next[i:i+1], _ = strategy.scheduler.step(
                velocity[i:i+1], t_cur[i], x_active[i:i+1], t_next[i],
            )

        x[active] = x_next
        lst_idx[active] += 1

        # --- Backstep rejection ---
        if backsteps > 0:
            _do_backstep_rejection(
                x, lst_idx, log_ratio_prev, per_sample_nfe,
                active, t_steps, adaptive, adaptive2,
                discriminator, device,
                backsteps=backsteps,
                min_backsteps=min_backsteps,
                max_backsteps=max_backsteps,
            )

        # --- Check for completed samples ---
        done = active & (lst_idx >= num_steps)
        if done.any():
            results[done] = x[done]
            finished[done] = True
            total_finished += done.sum().item()

        # --- NFE budget exceeded: full re-init ---
        over_budget = active & ~finished & (
            per_sample_nfe + (num_steps * 2 - 1 - lst_idx * 2) > max_iter
        )
        if over_budget.any():
            x[over_budget] = torch.randn_like(x[over_budget])
            lst_idx[over_budget] = 0
            per_sample_nfe[over_budget] = 0
            log_ratio_prev[over_budget] = 0.0

    return results


def _do_marginal_rejection(
    x: Tensor,
    lst_idx: Tensor,
    log_ratio_prev: Tensor,
    zero_indices: Tensor,
    t_steps: Tensor,
    adaptive: Tensor,
    discriminator: DiffRSDiscriminator,
    device: torch.device,
) -> None:
    """Marginal rejection sampling at step 0 (in-place)."""
    remaining = torch.ones(len(zero_indices), dtype=torch.bool, device=device)

    for _ in range(100):  # Safety bound
        if not remaining.any():
            break

        check_idx = zero_indices[remaining]
        x_check = x[check_idx]
        t_check = t_steps[lst_idx[check_idx]]

        log_ratio = discriminator.get_log_ratio(x_check, t_check)
        threshold = adaptive[lst_idx[check_idx]]
        rand_term = torch.log(torch.rand_like(log_ratio) + 1e-7)

        rejected = log_ratio < threshold + rand_term

        if rejected.any():
            reject_global = check_idx[rejected]
            x[reject_global] = torch.randn_like(x[reject_global]) * t_steps[0]

        # Accept: update log_ratio_prev
        accepted = ~rejected
        if accepted.any():
            accept_global = check_idx[accepted]
            log_ratio_prev[accept_global] = log_ratio[accepted]

        # Update remaining mask
        remaining_local = remaining.clone()
        remaining_local[remaining] = rejected
        remaining = remaining_local


def _do_backstep_rejection(
    x: Tensor,
    lst_idx: Tensor,
    log_ratio_prev: Tensor,
    per_sample_nfe: Tensor,
    active: Tensor,
    t_steps: Tensor,
    adaptive: Tensor,
    adaptive2: Tensor,
    discriminator: DiffRSDiscriminator,
    device: torch.device,
    backsteps: int,
    min_backsteps: int,
    max_backsteps: int,
) -> None:
    """Backstep rejection at intermediate timesteps (in-place)."""
    # Which active samples are eligible for backstep checking
    eligible = active & (lst_idx > min_backsteps) & (lst_idx <= max_backsteps)

    count = 0
    for _ in range(100):  # Safety bound
        if not eligible.any():
            break

        check_idx = torch.where(eligible)[0]
        x_check = x[check_idx]
        step_check = lst_idx[check_idx]
        t_check = t_steps[step_check.clamp(max=len(t_steps) - 1)]

        log_ratio = discriminator.get_log_ratio(x_check, t_check)
        rand_term = torch.log(torch.rand_like(log_ratio) + 1e-7)

        if count == 0:
            # First check uses adaptive2 + previous log ratio
            threshold = adaptive2[step_check]
            rejected = log_ratio < threshold + rand_term + log_ratio_prev[check_idx]
        else:
            threshold = adaptive[step_check]
            rejected = log_ratio < threshold + rand_term

        # Handle rejections: backstep
        if rejected.any():
            reject_global = check_idx[rejected]
            reject_steps = lst_idx[reject_global]
            back_steps = (reject_steps - backsteps).clamp(min=0)
            t_back = t_steps[back_steps.clamp(max=len(t_steps) - 1)]
            t_cur = t_steps[reject_steps.clamp(max=len(t_steps) - 1)]

            # Forward diffuse back: add noise to go from t_cur to t_back
            # x_back = x + sqrt(t_back^2 - t_cur^2) * eps
            noise_scale = (t_back ** 2 - t_cur ** 2).clamp(min=0).sqrt()
            expand_dims = [1] * (x.dim() - 1)
            eps = torch.randn_like(x[reject_global])
            x[reject_global] = x[reject_global] + noise_scale.view(-1, *expand_dims) * eps
            lst_idx[reject_global] = back_steps

        # Handle accepts: update log_ratio_prev
        accepted = ~rejected
        if accepted.any():
            accept_global = check_idx[accepted]
            log_ratio_prev[accept_global] = log_ratio[accepted]

        # Update eligibility
        eligible[check_idx[accepted]] = False
        eligible[lst_idx <= min_backsteps] = False
        count += 1
