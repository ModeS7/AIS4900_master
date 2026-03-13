#!/usr/bin/env python3
"""Find optimal FreeU parameters for diffusion generation via grid search.

FreeU (Si et al., CVPR 2024) is a training-free inference technique that
improves diffusion model quality by reweighting skip connections and backbone
features in the UNet decoder:
  - Backbone features (hidden_states) scaled by factor b (>1, amplify semantics)
  - Skip features (res_hidden_states) scaled by factor s (<1, suppress noise)

This script applies FreeU scaling via monkey-patching MONAI's UpBlock forward
methods, then evaluates FID/KID over a grid of (b, s) values.

Only applies FreeU to decoder levels that have attention (typically the deepest
levels), matching the original paper's approach.

Usage:
    # Grid search over b and s values
    python -m medgen.scripts.find_optimal_freeu \
        --checkpoint runs/checkpoint_latest.pt \
        --data-root ~/MedicalDataSets/brainmetshare-3 \
        --num-volumes 25 --output-dir eval_freeu

    # Custom grid ranges
    python -m medgen.scripts.find_optimal_freeu \
        --checkpoint runs/checkpoint_latest.pt \
        --data-root ~/MedicalDataSets/brainmetshare-3 \
        --b-values 1.0 1.1 1.2 1.3 1.4 \
        --s-values 0.2 0.4 0.6 0.8 0.9 1.0

    # Smoke test (no GPU needed)
    python -m medgen.scripts.find_optimal_freeu --smoke-test
"""
import argparse
import gc
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.fft
import torch.nn as nn

from medgen.scripts.eval_ode_solvers import (
    SolverConfig,
    compute_all_metrics,
    discover_splits,
    generate_noise_tensors,
    generate_volumes,
    get_or_cache_reference_features,
    load_conditioning,
    save_volumes,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# FreeU configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Default grid values (from FreeU paper, adapted for medical imaging)
DEFAULT_B_VALUES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
DEFAULT_S_VALUES = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]

# Default inference steps (Euler 25 is optimal for RFlow)
DEFAULT_NUM_STEPS = 25


@dataclass
class FreeUConfig:
    """FreeU parameter configuration for one evaluation."""
    backbone_scale: float  # b: scale for backbone features (>1 = amplify)
    skip_scale: float      # s: scale for skip features (<1 = suppress)

    @property
    def label(self) -> str:
        return f"b{self.backbone_scale:.2f}_s{self.skip_scale:.2f}"

    @property
    def dir_name(self) -> str:
        return f"b{self.backbone_scale:.2f}_s{self.skip_scale:.2f}"

    @property
    def is_baseline(self) -> bool:
        """b=1.0, s=1.0 means no FreeU (baseline)."""
        return self.backbone_scale == 1.0 and self.skip_scale == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# FreeU monkey-patching
# ═══════════════════════════════════════════════════════════════════════════════

def _freeu_filter_skip(skip: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Apply spectral filtering to skip connections (FreeU v2 style).

    Attenuates high-frequency components in the skip connection using an FFT
    low-pass filter. This reduces texture noise while preserving structure.

    Args:
        skip: Skip connection tensor [B, C, ...spatial...]
        threshold: Fraction of frequencies to keep (1.0 = no filtering)
    """
    if threshold >= 1.0:
        return skip

    # FFT-based low-pass filter
    dtype = skip.dtype
    skip_f = skip.float()

    # Apply FFT across spatial dims
    spatial_dims = list(range(2, skip.dim()))
    freq = torch.fft.fftn(skip_f, dim=spatial_dims)

    # Create low-pass mask
    mask = torch.ones_like(freq, dtype=torch.float32)
    for dim in spatial_dims:
        n = freq.shape[dim]
        cutoff = int(n * threshold)
        # Zero out high frequencies
        idx = [slice(None)] * freq.dim()
        idx[dim] = slice(cutoff, n - cutoff + 1) if cutoff < n // 2 else slice(0, 0)
        if cutoff < n // 2:
            mask[tuple(idx)] = 0.0

    freq = freq * mask
    result = torch.fft.ifftn(freq, dim=spatial_dims).real
    return result.to(dtype)


def _freeu_pre_hook(backbone_scale: float, skip_scale: float):
    """Create a forward pre-hook that applies FreeU scaling.

    The hook intercepts (hidden_states, res_hidden_states_list, temb, context)
    and scales hidden_states (backbone) and each tensor in res_hidden_states_list
    (skip connections) before the UpBlock processes them.

    MONAI calls UpBlocks with keyword args, so we use with_kwargs=True.
    """
    def hook(module, args, kwargs):
        # MONAI calls UpBlocks with kwargs: hidden_states=h,
        # res_hidden_states_list=..., temb=..., context=...
        # Convert to positional args with modifications applied.
        if kwargs:
            hidden_states = kwargs['hidden_states']
            res_list = kwargs['res_hidden_states_list']
            temb = kwargs['temb']
            context = kwargs.get('context')

            if backbone_scale != 1.0:
                hidden_states = hidden_states * backbone_scale
            if skip_scale != 1.0:
                res_list = [r * skip_scale for r in res_list]

            return (hidden_states, res_list, temb, context), {}
        else:
            # Positional args fallback
            args = list(args)
            if backbone_scale != 1.0:
                args[0] = args[0] * backbone_scale
            if skip_scale != 1.0:
                args[1] = [r * skip_scale for r in args[1]]
            return tuple(args), kwargs

    return hook


@contextmanager
def apply_freeu(model: nn.Module, config: FreeUConfig):
    """Context manager that temporarily applies FreeU scaling to a UNet.

    Uses PyTorch forward pre-hooks on UpBlocks that have attention
    (deepest decoder levels). Hooks are removed on exit.

    Args:
        model: The diffusion model (DiffusionModelUNet or wrapped).
        config: FreeU parameters (backbone_scale, skip_scale).
    """
    if config.is_baseline:
        yield
        return

    # Unwrap to get the actual DiffusionModelUNet
    unet = model
    while hasattr(unet, '_wrapped'):
        unet = unet._wrapped
    while hasattr(unet, 'module'):
        unet = unet.module

    if not hasattr(unet, 'up_blocks'):
        logger.warning("Model has no up_blocks — FreeU not applicable")
        yield
        return

    # FreeU applies to the deepest decoder levels (those with attention).
    # In our 6-level UNet with attention_levels=[F,F,F,F,T,T],
    # up_blocks are in reverse order: up_blocks[0] is deepest (level 5),
    # up_blocks[1] is level 4, etc.
    hook_handles = []
    patched_indices = []

    for i, block in enumerate(unet.up_blocks):
        block_type = type(block).__name__
        has_attention = 'Attn' in block_type or 'CrossAttn' in block_type

        if has_attention:
            handle = block.register_forward_pre_hook(
                _freeu_pre_hook(config.backbone_scale, config.skip_scale),
                with_kwargs=True,
            )
            hook_handles.append(handle)
            patched_indices.append(i)

    if not hook_handles:
        # No attention blocks — apply to first 2 up_blocks (deepest) as fallback
        logger.info("  No attention up_blocks found, applying FreeU to deepest 2 levels")
        for i in range(min(2, len(unet.up_blocks))):
            handle = unet.up_blocks[i].register_forward_pre_hook(
                _freeu_pre_hook(config.backbone_scale, config.skip_scale),
                with_kwargs=True,
            )
            hook_handles.append(handle)
            patched_indices.append(i)

    logger.info(f"  FreeU applied to {len(hook_handles)} decoder levels "
                f"(blocks {patched_indices}, b={config.backbone_scale}, s={config.skip_scale})")

    try:
        yield
    finally:
        for handle in hook_handles:
            handle.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# FreeU-aware volume generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_volumes_freeu(
    model: nn.Module,
    strategy,
    noise_list: list[torch.Tensor],
    cond_list: list[tuple[str, torch.Tensor]] | None,
    freeu_cfg: FreeUConfig,
    device: torch.device,
    num_steps: int = DEFAULT_NUM_STEPS,
    is_seg: bool = False,
    encode_cond_fn=None,
    decode_fn=None,
    latent_channels: int = 1,
) -> tuple[list[np.ndarray], float]:
    """Generate volumes with FreeU applied.

    Returns:
        (volumes_list, wall_time_seconds)
    """
    solver_cfg = SolverConfig(solver='euler', steps=num_steps)

    with apply_freeu(model, freeu_cfg):
        volumes, _, wall_time = generate_volumes(
            model, strategy, noise_list, cond_list, solver_cfg, device,
            is_seg=is_seg,
            encode_cond_fn=encode_cond_fn,
            decode_fn=decode_fn,
            latent_channels=latent_channels,
        )

    return volumes, wall_time


# ═══════════════════════════════════════════════════════════════════════════════
# Grid search
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FreeUResult:
    """Result for one FreeU configuration."""
    backbone_scale: float
    skip_scale: float
    fid: float
    kid_mean: float
    kid_std: float
    cmmd: float
    fid_radimagenet: float
    kid_radimagenet_mean: float
    kid_radimagenet_std: float
    wall_time_s: float
    num_volumes: int


def run_grid_search(
    model: nn.Module,
    strategy,
    noise_list: list[torch.Tensor],
    cond_list: list[tuple[str, torch.Tensor]] | None,
    eval_ref: dict[str, dict[str, torch.Tensor]],
    ref_split: str,
    b_values: list[float],
    s_values: list[float],
    device: torch.device,
    output_dir: Path,
    num_steps: int = DEFAULT_NUM_STEPS,
    is_seg: bool = False,
    encode_cond_fn=None,
    decode_fn=None,
    latent_channels: int = 1,
    metric: str = 'fid',
    trim_slices: int = 10,
) -> list[FreeUResult]:
    """Run grid search over FreeU parameters.

    Returns list of FreeUResult sorted by target metric.
    """
    results: list[FreeUResult] = []
    total_configs = len(b_values) * len(s_values)

    for idx, (b, s) in enumerate([(b, s) for b in b_values for s in s_values]):
        freeu_cfg = FreeUConfig(backbone_scale=b, skip_scale=s)
        tag = "BASELINE" if freeu_cfg.is_baseline else freeu_cfg.label

        logger.info(f"\n{'='*60}")
        logger.info(f"[{idx+1}/{total_configs}] Evaluating: {tag}")
        logger.info(f"{'='*60}")

        t0 = time.time()

        volumes, wall_time = generate_volumes_freeu(
            model, strategy, noise_list, cond_list, freeu_cfg, device,
            num_steps=num_steps, is_seg=is_seg,
            encode_cond_fn=encode_cond_fn, decode_fn=decode_fn,
            latent_channels=latent_channels,
        )

        logger.info(f"  Generated {len(volumes)} volumes in {wall_time:.1f}s")

        # Compute metrics
        split_metrics = compute_all_metrics(volumes, eval_ref, device, trim_slices)
        ref_metrics = split_metrics.get(ref_split)
        if ref_metrics is None:
            raise ValueError(f"Reference split '{ref_split}' not in metrics")

        from dataclasses import asdict
        m = asdict(ref_metrics)

        result = FreeUResult(
            backbone_scale=b,
            skip_scale=s,
            fid=m['fid'],
            kid_mean=m['kid_mean'],
            kid_std=m['kid_std'],
            cmmd=m['cmmd'],
            fid_radimagenet=m['fid_radimagenet'],
            kid_radimagenet_mean=m['kid_radimagenet_mean'],
            kid_radimagenet_std=m['kid_radimagenet_std'],
            wall_time_s=wall_time,
            num_volumes=len(volumes),
        )
        results.append(result)

        elapsed = time.time() - t0
        logger.info(
            f"  {tag}: FID={result.fid:.2f}  KID={result.kid_mean:.6f}  "
            f"CMMD={result.cmmd:.6f}  FID_RIN={result.fid_radimagenet:.2f}  "
            f"KID_RIN={result.kid_radimagenet_mean:.6f}  ({elapsed:.0f}s)"
        )

        # Save incremental results
        _save_results(output_dir, results, b_values, s_values, metric, ref_split)

        del volumes
        gc.collect()
        torch.cuda.empty_cache()

    return results


def _save_results(
    output_dir: Path,
    results: list[FreeUResult],
    b_values: list[float],
    s_values: list[float],
    metric: str,
    ref_split: str,
    best_config: FreeUConfig | None = None,
) -> None:
    """Save grid search results to JSON."""
    from dataclasses import asdict
    data = {
        'b_values': b_values,
        's_values': s_values,
        'metric': metric,
        'ref_split': ref_split,
        'best': asdict(best_config) if best_config else None,
        'results': [asdict(r) for r in results],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "freeu_results.json", 'w') as f:
        json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

SEG_MODES = frozenset({'seg', 'seg_conditioned', 'seg_conditioned_input'})


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal FreeU parameters via grid search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Model
    parser.add_argument('--checkpoint', '--bravo-model', default=None, dest='checkpoint',
                        help='Path to trained model checkpoint')
    parser.add_argument('--mode', default=None,
                        help='Generation mode (auto-detected from checkpoint if omitted)')
    parser.add_argument('--strategy', default=None, choices=['rflow', 'ddpm'],
                        help='Diffusion strategy (auto-detected from checkpoint)')

    # Space (pixel/latent/wavelet)
    parser.add_argument('--space', default='pixel', choices=['pixel', 'latent', 'wavelet'],
                        help='Diffusion space (default: pixel)')
    parser.add_argument('--compression-checkpoint', default=None,
                        help='Compression model checkpoint for --space latent')
    parser.add_argument('--compression-type', default=None,
                        help='Compression type: auto/vae/vqvae/dcae (default: auto)')

    # Data
    parser.add_argument('--ref-modality', default=None,
                        help='Reference file modality (auto: bravo for image modes)')
    parser.add_argument('--data-root', default=None, help='Root of dataset')
    parser.add_argument('--output-dir', default='eval_freeu',
                        help='Output directory (default: eval_freeu)')
    parser.add_argument('--num-volumes', type=int, default=25,
                        help='Volumes per evaluation (default: 25)')

    # FreeU grid
    parser.add_argument('--b-values', type=float, nargs='+', default=DEFAULT_B_VALUES,
                        help=f'Backbone scale values to search (default: {DEFAULT_B_VALUES})')
    parser.add_argument('--s-values', type=float, nargs='+', default=DEFAULT_S_VALUES,
                        help=f'Skip scale values to search (default: {DEFAULT_S_VALUES})')
    parser.add_argument('--num-steps', type=int, default=DEFAULT_NUM_STEPS,
                        help=f'Euler steps for generation (default: {DEFAULT_NUM_STEPS})')

    # Volume config
    parser.add_argument('--cond-split', default='val',
                        help='Split for conditioning seg masks (default: val)')
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--trim-slices', type=int, default=10)
    parser.add_argument('--fov-mm', type=float, default=240.0)
    parser.add_argument('--seed', type=int, default=42)

    # Metric
    parser.add_argument('--metric', choices=[
        'fid', 'kid', 'cmmd', 'fid_radimagenet', 'kid_radimagenet',
    ], default='fid',
                        help='Metric to minimize (default: fid)')
    parser.add_argument('--ref-split', default='all',
                        help='Reference split for metric computation (default: all)')
    parser.add_argument('--save-volumes', action='store_true',
                        help='Save generated volumes as NIfTI')

    # Pixel normalization overrides
    parser.add_argument('--rescale', action='store_true',
                        help='Pixel rescale [-1, 1] -> [0, 1]')
    parser.add_argument('--pixel-shift', type=float, nargs='+', default=None)
    parser.add_argument('--pixel-scale', type=float, nargs='+', default=None)

    # DDPM-specific
    parser.add_argument('--prediction-type', default=None)
    parser.add_argument('--schedule', default=None)

    parser.add_argument('--smoke-test', action='store_true',
                        help='Smoke test with synthetic data')
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required (unless --smoke-test)")
    if not args.data_root:
        parser.error("--data-root is required (unless --smoke-test)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Auto-detect config from checkpoint ────────────────────────────────
    logger.info("Loading checkpoint metadata...")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_cfg = ckpt.get('config', {})

    base_in_channels = ckpt_cfg.get('in_channels')
    base_out_channels = ckpt_cfg.get('out_channels')
    mode = args.mode or ckpt_cfg.get('mode')
    if isinstance(mode, dict):
        mode = mode.get('name', None)
    spatial_dims = ckpt_cfg.get('spatial_dims', 3)

    model_type = ckpt_cfg.get('model_type', 'unet')
    is_transformer = model_type in ('dit', 'sit', 'uvit', 'hdit')
    if is_transformer:
        logger.warning("FreeU is designed for UNet skip connections. "
                       "Transformer models may not benefit.")
        pixel_image_size = args.image_size or 256
        pixel_depth = args.depth or 160
    else:
        pixel_image_size = args.image_size or ckpt_cfg.get('image_size', 256)
        pixel_depth = args.depth or ckpt_cfg.get('depth_size', 160)

    strategy_name = args.strategy or ckpt_cfg.get('strategy', 'rflow')

    if base_in_channels is None:
        base_in_channels = 2
    if base_out_channels is None:
        base_out_channels = 1
    del ckpt

    # ── Setup diffusion space ─────────────────────────────────────────────
    # (Reuses same logic as find_optimal_steps.py)
    from medgen.scripts.find_optimal_steps import _setup_latent_space, _setup_wavelet_space

    encode_cond_fn = None
    decode_fn = None
    data_root = Path(args.data_root)

    if args.space == 'latent':
        space, latent_ch, sf, depth_sf = _setup_latent_space(args, ckpt_cfg, device, spatial_dims)
        model_out_channels = base_out_channels * latent_ch
        model_in_channels = base_in_channels * latent_ch
        noise_image_size = pixel_image_size // sf
        noise_depth = pixel_depth // depth_sf
        encode_cond_fn = space.encode_normalized_seg
        decode_fn = space.decode
    elif args.space == 'wavelet':
        space, wav_ch, sf, depth_sf = _setup_wavelet_space(args, ckpt_cfg)
        model_out_channels = base_out_channels * wav_ch
        model_in_channels = base_in_channels * wav_ch
        noise_image_size = pixel_image_size // sf
        noise_depth = pixel_depth // depth_sf
        encode_cond_fn = space.encode
        decode_fn = space.decode
    else:
        model_in_channels = base_in_channels
        model_out_channels = base_out_channels
        noise_image_size = pixel_image_size
        noise_depth = pixel_depth

        pixel_cfg = ckpt_cfg.get('pixel', {})
        pixel_shift = args.pixel_shift or pixel_cfg.get('pixel_shift')
        pixel_scale = args.pixel_scale or pixel_cfg.get('pixel_scale')
        pixel_rescale = args.rescale or pixel_cfg.get('rescale', False)
        if pixel_shift is not None or pixel_rescale:
            from medgen.diffusion.spaces import PixelSpace
            space = PixelSpace(rescale=pixel_rescale, shift=pixel_shift, scale=pixel_scale)
            decode_fn = space.decode
            encode_cond_fn = space.encode

    # ── Mode config ───────────────────────────────────────────────────────
    is_seg = mode in SEG_MODES if mode else False
    ref_modality = args.ref_modality or ('seg' if is_seg else 'bravo')
    cond_channels = base_in_channels - base_out_channels

    total_configs = len(args.b_values) * len(args.s_values)

    logger.info("=" * 70)
    logger.info("FreeU Parameter Optimization (Grid Search)")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Mode: {mode or 'unknown'} | Strategy: {strategy_name} | Space: {args.space}")
    logger.info(f"Pixel volume: {pixel_image_size}x{pixel_image_size}x{pixel_depth}")
    logger.info(f"Euler steps: {args.num_steps}")
    logger.info(f"Backbone scales (b): {args.b_values}")
    logger.info(f"Skip scales (s): {args.s_values}")
    logger.info(f"Total configs: {total_configs}")
    logger.info(f"Volumes per eval: {args.num_volumes}")
    logger.info(f"Metric: {args.metric} (vs '{args.ref_split}')")
    logger.info("=" * 70)

    # ── Data setup ────────────────────────────────────────────────────────
    splits = discover_splits(data_root, modality=ref_modality)

    cond_list = None
    if cond_channels > 0:
        if args.cond_split not in splits:
            raise ValueError(f"Split '{args.cond_split}' not found. Available: {list(splits.keys())}")
        logger.info(f"Loading {args.num_volumes} conditioning masks...")
        cond_list = load_conditioning(splits[args.cond_split], args.num_volumes, pixel_depth)

        sample_hw = cond_list[0][1].shape[-1]
        if sample_hw != pixel_image_size:
            from torch.nn.functional import interpolate
            logger.info(f"Resizing conditioning from {sample_hw} to {pixel_image_size}")
            cond_list = [
                (pid, interpolate(
                    seg, size=(pixel_depth, pixel_image_size, pixel_image_size),
                    mode='nearest',
                ))
                for pid, seg in cond_list
            ]

    # ── Reference features ────────────────────────────────────────────────
    logger.info("Preparing reference features...")
    cache_dir = output_dir / "reference_features"

    if args.ref_split == 'all':
        ref_features = get_or_cache_reference_features(
            splits, cache_dir, device, pixel_depth, args.trim_slices, pixel_image_size,
            modality=ref_modality, build_all=False,
        )
        eval_ref = {'all': {
            key: torch.cat([ref_features[s][key] for s in splits], dim=0)
            for key in ('resnet', 'resnet_radimagenet', 'clip')
        }}
        del ref_features
        gc.collect()
    elif args.ref_split in splits:
        needed_splits = {args.ref_split: splits[args.ref_split]}
        ref_features = get_or_cache_reference_features(
            needed_splits, cache_dir, device, pixel_depth, args.trim_slices, pixel_image_size,
            modality=ref_modality, build_all=False,
        )
        eval_ref = {args.ref_split: ref_features[args.ref_split]}
        del ref_features
        gc.collect()
    else:
        available = list(splits.keys()) + ['all']
        raise ValueError(f"Reference split '{args.ref_split}' not found. Available: {available}")

    # ── Load model ────────────────────────────────────────────────────────
    from medgen.diffusion import load_diffusion_model

    logger.info("Loading model...")
    model = load_diffusion_model(
        args.checkpoint, device=device,
        in_channels=model_in_channels, out_channels=model_out_channels,
        compile_model=False, spatial_dims=spatial_dims,
    )

    # ── Setup strategy ────────────────────────────────────────────────────
    if strategy_name == 'ddpm':
        from medgen.diffusion import DDPMStrategy
        strategy = DDPMStrategy()
        prediction_type = args.prediction_type or 'sample'
        schedule = args.schedule or 'linear_beta'
        strategy.setup_scheduler(
            num_timesteps=1000, image_size=noise_image_size,
            depth_size=noise_depth, spatial_dims=spatial_dims,
            prediction_type=prediction_type, schedule=schedule,
        )
    else:
        from medgen.diffusion import RFlowStrategy
        strategy = RFlowStrategy()
        strategy.setup_scheduler(
            num_timesteps=1000, image_size=noise_image_size,
            depth_size=noise_depth, spatial_dims=spatial_dims,
        )

    # EDM preconditioning
    sigma_data = ckpt_cfg.get('sigma_data', 0.0)
    if sigma_data > 0 and hasattr(strategy, 'set_preconditioning'):
        strategy.set_preconditioning(sigma_data, model_out_channels)

    # ── Pre-generate noise ────────────────────────────────────────────────
    logger.info(f"Pre-generating {args.num_volumes} noise tensors (seed={args.seed})...")
    noise_list = generate_noise_tensors(
        args.num_volumes, noise_depth, noise_image_size, device, args.seed,
        out_channels=model_out_channels,
    )

    # ── Run grid search ───────────────────────────────────────────────────
    total_start = time.time()

    results = run_grid_search(
        model, strategy, noise_list, cond_list, eval_ref, args.ref_split,
        args.b_values, args.s_values, device, output_dir,
        num_steps=args.num_steps, is_seg=is_seg,
        encode_cond_fn=encode_cond_fn, decode_fn=decode_fn,
        latent_channels=model_out_channels, metric=args.metric,
        trim_slices=args.trim_slices,
    )

    total_time = time.time() - total_start

    # ── Find best ─────────────────────────────────────────────────────────
    metric_key = {
        'fid': 'fid', 'kid': 'kid_mean', 'cmmd': 'cmmd',
        'fid_radimagenet': 'fid_radimagenet',
        'kid_radimagenet': 'kid_radimagenet_mean',
    }[args.metric]

    best = min(results, key=lambda r: getattr(r, metric_key))
    best_cfg = FreeUConfig(backbone_scale=best.backbone_scale, skip_scale=best.skip_scale)

    # Find baseline for comparison
    baseline = next((r for r in results if r.backbone_scale == 1.0 and r.skip_scale == 1.0), None)

    # Save final results
    _save_results(output_dir, results, args.b_values, args.s_values,
                  args.metric, args.ref_split, best_cfg)

    # ── Print results ─────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("RESULTS: FreeU Grid Search")
    logger.info(f"{'='*70}")
    logger.info(f"Total configs evaluated: {len(results)}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")

    if baseline:
        logger.info(f"\nBaseline (no FreeU): {args.metric}={getattr(baseline, metric_key):.4f}")

    logger.info(f"\nBest: b={best.backbone_scale:.2f}, s={best.skip_scale:.2f}")
    logger.info(f"  FID={best.fid:.2f}  KID={best.kid_mean:.6f}  CMMD={best.cmmd:.6f}")
    logger.info(f"  FID_RIN={best.fid_radimagenet:.2f}  KID_RIN={best.kid_radimagenet_mean:.6f}")

    if baseline and not best_cfg.is_baseline:
        improvement = getattr(baseline, metric_key) - getattr(best, metric_key)
        pct = 100 * improvement / getattr(baseline, metric_key)
        logger.info(f"  Improvement over baseline: {improvement:+.4f} ({pct:+.1f}%)")

    # Print grid table
    logger.info(f"\n{args.metric.upper()} grid (rows=b, cols=s):")
    header = f"{'b \\ s':>8}" + "".join(f"{s:>10.2f}" for s in args.s_values)
    logger.info(header)
    logger.info("-" * len(header))

    for b in args.b_values:
        row = f"{b:>8.2f}"
        for s in args.s_values:
            r = next((r for r in results if r.backbone_scale == b and r.skip_scale == s), None)
            if r:
                val = getattr(r, metric_key)
                marker = " *" if r is best else "  "
                row += f"{val:>8.2f}{marker}"
            else:
                row += f"{'—':>10}"
        logger.info(row)

    logger.info(f"\n* = best\nResults saved to: {output_dir}/freeu_results.json")


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _run_smoke_test():
    """Verify FreeU monkey-patching logic with a dummy model."""
    logger.info("=== SMOKE TEST (FreeU patching) ===")

    from monai.networks.nets import DiffusionModelUNet

    # Create tiny 2D UNet with random weights (default zero-init output
    # makes all outputs zero, so we need non-zero weights for testing)
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=[32, 64],
        attention_levels=[False, True],
        num_head_channels=32,
        num_res_blocks=1,
        norm_num_groups=32,
    )
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)

    x = torch.randn(1, 1, 32, 32)
    t = torch.tensor([500.0])

    # Baseline forward
    with torch.no_grad():
        out_baseline = model(x, t).clone()

    logger.info(f"Baseline output: mean={out_baseline.mean():.6f}, std={out_baseline.std():.6f}")

    # FreeU forward (b=1.2, s=0.8)
    cfg = FreeUConfig(backbone_scale=1.2, skip_scale=0.8)
    with apply_freeu(model, cfg):
        with torch.no_grad():
            out_freeu = model(x, t).clone()

    # After context manager, should be back to baseline
    with torch.no_grad():
        out_restored = model(x, t)

    # Check outputs differ with FreeU
    diff_freeu = (out_baseline - out_freeu).abs().mean().item()
    diff_restored = (out_baseline - out_restored).abs().mean().item()

    logger.info(f"Baseline vs FreeU diff: {diff_freeu:.6f} (should be >0)")
    logger.info(f"Baseline vs restored diff: {diff_restored:.6f} (should be ~0)")

    assert diff_freeu > 1e-7, f"FreeU should change the output (diff={diff_freeu})"
    assert diff_restored < 1e-5, "Output should be restored after context manager"

    # Test baseline config (b=1.0, s=1.0) should not change output
    cfg_baseline = FreeUConfig(backbone_scale=1.0, skip_scale=1.0)
    with apply_freeu(model, cfg_baseline):
        with torch.no_grad():
            out_noop = model(x, t)

    diff_noop = (out_baseline - out_noop).abs().mean().item()
    logger.info(f"Baseline vs noop FreeU diff: {diff_noop:.6f} (should be ~0)")
    assert diff_noop < 1e-5, "b=1.0, s=1.0 should be a no-op"

    logger.info("=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
