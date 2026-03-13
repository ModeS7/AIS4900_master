#!/usr/bin/env python3
"""Find optimal CFG scale for diffusion generation via grid search.

Classifier-Free Guidance (Ho & Salimans, 2022) steers generation toward
the conditioning signal by interpolating between unconditional and
conditional predictions:

    v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)

This script sweeps CFG scale values and evaluates FID/KID at each to find
the sweet spot. Supports both standard CFG and CFG-Zero* (Fan et al., 2025).

Requires a model trained WITH conditioning dropout (training.conditioning_dropout.prob > 0).

Usage:
    # Standard CFG sweep
    python -m medgen.scripts.find_optimal_cfg \
        --checkpoint runs/checkpoint_latest.pt \
        --data-root ~/MedicalDataSets/brainmetshare-3 \
        --num-volumes 25 --output-dir eval_cfg

    # CFG-Zero* sweep with custom scale range
    python -m medgen.scripts.find_optimal_cfg \
        --checkpoint runs/checkpoint_latest.pt \
        --data-root ~/MedicalDataSets/brainmetshare-3 \
        --cfg-mode zero_star \
        --cfg-values 1.0 1.5 2.0 2.5 3.0 3.5 4.0

    # Smoke test (no GPU needed)
    python -m medgen.scripts.find_optimal_cfg --smoke-test
"""
import argparse
import gc
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from medgen.data.utils import binarize_seg
from medgen.scripts.eval_ode_solvers import (
    compute_all_metrics,
    discover_splits,
    generate_noise_tensors,
    get_or_cache_reference_features,
    load_conditioning,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


# Default CFG scale values to sweep
DEFAULT_CFG_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]

# Default inference steps (Euler 25 is optimal for RFlow)
DEFAULT_NUM_STEPS = 25

SEG_MODES = frozenset({'seg', 'seg_conditioned', 'seg_conditioned_input'})


# ═══════════════════════════════════════════════════════════════════════════════
# CFG-aware volume generation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CFGResult:
    """Result for one CFG scale evaluation."""
    cfg_scale: float
    cfg_mode: str
    fid: float
    kid_mean: float
    kid_std: float
    cmmd: float
    fid_radimagenet: float
    kid_radimagenet_mean: float
    kid_radimagenet_std: float
    wall_time_s: float
    num_volumes: int


def generate_volumes_cfg(
    model: torch.nn.Module,
    strategy,
    noise_list: list[torch.Tensor],
    cond_list: list[tuple[str, torch.Tensor]],
    cfg_scale: float,
    cfg_mode: str,
    device: torch.device,
    num_steps: int = DEFAULT_NUM_STEPS,
    is_seg: bool = False,
    encode_cond_fn=None,
    decode_fn=None,
    latent_channels: int = 1,
    cfg_zero_init_steps: int = 1,
) -> tuple[list[np.ndarray], float]:
    """Generate volumes with classifier-free guidance.

    Unlike eval_ode_solvers.generate_volumes, this passes cfg_scale and
    cfg_mode through to strategy.generate() for guided sampling.

    Args:
        model: Diffusion model (trained with conditioning dropout).
        strategy: RFlowStrategy or DDPMStrategy instance.
        noise_list: Pre-generated noise tensors [1, C_out, D, H, W].
        cond_list: (patient_id, seg_tensor) pairs for conditioning.
        cfg_scale: Guidance scale (1.0 = no guidance).
        cfg_mode: 'standard' or 'zero_star'.
        device: CUDA device.
        num_steps: Euler steps for generation.
        is_seg: Threshold output for binary seg masks.
        encode_cond_fn: Optional space encoder for conditioning.
        decode_fn: Optional space decoder for output.
        latent_channels: Noise channels.
        cfg_zero_init_steps: Zero-velocity steps for CFG-Zero*.

    Returns:
        (volumes_list, wall_time_seconds)
    """
    # Configure solver
    if hasattr(strategy, 'ode_solver'):
        strategy.ode_solver = 'euler'

    volumes = []
    start_time = time.time()

    for i, noise in enumerate(noise_list):
        # Prepare conditioning
        seg_on_device = cond_list[i][1].to(device)
        if encode_cond_fn is not None:
            with torch.no_grad():
                seg_on_device = encode_cond_fn(seg_on_device)
        model_input = torch.cat([noise, seg_on_device], dim=1)

        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            with torch.no_grad():
                result = strategy.generate(
                    model, model_input, num_steps, device,
                    latent_channels=latent_channels,
                    cfg_scale=cfg_scale,
                    cfg_mode=cfg_mode,
                    cfg_zero_init_steps=cfg_zero_init_steps,
                )

        # Decode from latent/wavelet space to pixel space
        if decode_fn is not None:
            with torch.no_grad():
                result = decode_fn(result)

        if is_seg:
            vol = binarize_seg(result[0, 0])
        else:
            vol = torch.clamp(result[0, 0], 0, 1)
        volumes.append(vol.cpu().numpy())

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - start_time
            logger.info(f"    {i+1}/{len(noise_list)} volumes ({elapsed:.0f}s)")

    wall_time = time.time() - start_time
    return volumes, wall_time


# ═══════════════════════════════════════════════════════════════════════════════
# Grid search
# ═══════════════════════════════════════════════════════════════════════════════

def run_cfg_search(
    model: torch.nn.Module,
    strategy,
    noise_list: list[torch.Tensor],
    cond_list: list[tuple[str, torch.Tensor]],
    eval_ref: dict[str, dict[str, torch.Tensor]],
    ref_split: str,
    cfg_values: list[float],
    cfg_mode: str,
    device: torch.device,
    output_dir: Path,
    num_steps: int = DEFAULT_NUM_STEPS,
    is_seg: bool = False,
    encode_cond_fn=None,
    decode_fn=None,
    latent_channels: int = 1,
    metric: str = 'fid',
    trim_slices: int = 10,
    cfg_zero_init_steps: int = 1,
) -> list[CFGResult]:
    """Run grid search over CFG scale values.

    Returns list of CFGResult sorted by evaluation order.
    """
    results: list[CFGResult] = []

    for idx, cfg_scale in enumerate(cfg_values):
        tag = "NO GUIDANCE" if cfg_scale == 1.0 else f"cfg={cfg_scale:.2f}"

        logger.info(f"\n{'='*60}")
        logger.info(f"[{idx+1}/{len(cfg_values)}] Evaluating: {tag} ({cfg_mode})")
        logger.info(f"{'='*60}")

        t0 = time.time()

        volumes, wall_time = generate_volumes_cfg(
            model, strategy, noise_list, cond_list,
            cfg_scale=cfg_scale, cfg_mode=cfg_mode,
            device=device, num_steps=num_steps,
            is_seg=is_seg, encode_cond_fn=encode_cond_fn,
            decode_fn=decode_fn, latent_channels=latent_channels,
            cfg_zero_init_steps=cfg_zero_init_steps,
        )

        logger.info(f"  Generated {len(volumes)} volumes in {wall_time:.1f}s")

        # Compute metrics
        split_metrics = compute_all_metrics(volumes, eval_ref, device, trim_slices)
        ref_metrics = split_metrics.get(ref_split)
        if ref_metrics is None:
            raise ValueError(f"Reference split '{ref_split}' not in metrics")

        m = asdict(ref_metrics)

        result = CFGResult(
            cfg_scale=cfg_scale,
            cfg_mode=cfg_mode,
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
            f"  cfg={cfg_scale:.2f}: FID={result.fid:.2f}  KID={result.kid_mean:.6f}  "
            f"CMMD={result.cmmd:.6f}  FID_RIN={result.fid_radimagenet:.2f}  "
            f"KID_RIN={result.kid_radimagenet_mean:.6f}  ({elapsed:.0f}s)"
        )

        # Save incremental results
        _save_results(output_dir, results, cfg_values, cfg_mode, metric, ref_split)

        del volumes
        gc.collect()
        torch.cuda.empty_cache()

    return results


def _save_results(
    output_dir: Path,
    results: list[CFGResult],
    cfg_values: list[float],
    cfg_mode: str,
    metric: str,
    ref_split: str,
    best_scale: float | None = None,
) -> None:
    """Save search results to JSON."""
    data = {
        'cfg_values': cfg_values,
        'cfg_mode': cfg_mode,
        'metric': metric,
        'ref_split': ref_split,
        'best_cfg_scale': best_scale,
        'results': [asdict(r) for r in results],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "cfg_results.json", 'w') as f:
        json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Find optimal CFG scale via grid search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Model
    parser.add_argument('--checkpoint', default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--mode', default=None,
                        help='Generation mode (auto-detected from checkpoint if omitted)')
    parser.add_argument('--strategy', default=None, choices=['rflow', 'ddpm'],
                        help='Diffusion strategy (auto-detected from checkpoint)')

    # Space
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
    parser.add_argument('--output-dir', default='eval_cfg',
                        help='Output directory (default: eval_cfg)')
    parser.add_argument('--num-volumes', type=int, default=25,
                        help='Volumes per evaluation (default: 25)')

    # CFG search
    parser.add_argument('--cfg-values', type=float, nargs='+', default=DEFAULT_CFG_VALUES,
                        help=f'CFG scale values to search (default: {DEFAULT_CFG_VALUES})')
    parser.add_argument('--cfg-mode', default='standard', choices=['standard', 'zero_star'],
                        help='CFG mode (default: standard)')
    parser.add_argument('--cfg-zero-init-steps', type=int, default=1,
                        help='Zero-velocity init steps for CFG-Zero* (default: 1)')
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

    # Pixel normalization overrides
    parser.add_argument('--rescale', action='store_true',
                        help='Pixel rescale [-1, 1] -> [0, 1]')
    parser.add_argument('--pixel-shift', type=float, nargs='+', default=None)
    parser.add_argument('--pixel-scale', type=float, nargs='+', default=None)

    # DDPM-specific
    parser.add_argument('--prediction-type', default=None)
    parser.add_argument('--schedule', default=None)

    parser.add_argument('--smoke-test', action='store_true',
                        help='Smoke test without GPU')
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

    if cond_channels <= 0:
        parser.error(
            "CFG search requires a conditional model (in_channels > out_channels). "
            f"This model has in={base_in_channels}, out={base_out_channels}."
        )

    logger.info("=" * 70)
    logger.info("CFG Scale Optimization (Grid Search)")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Mode: {mode or 'unknown'} | Strategy: {strategy_name} | Space: {args.space}")
    logger.info(f"Channels: in={base_in_channels}, out={base_out_channels}, cond={cond_channels}")
    logger.info(f"Pixel volume: {pixel_image_size}x{pixel_image_size}x{pixel_depth}")
    logger.info(f"Euler steps: {args.num_steps}")
    logger.info(f"CFG mode: {args.cfg_mode}")
    logger.info(f"CFG scales: {args.cfg_values}")
    logger.info(f"Volumes per eval: {args.num_volumes}")
    logger.info(f"Metric: {args.metric} (vs '{args.ref_split}')")
    logger.info("=" * 70)

    # ── Data setup ────────────────────────────────────────────────────────
    splits = discover_splits(data_root, modality=ref_modality)

    if args.cond_split not in splits:
        raise ValueError(f"Split '{args.cond_split}' not found. Available: {list(splits.keys())}")
    logger.info(f"Loading {args.num_volumes} conditioning masks...")
    cond_list = load_conditioning(splits[args.cond_split], args.num_volumes, pixel_depth)

    # Resize conditioning H/W if needed
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

    results = run_cfg_search(
        model, strategy, noise_list, cond_list, eval_ref, args.ref_split,
        cfg_values=args.cfg_values, cfg_mode=args.cfg_mode,
        device=device, output_dir=output_dir,
        num_steps=args.num_steps, is_seg=is_seg,
        encode_cond_fn=encode_cond_fn, decode_fn=decode_fn,
        latent_channels=model_out_channels, metric=args.metric,
        trim_slices=args.trim_slices,
        cfg_zero_init_steps=args.cfg_zero_init_steps,
    )

    total_time = time.time() - total_start

    # ── Find best ─────────────────────────────────────────────────────────
    metric_key = {
        'fid': 'fid', 'kid': 'kid_mean', 'cmmd': 'cmmd',
        'fid_radimagenet': 'fid_radimagenet',
        'kid_radimagenet': 'kid_radimagenet_mean',
    }[args.metric]

    best = min(results, key=lambda r: getattr(r, metric_key))
    baseline = next((r for r in results if r.cfg_scale == 1.0), None)

    # Save final results
    _save_results(output_dir, results, args.cfg_values, args.cfg_mode,
                  args.metric, args.ref_split, best.cfg_scale)

    # ── Print results ─────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("RESULTS: CFG Scale Search")
    logger.info(f"{'='*70}")
    logger.info(f"CFG mode: {args.cfg_mode}")
    logger.info(f"Total scales evaluated: {len(results)}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")

    if baseline:
        logger.info(f"\nBaseline (cfg=1.0, no guidance): {args.metric}={getattr(baseline, metric_key):.4f}")

    logger.info(f"\nBest: cfg_scale={best.cfg_scale:.2f}")
    logger.info(f"  FID={best.fid:.2f}  KID={best.kid_mean:.6f}  CMMD={best.cmmd:.6f}")
    logger.info(f"  FID_RIN={best.fid_radimagenet:.2f}  KID_RIN={best.kid_radimagenet_mean:.6f}")

    if baseline and best.cfg_scale != 1.0:
        improvement = getattr(baseline, metric_key) - getattr(best, metric_key)
        pct = 100 * improvement / getattr(baseline, metric_key)
        logger.info(f"  Improvement over no-guidance: {improvement:+.4f} ({pct:+.1f}%)")

    # Print table
    logger.info(f"\n{'CFG':>8}  {'FID':>8}  {'KID':>10}  {'FID_RIN':>8}  "
                f"{'KID_RIN':>10}  {'CMMD':>10}  {'Time':>7}")
    logger.info(f"  {'-'*68}")
    for r in sorted(results, key=lambda x: x.cfg_scale):
        marker = " <--" if r.cfg_scale == best.cfg_scale else ""
        logger.info(
            f"{r.cfg_scale:>8.2f}  {r.fid:>8.2f}  {r.kid_mean:>10.6f}  "
            f"{r.fid_radimagenet:>8.2f}  {r.kid_radimagenet_mean:>10.6f}  "
            f"{r.cmmd:>10.6f}  {r.wall_time_s:>6.1f}s{marker}"
        )

    logger.info(f"\nResults saved to: {output_dir}/cfg_results.json")


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _run_smoke_test():
    """Verify CFG search logic with a tiny model."""
    logger.info("=== SMOKE TEST (CFG search) ===")

    from monai.networks.nets import DiffusionModelUNet

    # Create tiny 2D conditional UNet (in=2, out=1)
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=2,
        out_channels=1,
        channels=[32, 64],
        attention_levels=[False, True],
        num_head_channels=32,
        num_res_blocks=1,
        norm_num_groups=32,
    )
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)

    x = torch.randn(1, 2, 32, 32)  # noise + conditioning
    t = torch.tensor([500.0])

    # Verify model runs with different inputs
    with torch.no_grad():
        out = model(x, t)
    logger.info(f"Model output shape: {out.shape}")
    logger.info(f"Output stats: mean={out.mean():.6f}, std={out.std():.6f}")

    # Verify CFGResult dataclass
    result = CFGResult(
        cfg_scale=2.0, cfg_mode='zero_star',
        fid=25.0, kid_mean=0.01, kid_std=0.001,
        cmmd=0.5, fid_radimagenet=30.0,
        kid_radimagenet_mean=0.02, kid_radimagenet_std=0.002,
        wall_time_s=10.0, num_volumes=25,
    )
    d = asdict(result)
    assert d['cfg_scale'] == 2.0
    assert d['cfg_mode'] == 'zero_star'

    logger.info("=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
