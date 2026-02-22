#!/usr/bin/env python3
"""Find the optimal step count for diffusion generation via golden section search.

Uses golden section search to find the step count that minimizes FID
within a given range. Much faster than exhaustive sweep — typically
8-10 evaluations to find the optimum within ±1 step.

Golden section search: at each iteration, evaluates one new point and
narrows the interval by factor ~0.618. Requires the objective to be
unimodal (single minimum) in the search range.

Supports:
- Pixel-space models (UNet, default)
- Latent-space models (LDM with VAE/VQ-VAE/DC-AE decoder)
- Wavelet-space models (WDM with inverse Haar DWT)
- UNet and DiT architectures (auto-detected from checkpoint)
- RFlow and DDPM strategies (auto-detected from checkpoint)

Usage:
    # Pixel-space model (default)
    python -m medgen.scripts.find_optimal_steps \
        --checkpoint runs/checkpoint_latest.pt \
        --data-root ~/MedicalDataSets/brainmetshare-3 \
        --num-volumes 25 --output-dir eval_optimal_steps

    # Latent-space model (LDM)
    python -m medgen.scripts.find_optimal_steps \
        --checkpoint runs/ldm_checkpoint.pt \
        --data-root ~/MedicalDataSets/brainmetshare-3 \
        --space latent \
        --compression-checkpoint runs/vqvae_checkpoint.pt \
        --compression-type vqvae

    # Wavelet-space model (WDM)
    python -m medgen.scripts.find_optimal_steps \
        --checkpoint runs/wdm_checkpoint.pt \
        --data-root ~/MedicalDataSets/brainmetshare-3 \
        --space wavelet

    # Smoke test (no checkpoint needed)
    python -m medgen.scripts.find_optimal_steps --smoke-test
"""
import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch

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

GR = (math.sqrt(5) - 1) / 2  # Golden ratio conjugate ≈ 0.618


# ═══════════════════════════════════════════════════════════════════════════════
# Golden section search
# ═══════════════════════════════════════════════════════════════════════════════

def golden_section_search(
    evaluate_fn,
    lo: int,
    hi: int,
    tol: int = 1,
) -> tuple[int, float, dict[int, float]]:
    """Find integer minimizer of evaluate_fn on [lo, hi] via golden section.

    Args:
        evaluate_fn: Callable(int) -> float. Evaluates the objective at a step count.
        lo: Lower bound (inclusive).
        hi: Upper bound (inclusive).
        tol: Stop when hi - lo <= tol.

    Returns:
        (best_steps, best_fid, all_evaluations) where all_evaluations maps
        steps -> fid for every point evaluated.
    """
    cache: dict[int, float] = {}

    def eval_cached(n: int) -> float:
        if n not in cache:
            cache[n] = evaluate_fn(n)
        else:
            logger.info(f"  [cached] steps={n} -> FID={cache[n]:.2f}")
        return cache[n]

    a, b = lo, hi
    # Initial interior points
    x1 = round(b - GR * (b - a))
    x2 = round(a + GR * (b - a))

    f1 = eval_cached(x1)
    f2 = eval_cached(x2)

    iteration = 0
    prev_ab = None
    while b - a > tol:
        # Detect infinite loop: if interval hasn't changed, stop
        if (a, b) == prev_ab:
            logger.info(f"  Interval unchanged at [{a}, {b}], stopping.")
            break
        prev_ab = (a, b)

        iteration += 1
        logger.info(f"\n--- Iteration {iteration}: [{a}, {b}] (width={b-a}) ---")
        logger.info(f"  x1={x1} (FID={f1:.2f}), x2={x2} (FID={f2:.2f})")

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = round(b - GR * (b - a))
            # Handle degenerate case where rounding makes x1 == x2
            if x1 == x2:
                x1 = max(a, x1 - 1)
            if x1 == x2:
                break
            f1 = eval_cached(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = round(a + GR * (b - a))
            if x2 == x1:
                x2 = min(b, x2 + 1)
            if x2 == x1:
                break
            f2 = eval_cached(x2)

    # Find best among all evaluated points
    best_steps = min(cache, key=cache.get)
    return best_steps, cache[best_steps], cache


# ═══════════════════════════════════════════════════════════════════════════════
# Space setup helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_latent_space(
    args,
    ckpt_cfg: dict,
    device: torch.device,
    spatial_dims: int,
):
    """Load compression model and create LatentSpace for LDM decode.

    Reads latent normalization stats from checkpoint config (saved by
    profiling.py:get_model_config — exact match to training).

    Returns:
        (space, latent_channels, scale_factor, depth_scale_factor)
    """
    from medgen.data.loaders.latent import load_compression_model
    from medgen.diffusion.spaces import LatentSpace

    if not args.compression_checkpoint:
        raise ValueError("--compression-checkpoint required for --space latent")

    compression_type = args.compression_type or 'auto'
    logger.info(f"Loading compression model: {args.compression_checkpoint}")
    comp_model, detected_type, comp_spatial_dims, scale_factor, latent_channels = (
        load_compression_model(
            args.compression_checkpoint,
            compression_type,
            device,
            spatial_dims=spatial_dims,
        )
    )
    logger.info(
        f"  type={detected_type}, scale={scale_factor}x, "
        f"latent_ch={latent_channels}, spatial_dims={comp_spatial_dims}"
    )

    # Read latent normalization stats from checkpoint config
    latent_cfg = ckpt_cfg.get('latent', {})
    latent_shift = latent_cfg.get('latent_shift')
    latent_scale = latent_cfg.get('latent_scale')
    latent_seg_shift = latent_cfg.get('latent_seg_shift')
    latent_seg_scale = latent_cfg.get('latent_seg_scale')

    if latent_shift is not None:
        logger.info(f"  Latent normalization: shift={latent_shift}, scale={latent_scale}")
        if latent_seg_shift is not None:
            logger.info(f"  Seg normalization: shift={latent_seg_shift}, scale={latent_seg_scale}")
    else:
        raise ValueError(
            "Checkpoint has no latent normalization stats (missing config.latent.latent_shift). "
            "Retrain with the fixed profiling.py that saves stats in the checkpoint."
        )

    # Slicewise encoding: 2D compression model applied slice-by-slice to 3D volumes
    slicewise = (comp_spatial_dims == 2 and spatial_dims == 3)

    space = LatentSpace(
        compression_model=comp_model,
        device=device,
        deterministic=True,
        spatial_dims=comp_spatial_dims,
        compression_type=detected_type,
        scale_factor=scale_factor,
        latent_channels=latent_channels,
        slicewise_encoding=slicewise,
        latent_shift=latent_shift,
        latent_scale=latent_scale,
        latent_seg_shift=latent_seg_shift,
        latent_seg_scale=latent_seg_scale,
    )

    depth_sf = 1 if slicewise else scale_factor
    return space, latent_channels, scale_factor, depth_sf


def _setup_wavelet_space(
    args,
    ckpt_cfg: dict,
):
    """Create WaveletSpace from checkpoint stats.

    Reads wavelet normalization stats from checkpoint config (saved by
    profiling.py:get_model_config — exact match to training).

    Returns:
        (space, channels=8, scale_factor=2, depth_scale_factor=2)
    """
    from medgen.diffusion.spaces import WaveletSpace

    wavelet_cfg = ckpt_cfg.get('wavelet', {})
    shift = wavelet_cfg.get('wavelet_shift')
    scale = wavelet_cfg.get('wavelet_scale')
    rescale = wavelet_cfg.get('rescale', False)

    if shift is not None and scale is not None:
        logger.info("Using wavelet stats from checkpoint")
        space = WaveletSpace(shift=shift, scale=scale, rescale=rescale)
        names = ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']
        for i, name in enumerate(names):
            logger.info(f"  {name}: shift={shift[i]:.4f}, scale={scale[i]:.4f}")
        if rescale:
            logger.info("  rescale: [-1, 1]")
    else:
        raise ValueError(
            "Checkpoint has no wavelet normalization stats (missing config.wavelet.wavelet_shift). "
            "Retrain with the fixed profiling.py that saves stats in the checkpoint."
        )

    return space, 8, 2, 2



# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

SEG_MODES = frozenset({'seg', 'seg_conditioned', 'seg_conditioned_input'})


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal step count via golden section search",
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
                        help='Reference file modality, e.g. bravo or seg '
                             '(auto: bravo for image modes, seg for seg modes)')
    parser.add_argument('--data-root', default=None,
                        help='Root of dataset')
    parser.add_argument('--output-dir', default='eval_optimal_steps',
                        help='Output directory (default: eval_optimal_steps)')
    parser.add_argument('--num-volumes', type=int, default=25,
                        help='Volumes per evaluation (default: 25)')

    # Search params
    parser.add_argument('--lo', type=int, default=10,
                        help='Lower bound for step count search (default: 10)')
    parser.add_argument('--hi', type=int, default=50,
                        help='Upper bound for step count search (default: 50)')
    parser.add_argument('--tol', type=int, default=1,
                        help='Stop when interval width <= tol (default: 1)')

    # Volume config
    parser.add_argument('--cond-split', default='val',
                        help='Split for conditioning seg masks (default: val)')
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--trim-slices', type=int, default=10)
    parser.add_argument('--fov-mm', type=float, default=240.0)
    parser.add_argument('--seed', type=int, default=42)

    # Metric
    parser.add_argument('--metric', choices=['fid', 'kid', 'cmmd'], default='fid',
                        help='Metric to minimize (default: fid)')
    parser.add_argument('--ref-split', default='all',
                        help='Reference split for metric computation (default: all)')
    parser.add_argument('--save-volumes', action='store_true',
                        help='Save generated volumes as NIfTI (off by default to save disk)')

    # DDPM-specific
    parser.add_argument('--prediction-type', default=None,
                        help='DDPM prediction type: epsilon/sample (auto-detected)')
    parser.add_argument('--schedule', default=None,
                        help='DDPM noise schedule (auto-detected)')

    parser.add_argument('--smoke-test', action='store_true',
                        help='Smoke test with tiny dummy model')
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test(args)
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

    # New-style plain dict config (from profiling.py:get_model_config)
    base_in_channels = ckpt_cfg.get('in_channels')
    base_out_channels = ckpt_cfg.get('out_channels')
    mode = args.mode or ckpt_cfg.get('mode')
    if isinstance(mode, dict):
        mode = mode.get('name', None)
    spatial_dims = ckpt_cfg.get('spatial_dims', 3)

    # For DiT/transformer models, checkpoint 'image_size' is the model config
    # default (e.g. 32 for dit_3d), NOT the pixel size. Use 256 as default.
    model_type = ckpt_cfg.get('model_type', 'unet')
    is_transformer = model_type in ('dit', 'sit', 'uvit', 'hdit')
    if is_transformer:
        pixel_image_size = args.image_size or 256
        pixel_depth = args.depth or 160
    else:
        pixel_image_size = args.image_size or ckpt_cfg.get('image_size', 256)
        pixel_depth = args.depth or ckpt_cfg.get('depth_size', 160)

    # Auto-detect strategy from checkpoint
    strategy_name = args.strategy or ckpt_cfg.get('strategy', 'rflow')

    # Legacy Hydra config fallback
    if base_in_channels is None and hasattr(ckpt_cfg, 'model'):
        model_cfg = ckpt_cfg.model
        base_in_channels = getattr(model_cfg, 'in_channels', 2)
        base_out_channels = getattr(model_cfg, 'out_channels', 1)
        pixel_image_size = args.image_size or getattr(model_cfg, 'image_size', 256)
        pixel_depth = args.depth or getattr(model_cfg, 'depth_size', 160)

    if base_in_channels is None:
        base_in_channels = 2
    if base_out_channels is None:
        base_out_channels = 1
    del ckpt

    # ── Setup diffusion space ─────────────────────────────────────────────
    encode_cond_fn = None
    decode_fn = None
    data_root = Path(args.data_root)

    if args.space == 'latent':
        space, latent_ch, sf, depth_sf = _setup_latent_space(args, ckpt_cfg, device, spatial_dims)
        # Model operates in latent space
        model_out_channels = base_out_channels * latent_ch
        model_in_channels = base_in_channels * latent_ch
        noise_image_size = pixel_image_size // sf
        noise_depth = pixel_depth // depth_sf
        # Conditioning is seg masks → use seg-specific normalization stats
        # (matches LatentDataset.__getitem__ which normalizes latent_seg separately)
        encode_cond_fn = space.encode_normalized_seg
        decode_fn = space.decode
        logger.info(
            f"Latent space: {sf}x compression, {latent_ch} latent channels, "
            f"noise shape: {model_out_channels}x{noise_depth}x{noise_image_size}x{noise_image_size}"
        )
    elif args.space == 'wavelet':
        space, wav_ch, sf, depth_sf = _setup_wavelet_space(args, ckpt_cfg)
        model_out_channels = base_out_channels * wav_ch
        model_in_channels = base_in_channels * wav_ch
        noise_image_size = pixel_image_size // sf
        noise_depth = pixel_depth // depth_sf
        encode_cond_fn = space.encode
        decode_fn = space.decode
        logger.info(
            f"Wavelet space: 2x Haar DWT, 8 subbands, "
            f"noise shape: {model_out_channels}x{noise_depth}x{noise_image_size}x{noise_image_size}"
        )
    else:
        # Pixel space — channels as-is, optionally with brain-only normalization
        model_in_channels = base_in_channels
        model_out_channels = base_out_channels
        noise_image_size = pixel_image_size
        noise_depth = pixel_depth

        pixel_cfg = ckpt_cfg.get('pixel', {})
        pixel_shift = pixel_cfg.get('pixel_shift')
        pixel_scale = pixel_cfg.get('pixel_scale')
        if pixel_shift is not None:
            from medgen.diffusion.spaces import PixelSpace
            space = PixelSpace(shift=pixel_shift, scale=pixel_scale)
            decode_fn = space.decode
            encode_cond_fn = space.encode
            logger.info(f"Pixel normalization: shift={pixel_shift}, scale={pixel_scale}")

    # ── Derive mode-specific config ───────────────────────────────────────
    is_seg = mode in SEG_MODES if mode else False
    ref_modality = args.ref_modality or ('seg' if is_seg else 'bravo')
    cond_channels = base_in_channels - base_out_channels  # pixel-level conditioning check

    voxel_size = (args.fov_mm / pixel_image_size, args.fov_mm / pixel_image_size, 1.0)

    logger.info("=" * 70)
    logger.info("Optimal Step Search (Golden Section)")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Mode: {mode or 'unknown'} | Strategy: {strategy_name} | Space: {args.space}")
    logger.info(f"Base channels: in={base_in_channels}, out={base_out_channels}, cond={cond_channels}")
    if args.space != 'pixel':
        logger.info(f"Model channels: in={model_in_channels}, out={model_out_channels}")
    logger.info(f"Reference modality: {ref_modality}")
    logger.info(f"Pixel volume: {pixel_image_size}x{pixel_image_size}x{pixel_depth}")
    if args.space != 'pixel':
        logger.info(f"Model volume: {noise_image_size}x{noise_image_size}x{noise_depth}")
    logger.info(f"Search range: [{args.lo}, {args.hi}]")
    logger.info(f"Volumes per eval: {args.num_volumes}")
    logger.info(f"Metric: {args.metric} (vs '{args.ref_split}')")
    logger.info(f"Expected evaluations: ~{int(math.log(args.hi - args.lo) / math.log(1/GR)) + 1}")
    logger.info("=" * 70)

    # ── Data setup ────────────────────────────────────────────────────────
    splits = discover_splits(data_root, modality=ref_modality)

    # ── Load conditioning (only if model expects it) ──────────────────────
    cond_list = None
    if cond_channels > 0:
        if args.cond_split not in splits:
            raise ValueError(
                f"Split '{args.cond_split}' not found. Available: {list(splits.keys())}"
            )
        logger.info(f"Loading {args.num_volumes} conditioning masks...")
        # Load at PIXEL depth — encode_cond_fn handles space conversion
        cond_list = load_conditioning(splits[args.cond_split], args.num_volumes, pixel_depth)

        # Resize conditioning H/W to match pixel_image_size if needed
        # (NIfTI files may be 256x256 but model trains at smaller resolution, e.g. 128x128)
        sample_hw = cond_list[0][1].shape[-1]  # [1, 1, D, H, W] -> W
        if sample_hw != pixel_image_size:
            from torch.nn.functional import interpolate
            logger.info(f"Resizing conditioning from {sample_hw}x{sample_hw} to {pixel_image_size}x{pixel_image_size}")
            cond_list = [
                (pid, interpolate(
                    seg, size=(pixel_depth, pixel_image_size, pixel_image_size),
                    mode='nearest',
                ))
                for pid, seg in cond_list
            ]

    logger.info("Preparing reference features...")
    cache_dir = output_dir / "reference_features"
    ref_features = get_or_cache_reference_features(
        splits, cache_dir, device, pixel_depth, args.trim_slices, pixel_image_size,
        modality=ref_modality,
    )

    # ── Load model ────────────────────────────────────────────────────────
    from medgen.diffusion import load_diffusion_model

    logger.info("Loading model...")
    # For non-pixel spaces, pass actual model channels (not mode-level)
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
        scheduler_kwargs = dict(
            num_timesteps=1000,
            image_size=noise_image_size,
            depth_size=noise_depth,
            spatial_dims=spatial_dims,
            prediction_type=prediction_type,
            schedule=schedule,
        )
        # Disable sample clipping for non-pixel spaces (wavelet/latent values
        # are not bounded to [-1, 1])
        if args.space != 'pixel':
            scheduler_kwargs['clip_sample'] = False
        strategy.setup_scheduler(**scheduler_kwargs)
        logger.info(f"Strategy: DDPM (prediction={prediction_type}, schedule={schedule})")
    else:
        from medgen.diffusion import RFlowStrategy
        strategy = RFlowStrategy()
        strategy.setup_scheduler(
            num_timesteps=1000, image_size=noise_image_size,
            depth_size=noise_depth, spatial_dims=spatial_dims,
        )
        logger.info("Strategy: RFlow")

    # ── Pre-generate noise (shared across all evaluations) ────────────────
    logger.info(f"Pre-generating {args.num_volumes} noise tensors (seed={args.seed})...")
    noise_list = generate_noise_tensors(
        args.num_volumes, noise_depth, noise_image_size, device, args.seed,
        out_channels=model_out_channels,
    )

    # ── Search history for logging ────────────────────────────────────────
    history = []

    def evaluate_steps(num_steps: int) -> float:
        """Generate volumes with euler/N and return the target metric."""
        solver_cfg = SolverConfig(solver='euler', steps=num_steps)
        logger.info(f"\n  Evaluating euler/{num_steps} ...")

        t0 = time.time()
        volumes, total_nfe, wall_time = generate_volumes(
            model, strategy, noise_list, cond_list, solver_cfg, device,
            is_seg=is_seg,
            encode_cond_fn=encode_cond_fn,
            decode_fn=decode_fn,
            latent_channels=model_out_channels,
        )
        logger.info(f"  Generated {len(volumes)} volumes in {wall_time:.1f}s "
                     f"(NFE={total_nfe}, {total_nfe/args.num_volumes:.0f}/vol)")

        # Save volumes (optional — skipped by default to save disk)
        if args.save_volumes:
            vol_dir = output_dir / "generated" / f"euler_steps{num_steps:03d}"
            save_volumes(
                volumes, cond_list, vol_dir, voxel_size, args.trim_slices,
                modality=ref_modality,
            )

        # Compute metrics (always in pixel space)
        split_metrics = compute_all_metrics(volumes, ref_features, device, args.trim_slices)

        ref_metrics = split_metrics.get(args.ref_split)
        if ref_metrics is None:
            raise ValueError(f"Reference split '{args.ref_split}' not in metrics")

        from dataclasses import asdict
        m = asdict(ref_metrics)
        fid = m['fid']
        kid = m['kid_mean']
        cmmd = m['cmmd']

        target = {'fid': fid, 'kid': kid, 'cmmd': cmmd}[args.metric]

        elapsed = time.time() - t0
        logger.info(f"  euler/{num_steps}: FID={fid:.2f}  KID={kid:.6f}  "
                     f"CMMD={cmmd:.6f}  ({elapsed:.0f}s)")

        history.append({
            'steps': num_steps,
            'fid': fid,
            'kid_mean': kid,
            'kid_std': m['kid_std'],
            'cmmd': cmmd,
            'wall_time_s': wall_time,
            'eval_time_s': elapsed,
        })

        # Save incremental results
        _save_history(output_dir, history, args)

        del volumes
        torch.cuda.empty_cache()

        return target

    # ── Run golden section search ─────────────────────────────────────────
    total_start = time.time()

    best_steps, best_val, all_evals = golden_section_search(
        evaluate_steps, args.lo, args.hi, tol=args.tol,
    )

    total_time = time.time() - total_start

    # ── Final report ──────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 70}")
    logger.info(f"RESULT: Optimal step count = {best_steps}")
    logger.info(f"  {args.metric.upper()} = {best_val:.4f}")
    logger.info(f"  Total evaluations: {len(all_evals)}")
    logger.info(f"  Total time: {total_time/60:.1f} minutes")
    logger.info(f"{'=' * 70}")

    # Print all evaluated points sorted by steps
    logger.info("\nAll evaluations (sorted by steps):")
    logger.info(f"  {'Steps':>5}  {args.metric.upper():>10}  {'Best':>5}")
    logger.info(f"  {'-'*25}")
    for steps in sorted(all_evals.keys()):
        marker = " <--" if steps == best_steps else ""
        logger.info(f"  {steps:>5}  {all_evals[steps]:>10.4f}{marker}")

    # Print full history with all metrics
    logger.info("\nFull history:")
    logger.info(f"  {'Steps':>5}  {'FID':>8}  {'KID':>10}  {'CMMD':>10}  {'Time':>7}")
    logger.info(f"  {'-'*48}")
    for h in sorted(history, key=lambda x: x['steps']):
        marker = " <--" if h['steps'] == best_steps else ""
        logger.info(f"  {h['steps']:>5}  {h['fid']:>8.2f}  {h['kid_mean']:>10.6f}  "
                     f"{h['cmmd']:>10.6f}  {h['wall_time_s']:>6.1f}s{marker}")

    _save_history(output_dir, history, args, best_steps=best_steps)


def _save_history(
    output_dir: Path,
    history: list[dict],
    args,
    best_steps: int | None = None,
) -> None:
    """Save search history to JSON."""
    data = {
        'checkpoint': getattr(args, 'checkpoint', None),
        'mode': getattr(args, 'mode', None),
        'space': getattr(args, 'space', 'pixel'),
        'strategy': getattr(args, 'strategy', None),
        'search_range': [args.lo, args.hi],
        'metric': args.metric,
        'ref_split': args.ref_split,
        'num_volumes': args.num_volumes,
        'seed': args.seed,
        'best_steps': best_steps,
        'evaluations': sorted(history, key=lambda x: x['steps']),
    }
    with open(output_dir / "search_results.json", 'w') as f:
        json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _run_smoke_test(args) -> None:
    """Verify golden section search logic with a synthetic function."""
    logger.info("=== SMOKE TEST (no GPU needed) ===")

    # Synthetic unimodal function: f(x) = (x - 23)^2 + 50
    # Minimum at x=23
    call_count = 0

    def fake_evaluate(n: int) -> float:
        nonlocal call_count
        call_count += 1
        val = (n - 23) ** 2 + 50.0
        logger.info(f"  eval({n}) = {val:.1f}  [call #{call_count}]")
        return val

    best, best_val, all_evals = golden_section_search(
        fake_evaluate, lo=10, hi=50, tol=1,
    )

    logger.info(f"\nResult: optimal={best}, f(optimal)={best_val:.1f}")
    logger.info(f"Evaluations: {len(all_evals)} (expected ~9)")
    logger.info(f"All points: {sorted(all_evals.items())}")

    assert best == 23, f"Expected 23, got {best}"
    assert best_val == 50.0, f"Expected 50.0, got {best_val}"
    assert len(all_evals) <= 12, f"Too many evaluations: {len(all_evals)}"

    logger.info("=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
