#!/usr/bin/env python3
"""Evaluate time-shifted sampling schedules at inference for RFlow models.

SD3's Time-Shifted Sampler (from "Scaling Rectified Flow Transformers for
High-Resolution Image Synthesis") shifts the timestep schedule based on
resolution ratio, biasing sampling toward higher noise levels. MONAI's
RFlowScheduler implements this via timestep_transform(), but by default
base_img_size_numel = input_size → ratio=1.0 → no shift.

The shift formula:
    new_t = ratio * t / (1 + (ratio - 1) * t)
    ratio = (input_numel / base_numel)^(1/spatial_dim)

This script sweeps different base_img_size_numel values to test shift ratios
at inference, using 25 Euler steps (configurable), and compares generation
quality (FID/KID/CMMD) against no shift.

Usage:
    # Full evaluation
    python -m medgen.scripts.eval_time_shift \\
        --checkpoint runs/checkpoint_bravo.pt \\
        --data-root ~/MedicalDataSets/brainmetshare-3 \\
        --num-volumes 25 --output-dir eval_time_shift

    # Custom ratios
    python -m medgen.scripts.eval_time_shift \\
        --checkpoint runs/checkpoint_bravo.pt \\
        --data-root ~/MedicalDataSets/brainmetshare-3 \\
        --ratios 1.0 1.5 2.0 3.0

    # Smoke test (no checkpoint needed)
    python -m medgen.scripts.eval_time_shift --smoke-test
"""
import argparse
import gc
import json
import logging
import time
from dataclasses import asdict
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
from medgen.scripts.find_optimal_steps import (
    SEG_MODES,
    _setup_latent_space,
    _setup_wavelet_space,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


# Default shift ratios to evaluate
DEFAULT_RATIOS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.84]


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate time-shifted sampling schedules for RFlow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Model
    parser.add_argument('--checkpoint', '--bravo-model', default=None, dest='checkpoint',
                        help='Path to trained model checkpoint')
    parser.add_argument('--mode', default=None,
                        help='Generation mode (auto-detected from checkpoint if omitted)')

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
    parser.add_argument('--output-dir', default='eval_time_shift',
                        help='Output directory (default: eval_time_shift)')
    parser.add_argument('--num-volumes', type=int, default=25,
                        help='Volumes per evaluation (default: 25)')

    # Time shift params
    parser.add_argument('--ratios', type=float, nargs='+', default=DEFAULT_RATIOS,
                        help=f'Shift ratios to test (default: {DEFAULT_RATIOS})')
    parser.add_argument('--num-steps', type=int, default=25,
                        help='Number of Euler steps (default: 25)')

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
                        help='Primary metric for comparison (default: fid)')
    parser.add_argument('--ref-split', default='test',
                        help='Reference split for metric computation (default: test)')
    parser.add_argument('--save-volumes', action='store_true',
                        help='Save generated volumes as NIfTI (off by default)')

    # Pixel normalization overrides
    parser.add_argument('--rescale', action='store_true',
                        help='Pixel rescale [-1, 1] -> [0, 1] (override checkpoint config)')
    parser.add_argument('--pixel-shift', type=float, nargs='+', default=None,
                        help='Pixel shift value(s) for brain-only normalization')
    parser.add_argument('--pixel-scale', type=float, nargs='+', default=None,
                        help='Pixel scale value(s) for brain-only normalization')

    parser.add_argument('--smoke-test', action='store_true',
                        help='Smoke test with synthetic function')
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

    # ── Derive mode-specific config ───────────────────────────────────────
    is_seg = mode in SEG_MODES if mode else False
    ref_modality = args.ref_modality or ('seg' if is_seg else 'bravo')
    cond_channels = base_in_channels - base_out_channels

    voxel_size = (args.fov_mm / pixel_image_size, args.fov_mm / pixel_image_size, 1.0)

    # Compute input numel (in model space, not pixel space)
    if spatial_dims == 3:
        input_numel = noise_image_size * noise_image_size * noise_depth
    else:
        input_numel = noise_image_size * noise_image_size

    logger.info("=" * 70)
    logger.info("Time-Shifted Sampling Evaluation")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Mode: {mode or 'unknown'} | Space: {args.space}")
    logger.info(f"Spatial dims: {spatial_dims}")
    logger.info(f"Base channels: in={base_in_channels}, out={base_out_channels}")
    logger.info(f"Pixel volume: {pixel_image_size}x{pixel_image_size}x{pixel_depth}")
    if args.space != 'pixel':
        logger.info(f"Model volume: {noise_image_size}x{noise_image_size}x{noise_depth}")
    logger.info(f"Input numel: {input_numel:,}")
    logger.info(f"Euler steps: {args.num_steps}")
    logger.info(f"Ratios to test: {args.ratios}")
    logger.info(f"Volumes per eval: {args.num_volumes}")
    logger.info(f"Metric: {args.metric} (vs '{args.ref_split}')")
    logger.info("=" * 70)

    # ── Data setup ────────────────────────────────────────────────────────
    splits = discover_splits(data_root, modality=ref_modality)

    # ── Load conditioning ─────────────────────────────────────────────────
    cond_list = None
    if cond_channels > 0:
        if args.cond_split not in splits:
            raise ValueError(
                f"Split '{args.cond_split}' not found. Available: {list(splits.keys())}"
            )
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
        logger.info("Building combined 'all' reference features...")
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
    from medgen.diffusion import RFlowStrategy, load_diffusion_model

    logger.info("Loading model...")
    model = load_diffusion_model(
        args.checkpoint, device=device,
        in_channels=model_in_channels, out_channels=model_out_channels,
        compile_model=False, spatial_dims=spatial_dims,
    )

    # ── Setup RFlow strategy ──────────────────────────────────────────────
    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000, image_size=noise_image_size,
        depth_size=noise_depth, spatial_dims=spatial_dims,
    )
    logger.info("Strategy: RFlow")

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

    # ── Evaluate each shift ratio ─────────────────────────────────────────
    solver_cfg = SolverConfig(solver='euler', steps=args.num_steps)
    history = []
    total_start = time.time()

    for ratio in args.ratios:
        # Compute base_numel from ratio: ratio = (input_numel / base_numel)^(1/spatial_dims)
        # → base_numel = input_numel / ratio^spatial_dims
        base_numel = int(input_numel / (ratio ** spatial_dims))
        # Ensure base_numel >= 1
        base_numel = max(1, base_numel)

        # Set the scheduler's base_img_size_numel to induce the desired ratio
        strategy.scheduler.base_img_size_numel = base_numel

        ratio_label = f"ratio_{ratio:.2f}"
        if ratio == 1.0:
            ratio_label = "no_shift"
        elif abs(ratio - 6.84) < 0.1:
            ratio_label = "monai_default"

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {ratio_label} (ratio={ratio:.2f}, base_numel={base_numel:,})")
        logger.info(f"{'='*60}")

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

        if args.save_volumes:
            vol_dir = output_dir / "generated" / ratio_label
            save_volumes(
                volumes, cond_list, vol_dir, voxel_size, args.trim_slices,
                modality=ref_modality,
            )

        split_metrics = compute_all_metrics(volumes, eval_ref, device, args.trim_slices)
        ref_metrics = split_metrics.get(args.ref_split)
        if ref_metrics is None:
            raise ValueError(f"Reference split '{args.ref_split}' not in metrics")

        m = asdict(ref_metrics)
        fid = m['fid']
        kid = m['kid_mean']
        cmmd = m['cmmd']
        fid_rin = m['fid_radimagenet']
        kid_rin = m['kid_radimagenet_mean']

        elapsed = time.time() - t0
        logger.info(
            f"  {ratio_label}: FID={fid:.2f}  KID={kid:.6f}  CMMD={cmmd:.6f}  "
            f"FID_RIN={fid_rin:.2f}  KID_RIN={kid_rin:.6f}  ({elapsed:.0f}s)"
        )

        history.append({
            'ratio': ratio,
            'label': ratio_label,
            'base_numel': base_numel,
            'fid': fid,
            'kid_mean': kid,
            'kid_std': m['kid_std'],
            'cmmd': cmmd,
            'fid_radimagenet': fid_rin,
            'kid_radimagenet_mean': kid_rin,
            'kid_radimagenet_std': m['kid_radimagenet_std'],
            'wall_time_s': wall_time,
            'eval_time_s': elapsed,
        })

        # Save incremental results
        _save_results(output_dir, history, args, input_numel)

        del volumes
        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # ── Final report ──────────────────────────────────────────────────────
    _print_comparison_table(history, args)

    logger.info(f"\nTotal time: {total_time/60:.1f} minutes")
    logger.info(f"Results saved to: {output_dir / 'time_shift_results.json'}")


def _print_comparison_table(history: list[dict], args) -> None:
    """Print formatted comparison table."""
    metric_key = {
        'fid': 'fid', 'kid': 'kid_mean', 'cmmd': 'cmmd',
        'fid_radimagenet': 'fid_radimagenet',
        'kid_radimagenet': 'kid_radimagenet_mean',
    }[args.metric]

    best_entry = min(history, key=lambda h: h[metric_key])

    logger.info(f"\n{'='*100}")
    logger.info(f"Time-Shift Evaluation Results (vs '{args.ref_split}', {args.num_steps} Euler steps)")
    logger.info(f"{'='*100}")
    logger.info(
        f"  {'Label':<18} {'Ratio':>6} {'Base Numel':>14} "
        f"{'FID':>8} {'KID':>10} {'FID_RIN':>8} {'KID_RIN':>10} {'CMMD':>10} {'Time':>7}"
    )
    logger.info(f"  {'-'*95}")

    for h in history:
        marker = " <--" if h['ratio'] == best_entry['ratio'] else ""
        logger.info(
            f"  {h['label']:<18} {h['ratio']:>6.2f} {h['base_numel']:>14,} "
            f"{h['fid']:>8.2f} {h['kid_mean']:>10.6f} "
            f"{h.get('fid_radimagenet', 0):>8.2f} "
            f"{h.get('kid_radimagenet_mean', 0):>10.6f} "
            f"{h['cmmd']:>10.6f} {h['wall_time_s']:>6.1f}s{marker}"
        )

    logger.info(f"{'='*100}")
    logger.info(
        f"Best by {args.metric.upper()}: {best_entry['label']} "
        f"(ratio={best_entry['ratio']:.2f}, {args.metric}={best_entry[metric_key]:.4f})"
    )


def _save_results(
    output_dir: Path,
    history: list[dict],
    args,
    input_numel: int,
) -> None:
    """Save results to JSON."""
    data = {
        'checkpoint': args.checkpoint,
        'mode': args.mode,
        'space': args.space,
        'num_steps': args.num_steps,
        'metric': args.metric,
        'ref_split': args.ref_split,
        'num_volumes': args.num_volumes,
        'seed': args.seed,
        'input_numel': input_numel,
        'evaluations': history,
    }
    with open(output_dir / "time_shift_results.json", 'w') as f:
        json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _run_smoke_test() -> None:
    """Verify ratio-to-base_numel computation logic."""
    logger.info("=== SMOKE TEST (no GPU needed) ===")

    # 3D volume: 256 x 256 x 160 = 10,485,760
    input_numel = 256 * 256 * 160
    spatial_dims = 3

    test_cases = [
        (1.0, input_numel),   # no shift: base = input
        (2.0, input_numel // 8),  # 2^3 = 8
        (4.0, input_numel // 64),  # 4^3 = 64
    ]

    for ratio, expected_base in test_cases:
        base_numel = int(input_numel / (ratio ** spatial_dims))
        actual_ratio = (input_numel / base_numel) ** (1.0 / spatial_dims)
        logger.info(
            f"  ratio={ratio:.1f}: base_numel={base_numel:,} "
            f"(expected={expected_base:,}), actual_ratio={actual_ratio:.4f}"
        )
        assert base_numel == expected_base, (
            f"ratio={ratio}: expected base={expected_base}, got {base_numel}"
        )

    # Verify MONAI default (32^3 = 32768)
    monai_base = 32 ** 3
    monai_ratio = (input_numel / monai_base) ** (1.0 / spatial_dims)
    logger.info(f"\n  MONAI default: base=32^3={monai_base:,}, ratio={monai_ratio:.2f}")
    assert abs(monai_ratio - 6.84) < 0.1, f"Expected ~6.84, got {monai_ratio:.2f}"

    # Verify reverse: ratio → base → ratio roundtrip
    for ratio in [1.0, 1.5, 2.0, 3.0, 4.0]:
        base = int(input_numel / (ratio ** spatial_dims))
        if base < 1:
            base = 1
        recovered = (input_numel / base) ** (1.0 / spatial_dims)
        logger.info(f"  roundtrip ratio={ratio:.1f} → base={base:,} → ratio={recovered:.4f}")

    logger.info("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
