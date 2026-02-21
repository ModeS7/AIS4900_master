#!/usr/bin/env python3
"""Find the optimal Euler step count for RFlow generation via golden section search.

Uses golden section search to find the step count that minimizes FID
within a given range. Much faster than exhaustive sweep — typically
8-10 evaluations to find the optimum within ±1 step.

Golden section search: at each iteration, evaluates one new point and
narrows the interval by factor ~0.618. Requires the objective to be
unimodal (single minimum) in the search range.

Supports all modes (seg, bravo, dual, multi, seg_conditioned, etc.)
with auto-detection from checkpoint config. Currently limited to UNet
checkpoints (transformer model loading requires full Hydra config).

Usage:
    # Auto-detect mode from checkpoint
    python -m medgen.scripts.find_optimal_steps \
        --checkpoint runs/checkpoint_latest.pt \
        --data-root ~/MedicalDataSets/brainmetshare-3 \
        --num-volumes 25 --output-dir eval_optimal_steps

    # Explicit mode override
    python -m medgen.scripts.find_optimal_steps \
        --checkpoint runs/checkpoint_seg.pt \
        --data-root ~/MedicalDataSets/brainmetshare-3 \
        --mode seg --num-volumes 25

    # Legacy --bravo-model still works
    python -m medgen.scripts.find_optimal_steps \
        --bravo-model runs/checkpoint_bravo.pt \
        --data-root ~/MedicalDataSets/brainmetshare-3

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
# Main
# ═══════════════════════════════════════════════════════════════════════════════

SEG_MODES = frozenset({'seg', 'seg_conditioned', 'seg_conditioned_input'})


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal Euler step count via golden section search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--checkpoint', '--bravo-model', default=None, dest='checkpoint',
                        help='Path to trained model checkpoint')
    parser.add_argument('--mode', default=None,
                        help='Generation mode (auto-detected from checkpoint if omitted)')
    parser.add_argument('--ref-modality', default=None,
                        help='Reference file modality, e.g. bravo or seg '
                             '(auto: bravo for image modes, seg for seg modes)')
    parser.add_argument('--data-root', default=None,
                        help='Root of dataset')
    parser.add_argument('--output-dir', default='eval_optimal_steps',
                        help='Output directory (default: eval_optimal_steps)')
    parser.add_argument('--num-volumes', type=int, default=25,
                        help='Volumes per evaluation (default: 25)')
    parser.add_argument('--lo', type=int, default=10,
                        help='Lower bound for step count search (default: 10)')
    parser.add_argument('--hi', type=int, default=50,
                        help='Upper bound for step count search (default: 50)')
    parser.add_argument('--tol', type=int, default=1,
                        help='Stop when interval width <= tol (default: 1)')
    parser.add_argument('--cond-split', default='val',
                        help='Split for conditioning seg masks (default: val)')
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--trim-slices', type=int, default=10)
    parser.add_argument('--fov-mm', type=float, default=240.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--metric', choices=['fid', 'kid', 'cmmd'], default='fid',
                        help='Metric to minimize (default: fid)')
    parser.add_argument('--ref-split', default='all',
                        help='Reference split for metric computation (default: all)')
    parser.add_argument('--save-volumes', action='store_true',
                        help='Save generated volumes as NIfTI (off by default to save disk)')
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
    in_channels = ckpt_cfg.get('in_channels')
    out_channels = ckpt_cfg.get('out_channels')
    mode = args.mode or ckpt_cfg.get('mode')
    # Hydra stores mode as a config group (dict with 'name' key), not a string
    if isinstance(mode, dict):
        mode = mode.get('name', None)
    spatial_dims = ckpt_cfg.get('spatial_dims', 3)
    image_size = args.image_size or ckpt_cfg.get('image_size', 256)
    depth = args.depth or ckpt_cfg.get('depth_size', 160)

    # Legacy Hydra config fallback
    if in_channels is None and hasattr(ckpt_cfg, 'model'):
        model_cfg = ckpt_cfg.model
        in_channels = getattr(model_cfg, 'in_channels', 2)
        out_channels = getattr(model_cfg, 'out_channels', 1)
        image_size = args.image_size or getattr(model_cfg, 'image_size', 256)
        depth = args.depth or getattr(model_cfg, 'depth_size', 160)

    # Defaults when nothing is available
    if in_channels is None:
        in_channels = 2
    if out_channels is None:
        out_channels = 1
    del ckpt

    # ── Derive mode-specific config ───────────────────────────────────────
    is_seg = mode in SEG_MODES if mode else False
    ref_modality = args.ref_modality or ('seg' if is_seg else 'bravo')
    cond_channels = in_channels - out_channels  # >0 means conditioning needed

    voxel_size = (args.fov_mm / image_size, args.fov_mm / image_size, 1.0)

    logger.info("=" * 70)
    logger.info("Optimal Euler Step Search (Golden Section)")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Mode: {mode or 'unknown'} (in_ch={in_channels}, out_ch={out_channels}, "
                f"cond_ch={cond_channels})")
    logger.info(f"Reference modality: {ref_modality}")
    logger.info(f"Seg mode: {is_seg}")
    logger.info(f"Volume: {image_size}x{image_size}x{depth} (spatial_dims={spatial_dims})")
    logger.info(f"Search range: [{args.lo}, {args.hi}]")
    logger.info(f"Volumes per eval: {args.num_volumes}")
    logger.info(f"Metric: {args.metric} (vs '{args.ref_split}')")
    logger.info(f"Expected evaluations: ~{int(math.log(args.hi - args.lo) / math.log(1/GR)) + 1}")
    logger.info("=" * 70)

    # ── Data setup ────────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    splits = discover_splits(data_root, modality=ref_modality)

    # ── Load conditioning (only if model expects it) ──────────────────────
    cond_list = None
    if cond_channels > 0:
        if args.cond_split not in splits:
            raise ValueError(
                f"Split '{args.cond_split}' not found. Available: {list(splits.keys())}"
            )
        logger.info(f"Loading {args.num_volumes} conditioning masks...")
        cond_list = load_conditioning(splits[args.cond_split], args.num_volumes, depth)

    logger.info("Preparing reference features...")
    cache_dir = output_dir / "reference_features"
    ref_features = get_or_cache_reference_features(
        splits, cache_dir, device, depth, args.trim_slices, image_size,
        modality=ref_modality,
    )

    # ── Load model ────────────────────────────────────────────────────────
    from medgen.diffusion import RFlowStrategy, load_diffusion_model

    logger.info("Loading model...")
    model = load_diffusion_model(
        args.checkpoint, device=device,
        in_channels=in_channels, out_channels=out_channels,
        compile_model=False, spatial_dims=spatial_dims,
    )

    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000, image_size=image_size,
        depth_size=depth, spatial_dims=spatial_dims,
    )

    # ── Pre-generate noise (shared across all evaluations) ────────────────
    logger.info(f"Pre-generating {args.num_volumes} noise tensors (seed={args.seed})...")
    noise_list = generate_noise_tensors(
        args.num_volumes, depth, image_size, device, args.seed,
        out_channels=out_channels,
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

        # Compute metrics
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
